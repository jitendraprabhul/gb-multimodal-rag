"""
Named Entity Recognition (NER) extraction.

Supports multiple NER backends:
- spaCy (general purpose)
- SciSpaCy (healthcare/biomedical)
- Transformers (BERT-based NER)

Domain-specific entity normalization for finance and healthcare.
"""

import re
from abc import ABC, abstractmethod
from typing import Any

import spacy
from spacy.tokens import Doc

from src.core.exceptions import NERError
from src.core.logging import LoggerMixin
from src.core.types import Chunk, Entity, EntityType


class NERExtractor(ABC, LoggerMixin):
    """Abstract base class for NER extractors."""

    # Mapping from NER labels to our EntityType enum
    LABEL_MAP: dict[str, EntityType] = {}

    def __init__(self, **config: Any) -> None:
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize NER model."""
        pass

    @abstractmethod
    async def extract(self, text: str) -> list[Entity]:
        """Extract entities from text."""
        pass

    async def extract_from_chunk(self, chunk: Chunk) -> list[Entity]:
        """
        Extract entities from a chunk.

        Args:
            chunk: Document chunk

        Returns:
            List of extracted entities
        """
        entities = await self.extract(chunk.content)

        # Add chunk reference to entities
        for entity in entities:
            entity.source_chunk_ids.append(chunk.id)

        return entities

    async def extract_batch(self, texts: list[str]) -> list[list[Entity]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of entity lists (one per text)
        """
        results = []
        for text in texts:
            entities = await self.extract(text)
            results.append(entities)
        return results

    def normalize_entity(self, name: str, entity_type: EntityType) -> str:
        """
        Normalize entity name for deduplication.

        Override in subclasses for domain-specific normalization.
        """
        return name.lower().strip()

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._initialized = False


class SpaCyNERExtractor(NERExtractor):
    """
    NER extractor using spaCy.

    Supports en_core_web_trf for general-purpose NER.
    """

    LABEL_MAP = {
        "PERSON": EntityType.PERSON,
        "ORG": EntityType.ORGANIZATION,
        "GPE": EntityType.LOCATION,
        "LOC": EntityType.LOCATION,
        "DATE": EntityType.DATE,
        "MONEY": EntityType.MONEY,
        "PERCENT": EntityType.PERCENT,
        "QUANTITY": EntityType.QUANTITY,
        "CARDINAL": EntityType.QUANTITY,
        "ORDINAL": EntityType.QUANTITY,
    }

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.model_name = model_name
        self._nlp: spacy.Language | None = None

    async def initialize(self) -> None:
        """Load spaCy model."""
        if self._initialized:
            return

        try:
            self._nlp = spacy.load(self.model_name)
            self._initialized = True
            self.logger.info("spaCy model loaded", model=self.model_name)
        except OSError:
            # Try downloading the model
            self.logger.info("Downloading spaCy model", model=self.model_name)
            spacy.cli.download(self.model_name)
            self._nlp = spacy.load(self.model_name)
            self._initialized = True

    async def extract(self, text: str) -> list[Entity]:
        """Extract entities using spaCy."""
        if not self._initialized:
            await self.initialize()

        if not text or not text.strip():
            return []

        try:
            doc = self._nlp(text)
            entities = []

            for ent in doc.ents:
                entity_type = self.LABEL_MAP.get(ent.label_)
                if entity_type is None:
                    continue

                entity = Entity(
                    name=ent.text,
                    normalized_name=self.normalize_entity(ent.text, entity_type),
                    entity_type=entity_type,
                    attributes={
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "label": ent.label_,
                    },
                )
                entities.append(entity)

            return entities

        except Exception as e:
            raise NERError(
                f"NER extraction failed: {e}",
                model=self.model_name,
                text_length=len(text),
                cause=e,
            )


class FinanceNERExtractor(NERExtractor):
    """
    NER extractor specialized for finance domain.

    Combines spaCy NER with pattern-based extraction for:
    - Company names and tickers
    - Financial metrics
    - Filing types
    - Monetary values
    """

    LABEL_MAP = {
        **SpaCyNERExtractor.LABEL_MAP,
        "COMPANY": EntityType.COMPANY,
        "TICKER": EntityType.TICKER,
        "FILING": EntityType.FILING,
        "METRIC": EntityType.METRIC,
        "EVENT": EntityType.EVENT,
    }

    # Patterns for finance entities
    PATTERNS = {
        "ticker": r"\b[A-Z]{1,5}\b(?:\.[A-Z]{1,2})?",  # e.g., AAPL, BRK.A
        "filing": r"\b(10-K|10-Q|8-K|S-1|DEF 14A|13F|13D|13G)\b",
        "metric": r"\b(revenue|earnings|EPS|EBITDA|gross margin|net income|operating income|free cash flow|ROE|ROA|P/E|debt-to-equity)\b",
        "money": r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?\b",
        "percent": r"[\d.]+%",
    }

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        transformer_model: str | None = "dslim/bert-base-NER",
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.spacy_model = spacy_model
        self.transformer_model = transformer_model
        self._spacy_extractor: SpaCyNERExtractor | None = None
        self._transformer_pipeline = None

    async def initialize(self) -> None:
        """Initialize NER models."""
        if self._initialized:
            return

        # Initialize spaCy
        self._spacy_extractor = SpaCyNERExtractor(model_name=self.spacy_model)
        await self._spacy_extractor.initialize()

        # Initialize transformer model if specified
        if self.transformer_model:
            try:
                from transformers import pipeline

                self._transformer_pipeline = pipeline(
                    "ner",
                    model=self.transformer_model,
                    aggregation_strategy="simple",
                )
                self.logger.info(
                    "Transformer NER loaded",
                    model=self.transformer_model,
                )
            except Exception as e:
                self.logger.warning(
                    "Transformer NER unavailable",
                    error=str(e),
                )

        self._initialized = True

    async def extract(self, text: str) -> list[Entity]:
        """Extract finance entities from text."""
        if not self._initialized:
            await self.initialize()

        if not text or not text.strip():
            return []

        entities = []

        # Extract with spaCy
        spacy_entities = await self._spacy_extractor.extract(text)
        entities.extend(spacy_entities)

        # Extract with patterns
        pattern_entities = self._extract_patterns(text)
        entities.extend(pattern_entities)

        # Extract with transformer if available
        if self._transformer_pipeline:
            transformer_entities = self._extract_transformer(text)
            entities.extend(transformer_entities)

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        return entities

    def _extract_patterns(self, text: str) -> list[Entity]:
        """Extract entities using regex patterns."""
        entities = []

        # Ticker symbols
        for match in re.finditer(self.PATTERNS["ticker"], text):
            # Skip if it's a common word
            ticker = match.group()
            if ticker.lower() in {"a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we"}:
                continue

            entity = Entity(
                name=ticker,
                normalized_name=ticker.upper(),
                entity_type=EntityType.TICKER,
                attributes={"start_char": match.start(), "end_char": match.end()},
            )
            entities.append(entity)

        # Filing types
        for match in re.finditer(self.PATTERNS["filing"], text, re.IGNORECASE):
            entity = Entity(
                name=match.group(),
                normalized_name=match.group().upper(),
                entity_type=EntityType.FILING,
                attributes={"start_char": match.start(), "end_char": match.end()},
            )
            entities.append(entity)

        # Financial metrics
        for match in re.finditer(self.PATTERNS["metric"], text, re.IGNORECASE):
            entity = Entity(
                name=match.group(),
                normalized_name=match.group().lower().replace(" ", "_"),
                entity_type=EntityType.METRIC,
                attributes={"start_char": match.start(), "end_char": match.end()},
            )
            entities.append(entity)

        return entities

    def _extract_transformer(self, text: str) -> list[Entity]:
        """Extract entities using transformer model."""
        entities = []

        try:
            results = self._transformer_pipeline(text)

            for result in results:
                # Map transformer labels to our types
                label = result.get("entity_group", "")
                entity_type = self.LABEL_MAP.get(label)

                if entity_type is None:
                    continue

                entity = Entity(
                    name=result["word"],
                    normalized_name=result["word"].lower().strip(),
                    entity_type=entity_type,
                    confidence=result.get("score", 1.0),
                    attributes={
                        "start_char": result.get("start", 0),
                        "end_char": result.get("end", 0),
                    },
                )
                entities.append(entity)

        except Exception as e:
            self.logger.warning("Transformer extraction failed", error=str(e))

        return entities

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """Deduplicate entities by normalized name and type."""
        seen = {}

        for entity in entities:
            key = (entity.normalized_name, entity.entity_type)
            if key in seen:
                # Merge with existing
                existing = seen[key]
                existing.source_chunk_ids.extend(entity.source_chunk_ids)
                existing.confidence = max(existing.confidence, entity.confidence)
            else:
                seen[key] = entity

        return list(seen.values())

    def normalize_entity(self, name: str, entity_type: EntityType) -> str:
        """Normalize finance entity names."""
        name = name.strip()

        if entity_type == EntityType.TICKER:
            return name.upper()
        elif entity_type == EntityType.COMPANY:
            # Remove common suffixes
            for suffix in [" Inc.", " Inc", " Corp.", " Corp", " LLC", " Ltd.", " Ltd", " Co."]:
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
            return name.title()
        elif entity_type == EntityType.METRIC:
            return name.lower().replace(" ", "_")

        return name.lower()


class HealthcareNERExtractor(NERExtractor):
    """
    NER extractor specialized for healthcare/biomedical domain.

    Uses SciSpaCy models for:
    - Diseases and conditions
    - Drugs and chemicals
    - Genes and proteins
    - Anatomical terms
    """

    LABEL_MAP = {
        "DISEASE": EntityType.CONDITION,
        "CHEMICAL": EntityType.CHEMICAL,
        "DRUG": EntityType.DRUG,
        "GENE_OR_GENE_PRODUCT": EntityType.GENE,
        "CELL": EntityType.ANATOMY,
        "CELL_TYPE": EntityType.ANATOMY,
        "CELL_LINE": EntityType.ANATOMY,
        "ORGAN": EntityType.ANATOMY,
        "TISSUE": EntityType.ANATOMY,
        "SPECIES": EntityType.ORGANIZATION,
        # BC5CDR labels
        "Chemical": EntityType.CHEMICAL,
        "Disease": EntityType.CONDITION,
    }

    def __init__(
        self,
        model_name: str = "en_ner_bc5cdr_md",
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.model_name = model_name
        self._nlp: spacy.Language | None = None

    async def initialize(self) -> None:
        """Load SciSpaCy model."""
        if self._initialized:
            return

        try:
            self._nlp = spacy.load(self.model_name)
            self._initialized = True
            self.logger.info("SciSpaCy model loaded", model=self.model_name)
        except OSError:
            # Try alternative models
            alternative_models = [
                "en_core_sci_sm",
                "en_core_sci_md",
                "en_core_web_sm",
            ]

            for alt_model in alternative_models:
                try:
                    self._nlp = spacy.load(alt_model)
                    self.model_name = alt_model
                    self._initialized = True
                    self.logger.info(
                        "Loaded alternative model",
                        model=alt_model,
                    )
                    break
                except OSError:
                    continue

            if not self._initialized:
                raise NERError(
                    f"Failed to load SciSpaCy model: {self.model_name}",
                    model=self.model_name,
                )

    async def extract(self, text: str) -> list[Entity]:
        """Extract healthcare entities from text."""
        if not self._initialized:
            await self.initialize()

        if not text or not text.strip():
            return []

        try:
            doc = self._nlp(text)
            entities = []

            for ent in doc.ents:
                entity_type = self.LABEL_MAP.get(ent.label_)
                if entity_type is None:
                    continue

                entity = Entity(
                    name=ent.text,
                    normalized_name=self.normalize_entity(ent.text, entity_type),
                    entity_type=entity_type,
                    attributes={
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "label": ent.label_,
                    },
                )
                entities.append(entity)

            return entities

        except Exception as e:
            raise NERError(
                f"Healthcare NER extraction failed: {e}",
                model=self.model_name,
                text_length=len(text),
                cause=e,
            )

    def normalize_entity(self, name: str, entity_type: EntityType) -> str:
        """Normalize healthcare entity names."""
        name = name.strip().lower()

        # Remove common prefixes/suffixes
        if entity_type in (EntityType.CONDITION, EntityType.DRUG):
            # Handle plural forms
            if name.endswith("s") and not name.endswith("ss"):
                name = name[:-1]

        return name
