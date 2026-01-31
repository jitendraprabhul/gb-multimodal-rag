"""
Relation extraction for knowledge graph construction.

Supports:
- Pattern-based extraction (rules and regex)
- LLM-based extraction (using local LLM)
- Co-occurrence based relations
"""

import re
from abc import ABC, abstractmethod
from typing import Any

from src.core.logging import LoggerMixin
from src.core.types import Chunk, Entity, EntityType, Relation, RelationType


class RelationExtractor(ABC, LoggerMixin):
    """Abstract base class for relation extractors."""

    def __init__(self, **config: Any) -> None:
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize relation extraction model."""
        pass

    @abstractmethod
    async def extract(
        self,
        text: str,
        entities: list[Entity],
    ) -> list[Relation]:
        """
        Extract relations from text given known entities.

        Args:
            text: Source text
            entities: Entities found in the text

        Returns:
            List of extracted relations
        """
        pass

    async def extract_from_chunk(
        self,
        chunk: Chunk,
        entities: list[Entity],
    ) -> list[Relation]:
        """
        Extract relations from a chunk.

        Args:
            chunk: Document chunk
            entities: Entities in the chunk

        Returns:
            List of relations
        """
        relations = await self.extract(chunk.content, entities)

        # Add chunk reference to relations
        for relation in relations:
            relation.source_chunk_ids.append(chunk.id)

        return relations

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._initialized = False


class PatternRelationExtractor(RelationExtractor):
    """
    Pattern-based relation extraction.

    Uses regex patterns and entity type constraints to
    identify relationships between entities.
    """

    # Finance domain patterns
    FINANCE_PATTERNS = [
        # Company filed document
        {
            "pattern": r"(\w+(?:\s+\w+)*)\s+filed\s+(?:an?\s+)?(\d{1,2}-[KQ]|8-K|S-1)",
            "source_types": [EntityType.COMPANY, EntityType.ORGANIZATION],
            "target_types": [EntityType.FILING],
            "relation": RelationType.FILED,
        },
        # Acquisition patterns
        {
            "pattern": r"(\w+(?:\s+\w+)*)\s+(?:acquired|bought|purchased)\s+(\w+(?:\s+\w+)*)",
            "source_types": [EntityType.COMPANY, EntityType.ORGANIZATION],
            "target_types": [EntityType.COMPANY, EntityType.ORGANIZATION],
            "relation": RelationType.ACQUIRED,
        },
        # Merger patterns
        {
            "pattern": r"(\w+(?:\s+\w+)*)\s+(?:merged with|merging with)\s+(\w+(?:\s+\w+)*)",
            "source_types": [EntityType.COMPANY, EntityType.ORGANIZATION],
            "target_types": [EntityType.COMPANY, EntityType.ORGANIZATION],
            "relation": RelationType.MERGED_WITH,
        },
        # Revenue/metric reporting
        {
            "pattern": r"(\w+(?:\s+\w+)*)\s+reported\s+(\w+(?:\s+\w+)*)\s+of\s+\$?[\d,]+",
            "source_types": [EntityType.COMPANY, EntityType.ORGANIZATION],
            "target_types": [EntityType.METRIC],
            "relation": RelationType.REPORTED,
        },
        # Subsidiary relationship
        {
            "pattern": r"(\w+(?:\s+\w+)*)\s+(?:is\s+a\s+)?subsidiary\s+of\s+(\w+(?:\s+\w+)*)",
            "source_types": [EntityType.COMPANY, EntityType.ORGANIZATION],
            "target_types": [EntityType.COMPANY, EntityType.ORGANIZATION],
            "relation": RelationType.SUBSIDIARY_OF,
        },
    ]

    # Healthcare domain patterns
    HEALTHCARE_PATTERNS = [
        # Drug treats condition
        {
            "pattern": r"(\w+(?:\s+\w+)*)\s+(?:treats|treating|treatment for)\s+(\w+(?:\s+\w+)*)",
            "source_types": [EntityType.DRUG, EntityType.CHEMICAL],
            "target_types": [EntityType.CONDITION],
            "relation": RelationType.TREATED_WITH,
            "reverse": True,  # Condition treated with drug
        },
        # Patient diagnosed with condition
        {
            "pattern": r"(?:patient|subject)\s+(?:was\s+)?(?:diagnosed with|has)\s+(\w+(?:\s+\w+)*)",
            "source_types": [EntityType.PATIENT],
            "target_types": [EntityType.CONDITION],
            "relation": RelationType.HAS_CONDITION,
        },
        # Drug prescribed for condition
        {
            "pattern": r"(\w+(?:\s+\w+)*)\s+(?:prescribed|recommended)\s+for\s+(\w+(?:\s+\w+)*)",
            "source_types": [EntityType.DRUG],
            "target_types": [EntityType.CONDITION],
            "relation": RelationType.PRESCRIBED,
        },
        # Drug interaction
        {
            "pattern": r"(\w+(?:\s+\w+)*)\s+(?:interacts with|interaction with)\s+(\w+(?:\s+\w+)*)",
            "source_types": [EntityType.DRUG, EntityType.CHEMICAL],
            "target_types": [EntityType.DRUG, EntityType.CHEMICAL],
            "relation": RelationType.INTERACTS_WITH,
        },
        # Contraindication
        {
            "pattern": r"(\w+(?:\s+\w+)*)\s+(?:contraindicated|should not be used)\s+(?:in|with)\s+(\w+(?:\s+\w+)*)",
            "source_types": [EntityType.DRUG],
            "target_types": [EntityType.CONDITION],
            "relation": RelationType.CONTRAINDICATES,
        },
        # Gene affects condition
        {
            "pattern": r"(\w+(?:\s+\w+)*)\s+(?:gene|mutation)\s+(?:associated with|causes|affects)\s+(\w+(?:\s+\w+)*)",
            "source_types": [EntityType.GENE],
            "target_types": [EntityType.CONDITION],
            "relation": RelationType.AFFECTS,
        },
    ]

    def __init__(
        self,
        domain: str = "finance",
        min_confidence: float = 0.5,
        **config: Any,
    ) -> None:
        """
        Initialize pattern-based relation extractor.

        Args:
            domain: Domain ("finance" or "healthcare")
            min_confidence: Minimum confidence threshold
            **config: Additional configuration
        """
        super().__init__(**config)
        self.domain = domain
        self.min_confidence = min_confidence
        self.patterns = (
            self.FINANCE_PATTERNS if domain == "finance" else self.HEALTHCARE_PATTERNS
        )

    async def initialize(self) -> None:
        """Initialize extractor."""
        self._initialized = True

    async def extract(
        self,
        text: str,
        entities: list[Entity],
    ) -> list[Relation]:
        """
        Extract relations using patterns.

        Args:
            text: Source text
            entities: Entities in the text

        Returns:
            List of relations
        """
        if not self._initialized:
            await self.initialize()

        relations = []

        # Build entity lookup by name variants
        entity_lookup = self._build_entity_lookup(entities)

        # Apply patterns
        for pattern_config in self.patterns:
            pattern = pattern_config["pattern"]
            source_types = pattern_config["source_types"]
            target_types = pattern_config["target_types"]
            relation_type = pattern_config["relation"]
            reverse = pattern_config.get("reverse", False)

            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) < 2:
                    continue

                source_text = groups[0].strip()
                target_text = groups[1].strip() if len(groups) > 1 else ""

                # Find matching entities
                source_entity = self._find_entity(
                    source_text, entity_lookup, source_types
                )
                target_entity = self._find_entity(
                    target_text, entity_lookup, target_types
                )

                if source_entity and target_entity:
                    if reverse:
                        source_entity, target_entity = target_entity, source_entity

                    relation = Relation(
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        relation_type=relation_type,
                        confidence=0.8,
                        attributes={
                            "pattern": pattern,
                            "match": match.group(),
                        },
                    )
                    relations.append(relation)

        # Add co-occurrence based relations
        cooccurrence_relations = self._extract_cooccurrence(entities)
        relations.extend(cooccurrence_relations)

        return relations

    def _build_entity_lookup(
        self,
        entities: list[Entity],
    ) -> dict[str, Entity]:
        """Build lookup dict for entities by name variants."""
        lookup = {}

        for entity in entities:
            # Add original name
            lookup[entity.name.lower()] = entity
            lookup[entity.normalized_name] = entity

            # Add without common suffixes
            name = entity.name.lower()
            for suffix in [" inc", " corp", " llc", " ltd", " co"]:
                if name.endswith(suffix):
                    lookup[name[: -len(suffix)]] = entity

        return lookup

    def _find_entity(
        self,
        text: str,
        lookup: dict[str, Entity],
        allowed_types: list[EntityType],
    ) -> Entity | None:
        """Find entity matching text and type constraints."""
        text_lower = text.lower()

        # Direct match
        entity = lookup.get(text_lower)
        if entity and entity.entity_type in allowed_types:
            return entity

        # Partial match
        for key, entity in lookup.items():
            if entity.entity_type not in allowed_types:
                continue

            if text_lower in key or key in text_lower:
                return entity

        return None

    def _extract_cooccurrence(
        self,
        entities: list[Entity],
    ) -> list[Relation]:
        """
        Extract relations based on entity co-occurrence.

        Entities appearing in the same context are likely related.
        """
        relations = []

        # Define meaningful co-occurrence pairs by type
        cooccurrence_rules = [
            (EntityType.COMPANY, EntityType.FILING, RelationType.FILED),
            (EntityType.COMPANY, EntityType.METRIC, RelationType.MENTIONS_METRIC),
            (EntityType.CONDITION, EntityType.DRUG, RelationType.TREATED_WITH),
            (EntityType.PATIENT, EntityType.CONDITION, RelationType.HAS_CONDITION),
        ]

        # Group entities by type
        entities_by_type: dict[EntityType, list[Entity]] = {}
        for entity in entities:
            if entity.entity_type not in entities_by_type:
                entities_by_type[entity.entity_type] = []
            entities_by_type[entity.entity_type].append(entity)

        # Check co-occurrence rules
        for source_type, target_type, relation_type in cooccurrence_rules:
            source_entities = entities_by_type.get(source_type, [])
            target_entities = entities_by_type.get(target_type, [])

            for source in source_entities:
                for target in target_entities:
                    if source.id == target.id:
                        continue

                    # Check if entities share chunks
                    shared_chunks = set(source.source_chunk_ids) & set(
                        target.source_chunk_ids
                    )

                    if shared_chunks:
                        relation = Relation(
                            source_entity_id=source.id,
                            target_entity_id=target.id,
                            relation_type=relation_type,
                            confidence=0.5,  # Lower confidence for co-occurrence
                            attributes={
                                "extraction_method": "cooccurrence",
                                "shared_chunks": list(shared_chunks),
                            },
                        )
                        relations.append(relation)

        return relations


class LLMRelationExtractor(RelationExtractor):
    """
    LLM-based relation extraction.

    Uses a local LLM (via Ollama) to extract relationships
    from text, given entity context.
    """

    EXTRACTION_PROMPT = """Extract relationships between the given entities from the text.

Text: {text}

Entities:
{entities}

For each relationship found, output in the format:
RELATION: <source_entity> -> <relation_type> -> <target_entity>

Valid relation types: {relation_types}

Only output relationships that are explicitly stated or strongly implied in the text.
If no relationships are found, output: NO_RELATIONS

Output:"""

    def __init__(
        self,
        ollama_client: Any = None,
        model: str = "mistral:7b-instruct",
        domain: str = "finance",
        **config: Any,
    ) -> None:
        """
        Initialize LLM relation extractor.

        Args:
            ollama_client: Ollama client instance
            model: LLM model to use
            domain: Domain for relation types
            **config: Additional configuration
        """
        super().__init__(**config)
        self.ollama_client = ollama_client
        self.model = model
        self.domain = domain

        # Get valid relation types for domain
        if domain == "finance":
            self.valid_relations = [
                RelationType.FILED,
                RelationType.ACQUIRED,
                RelationType.MERGED_WITH,
                RelationType.SUBSIDIARY_OF,
                RelationType.REPORTED,
                RelationType.MENTIONS_METRIC,
            ]
        else:
            self.valid_relations = [
                RelationType.HAS_CONDITION,
                RelationType.TREATED_WITH,
                RelationType.PRESCRIBED,
                RelationType.CONTRAINDICATES,
                RelationType.INTERACTS_WITH,
                RelationType.AFFECTS,
            ]

    async def initialize(self) -> None:
        """Initialize LLM client."""
        if self.ollama_client is None:
            from src.llm.ollama_client import OllamaClient

            self.ollama_client = OllamaClient()
            await self.ollama_client.initialize()

        self._initialized = True

    async def extract(
        self,
        text: str,
        entities: list[Entity],
    ) -> list[Relation]:
        """
        Extract relations using LLM.

        Args:
            text: Source text
            entities: Entities in the text

        Returns:
            List of relations
        """
        if not self._initialized:
            await self.initialize()

        if not entities or len(entities) < 2:
            return []

        # Build entity string
        entity_str = "\n".join(
            f"- {e.name} ({e.entity_type.value})" for e in entities
        )

        # Build relation types string
        relation_str = ", ".join(r.value for r in self.valid_relations)

        # Create prompt
        prompt = self.EXTRACTION_PROMPT.format(
            text=text[:2000],  # Limit text length
            entities=entity_str,
            relation_types=relation_str,
        )

        try:
            # Get LLM response
            response = await self.ollama_client.generate(
                prompt=prompt,
                max_tokens=500,
            )

            # Parse response
            relations = self._parse_llm_response(response, entities)
            return relations

        except Exception as e:
            self.logger.warning("LLM relation extraction failed", error=str(e))
            return []

    def _parse_llm_response(
        self,
        response: str,
        entities: list[Entity],
    ) -> list[Relation]:
        """Parse LLM response into relations."""
        relations = []

        if "NO_RELATIONS" in response.upper():
            return []

        # Build entity name to ID mapping
        entity_map = {e.name.lower(): e for e in entities}
        entity_map.update({e.normalized_name: e for e in entities})

        # Parse RELATION: lines
        relation_pattern = r"RELATION:\s*(.+?)\s*->\s*(\w+)\s*->\s*(.+)"

        for match in re.finditer(relation_pattern, response, re.IGNORECASE):
            source_name = match.group(1).strip().lower()
            relation_name = match.group(2).strip().upper()
            target_name = match.group(3).strip().lower()

            # Find entities
            source_entity = entity_map.get(source_name)
            target_entity = entity_map.get(target_name)

            if not source_entity or not target_entity:
                continue

            # Map relation type
            try:
                relation_type = RelationType(relation_name)
            except ValueError:
                # Try to find closest match
                for rt in self.valid_relations:
                    if rt.value.upper() == relation_name:
                        relation_type = rt
                        break
                else:
                    continue

            relation = Relation(
                source_entity_id=source_entity.id,
                target_entity_id=target_entity.id,
                relation_type=relation_type,
                confidence=0.7,  # LLM extraction confidence
                attributes={
                    "extraction_method": "llm",
                    "model": self.model,
                },
            )
            relations.append(relation)

        return relations
