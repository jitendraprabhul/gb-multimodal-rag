"""
Table extraction from various sources.

Supports:
- PDF tables (via Camelot/Tabula)
- Excel/CSV files
- HTML tables
- Image-based tables (via OCR)
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src.core.exceptions import DocumentProcessingError
from src.core.types import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentType,
    Modality,
    Table,
)
from src.etl.base import BaseProcessor


class TableExtractor(BaseProcessor[list[Table]]):
    """
    Extracts tables from documents and converts them to structured format.

    Uses Camelot for PDF tables and pandas for spreadsheets.
    """

    supported_types = [DocumentType.CSV, DocumentType.XLSX]

    def __init__(
        self,
        max_rows: int = 1000,
        max_columns: int = 50,
        **config: Any,
    ) -> None:
        """
        Initialize table extractor.

        Args:
            max_rows: Maximum rows to extract per table
            max_columns: Maximum columns to extract per table
            **config: Additional configuration
        """
        super().__init__(**config)
        self.max_rows = max_rows
        self.max_columns = max_columns

    async def process(self, file_path: Path, **kwargs: Any) -> list[Table]:
        """
        Extract tables from a file.

        Args:
            file_path: Path to file
            **kwargs: Additional options

        Returns:
            List of extracted tables
        """
        self.validate_file(file_path)
        self.logger.info("Extracting tables", file=str(file_path))

        try:
            suffix = file_path.suffix.lower()

            if suffix == ".csv":
                tables = await self._extract_csv(file_path)
            elif suffix in (".xlsx", ".xls"):
                tables = await self._extract_excel(file_path)
            else:
                raise DocumentProcessingError(
                    f"Unsupported table format: {suffix}",
                    document_path=str(file_path),
                )

            self.logger.info(
                "Tables extracted",
                file=str(file_path),
                count=len(tables),
            )
            return tables

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to extract tables: {e}",
                document_path=str(file_path),
                cause=e,
            )

    async def _extract_csv(self, file_path: Path) -> list[Table]:
        """Extract table from CSV file."""
        try:
            df = pd.read_csv(file_path, nrows=self.max_rows)
            df = df.iloc[:, : self.max_columns]

            return [self._dataframe_to_table(df)]
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to read CSV: {e}",
                document_path=str(file_path),
                cause=e,
            )

    async def _extract_excel(self, file_path: Path) -> list[Table]:
        """Extract tables from Excel file (all sheets)."""
        tables = []

        try:
            xlsx = pd.ExcelFile(file_path)

            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(
                    xlsx,
                    sheet_name=sheet_name,
                    nrows=self.max_rows,
                )
                df = df.iloc[:, : self.max_columns]

                table = self._dataframe_to_table(df)
                table.caption = f"Sheet: {sheet_name}"
                tables.append(table)

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to read Excel: {e}",
                document_path=str(file_path),
                cause=e,
            )

        return tables

    def _dataframe_to_table(self, df: pd.DataFrame) -> Table:
        """Convert pandas DataFrame to Table."""
        # Clean up column names
        headers = [str(col).strip() for col in df.columns]

        # Convert rows, handling NaN values
        rows = []
        for _, row in df.iterrows():
            row_data = []
            for val in row:
                if pd.isna(val):
                    row_data.append("")
                else:
                    row_data.append(str(val).strip())
            rows.append(row_data)

        return Table(headers=headers, rows=rows)

    async def extract_pdf_tables(
        self,
        file_path: Path,
        pages: str = "all",
        flavor: str = "lattice",
    ) -> list[Table]:
        """
        Extract tables from PDF using Camelot.

        Args:
            file_path: Path to PDF
            pages: Page specification (e.g., "1,3-5" or "all")
            flavor: Extraction method ("lattice" or "stream")

        Returns:
            List of extracted tables
        """
        try:
            import camelot
        except ImportError:
            self.logger.warning("Camelot not available, skipping PDF table extraction")
            return []

        self.validate_file(file_path)
        tables = []

        try:
            # Extract tables with Camelot
            camelot_tables = camelot.read_pdf(
                str(file_path),
                pages=pages,
                flavor=flavor,
            )

            for i, ct in enumerate(camelot_tables):
                df = ct.df

                # Skip empty or very small tables
                if df.empty or len(df) < 2:
                    continue

                # Use first row as header if it looks like headers
                if self._looks_like_header(df.iloc[0]):
                    headers = [str(v) for v in df.iloc[0]]
                    rows = [[str(v) for v in row] for row in df.iloc[1:].values]
                else:
                    headers = [f"Col_{i}" for i in range(len(df.columns))]
                    rows = [[str(v) for v in row] for row in df.values]

                table = Table(
                    headers=headers,
                    rows=rows,
                    page_number=ct.page,
                )
                tables.append(table)

        except Exception as e:
            self.logger.warning(
                "Camelot extraction failed, trying Tabula",
                error=str(e),
            )
            tables = await self._extract_pdf_tabula(file_path, pages)

        return tables

    async def _extract_pdf_tabula(
        self,
        file_path: Path,
        pages: str = "all",
    ) -> list[Table]:
        """Fallback PDF table extraction using Tabula."""
        try:
            import tabula
        except ImportError:
            self.logger.warning("Tabula not available")
            return []

        tables = []

        try:
            dfs = tabula.read_pdf(
                str(file_path),
                pages=pages,
                multiple_tables=True,
            )

            for i, df in enumerate(dfs):
                if df.empty or len(df) < 2:
                    continue

                table = self._dataframe_to_table(df)
                tables.append(table)

        except Exception as e:
            self.logger.warning("Tabula extraction failed", error=str(e))

        return tables

    def _looks_like_header(self, row: pd.Series) -> bool:
        """Check if a row looks like a header row."""
        # Headers typically have more text and fewer numbers
        text_count = sum(1 for v in row if isinstance(v, str) and v.strip())
        return text_count > len(row) * 0.5


class SpreadsheetProcessor(BaseProcessor[Document]):
    """
    Processor for spreadsheet files (CSV, Excel).

    Converts spreadsheets to Document with table chunks.
    """

    supported_types = [DocumentType.CSV, DocumentType.XLSX]

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self.table_extractor = TableExtractor(**config)

    async def initialize(self) -> None:
        await super().initialize()
        await self.table_extractor.initialize()

    async def process(self, file_path: Path, **kwargs: Any) -> Document:
        """Process a spreadsheet file."""
        self.validate_file(file_path)

        try:
            tables = await self.table_extractor.process(file_path)

            # Convert tables to chunks
            chunks = []
            for i, table in enumerate(tables):
                # Create text representation of table
                text_content = table.as_text
                if not text_content:
                    continue

                chunk = Chunk(
                    content=text_content,
                    modality=Modality.TABLE,
                    metadata=ChunkMetadata(
                        doc_id="",
                        table_id=table.id,
                        source_file=str(file_path),
                        section=table.caption,
                    ),
                )
                chunks.append(chunk)

            suffix = file_path.suffix.lower()
            doc_type = DocumentType.CSV if suffix == ".csv" else DocumentType.XLSX

            return Document(
                filename=file_path.name,
                doc_type=doc_type,
                chunks=chunks,
                metadata={
                    "table_count": len(tables),
                    "tables": [t.model_dump() for t in tables],
                },
            )

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process spreadsheet: {e}",
                document_path=str(file_path),
                cause=e,
            )
