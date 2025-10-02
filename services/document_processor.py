"""
Document processing service using LandingAI ADE
Extracts structured data from financial documents with high accuracy
"""

import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from landingai_ade import LandingAIADE
    LANDINGAI_AVAILABLE = True
except ImportError:
    logger.warning("LandingAI ADE not available. Install with: pip install landingai-ade")
    LANDINGAI_AVAILABLE = False


class DocumentProcessor:
    """Process financial documents using LandingAI ADE"""

    def __init__(self, api_key: Optional[str] = None):
        # LandingAI official docs use VISION_AGENT_API_KEY
        self.api_key = api_key or os.getenv("VISION_AGENT_API_KEY") or os.getenv("LANDINGAI_API_KEY")
        self.client = None

        if LANDINGAI_AVAILABLE and self.api_key:
            try:
                self.client = LandingAIADE(apikey=self.api_key)
                logger.info("LandingAI ADE initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LandingAI ADE: {e}")
        elif not LANDINGAI_AVAILABLE:
            logger.error("LandingAI ADE library not installed. Run: pip install landingai-ade")
        elif not self.api_key:
            logger.error("VISION_AGENT_API_KEY not found in environment variables")

    async def extract_document_data(
        self,
        file_path: str,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from a document using LandingAI ADE

        Args:
            file_path: Path to the document
            schema: Optional extraction schema for specific fields

        Returns:
            Extracted data with metadata, tables, and coordinates

        Raises:
            Exception: If LandingAI is not available or extraction fails
        """
        if not self.client:
            error_msg = "LandingAI ADE not configured. Please set LANDINGAI_API_KEY environment variable."
            logger.error(error_msg)
            raise Exception(error_msg)

        try:
            # Use LandingAI DPT-2 for document parsing with timeout
            logger.info(f"Using LandingAI ADE to parse: {file_path}")

            import asyncio

            # Set 30 second timeout for LandingAI API
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.parse,
                    document_url=file_path,
                    model="dpt-2-latest"
                ),
                timeout=30.0
            )

            logger.info(f"LandingAI ADE parsing completed for: {file_path}")
            result = self._format_landingai_response(response, file_path)

            # If LandingAI returned no text, raise error
            if not result.get("data", {}).get("text", "").strip():
                error_msg = "LandingAI returned empty text. Document may be corrupted or unsupported format."
                logger.error(f"{error_msg} File: {file_path}")
                raise Exception(error_msg)

            return result

        except asyncio.TimeoutError:
            error_msg = "LandingAI API request timed out after 30 seconds"
            logger.error(f"{error_msg}. File: {file_path}")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"LandingAI extraction failed: {str(e)}"
            logger.error(f"{error_msg}. File: {file_path}")
            raise Exception(error_msg)

    def _format_landingai_response(
        self,
        response: Any,
        file_path: str
    ) -> Dict[str, Any]:
        """Format LandingAI ADE parse response"""

        # LandingAI API returns markdown text in response.markdown and response.chunks[].markdown
        text = ""
        tables = []

        # First try to get full document markdown
        if hasattr(response, "markdown") and response.markdown:
            text = response.markdown
            logger.info(f"LandingAI extracted {len(text)} chars from full markdown")

        # If no full markdown, try chunks
        elif hasattr(response, "chunks") and response.chunks:
            logger.info(f"LandingAI returned {len(response.chunks)} chunks")
            for i, chunk in enumerate(response.chunks):
                # Access chunk.markdown (not chunk.text)
                chunk_text = getattr(chunk, "markdown", "")
                if chunk_text:
                    text += chunk_text + "\n\n"
                    logger.debug(f"Chunk {i}: {len(chunk_text)} chars")
                else:
                    logger.warning(f"Chunk {i} has no markdown")

                # Extract tables if present
                if hasattr(chunk, "tables"):
                    for table in chunk.tables:
                        tables.append({
                            "data": getattr(table, "data", []),
                            "rows": getattr(table, "rows", None),
                            "columns": getattr(table, "columns", None),
                        })

        # Log result
        if not text.strip():
            logger.error(f"LandingAI returned no text for {file_path}")
            logger.error(f"Response attributes: {dir(response)}")
        else:
            logger.info(f"LandingAI successfully extracted {len(text)} chars from {file_path}")

        return {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "status": "success",
            "data": {
                "text": text.strip(),
                "tables": tables,
                "fields": {},
                "metadata": {
                    "pages": len(response.chunks) if hasattr(response, "chunks") else 1,
                    "coordinates": [],
                }
            },
            "extracted_metrics": {}
        }

    def _extract_tables(self, result: Any) -> List[Dict[str, Any]]:
        """Extract tables from document"""
        tables = []

        if hasattr(result, "tables"):
            for i, table in enumerate(result.tables):
                tables.append({
                    "table_id": i,
                    "rows": getattr(table, "rows", []),
                    "columns": getattr(table, "columns", []),
                    "data": getattr(table, "data", []),
                    "page": getattr(table, "page", 1),
                })

        return tables

    def _extract_financial_metrics(self, result: Any) -> Dict[str, Any]:
        """Extract common financial metrics from extracted data"""
        metrics = {
            "revenue": None,
            "expenses": None,
            "profit": None,
            "assets": None,
            "liabilities": None,
            "equity": None,
        }

        # This would use NLP/pattern matching on extracted text
        # For now, return placeholder structure
        return metrics


# Singleton instance
_processor_instance = None

def get_document_processor() -> DocumentProcessor:
    """Get or create document processor instance"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = DocumentProcessor()
    return _processor_instance
