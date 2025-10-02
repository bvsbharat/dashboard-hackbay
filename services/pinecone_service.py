"""
Pinecone Vector Database Service
Handles vector storage and similarity search
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


class PineconeService:
    """Service for managing vectors in Pinecone"""

    def __init__(self, api_key: Optional[str] = None, index_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "due-diligence")
        self.pc = None
        self.index = None

        if self.api_key:
            self._initialize()
        else:
            logger.warning("Pinecone API key not found")

    def _initialize(self):
        """Initialize Pinecone client and index"""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.api_key)

            # Check if index exists, create if not
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # text-embedding-3-small or text-embedding-ada-002 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Pinecone index created: {self.index_name}")

            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    async def upsert_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        """
        Upsert document chunks with embeddings to Pinecone

        Args:
            document_id: Unique document identifier
            chunks: List of chunks with text, embedding, chunk_index, metadata

        Returns:
            Success status
        """
        try:
            if not self.index:
                logger.error("Pinecone index not initialized")
                return False

            # Prepare vectors for upsert
            vectors = []
            for chunk in chunks:
                chunk_id = f"{document_id}_chunk_{chunk['chunk_index']}"

                # Metadata to store with vector
                metadata = {
                    "document_id": document_id,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"][:1000],  # Limit metadata text size
                    **chunk.get("metadata", {})
                }

                vectors.append({
                    "id": chunk_id,
                    "values": chunk["embedding"],
                    "metadata": metadata
                })

            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)

            logger.info(f"Upserted {len(vectors)} chunks to Pinecone for document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert chunks to Pinecone: {e}")
            return False

    async def query_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query similar chunks from Pinecone

        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of similar chunks with metadata and scores
        """
        try:
            if not self.index:
                logger.error("Pinecone index not initialized")
                return []

            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )

            # Format results
            chunks = []
            for match in results.matches:
                chunks.append({
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "document_id": match.metadata.get("document_id", ""),
                    "chunk_index": match.metadata.get("chunk_index", 0),
                    "metadata": match.metadata
                })

            logger.info(f"Retrieved {len(chunks)} similar chunks from Pinecone")
            return chunks

        except Exception as e:
            logger.error(f"Failed to query Pinecone: {e}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document from Pinecone

        Args:
            document_id: Document identifier

        Returns:
            Success status
        """
        try:
            if not self.index:
                logger.error("Pinecone index not initialized")
                return False

            # Delete by metadata filter
            self.index.delete(filter={"document_id": document_id})
            logger.info(f"Deleted chunks for document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document from Pinecone: {e}")
            return False

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index"""
        try:
            if not self.index:
                return {"error": "Index not initialized"}

            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}


# Singleton instance
_pinecone_instance = None


def get_pinecone_service() -> PineconeService:
    """Get or create Pinecone service instance"""
    global _pinecone_instance
    if _pinecone_instance is None:
        _pinecone_instance = PineconeService()
    return _pinecone_instance
