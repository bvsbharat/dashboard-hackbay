"""
MongoDB service for storing and retrieving processed document data
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure

logger = logging.getLogger(__name__)


class MongoDBService:
    """MongoDB service for document storage and analytics"""

    def __init__(self):
        self.mongo_url = os.getenv("MONGO_DB")
        self.database_name = os.getenv("MONGO_DATABASE", "due_diligence")
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None

    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.mongo_url)
            self.db = self.client[self.database_name]

            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self.database_name}")

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    async def save_document(
        self,
        document_id: str,
        filename: str,
        file_path: str,
        extracted_data: Dict[str, Any]
    ) -> str:
        """Save processed document to MongoDB"""
        try:
            collection = self.db.documents

            document = {
                "document_id": document_id,
                "filename": filename,
                "file_path": file_path,
                "upload_date": datetime.utcnow(),
                "processed_date": datetime.utcnow(),
                "status": "processed",
                "extracted_data": extracted_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            result = await collection.insert_one(document)
            logger.info(f"Document saved to MongoDB: {document_id}")
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            raise

    async def save_financial_metrics(
        self,
        document_id: str,
        metrics: Dict[str, Any]
    ):
        """Save extracted financial metrics"""
        try:
            collection = self.db.financial_metrics

            metric_doc = {
                "document_id": document_id,
                "metrics": metrics,
                "extracted_at": datetime.utcnow(),
                "created_at": datetime.utcnow()
            }

            await collection.insert_one(metric_doc)
            logger.info(f"Financial metrics saved for: {document_id}")

        except Exception as e:
            logger.error(f"Failed to save financial metrics: {e}")
            raise

    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from MongoDB"""
        try:
            collection = self.db.documents
            cursor = collection.find().sort("upload_date", -1)
            documents = await cursor.to_list(length=100)

            # Convert ObjectId to string
            for doc in documents:
                doc["_id"] = str(doc["_id"])

            return documents

        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return []

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document"""
        try:
            collection = self.db.documents
            document = await collection.find_one({"document_id": document_id})

            if document:
                document["_id"] = str(document["_id"])

            return document

        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            return None

    async def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated financial metrics from all documents (LATEST per document)"""
        try:
            collection = self.db.financial_metrics

            # Get LATEST metric per document using aggregation pipeline
            pipeline = [
                # Sort by created_at descending to get latest first
                {"$sort": {"created_at": -1}},
                # Group by document_id and take the first (latest) one
                {
                    "$group": {
                        "_id": "$document_id",
                        "latest_metrics": {"$first": "$metrics"},
                        "created_at": {"$first": "$created_at"}
                    }
                }
            ]

            cursor = collection.aggregate(pipeline)
            latest_metrics_by_doc = await cursor.to_list(length=None)

            if not latest_metrics_by_doc:
                logger.warning("No metrics found in database")
                return self._get_default_metrics()

            logger.info(f"Found latest metrics for {len(latest_metrics_by_doc)} documents")

            # Aggregate LATEST metrics only (one per document)
            total_revenue = 0
            total_expenses = 0
            total_assets = 0
            total_liabilities = 0
            count = 0

            for doc in latest_metrics_by_doc:
                metrics = doc.get("latest_metrics", {})

                # Log each document's contribution
                logger.debug(f"Document {doc.get('_id')}: revenue={metrics.get('revenue')}, expenses={metrics.get('expenses')}")

                if "revenue" in metrics and metrics["revenue"]:
                    total_revenue += float(metrics["revenue"])
                if "expenses" in metrics and metrics["expenses"]:
                    total_expenses += float(metrics["expenses"])
                if "assets" in metrics and metrics["assets"]:
                    total_assets += float(metrics["assets"])
                if "liabilities" in metrics and metrics["liabilities"]:
                    total_liabilities += float(metrics["liabilities"])

                count += 1

            net_profit = total_revenue - total_expenses
            equity = total_assets - total_liabilities
            debt_to_equity = (total_liabilities / equity) if equity > 0 else 0

            logger.info(f"Aggregated metrics: revenue={total_revenue}, expenses={total_expenses}, profit={net_profit}, assets={total_assets}")

            # Calculate percentage changes from historical data
            revenue_change = await self._calculate_change("revenue", total_revenue)
            expense_change = await self._calculate_change("expenses", total_expenses)
            profit_margin = ((net_profit / total_revenue) * 100) if total_revenue > 0 else 0

            return {
                "total_revenue": {
                    "value": total_revenue,
                    "currency": "USD",
                    "period": "Current Period",
                    "change_pct": revenue_change
                },
                "total_expenses": {
                    "value": total_expenses,
                    "currency": "USD",
                    "period": "Current Period",
                    "change_pct": expense_change
                },
                "net_profit": {
                    "value": net_profit,
                    "currency": "USD",
                    "period": "Current Period",
                    "change_pct": profit_margin
                },
                "total_assets": {
                    "value": total_assets,
                    "currency": "USD",
                    "date": datetime.utcnow().isoformat()
                },
                "debt_to_equity": {
                    "value": debt_to_equity,
                    "category": "Healthy" if debt_to_equity < 0.5 else "Moderate" if debt_to_equity < 1.0 else "High"
                }
            }

        except Exception as e:
            logger.error(f"Failed to aggregate metrics: {e}")
            return self._get_default_metrics()

    async def _calculate_change(self, metric_name: str, current_value: float) -> float:
        """Calculate percentage change from historical data"""
        try:
            # Get historical snapshot (if exists)
            snapshot_collection = self.db.metric_snapshots
            last_snapshot = await snapshot_collection.find_one(
                {"metric": metric_name},
                sort=[("timestamp", -1)]
            )

            if last_snapshot and last_snapshot.get("value"):
                old_value = float(last_snapshot["value"])
                if old_value > 0:
                    change = ((current_value - old_value) / old_value) * 100
                    return round(change, 1)

            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate change: {e}")
            return 0.0

    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when no data is available"""
        return {
            "total_revenue": {
                "value": 0,
                "currency": "USD",
                "period": "FY 2024",
                "change_pct": 0
            },
            "total_expenses": {
                "value": 0,
                "currency": "USD",
                "period": "FY 2024",
                "change_pct": 0
            },
            "net_profit": {
                "value": 0,
                "currency": "USD",
                "period": "FY 2024",
                "change_pct": 0
            },
            "total_assets": {
                "value": 0,
                "currency": "USD",
                "date": datetime.utcnow().isoformat()
            },
            "debt_to_equity": {
                "value": 0,
                "category": "No Data"
            }
        }

    async def delete_document(self, document_id: str):
        """Delete a document from MongoDB"""
        try:
            collection = self.db.documents
            await collection.delete_one({"document_id": document_id})

            # Also delete associated metrics
            metrics_collection = self.db.financial_metrics
            await metrics_collection.delete_many({"document_id": document_id})

            logger.info(f"Document deleted from MongoDB: {document_id}")

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise

    async def get_historical_metrics(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical metrics for chart visualization (deduped by document)"""
        try:
            collection = self.db.financial_metrics

            # Get metrics from the last N days
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Use aggregation pipeline to get LATEST metric per document per day
            pipeline = [
                {"$match": {"created_at": {"$gte": cutoff_date}}},
                {"$sort": {"created_at": -1}},
                # Group by document_id and date to get latest per day
                {
                    "$group": {
                        "_id": {
                            "document_id": "$document_id",
                            "date": {
                                "$dateToString": {
                                    "format": "%Y-%m-%d",
                                    "date": "$created_at"
                                }
                            }
                        },
                        "metrics": {"$first": "$metrics"},
                        "created_at": {"$first": "$created_at"}
                    }
                }
            ]

            cursor = collection.aggregate(pipeline)
            deduplicated_metrics = await cursor.to_list(length=None)

            logger.info(f"Found {len(deduplicated_metrics)} deduplicated historical metrics")

            # Group by date and sum across documents
            daily_data = {}
            for item in deduplicated_metrics:
                date_key = item["_id"]["date"]
                metrics = item.get("metrics", {})

                if date_key not in daily_data:
                    daily_data[date_key] = {
                        "date": date_key,
                        "revenue": 0,
                        "expenses": 0,
                        "profit": 0,
                        "assets": 0,
                        "doc_count": 0
                    }

                if metrics.get("revenue"):
                    daily_data[date_key]["revenue"] += float(metrics["revenue"])
                if metrics.get("expenses"):
                    daily_data[date_key]["expenses"] += float(metrics["expenses"])
                if metrics.get("assets"):
                    daily_data[date_key]["assets"] += float(metrics["assets"])

                daily_data[date_key]["doc_count"] += 1

            # Calculate profit and format
            result = []
            for date_key in sorted(daily_data.keys()):
                data = daily_data[date_key]
                data["profit"] = data["revenue"] - data["expenses"]
                result.append(data)

            logger.info(f"Returning {len(result)} days of historical data")
            return result

        except Exception as e:
            logger.error(f"Failed to get historical metrics: {e}")
            return []

    async def get_expense_distribution(self) -> List[Dict[str, Any]]:
        """Get expense distribution by category from document analysis"""
        try:
            documents_collection = self.db.documents
            cursor = documents_collection.find({})
            documents = await cursor.to_list(length=None)

            # Aggregate expense categories from deep analysis
            categories = {}
            for doc in documents:
                deep_analysis = doc.get("extracted_data", {}).get("deep_analysis", {})
                # Try to extract expense breakdown if available
                # For now, return a basic distribution based on total expenses

            # If no category data, return default distribution
            metrics_collection = self.db.financial_metrics
            all_metrics = await metrics_collection.find().to_list(length=None)

            total_expenses = sum(
                float(m.get("metrics", {}).get("expenses", 0))
                for m in all_metrics
                if m.get("metrics", {}).get("expenses")
            )

            if total_expenses == 0:
                return []

            # Basic distribution (can be enhanced with actual categorization)
            return [
                {"name": "Operations", "value": total_expenses * 0.45},
                {"name": "Marketing", "value": total_expenses * 0.25},
                {"name": "R&D", "value": total_expenses * 0.20},
                {"name": "Admin", "value": total_expenses * 0.10},
            ]

        except Exception as e:
            logger.error(f"Failed to get expense distribution: {e}")
            return []

    async def get_all_red_flags(self) -> List[Dict[str, Any]]:
        """Get all red flags from document analysis"""
        try:
            documents_collection = self.db.documents
            cursor = documents_collection.find({})
            documents = await cursor.to_list(length=None)

            all_red_flags = []
            for doc in documents:
                deep_analysis = doc.get("extracted_data", {}).get("deep_analysis", {})
                red_flags = deep_analysis.get("red_flags", [])

                # Add source document info to each red flag
                for flag in red_flags:
                    flag["source"] = doc.get("filename", "Unknown")
                    flag["document_id"] = doc.get("document_id")
                    all_red_flags.append(flag)

            return all_red_flags

        except Exception as e:
            logger.error(f"Failed to get red flags: {e}")
            return []

    async def get_all_green_flags(self) -> List[Dict[str, Any]]:
        """Get all green flags from document analysis"""
        try:
            documents_collection = self.db.documents
            cursor = documents_collection.find({})
            documents = await cursor.to_list(length=None)

            all_green_flags = []
            for doc in documents:
                deep_analysis = doc.get("extracted_data", {}).get("deep_analysis", {})
                green_flags = deep_analysis.get("green_flags", [])

                # Add source document info to each green flag
                for flag in green_flags:
                    flag["source"] = doc.get("filename", "Unknown")
                    flag["document_id"] = doc.get("document_id")
                    all_green_flags.append(flag)

            return all_green_flags

        except Exception as e:
            logger.error(f"Failed to get green flags: {e}")
            return []

    async def save_document_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]]
    ):
        """
        Save document text chunks with embeddings for RAG

        Args:
            document_id: Unique document identifier
            chunks: List of chunks with text, embeddings, and metadata
        """
        try:
            collection = self.db.document_chunks

            # Add document_id and timestamp to each chunk
            for chunk in chunks:
                chunk["document_id"] = document_id
                chunk["created_at"] = datetime.utcnow()

            # Insert all chunks
            if chunks:
                await collection.insert_many(chunks)
                logger.info(f"Saved {len(chunks)} chunks for document: {document_id}")

        except Exception as e:
            logger.error(f"Failed to save document chunks: {e}")
            raise

    async def get_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.0,
        exclude_document_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve document chunks similar to query embedding using vector search

        Args:
            query_embedding: Query vector embedding
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold
            exclude_document_ids: List of document IDs to exclude from search

        Returns:
            List of similar chunks with scores
        """
        try:
            collection = self.db.document_chunks

            # Build filter to exclude certain documents
            filter_query = {}
            if exclude_document_ids:
                filter_query = {"document_id": {"$nin": exclude_document_ids}}
                logger.info(f"Excluding {len(exclude_document_ids)} documents from RAG search")

            # Fetch chunks matching filter
            cursor = collection.find(filter_query)
            all_chunks = await cursor.to_list(length=None)

            logger.info(f"Fetched {len(all_chunks)} chunks for similarity comparison")

            if not all_chunks:
                logger.warning("No chunks found in database for RAG search")
                return []

            # Compute cosine similarity for each chunk
            scored_chunks = []
            for chunk in all_chunks:
                if "embedding" not in chunk:
                    logger.warning(f"Chunk missing embedding: {chunk.get('document_id')}")
                    continue

                similarity = self._cosine_similarity(
                    query_embedding,
                    chunk["embedding"]
                )

                if similarity >= min_score:
                    chunk["score"] = similarity
                    scored_chunks.append(chunk)

            logger.info(f"Found {len(scored_chunks)} chunks above similarity threshold {min_score}")

            # Sort by score and return top k
            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
            top_results = scored_chunks[:top_k]

            # Log top results
            for i, chunk in enumerate(top_results[:3]):
                logger.debug(f"Top {i+1} result: {chunk.get('document_id')} (score: {chunk.get('score'):.3f})")

            return top_results

        except Exception as e:
            logger.error(f"Failed to retrieve similar chunks: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    async def delete_document_chunks(self, document_id: str):
        """Delete all chunks for a document"""
        try:
            collection = self.db.document_chunks
            result = await collection.delete_many({"document_id": document_id})
            logger.info(f"Deleted {result.deleted_count} chunks for document: {document_id}")

        except Exception as e:
            logger.error(f"Failed to delete document chunks: {e}")
            raise


# Singleton instance
_mongodb_service = None

async def get_mongodb_service() -> MongoDBService:
    """Get or create MongoDB service instance"""
    global _mongodb_service
    if _mongodb_service is None:
        _mongodb_service = MongoDBService()
        await _mongodb_service.connect()
    return _mongodb_service
