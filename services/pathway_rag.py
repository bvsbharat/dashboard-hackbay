"""
Real Pathway RAG Pipeline with Streaming
Implements proper file watching and real-time indexing
"""

import os
import logging
import asyncio
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    logger.warning("Pathway not available. Install with: pip install pathway[all]")
    PATHWAY_AVAILABLE = False


class PathwayRAGPipeline:
    """Real-time RAG pipeline using Pathway streaming"""

    def __init__(
        self,
        data_dir: str = "./uploads",
        openai_api_key: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.pipeline_thread = None
        self.running = False

    async def start_pipeline(self):
        """Start Pathway pipeline in background thread"""
        if not PATHWAY_AVAILABLE:
            logger.error("Pathway not available - cannot start pipeline")
            return

        if self.running:
            logger.warning("Pathway pipeline already running")
            return

        try:
            logger.info(f"🚀 Starting Pathway pipeline watching: {self.data_dir}")

            # Start Pathway in background thread
            self.pipeline_thread = threading.Thread(
                target=self._run_pathway_pipeline,
                daemon=True
            )
            self.pipeline_thread.start()
            self.running = True

            logger.info("✅ Pathway pipeline started successfully")

        except Exception as e:
            logger.error(f"Failed to start Pathway pipeline: {e}", exc_info=True)

    def _run_pathway_pipeline(self):
        """Run the Pathway pipeline (blocking call in thread)"""
        # Get the main event loop (from FastAPI)
        import threading
        main_loop = None
        for thread_id, frame in threading._active.items():
            if hasattr(frame, 'loop'):
                main_loop = frame.loop
                break

        # If we can't find main loop, we'll use asyncio.create_task approach
        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            from .event_queue import get_event_queue
            from .pinecone_service import get_pinecone_service
            from .mongodb_service import get_mongodb_service
            from .document_processor import get_document_processor
            from .deep_agent import get_deep_agent

            logger.info("🔧 Initializing Pathway components...")

            # Create file watcher
            files = pw.io.fs.read(
                path=self.data_dir,
                format="binary",
                mode="streaming",
                with_metadata=True,
            )

            logger.info(f"👀 Pathway watching directory: {self.data_dir}")

            # Process each file
            def process_file(data, metadata):
                """Process a single file"""
                try:
                    # Extract file path from metadata - Pathway returns it as JSON object
                    # Need to access .value to get the actual string
                    file_path_obj = metadata["path"]
                    # Convert from Pathway Json type to Python string
                    file_path = file_path_obj.value if hasattr(file_path_obj, 'value') else str(file_path_obj)
                    # Remove surrounding quotes if present (Pathway sometimes adds them)
                    file_path = file_path.strip('"').strip("'")
                    logger.info(f"📄 Pathway detected new file: {file_path}")

                    # Publish event (using loop.run_until_complete with thread's event loop)
                    event_queue = get_event_queue()
                    try:
                        loop.run_until_complete(event_queue.publish({
                            "type": "file_detected",
                            "message": f"New file detected: {Path(file_path).name}",
                            "file_path": file_path
                        }))
                    except:
                        pass  # Event queue not critical for processing

                    # Extract document using LandingAI
                    doc_processor = get_document_processor()
                    try:
                        extracted = loop.run_until_complete(doc_processor.extract_document_data(file_path))
                    except Exception as extraction_error:
                        error_msg = str(extraction_error)
                        logger.error(f"❌ Document extraction failed for {file_path}: {error_msg}")

                        # Publish error event to frontend
                        try:
                            loop.run_until_complete(event_queue.publish({
                                "type": "document_error",
                                "message": f"Failed to process: {Path(file_path).name}",
                                "error": error_msg,
                                "file_path": file_path
                            }))
                        except:
                            pass

                        return {"status": "error", "error": error_msg}

                    document_text = extracted.get("data", {}).get("text", "")
                    if not document_text:
                        error_msg = "No text extracted from document"
                        logger.error(f"❌ {error_msg}: {file_path}")

                        # Publish error event
                        try:
                            loop.run_until_complete(event_queue.publish({
                                "type": "document_error",
                                "message": f"Failed to process: {Path(file_path).name}",
                                "error": error_msg,
                                "file_path": file_path
                            }))
                        except:
                            pass

                        return {"status": "error", "error": error_msg}

                    logger.info(f"✅ Extracted {len(document_text)} chars from {Path(file_path).name}")

                    # Analyze with DeepAgent
                    deep_agent = get_deep_agent()
                    analysis = loop.run_until_complete(deep_agent.analyze_with_context(
                        document_text=document_text,
                        context_documents=[],  # No context for first doc
                        document_type="financial"
                    ))

                    logger.info(f"🧠 DeepAgent analysis complete for {Path(file_path).name}")

                    # Generate chunks and embeddings
                    chunks = self._create_chunks(document_text)
                    logger.info(f"📦 Created {len(chunks)} chunks")

                    # Generate embeddings
                    import openai
                    client = openai.OpenAI(api_key=self.openai_api_key)

                    chunk_records = []
                    for i, chunk_text in enumerate(chunks):
                        if len(chunk_text.strip()) < 50:
                            continue

                        try:
                            # Use text-embedding-3-small (1536 dimensions)
                            embedding_response = client.embeddings.create(
                                model="text-embedding-3-small",
                                input=chunk_text[:8000]
                            )
                            embedding = embedding_response.data[0].embedding

                            chunk_records.append({
                                "text": chunk_text,
                                "embedding": embedding,
                                "chunk_index": i,
                                "metadata": {
                                    "filename": Path(file_path).name,
                                    "file_path": file_path
                                }
                            })
                        except Exception as e:
                            logger.error(f"Failed to generate embedding for chunk {i}: {e}")

                    logger.info(f"🔢 Generated {len(chunk_records)} embeddings")

                    # Save to Pinecone (sync wrapper)
                    document_id = Path(file_path).name
                    pinecone = get_pinecone_service()

                    # Run async function in this thread's event loop
                    import nest_asyncio
                    nest_asyncio.apply()
                    success = loop.run_until_complete(pinecone.upsert_chunks(document_id, chunk_records))

                    if success:
                        logger.info(f"✅ Pinecone: Indexed {len(chunk_records)} vectors for {document_id}")
                    else:
                        logger.error(f"❌ Pinecone: Failed to index {document_id}")

                    # Save to MongoDB using sync wrapper
                    try:
                        mongodb = loop.run_until_complete(get_mongodb_service())

                        # Save document metadata
                        loop.run_until_complete(mongodb.save_document(
                            document_id=document_id,
                            filename=Path(file_path).name,
                            file_path=file_path,
                            extracted_data={
                                **extracted,
                                "deep_analysis": analysis
                            }
                        ))
                        logger.info(f"✅ MongoDB: Saved document metadata for {document_id}")

                        # Save financial metrics with detailed logging
                        metrics = analysis.get("metrics")
                        logger.info(f"📊 Analysis metrics from DeepAgent: {metrics}")

                        if metrics:
                            # Validate metrics are not all None/null
                            has_data = any(v is not None and v != 0 for v in metrics.values())
                            if has_data:
                                logger.info(f"💰 Saving financial metrics: {metrics}")
                                loop.run_until_complete(mongodb.save_financial_metrics(
                                    document_id=document_id,
                                    metrics=metrics
                                ))
                                logger.info(f"✅ MongoDB: Successfully saved financial metrics for {document_id}")
                            else:
                                logger.warning(f"⚠️  Metrics contain only null/zero values, skipping save: {metrics}")
                        else:
                            logger.warning(f"⚠️  No metrics found in analysis for {document_id}")
                            logger.warning(f"⚠️  Full analysis keys: {list(analysis.keys())}")

                    except Exception as db_error:
                        logger.error(f"❌ MongoDB save failed: {db_error}", exc_info=True)

                    # Publish completion event
                    try:
                        loop.run_until_complete(event_queue.publish({
                            "type": "document_processed",
                            "message": f"Document processed: {Path(file_path).name}",
                            "document_id": document_id,
                            "metrics": analysis.get("metrics", {}),
                            "red_flags_count": len(analysis.get("red_flags", [])),
                            "green_flags_count": len(analysis.get("green_flags", []))
                        }))
                    except:
                        pass  # Event queue not critical for processing

                    logger.info(f"🎉 Processing complete for {Path(file_path).name}")

                    return {"status": "success", "document_id": document_id}

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                    return {"status": "error", "error": str(e)}

            # Apply processing to each file
            processed = files.select(
                result=pw.apply(process_file, pw.this.data, pw.this._metadata)
            )

            # Subscribe to results
            def on_result(key, row, time, is_addition):
                if is_addition:
                    # row is a dict when using select()
                    result = row.get("result", {}) if isinstance(row, dict) else row.result
                    logger.info(f"Pathway result: {result}")

            pw.io.subscribe(processed, on_result)

            # Run pipeline (blocking)
            logger.info("▶️  Starting Pathway run...")
            pw.run()

        except Exception as e:
            logger.error(f"Pathway pipeline error: {e}", exc_info=True)

    def _create_chunks(self, text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation

        Args:
            text: Text to chunk
            chunk_size: Number of words per chunk (default: 150)
            overlap: Number of words to overlap between chunks (default: 30)

        Returns:
            List of text chunks with overlap
        """
        words = text.split()
        chunks = []

        # Create overlapping chunks
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) > 10:  # Only include chunks with substantial content
                chunks.append(" ".join(chunk_words))

        return chunks

    async def query(
        self,
        question: str,
        top_k: int = 5,
        with_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system using Pinecone vectors
        """
        try:
            from .pinecone_service import get_pinecone_service
            import openai

            logger.info(f"🔍 RAG Query: {question[:100]}...")

            # Generate query embedding (must match model used for indexing)
            client = openai.OpenAI(api_key=self.openai_api_key)
            query_embedding_response = client.embeddings.create(
                model="text-embedding-3-small",  # Must match indexing model
                input=question
            )
            query_embedding = query_embedding_response.data[0].embedding

            # Query Pinecone
            pinecone = get_pinecone_service()
            similar_chunks = await pinecone.query_similar_chunks(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=None
            )

            logger.info(f"📊 Retrieved {len(similar_chunks)} similar chunks from Pinecone")

            if not similar_chunks:
                return {
                    "answer": "I don't have enough information in the indexed documents to answer this question. Please upload relevant documents first.",
                    "sources": [],
                    "confidence": 0.0
                }

            # Format context
            context_parts = []
            sources = []

            for i, chunk in enumerate(similar_chunks):
                chunk_text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                score = chunk.get("score", 0)

                context_parts.append(f"[Document {i+1}: {metadata.get('filename', 'Unknown')}]\n{chunk_text}\n")

                if with_citations:
                    sources.append({
                        "document": metadata.get("filename", "Unknown"),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "relevance_score": round(score, 3),
                        "excerpt": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                    })

            context = "\n\n".join(context_parts)

            # Generate answer
            prompt = f"""You are a financial due diligence assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Provide a clear, accurate answer with specific numbers and facts from the context.
If the context doesn't contain enough information, say so clearly.

Answer:"""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )

            answer = response.choices[0].message.content
            logger.info(f"✅ Generated answer ({len(answer)} chars)")

            return {
                "answer": answer,
                "sources": sources[:5],  # Limit to top 5 sources
                "confidence": 0.85,
                "documents": similar_chunks  # Include full chunks for context
            }

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return {
                "answer": f"I encountered an error: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }

    async def add_document(
        self,
        document_id: str,
        document_text: str,
        metadata: dict = None
    ) -> bool:
        """
        Add document to index (called from upload endpoint)
        Note: With Pathway watching, this is redundant but kept for compatibility
        """
        logger.info(f"📝 Manual add_document called for: {document_id}")
        logger.info("ℹ️  Pathway file watcher should handle this automatically")
        return True


# Singleton instance
_rag_pipeline = None


def get_rag_pipeline() -> PathwayRAGPipeline:
    """Get or create the RAG pipeline instance"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = PathwayRAGPipeline()
    return _rag_pipeline
