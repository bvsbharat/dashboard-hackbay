"""
FastAPI routes for document management and chat
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

from services import get_document_processor, get_rag_pipeline, get_mongodb_service, get_deep_agent, get_event_queue

logger = logging.getLogger(__name__)

router = APIRouter()

# Data models
class ChatQuery(BaseModel):
    question: str
    top_k: Optional[int] = 5
    with_citations: Optional[bool] = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []
    confidence: float = 0.0
    timestamp: str

class DocumentInfo(BaseModel):
    id: str
    name: str
    size: int
    upload_date: str
    status: str
    extracted_data: Optional[dict] = None


# Upload directory
UPLOAD_DIR = Path(os.getenv("DATA_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Due Diligence Copilot API",
        "version": "1.0.0"
    }


@router.get("/api/health")
async def health_check():
    """
    System health and status check

    Returns status of all services:
    - Pathway pipeline (running/stopped)
    - MongoDB connection
    - Pinecone connection
    - Document and vector counts
    """
    try:
        from services import get_pinecone_service

        rag_pipeline = get_rag_pipeline()
        mongodb = await get_mongodb_service()

        # Get document count from MongoDB - use db.collection syntax
        doc_count = 0
        try:
            doc_count = await mongodb.db.documents.count_documents({})
        except Exception as e:
            logger.warning(f"Could not get document count: {e}")

        # Get Pinecone stats
        try:
            pinecone = get_pinecone_service()
            pinecone_status = "connected"
            # Get vector count from Pinecone index stats
            stats = pinecone.index.describe_index_stats()
            vector_count = stats.get("total_vector_count", 0)
        except Exception as e:
            logger.error(f"Pinecone health check failed: {e}")
            pinecone_status = "error"
            vector_count = 0

        return {
            "status": "healthy",
            "pathway": {
                "status": "running" if rag_pipeline.running else "stopped",
                "watching": rag_pipeline.data_dir
            },
            "mongodb": {
                "status": "connected",
                "documents_indexed": doc_count
            },
            "pinecone": {
                "status": pinecone_status,
                "vectors_count": vector_count
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/api/events/stream")
async def stream_events(request: Request):
    """
    Server-Sent Events endpoint for real-time dashboard updates

    Streams events from the Pathway pipeline to the frontend:
    - file_detected: New file added to uploads
    - document_processed: Document analysis complete
    - metrics_updated: Financial metrics updated
    """
    async def event_generator():
        event_queue = get_event_queue()
        logger.info("SSE client connected")

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info("SSE client disconnected")
                    break

                # Get event from queue (with timeout for heartbeat)
                event = await event_queue.subscribe()

                # Format as SSE
                event_data = json.dumps(event)
                yield f"data: {event_data}\n\n"

        except asyncio.CancelledError:
            logger.info("SSE connection cancelled")
        except Exception as e:
            logger.error(f"SSE error: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload document - Pathway file watcher handles processing automatically

    This endpoint simply saves the file to disk. The Pathway streaming pipeline
    will detect the new file and process it automatically.
    """
    try:
        import unicodedata
        import re

        # Sanitize filename - remove special characters and Unicode
        # Normalize Unicode characters
        normalized_filename = unicodedata.normalize('NFKD', file.filename)
        # Remove non-ASCII characters
        ascii_filename = normalized_filename.encode('ascii', 'ignore').decode('ascii')
        # Replace spaces with underscores and remove special characters
        clean_filename = re.sub(r'[^\w\s.-]', '', ascii_filename)
        clean_filename = re.sub(r'\s+', '_', clean_filename)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{clean_filename}"
        file_path = UPLOAD_DIR / safe_filename

        # Save file to uploads directory
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"📤 File uploaded: {safe_filename}")
        logger.info(f"ℹ️  Pathway will auto-process this file")

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded - processing will begin automatically",
                "document": {
                    "id": safe_filename,
                    "name": file.filename,
                    "size": len(content),
                    "upload_date": datetime.now().isoformat(),
                    "status": "uploaded",
                    "path": str(file_path)
                }
            }
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/list")
async def list_documents():
    """List all uploaded documents from MongoDB"""
    try:
        mongodb = await get_mongodb_service()
        documents = await mongodb.get_all_documents()

        # Format for API response
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                "id": doc.get("document_id"),
                "name": doc.get("filename"),
                "size": len(open(doc.get("file_path"), 'rb').read()) if Path(doc.get("file_path")).exists() else 0,
                "upload_date": doc.get("upload_date").isoformat() if doc.get("upload_date") else None,
                "status": doc.get("status", "processed"),
                "path": doc.get("file_path")
            })

        return {
            "documents": formatted_docs,
            "total": len(formatted_docs)
        }

    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/{document_id}/download")
async def download_document(document_id: str):
    """Download a document"""
    try:
        file_path = UPLOAD_DIR / document_id

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")

        # Get original filename (remove timestamp prefix)
        original_name = "_".join(document_id.split("_")[2:]) if "_" in document_id else document_id

        return FileResponse(
            path=str(file_path),
            filename=original_name,
            media_type="application/octet-stream"
        )

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its associated data"""
    try:
        file_path = UPLOAD_DIR / document_id

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete file
        file_path.unlink()

        # Delete from MongoDB (document, metrics, and chunks)
        mongodb = await get_mongodb_service()
        await mongodb.delete_document(document_id)
        await mongodb.delete_document_chunks(document_id)

        logger.info(f"Document and all associated data deleted: {document_id}")

        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }

    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/chat/query", response_model=ChatResponse)
async def chat_query(query: ChatQuery):
    """
    Ask a question about uploaded documents using RAG retrieval + DeepAgent

    Returns answer with citations and confidence score
    """
    try:
        logger.info(f"Chat query received: {query.question[:100]}... (top_k={query.top_k}, with_citations={query.with_citations})")

        # Use RAG pipeline for retrieval
        rag_pipeline = get_rag_pipeline()
        result = await rag_pipeline.query(
            question=query.question,
            top_k=query.top_k,
            with_citations=query.with_citations
        )

        # Log result summary
        answer_preview = result.get("answer", "")[:100]
        logger.info(f"Chat query completed. Answer preview: {answer_preview}... Sources: {len(result.get('sources', []))}")

        return ChatResponse(
            answer=result.get("answer", "No answer generated"),
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Chat query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/summary")
async def get_analytics_summary():
    """
    Get analytics summary from MongoDB

    Returns real aggregated financial metrics
    """
    try:
        logger.info("Fetching analytics summary...")
        mongodb = await get_mongodb_service()
        metrics = await mongodb.get_aggregated_metrics()
        documents = await mongodb.get_all_documents()

        logger.info(f"Analytics summary: {len(documents)} documents analyzed, revenue={metrics.get('total_revenue', {}).get('value', 0)}")

        return {
            "metrics": metrics,
            "documents_analyzed": len(documents),
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Analytics summary failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/red-flags")
async def detect_red_flags():
    """
    Detect potential red flags in financial documents

    Returns list of concerns with severity and evidence from actual document analysis
    """
    try:
        mongodb = await get_mongodb_service()
        red_flags = await mongodb.get_all_red_flags()

        return {
            "red_flags": red_flags,
            "total_flags": len(red_flags),
            "last_analysis": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Red flag detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/green-flags")
async def detect_green_flags():
    """
    Identify positive indicators and strengths in financial documents

    Returns list of green flags with evidence from actual document analysis
    """
    try:
        mongodb = await get_mongodb_service()
        green_flags = await mongodb.get_all_green_flags()

        return {
            "green_flags": green_flags,
            "total_flags": len(green_flags),
            "last_analysis": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Green flag detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/historical")
async def get_historical_data(days: int = 30):
    """
    Get historical financial metrics for chart visualization

    Args:
        days: Number of days to retrieve (default 30)

    Returns:
        Time-series data for revenue, expenses, and profit
    """
    try:
        mongodb = await get_mongodb_service()
        historical_data = await mongodb.get_historical_metrics(days=days)

        return {
            "data": historical_data,
            "period_days": days,
            "total_points": len(historical_data)
        }

    except Exception as e:
        logger.error(f"Historical data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/distribution")
async def get_expense_distribution():
    """
    Get expense distribution by category

    Returns:
        Breakdown of expenses by category
    """
    try:
        mongodb = await get_mongodb_service()
        distribution = await mongodb.get_expense_distribution()

        return {
            "distribution": distribution,
            "total_categories": len(distribution)
        }

    except Exception as e:
        logger.error(f"Distribution retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/processing")
async def websocket_processing(websocket: WebSocket):
    """
    WebSocket for real-time document processing updates
    """
    await websocket.accept()

    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()

            # Send processing status
            await websocket.send_json({
                "type": "status",
                "message": "Processing document...",
                "progress": 50
            })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


async def _count_documents() -> int:
    """Count total documents"""
    try:
        return len(list(UPLOAD_DIR.glob("*")))
    except:
        return 0
