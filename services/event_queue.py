"""
Event Queue for Real-Time Updates
Bridges Pathway pipeline events to frontend via SSE
"""
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class EventQueue:
    """Thread-safe event queue for real-time updates"""

    def __init__(self, maxsize: int = 100):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.history = deque(maxlen=50)  # Keep last 50 events
        self.subscribers = []

    async def publish(self, event: Dict[str, Any]):
        """Publish an event to all subscribers"""
        try:
            # Add timestamp
            event["timestamp"] = datetime.utcnow().isoformat()

            # Add to history
            self.history.append(event)

            # Put in queue
            await self.queue.put(event)

            logger.info(f"Published event: {event.get('type')} - {event.get('message', '')}")

        except asyncio.QueueFull:
            logger.warning("Event queue is full, dropping event")
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")

    async def subscribe(self) -> Dict[str, Any]:
        """Subscribe to events (blocking until event available)"""
        try:
            event = await asyncio.wait_for(self.queue.get(), timeout=30.0)
            return event
        except asyncio.TimeoutError:
            # Return heartbeat to keep connection alive
            return {"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()}

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent event history"""
        return list(self.history)[-limit:]


# Global singleton
_event_queue = None


def get_event_queue() -> EventQueue:
    """Get or create the global event queue"""
    global _event_queue
    if _event_queue is None:
        _event_queue = EventQueue()
    return _event_queue
