"""
Service for real-time status tracking via Server-Sent Events.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional

from ..models.conversion import StatusEvent

logger = logging.getLogger(__name__)

# In-memory event store for demo
# In production, this would poll from the status_events UC table
_event_store: Dict[str, list] = {}


class StatusTracker:
    """Tracks and streams real-time status updates."""

    def __init__(self):
        self.poll_interval = 1.0  # seconds

    def add_event(
        self,
        job_id: str,
        event_type: str,
        event_data: Dict[str, Any],
    ) -> StatusEvent:
        """Add a status event for a job.

        Args:
            job_id: Job identifier
            event_type: Type of event (status_change, progress, log, error)
            event_data: Event payload

        Returns:
            Created StatusEvent
        """
        event = StatusEvent(
            job_id=job_id,
            event_type=event_type,
            event_data=event_data,
        )

        if job_id not in _event_store:
            _event_store[job_id] = []
        _event_store[job_id].append(event)

        logger.debug(f"Added event for {job_id}: {event_type}")

        return event

    def get_events(
        self,
        job_id: str,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[StatusEvent]:
        """Get events for a job.

        Args:
            job_id: Job identifier
            since: Only return events after this timestamp
            limit: Maximum number of events to return

        Returns:
            List of StatusEvent objects
        """
        events = _event_store.get(job_id, [])

        if since:
            events = [e for e in events if e.created_at > since]

        return events[-limit:]

    async def stream_status(self, job_id: str) -> AsyncGenerator[dict, None]:
        """Stream status updates for a job via SSE.

        Args:
            job_id: Job identifier

        Yields:
            SSE event dicts with event type and data
        """
        logger.info(f"Starting SSE stream for job {job_id}")

        last_event_id = None
        no_event_count = 0
        max_idle_iterations = 300  # ~5 minutes with 1s poll

        while no_event_count < max_idle_iterations:
            # Get new events since last check
            events = _event_store.get(job_id, [])

            if last_event_id:
                # Find events after the last one we sent
                new_events = []
                found_last = False
                for event in events:
                    if found_last:
                        new_events.append(event)
                    elif event.event_id == last_event_id:
                        found_last = True
            else:
                # First iteration - send all recent events
                new_events = events[-10:]  # Last 10 events

            if new_events:
                no_event_count = 0
                for event in new_events:
                    last_event_id = event.event_id
                    yield {
                        "event": event.event_type,
                        "id": event.event_id,
                        "data": json.dumps({
                            "job_id": event.job_id,
                            "timestamp": event.created_at.isoformat(),
                            **event.event_data,
                        }),
                    }

                # Check for terminal events
                for event in new_events:
                    if event.event_type == "status_change":
                        status = event.event_data.get("status")
                        if status in ("completed", "failed", "cancelled"):
                            logger.info(f"Job {job_id} reached terminal state: {status}")
                            yield {
                                "event": "close",
                                "data": json.dumps({"reason": f"Job {status}"}),
                            }
                            return
            else:
                no_event_count += 1

            await asyncio.sleep(self.poll_interval)

        logger.info(f"SSE stream for job {job_id} timed out (idle)")
        yield {
            "event": "close",
            "data": json.dumps({"reason": "Timeout - no activity"}),
        }

    def emit_status_change(
        self,
        job_id: str,
        old_status: str,
        new_status: str,
        message: Optional[str] = None,
    ) -> StatusEvent:
        """Emit a status change event.

        Args:
            job_id: Job identifier
            old_status: Previous status
            new_status: New status
            message: Optional status message

        Returns:
            Created StatusEvent
        """
        return self.add_event(
            job_id=job_id,
            event_type="status_change",
            event_data={
                "old_status": old_status,
                "status": new_status,
                "message": message or f"Status changed to {new_status}",
            },
        )

    def emit_progress(
        self,
        job_id: str,
        progress: float,
        stage: str,
        message: Optional[str] = None,
    ) -> StatusEvent:
        """Emit a progress update event.

        Args:
            job_id: Job identifier
            progress: Progress percentage (0-100)
            stage: Current stage name
            message: Optional progress message

        Returns:
            Created StatusEvent
        """
        return self.add_event(
            job_id=job_id,
            event_type="progress",
            event_data={
                "progress": progress,
                "stage": stage,
                "message": message or f"Processing: {stage}",
            },
        )

    def emit_log(
        self,
        job_id: str,
        level: str,
        message: str,
    ) -> StatusEvent:
        """Emit a log event.

        Args:
            job_id: Job identifier
            level: Log level (info, warning, error)
            message: Log message

        Returns:
            Created StatusEvent
        """
        return self.add_event(
            job_id=job_id,
            event_type="log",
            event_data={
                "level": level,
                "message": message,
            },
        )

    def emit_error(
        self,
        job_id: str,
        error: str,
        details: Optional[str] = None,
    ) -> StatusEvent:
        """Emit an error event.

        Args:
            job_id: Job identifier
            error: Error message
            details: Optional error details/stack trace

        Returns:
            Created StatusEvent
        """
        return self.add_event(
            job_id=job_id,
            event_type="error",
            event_data={
                "error": error,
                "details": details,
            },
        )


# Global tracker instance
_tracker = StatusTracker()


def get_tracker() -> StatusTracker:
    """Get the global status tracker."""
    return _tracker
