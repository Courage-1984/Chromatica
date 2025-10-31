# src/chromatica/api/routers/system.py

import logging
import os
import signal
import threading
import time
from pathlib import Path
from fastapi import APIRouter
from ..models import StatusResponse

router = APIRouter(
    tags=["System"],
)

system_logger = logging.getLogger("chromatica.system")


# This is the function called by the POST /restart endpoint
# It must return a success response immediately and then shut down.
@router.post("/restart", response_model=StatusResponse)
async def restart_server():
    """
    Triggers a graceful self-restart of the Uvicorn application.
    Relies on Uvicorn's --reload feature to detect the process exit and restart.
    """
    system_logger.info("Restart request received. Scheduling server shutdown...")

    # Define the shutdown function in a separate thread
    def shutdown_and_exit():
        # Wait longer (1 second) to ensure the HTTP response is fully sent
        # This gives Uvicorn time to complete the response before exit
        time.sleep(1.0)

        # Trigger Uvicorn's reloader by updating a watched file's modification time
        # This is the most reliable way to trigger a reload with --reload flag
        try:
            # Get the path to this file (system.py)
            current_file = Path(__file__).resolve()
            system_logger.warning(
                f"Triggering reload by updating mtime of {current_file}"
            )

            # Update the file's modification time without changing contents
            # This triggers Uvicorn's reloader to detect a change and restart
            os.utime(current_file, None)  # None means use current time

            system_logger.warning("File mtime updated, reloader should trigger restart")
        except Exception as e:
            system_logger.error(f"Failed to trigger file-based reload: {e}")
            # Fallback: Try signal-based approach
            try:
                system_logger.warning("Falling back to signal.SIGTERM")
                os.kill(os.getpid(), signal.SIGTERM)
            except (OSError, AttributeError):
                system_logger.warning("SIGTERM not available, using os._exit(0)")
                os._exit(0)

    # Schedule the shutdown in a separate thread to not block the response
    thread = threading.Thread(target=shutdown_and_exit, daemon=False)
    thread.start()

    # Return the success response immediately
    return StatusResponse(status="ok", message="Server scheduled for restart.")
