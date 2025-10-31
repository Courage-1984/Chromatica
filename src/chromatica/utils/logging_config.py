import logging
import os
from pathlib import Path
from datetime import datetime


def setup_logging():
    """Setup comprehensive logging for console and file output."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"chromatica_api_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    # Create specific loggers for different components
    api_logger = logging.getLogger("chromatica.api")
    webui_logger = logging.getLogger("chromatica.webui")
    search_logger = logging.getLogger("chromatica.search")
    visualization_logger = logging.getLogger("chromatica.visualization")

    # Set levels
    api_logger.setLevel(logging.INFO)
    webui_logger.setLevel(logging.INFO)
    search_logger.setLevel(logging.INFO)
    visualization_logger.setLevel(logging.INFO)

    # Log startup information
    api_logger.info(f"Logging initialized - Console and file: {log_file}")
    api_logger.info(f"Log directory: {log_dir.absolute()}")

    return api_logger, webui_logger, search_logger, visualization_logger
