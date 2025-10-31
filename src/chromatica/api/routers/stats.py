# src/chromatica/api/routers/stats.py

import time
import logging
from fastapi import APIRouter
from ..models import StatsResponse, PerformanceStats
from ...utils.config import get_global_state

# Define the router instance
router = APIRouter(
    tags=["Statistics"],
)

stats_logger = logging.getLogger("chromatica.stats")


@router.get("/get_stats", response_model=StatsResponse)
async def get_stats():
    """
    Retrieves current performance and system statistics.
    """
    stats_logger.info("Retrieving system statistics...")

    # CRITICAL: Get global state components
    state = get_global_state()
    store = state.store
    # Access the raw performance dictionary managed in main.py
    performance_stats = state.performance_stats
    start_time = state.start_time

    # Acquire the lock to safely read performance metrics
    with performance_stats["lock"]:
        total_searches = performance_stats["total_searches"]
        total_time = performance_stats["total_time"]
        concurrent_searches = performance_stats["concurrent_searches"]
        # 💡 CRITICAL: Ensure this key name is an exact match
        max_concurrent_searches = performance_stats["max_concurrent_searches"]

    # Calculate metrics
    uptime_seconds = int(
        time.time() - start_time
    )  # start_time needs to be passed via set_global_state

    if total_searches > 0:
        average_latency_ms = int((total_time / total_searches) * 1000)
    else:
        average_latency_ms = 0

    index_size = store.get_image_count() if store else 0

    stats_logger.info(
        f"Stats retrieved: Total searches={total_searches}, Index size={index_size}"
    )

    return StatsResponse(
        status="ok",
        data=PerformanceStats(
            total_searches=total_searches,
            average_latency_ms=average_latency_ms,
            concurrent_searches=concurrent_searches,
            max_concurrent_searches=max_concurrent_searches,
            index_size=index_size,
            uptime_seconds=uptime_seconds,
        ),
    )
