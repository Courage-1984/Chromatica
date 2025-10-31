from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass, asdict

# --- Data Structures from search.py (dataclasses) ---
# NOTE: RerankResult is likely defined in core.rerank, but SearchResult is often
# a hybrid used for the API response. Let's assume you've kept the API-facing
# dataclasses in main.py, but include the core SearchResult here for completeness.


@dataclass
class SearchResult:
    """Result of a complete search operation for a single image."""

    # --- REQUIRED FIELDS (Must be first) ---
    image_id: str
    file_path: str
    distance: float
    rank: int
    ann_score: float
    dominant_colors: List[str]

    # --- OPTIONAL FIELDS (Must be last) ---
    confidence: float = 1.0
    image_url: Optional[str] = None

    # Allows easy conversion to a dict for API response
    def to_dict(self):
        return asdict(self)


# class SearchResult(BaseModel):
#     """Individual search result with image information."""

#     image_id: str = Field(..., description="Unique identifier for the image")
#     distance: float = Field(..., description="Sinkhorn-EMD distance from query")
#     dominant_colors: List[str] = Field(..., description="Dominant colors in the image")
#     file_path: Optional[str] = Field(None, description="Path to the image file")
#     image_url: Optional[str] = Field(None, description="URL of the image")


# Pydantic models for request/response validation
class QueryColors(BaseModel):
    """Query colors and weights for the search."""

    colors: List[str] = Field(..., description="List of hex color codes")
    weights: List[float] = Field(..., description="List of corresponding weights")


class SearchMetadata(BaseModel):
    """Performance metadata for the search operation."""

    ann_time_ms: int = Field(..., description="ANN search time in milliseconds")
    rerank_time_ms: int = Field(..., description="Reranking time in milliseconds")
    total_time_ms: int = Field(..., description="Total search time in milliseconds")
    index_size: int = Field(
        ..., description="Total number of images in the search index"
    )


class SearchResponse(BaseModel):
    """Complete search response as specified in Section H."""

    query_id: str = Field(..., description="Unique identifier for this search query")
    query: QueryColors = Field(..., description="Original query colors and weights")
    results_count: int = Field(..., description="Number of results returned")
    results: List[Dict[str, Any]] = Field(..., description="Ranked search results")
    metadata: SearchMetadata = Field(..., description="Performance timing metadata")


class VisualizationResponse(BaseModel):
    """Response for visualization endpoints."""

    query_id: str = Field(..., description="Unique identifier for this visualization")
    query: QueryColors = Field(..., description="Original query colors and weights")
    visualization_type: str = Field(..., description="Type of visualization generated")
    image_data: str = Field(..., description="Base64 encoded image data")
    mime_type: str = Field(..., description="MIME type of the image")


# Request model for command execution
class CommandRequest(BaseModel):
    command: str = Field(..., description="Command to execute")
    args: List[str] = Field(..., description="Command arguments")
    tool_type: str = Field(..., description="Type of tool being executed")
    dataset: str = Field(..., description="Dataset path for the tool")


# Response model for command execution
class CommandResponse(BaseModel):
    success: bool = Field(..., description="Whether the command executed successfully")
    output: str = Field(..., description="Command output or error message")
    error: Optional[str] = Field(None, description="Error message if command failed")


class ParallelSearchRequest(BaseModel):
    """Request model for parallel search operations"""

    queries: List[Dict[str, Any]] = Field(..., description="List of search queries")
    max_concurrent: int = Field(
        10, description="Maximum number of concurrent searches", ge=1, le=50
    )


class ParallelSearchResponse(BaseModel):
    """Response model for parallel search operations"""

    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    total_time_ms: int = Field(..., description="Total processing time in milliseconds")
    results: List[Dict[str, Any]] = Field(
        ..., description="Search results for each query"
    )


class PerformanceStatsResponse(BaseModel):
    """Response model for performance statistics"""

    total_searches: int = Field(..., description="Total number of searches performed")
    concurrent_searches: int = Field(
        ..., description="Current number of concurrent searches"
    )
    max_concurrent_searches: int = Field(
        ..., description="Maximum concurrent searches reached"
    )
    average_search_time: float = Field(
        ..., description="Average search time in seconds"
    )
    recent_search_times: List[float] = Field(..., description="Recent search times")


# --- Statistics Models ---


class PerformanceStats(BaseModel):
    """
    Detailed performance statistics for the search system.
    """

    total_searches: int = Field(
        ..., description="Total number of successful searches performed since startup."
    )
    average_latency_ms: int = Field(
        ..., description="Average search latency in milliseconds."
    )
    concurrent_searches: int = Field(
        ..., description="Current number of concurrent search operations."
    )
    max_concurrent_searches: int = Field(
        ..., description="Maximum allowed concurrent search operations."
    )
    index_size: int = Field(
        ...,
        description="Total number of images currently indexed in the FAISS index and metadata store.",
    )
    uptime_seconds: int = Field(..., description="Total server uptime in seconds.")


class StatsResponse(BaseModel):
    """
    Response model for the /get_stats endpoint.
    """

    status: str = Field(..., description="Server status (e.g., 'ok').")
    data: PerformanceStats


class StatusResponse(BaseModel):
    """
    Simple status response model for system operations.
    """

    status: str = Field(..., description="Status of the operation (e.g., 'ok').")
    message: str = Field(
        ..., description="Human-readable message describing the operation result."
    )


class ColorExtractionResponse(BaseModel):
    """
    Response model for the /extract_colors endpoint, returning hex colors and weights.
    """

    status: str = Field(..., description="Extraction status (e.g., 'ok').")
    colors: List[str] = Field(
        ...,
        description="List of dominant colors in hex format (e.g., ['#FF0000', '#00FF00']).",
    )
    weights: List[float] = Field(
        ...,
        description="List of normalized weights (0.0 to 1.0) corresponding to each color.",
    )
    num_colors: int = Field(
        ..., description="The number of colors successfully extracted."
    )
