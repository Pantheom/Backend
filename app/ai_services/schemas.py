from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# =========================================
# REQUEST
# =========================================

class AIQueryRequest(BaseModel):
    prompt: str = Field(min_length=1, description="The user's prompt to process")
    session_id: Optional[UUID] = Field(
        default=None,
        description="Optional session UUID for traceability",
    )


# =========================================
# RESPONSE
# =========================================

class CacheDebug(BaseModel):
    tier: Optional[str] = None
    similarity_score: Optional[float] = None
    reranker_score: Optional[float] = None
    hit_count: Optional[int] = None
    classifier_called: Optional[bool] = None
    decision_layer: Optional[str] = None
    heuristic_reason: Optional[str] = None


class AIQueryResponse(BaseModel):
    cache_hit: bool = Field(description="True if the cache returned a usable response")
    source: str = Field(
        description="RAM_Exact_Hit | DB_Semantic_Hit | Cache_Miss | Cache_Unavailable"
    )
    response: Optional[str] = Field(
        default=None,
        description="Cached answer on a hit, null on a miss or unavailable",
    )
    classification: Optional[str] = Field(
        default=None,
        description="GENERAL or PERSONAL — describes the nature of the query",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Round-trip latency to the cache service in milliseconds",
    )
    debug: Optional[CacheDebug] = None
    message: Optional[str] = Field(
        default=None,
        description="Human-readable status message",
    )
