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
# RESPONSE — sub-models
# =========================================

class CacheDebug(BaseModel):
    tier: Optional[str] = None
    similarity_score: Optional[float] = None
    reranker_score: Optional[float] = None
    hit_count: Optional[int] = None
    classifier_called: Optional[bool] = None
    decision_layer: Optional[str] = None
    heuristic_reason: Optional[str] = None


class RoutingResult(BaseModel):
    """Result from the Model Cascader — which LLM tier/model to use."""
    tier: Optional[int] = Field(default=None, description="LLM tier: 1 (small), 2 (medium), 3 (large)")
    model: Optional[str] = Field(default=None, description="Model identifier chosen for this tier")
    score: Optional[float] = Field(default=None, description="Gatekeeper score that decided the tier")


class ContextResult(BaseModel):
    """Result from the Context Classifier + Summarizer."""
    needs_context: Optional[bool] = Field(
        default=None,
        description="True if the prompt requires conversation history context",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Rolling conversation summary to inject into the LLM prompt, if needed",
    )


# =========================================
# RESPONSE — top-level
# =========================================

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
    # Phase 2 fields — populated on cache miss
    routing: Optional[RoutingResult] = Field(
        default=None,
        description="Model routing decision (tier + model). Populated on cache miss.",
    )
    context: Optional[ContextResult] = Field(
        default=None,
        description="Context classification + summary. Populated on cache miss.",
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
