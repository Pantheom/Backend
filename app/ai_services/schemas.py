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
    """Result from the Context Service production API (POST /v1/process)."""
    needs_context: Optional[bool] = Field(
        default=None,
        description="True if prior conversation history context was injected",
    )
    context: Optional[str] = Field(
        default=None,
        description="The raw context block (summary + recent turns) that was prepended, or null",
    )
    combined_prompt: Optional[str] = Field(
        default=None,
        description="LLM-ready prompt with context already injected. Send this directly to the LLM.",
    )
    turn_index: Optional[int] = Field(
        default=None,
        description="Sequential index of the user turn in session history",
    )


# =========================================
# TOKEN USAGE
# =========================================

class TokenUsage(BaseModel):
    """Token consumption from the LLM call. Null on cache hits (no LLM was invoked)."""
    prompt_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens in the input prompt sent to the LLM",
    )
    completion_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens in the LLM's output response",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        description="Total tokens consumed by this LLM call (prompt + completion)",
    )
    provider: Optional[str] = Field(
        default=None,
        description="LLM provider that processed the request — 'Groq' or 'Google'",
    )
    model: Optional[str] = Field(
        default=None,
        description="Exact model identifier used for generation",
    )


# =========================================
# RESPONSE — top-level
# =========================================

class AIQueryResponse(BaseModel):
    cache_hit: bool = Field(description="True if the cache returned a usable response")
    source: str = Field(
        description="DB_Semantic_Hit | Cache_Miss | Cache_Unavailable"
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
    token_usage: Optional[TokenUsage] = Field(
        default=None,
        description="Token consumption from the LLM. Null on cache hits.",
    )
    debug: Optional[CacheDebug] = None
    message: Optional[str] = Field(
        default=None,
        description="Human-readable status message",
    )
