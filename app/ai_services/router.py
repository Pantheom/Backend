import time

from fastapi import APIRouter, Depends

from app.dependencies import get_current_user
from app.ai_services.client import ping_cache, query_cache
from app.ai_services.schemas import AIQueryRequest, AIQueryResponse, CacheDebug


router = APIRouter(
    prefix="/ai",
    tags=["AI Services"],
)


# =========================================
# HEALTH — no auth required
# =========================================

@router.get("/health")
async def ai_health():
    """
    Pings the upstream Semantic Cache and reports its status.
    Does not require authentication.
    """
    cache_ok = await ping_cache()
    return {
        "status": "ok",
        "services": {
            "semantic_cache": "ok" if cache_ok else "unavailable",
        },
    }


# =========================================
# QUERY — auth required
# =========================================

@router.post("/query", response_model=AIQueryResponse)
async def ai_query(
    data: AIQueryRequest,
    current_user=Depends(get_current_user),
):
    """
    Phase 1: Semantic Cache lookup.

    Workflow:
      1. Send user prompt to the Semantic Cache (AWS EC2).
      2a. Cache HIT  → return cached response immediately (no LLM call needed).
      2b. Cache MISS → return miss metadata; Phase 2 will add model routing here.
      2c. Cache UNAVAILABLE → return graceful degradation response (never a 500).

    Authentication: Bearer token required (same JWT used for all other endpoints).
    """
    uid: str = current_user["uid"]
    session_id_str = str(data.session_id) if data.session_id else None

    t0 = time.monotonic()
    cache_result = await query_cache(
        prompt=data.prompt,
        user_id=uid,
        session_id=session_id_str,
    )
    elapsed_ms = round((time.monotonic() - t0) * 1000, 2)

    # ------------------------------------------------------------------
    # Case 1: Cache is completely unreachable
    # ------------------------------------------------------------------
    if cache_result is None:
        return AIQueryResponse(
            cache_hit=False,
            source="Cache_Unavailable",
            response=None,
            classification=None,
            latency_ms=elapsed_ms,
            message=(
                "Semantic cache is currently unavailable. "
                "Model routing pipeline will be triggered in Phase 2."
            ),
        )

    # ------------------------------------------------------------------
    # Case 2: Cache HIT — return the cached answer
    # ------------------------------------------------------------------
    if cache_result["cache_hit"] and cache_result.get("response"):
        return AIQueryResponse(
            cache_hit=True,
            source=cache_result["source"],
            response=cache_result["response"],
            classification=cache_result.get("classification"),
            latency_ms=cache_result.get("latency_ms", elapsed_ms),
            debug=CacheDebug(**cache_result.get("debug", {}))
            if cache_result.get("debug")
            else None,
            message="Served from semantic cache.",
        )

    # ------------------------------------------------------------------
    # Case 3: Cache MISS — no cached answer available
    # Phase 2 will send this to modelrouter/cascader + contextclassifysum
    # ------------------------------------------------------------------
    return AIQueryResponse(
        cache_hit=False,
        source=cache_result.get("source", "Cache_Miss"),
        response=None,
        classification=cache_result.get("classification"),
        latency_ms=cache_result.get("latency_ms", elapsed_ms),
        debug=CacheDebug(**cache_result.get("debug", {}))
        if cache_result.get("debug")
        else None,
        message=(
            "Cache miss. "
            f"Query classified as {cache_result.get('classification', 'UNKNOWN')}. "
            "Model routing pipeline will be triggered in Phase 2."
        ),
    )
