import asyncio
import time
import uuid

from fastapi import APIRouter, Depends

from app.dependencies import get_current_user
from app.ai_services.client import ping_cache, query_cache
from app.ai_services.cascader_client import ping_cascader, route_prompt
from app.ai_services.context_client import ping_context, get_context
from app.ai_services.schemas import (
    AIQueryRequest,
    AIQueryResponse,
    CacheDebug,
    RoutingResult,
    ContextResult,
)


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
    Pings all upstream AI services and reports their status.
    Does not require authentication.
    """
    cache_ok, cascader_ok, context_ok = await asyncio.gather(
        ping_cache(),
        ping_cascader(),
        ping_context(),
    )
    return {
        "status": "ok",
        "services": {
            "semantic_cache": "ok" if cache_ok else "unavailable",
            "model_cascader": "ok" if cascader_ok else "unavailable",
            "context_classifier": "ok" if context_ok else "unavailable",
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
    Full AI pipeline endpoint.

    Workflow:
      1. Send user prompt to the Semantic Cache (AWS EC2).

      2a. Cache HIT  → return cached response immediately. No further calls made.

      2b. Cache MISS → fire Model Cascader + Context Classifier **in parallel**
                       using asyncio.gather(). Total latency = max(cascader, context).
                       Returns unified response with routing + context data.

      2c. Cache UNAVAILABLE → fall through to the parallel pipeline anyway,
                              so the user still gets routing + context info.

    Partial failure: if one downstream service is down, its field is null.
    The other result is still returned. Never a 500.

    Authentication: Bearer token required.
    """
    uid: str = current_user["uid"]

    # Use provided session_id, or generate a stable per-request UUID so the
    # context service always receives a valid session identifier.
    session_id: str = str(data.session_id) if data.session_id else str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Step 1: Semantic Cache lookup
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    cache_result = await query_cache(
        prompt=data.prompt,
        user_id=uid,
        session_id=session_id,
    )
    cache_latency_ms = round((time.monotonic() - t0) * 1000, 2)

    # ------------------------------------------------------------------
    # Case: Cache HIT — return immediately, no further calls needed
    # ------------------------------------------------------------------
    if cache_result and cache_result["cache_hit"] and cache_result.get("response"):
        return AIQueryResponse(
            cache_hit=True,
            source=cache_result["source"],
            response=cache_result["response"],
            classification=cache_result.get("classification"),
            routing=None,
            context=None,
            latency_ms=cache_result.get("latency_ms", cache_latency_ms),
            debug=CacheDebug(**cache_result.get("debug", {}))
            if cache_result.get("debug")
            else None,
            message="Served from semantic cache.",
        )

    # ------------------------------------------------------------------
    # Case: Cache MISS or Cache UNAVAILABLE
    # Fire Model Cascader + Context Classifier in parallel
    # ------------------------------------------------------------------
    t1 = time.monotonic()

    routing_raw, context_raw = await asyncio.gather(
        route_prompt(data.prompt),
        get_context(session_id, data.prompt),
        return_exceptions=True,   # ensures one failure doesn't cancel the other
    )

    pipeline_latency_ms = round((time.monotonic() - t1) * 1000, 2)
    total_latency_ms = round((time.monotonic() - t0) * 1000, 2)

    # Safely handle exceptions returned by return_exceptions=True
    if isinstance(routing_raw, Exception):
        routing_raw = None
    if isinstance(context_raw, Exception):
        context_raw = None

    routing = RoutingResult(**routing_raw) if routing_raw else None
    context = ContextResult(**context_raw) if context_raw else None

    # Determine source and classification from cache result (if available)
    source = cache_result.get("source", "Cache_Miss") if cache_result else "Cache_Unavailable"
    classification = cache_result.get("classification") if cache_result else None

    # Build a human-readable status message
    routing_status = f"tier={routing.tier} model={routing.model}" if routing else "unavailable"
    context_status = (
        f"needs_context={context.needs_context}"
        if context else "unavailable"
    )
    message = (
        f"Cache miss. "
        f"Routing: {routing_status}. "
        f"Context: {context_status}. "
        f"Pipeline latency: {pipeline_latency_ms}ms."
    )

    return AIQueryResponse(
        cache_hit=False,
        source=source,
        response=None,
        classification=classification,
        routing=routing,
        context=context,
        latency_ms=total_latency_ms,
        debug=CacheDebug(**cache_result.get("debug", {}))
        if cache_result and cache_result.get("debug")
        else None,
        message=message,
    )
