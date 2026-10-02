"""
storage_tasks.py
================
Fire-and-forget async helpers for persisting conversation state.

Both functions are designed to be called via:
    asyncio.create_task(...)

They NEVER raise and NEVER block the response path.

Functions
---------
push_to_cache    - POST the prompt+response to the semantic cache store
                   endpoint.  Called on MISS only, and ONLY when the query
                   classification is GENERAL (PERSONAL queries must never
                   be cached for privacy reasons).

NOTE: save_chat_turn was removed. All chat_history writes now go through
the context service (context_client.py -> /api/session/{id}/turn).
This ensures the summarizer's unsummarized-turn counter is always accurate
and prevents double-writes that were causing token overflow.
"""

import logging

from app.ai_services.client import store_cache

log = logging.getLogger(__name__)


async def push_to_cache(
    prompt: str,
    response: str,
    classification: str | None,
) -> None:
    """
    Stores a prompt+response in the semantic cache DB via the bridge endpoint.

    PRIVACY RULE: PERSONAL queries are NEVER stored in the shared cache.
    Only call this on cache MISS after LLM response is available.
    """
    if not response:
        log.debug("[STORAGE] push_to_cache skipped - no LLM response.")
        return

    if classification and classification.upper() == "PERSONAL":
        log.debug("[STORAGE] push_to_cache skipped - PERSONAL query, not caching.")
        return

    success = await store_cache(prompt=prompt, response=response)
    if success:
        log.debug("[STORAGE] Cache store OK for prompt=%.60s", prompt)
    else:
        log.warning("[STORAGE] Cache store failed - continuing without caching.")
