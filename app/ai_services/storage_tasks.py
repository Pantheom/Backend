"""
storage_tasks.py
================
Fire-and-forget async helpers for persisting conversation state.

Both functions are designed to be called via:
    asyncio.create_task(...)

They NEVER raise and NEVER block the response path.

Functions
---------
save_chat_turn   - write a user+assistant message pair to the Supabase
                   chat_history table.  Always called (HIT + MISS).

push_to_cache    - POST the prompt+response to the semantic cache store
                   endpoint.  Called on MISS only, and ONLY when the query
                   classification is GENERAL (PERSONAL queries must never
                   be cached for privacy reasons).
"""

import logging
from datetime import datetime, timezone

from app.database import supabase
from app.ai_services.client import store_cache

log = logging.getLogger(__name__)


async def save_chat_turn(
    uid: str,
    session_id: str,
    user_prompt: str,
    assistant_response: str | None,
) -> None:
    """
    Inserts a user message + assistant reply into the chat_history table.

    Schema expected:
        chat_history (
            uid          TEXT  NOT NULL,
            session_id   TEXT  NOT NULL,
            role         TEXT  NOT NULL,
            content      TEXT  NOT NULL,
            created_at   TIMESTAMPTZ DEFAULT now()
        )

    Called on BOTH cache hit and cache miss - it is a conversation log.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "uid":        uid,
            "session_id": session_id,
            "role":       "user",
            "message":    user_prompt,
            "created_at": now,
        },
    ]
    if assistant_response:
        rows.append({
            "uid":        uid,
            "session_id": session_id,
            "role":       "assistant",
            "message":    assistant_response,
            "created_at": now,
        })

    try:
        supabase.table("chat_history").insert(rows).execute()
        log.debug(
            "[STORAGE] Saved %d chat row(s) uid=%s session=%s",
            len(rows), uid, session_id,
        )
    except Exception as exc:  # noqa: BLE001
        log.error(
            "[STORAGE] save_chat_turn failed uid=%s session=%s: %s",
            uid, session_id, exc, exc_info=True,
        )


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
