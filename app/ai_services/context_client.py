"""
context_client.py
-----------------
Async HTTP client for the Context Service production API (apicreation branch).

API contract (POST /v1/process):
  Request:  { "session_id": "<str>", "prompt": "<str>" }
  Header:   X-API-Key: <CONTEXT_API_KEY>
  Response: {
      "needs_context":   true | false,
      "context":         "<context block>" | null,
      "combined_prompt": "<ready-to-use prompt for LLM>",
      "turn_index":      <int>
  }

The caller (router.py) should:
  1. Call process_prompt() to get the combined_prompt, then send that to the LLM.
  2. After the LLM responds, call log_reply() so conversation history stays complete.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# =========================================
# CONFIG
# =========================================

CONTEXT_BASE_URL: str = os.getenv(
    "CONTEXT_URL",
    "http://51.20.142.86:8000",
)

CONTEXT_API_KEY: Optional[str] = os.getenv("CONTEXT_API_KEY")

CONTEXT_TIMEOUT: float = float(os.getenv("CONTEXT_TIMEOUT_S", "15.0"))
# Context service may run Phi-4-mini summarization — allow generous timeout.


# =========================================
# SINGLETON ASYNC CLIENT
# =========================================

_client: Optional[httpx.AsyncClient] = None


def _build_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if CONTEXT_API_KEY:
        headers["X-API-Key"] = CONTEXT_API_KEY
    return headers


async def get_context_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=CONTEXT_BASE_URL,
            headers=_build_headers(),
            timeout=CONTEXT_TIMEOUT,
        )
    return _client


async def close_context_client() -> None:
    """Call on app shutdown to cleanly close the connection pool."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


# =========================================
# PUBLIC INTERFACE
# =========================================

async def process_prompt(session_id: str, prompt: str) -> Optional[dict]:
    """
    Call POST /v1/process on the Context Service.

    Returns None on any error — never raises.

    Success response shape:
      {
        "needs_context":   true | false,
        "context":         "<context block>" | null,
        "combined_prompt": "<LLM-ready prompt>",
        "turn_index":      <int>
      }

    The combined_prompt field is the one to send directly to the LLM —
    it already has context injected if needs_context is True.
    """
    try:
        client = await get_context_client()
        resp = await client.post(
            "/v1/process",
            json={"session_id": session_id, "prompt": prompt},
        )

        if resp.status_code == 503:
            log.warning("[CONTEXT] Service still loading models (503) — will retry on next request")
            return None

        if resp.status_code == 401:
            log.error("[CONTEXT] Auth failed — check CONTEXT_API_KEY")
            return None

        resp.raise_for_status()
        data = resp.json()

        return {
            "needs_context":   data.get("needs_context", False),
            "context":         data.get("context"),
            "combined_prompt": data.get("combined_prompt", prompt),
            "turn_index":      data.get("turn_index"),
        }

    except httpx.TimeoutException:
        log.warning("[CONTEXT] Timeout after %.1fs", CONTEXT_TIMEOUT)
        return None

    except httpx.ConnectError:
        log.warning("[CONTEXT] Connection refused — is the context service running at %s?", CONTEXT_BASE_URL)
        return None

    except httpx.HTTPStatusError as exc:
        log.warning("[CONTEXT] HTTP %s: %s", exc.response.status_code, exc.response.text[:200])
        return None

    except Exception as exc:  # noqa: BLE001
        log.error("[CONTEXT] Unexpected error: %s", exc, exc_info=True)
        return None


async def log_reply(session_id: str, reply_text: str) -> None:
    """
    Call POST /v1/sessions/{session_id}/reply to store the assistant response.

    Should be called after every LLM response so conversation history stays
    complete for the next process_prompt() call.

    Fire-and-forget — errors are logged but never propagated.
    """
    try:
        client = await get_context_client()
        resp = await client.post(
            f"/v1/sessions/{session_id}/reply",
            json={"text": reply_text},
        )
        resp.raise_for_status()
        log.debug("[CONTEXT] Reply logged for session=%s", session_id)

    except Exception as exc:  # noqa: BLE001
        # Non-critical — history will be incomplete but we never surface this as an error.
        log.warning("[CONTEXT] Failed to log reply for session=%s: %s", session_id, exc)


async def ping_context() -> bool:
    """Returns True if the context service health endpoint responds and models are loaded."""
    try:
        client = await get_context_client()
        resp = await client.get("/v1/health", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("models_loaded", False)
        return False
    except Exception:
        return False
