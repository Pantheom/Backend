import os
import logging
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
    "http://51.20.142.86:8002",
)

CONTEXT_TIMEOUT: float = float(os.getenv("CONTEXT_TIMEOUT_S", "10.0"))
# Context service may need to run Phi-4-mini summarization — allow more time.


# =========================================
# SINGLETON ASYNC CLIENT
# =========================================

_client: Optional[httpx.AsyncClient] = None


async def get_context_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=CONTEXT_BASE_URL,
            headers={"Content-Type": "application/json"},
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

async def get_context(session_id: str, prompt: str) -> Optional[dict]:
    """
    Calls the Context Classifier + Summarizer's full pipeline endpoint.

    Endpoint: POST /api/session/{session_id}/get_context
    Body:     { "prompt": "..." }

    Returns None on any error — never raises.

    Success response shape:
      {
        "needs_context": true | false,
        "summary":       "<rolling summary text>" | null
      }
    """
    try:
        client = await get_context_client()
        resp = await client.post(
            f"/api/session/{session_id}/get_context",
            json={"prompt": prompt},
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "needs_context": data.get("needs_context", False),
            "summary":       data.get("summary"),
        }

    except httpx.TimeoutException:
        log.warning("[CONTEXT] Timeout after %.1fs", CONTEXT_TIMEOUT)
        return None

    except httpx.ConnectError:
        log.warning("[CONTEXT] Connection refused — is the context service running?")
        return None

    except httpx.HTTPStatusError as exc:
        log.warning("[CONTEXT] HTTP %s", exc.response.status_code)
        return None

    except Exception as exc:  # noqa: BLE001
        log.error("[CONTEXT] Unexpected error: %s", exc, exc_info=True)
        return None


async def ping_context() -> bool:
    """Returns True if the context service health endpoint responds."""
    try:
        client = await get_context_client()
        # The context service doesn't have a /health — use /api/status instead
        resp = await client.get("/api/status")
        return resp.status_code == 200
    except Exception:
        return False
