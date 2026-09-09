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

SEMANTIC_CACHE_BASE_URL: str = os.getenv(
    "SEMANTIC_CACHE_URL",
    "http://13.53.130.85:8002",
)

SEMANTIC_CACHE_API_KEY: Optional[str] = os.getenv("SEMANTIC_CACHE_API_KEY")

SEMANTIC_CACHE_TIMEOUT: float = float(
    os.getenv("SEMANTIC_CACHE_TIMEOUT_S", "5.0")
)


# =========================================
# SINGLETON ASYNC CLIENT
# =========================================

_client: Optional[httpx.AsyncClient] = None


def _build_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if SEMANTIC_CACHE_API_KEY:
        headers["X-API-Key"] = SEMANTIC_CACHE_API_KEY
    return headers


async def get_cache_client() -> httpx.AsyncClient:
    """
    Returns a singleton AsyncClient.  Re-creates it if it has been closed.
    Using a single client gives us TCP connection pooling — much faster than
    creating a new connection for every request.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=SEMANTIC_CACHE_BASE_URL,
            headers=_build_headers(),
            timeout=SEMANTIC_CACHE_TIMEOUT,
        )
    return _client


async def close_cache_client() -> None:
    """Call this on app shutdown to cleanly close the connection pool."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


# =========================================
# PUBLIC INTERFACE
# =========================================

async def query_cache(
    prompt: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[dict]:
    """
    Sends a prompt to the Semantic Cache and returns the parsed response.

    Returns None on any error (timeout, connection refused, bad HTTP status).
    Never raises — the LLM pipeline is always the fallback.

    Response shape on success:
      {
        "cache_hit":      bool,
        "source":         "RAM_Exact_Hit" | "DB_Semantic_Hit" | "Cache_Miss",
        "response":       str | None,       # cached answer, or None on miss
        "classification": "GENERAL" | "PERSONAL",
        "latency_ms":     float,
        "debug":          { ... }
      }
    """
    payload: dict = {"prompt": prompt}
    if user_id:
        payload["user_id"] = user_id
    if session_id:
        payload["session_id"] = session_id

    try:
        client = await get_cache_client()
        resp = await client.post("/v1/cache/query", json=payload)

        if resp.status_code == 401:
            log.error(
                "[CACHE] Authentication failed — check SEMANTIC_CACHE_API_KEY"
            )
            return None

        resp.raise_for_status()
        data = resp.json()

        # Normalise the response: the local main.py uses a slightly different
        # shape than the bridge (axiom_bridge).  We unify both here so the
        # router never has to worry about which version is deployed.
        cache_hit = (
            data.get("cache_hit")                          # bridge format
            if "cache_hit" in data
            else (data.get("source", "") != "Cache_Miss")  # direct format
        )
        return {
            "cache_hit":      cache_hit,
            "source":         data.get("source", "unknown"),
            "response":       data.get("response"),
            "classification": data.get("classification", "GENERAL"),
            "latency_ms":     data.get("latency_ms", 0.0),
            "debug":          data.get("debug", {}),
        }

    except httpx.TimeoutException:
        log.warning("[CACHE] Timeout after %.1fs — will fall through", SEMANTIC_CACHE_TIMEOUT)
        return None

    except httpx.ConnectError:
        log.warning("[CACHE] Connection refused — is the cache server running?")
        return None

    except httpx.HTTPStatusError as exc:
        log.warning("[CACHE] HTTP %s — %s", exc.response.status_code, exc.response.text[:200])
        return None

    except Exception as exc:  # noqa: BLE001
        log.error("[CACHE] Unexpected error: %s", exc, exc_info=True)
        return None


async def ping_cache() -> bool:
    """Returns True if the cache health endpoint responds successfully."""
    try:
        client = await get_cache_client()
        resp = await client.get("/health")
        return resp.status_code == 200
    except Exception:
        return False


async def store_cache(prompt: str, response: str) -> bool:
    """
    Persists a prompt+response pair into the semantic cache DB.

    Called only on cache MISS and only for GENERAL (non-personal) queries.
    Never raises — returns True on success, False on any failure.
    Intended to be called with asyncio.create_task() so it never blocks.
    """
    payload = {"prompt": prompt, "response": response}
    try:
        client = await get_cache_client()
        resp = await client.post("/v1/cache/store", json=payload)
        if resp.status_code == 200:
            log.debug("[CACHE] Stored prompt in semantic cache DB.")
            return True
        log.warning("[CACHE] store_cache got HTTP %s: %s", resp.status_code, resp.text[:200])
        return False
    except httpx.TimeoutException:
        log.warning("[CACHE] store_cache timed out — skipping cache write.")
        return False
    except httpx.ConnectError:
        log.warning("[CACHE] store_cache connection refused — cache may be down.")
        return False
    except Exception as exc:  # noqa: BLE001
        log.error("[CACHE] store_cache unexpected error: %s", exc, exc_info=True)
        return False
