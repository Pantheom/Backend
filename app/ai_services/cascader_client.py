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

CASCADER_BASE_URL: str = os.getenv(
    "CASCADER_URL",
    "http://16.16.28.68:8000",
)

CASCADER_API_KEY: Optional[str] = os.getenv("CASCADER_API_KEY")

CASCADER_TIMEOUT: float = float(os.getenv("CASCADER_TIMEOUT_S", "5.0"))


# =========================================
# SINGLETON ASYNC CLIENT
# =========================================

_client: Optional[httpx.AsyncClient] = None


def _build_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if CASCADER_API_KEY:
        headers["X-API-Key"] = CASCADER_API_KEY
    return headers


async def get_cascader_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=CASCADER_BASE_URL,
            headers=_build_headers(),
            timeout=CASCADER_TIMEOUT,
        )
    return _client


async def close_cascader_client() -> None:
    """Call on app shutdown to cleanly close the connection pool."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


# =========================================
# PUBLIC INTERFACE
# =========================================

async def route_prompt(prompt: str) -> Optional[dict]:
    """
    Sends prompt to the Model Cascader and returns routing decision.

    Returns None on any error — never raises.

    Success response shape:
      {
        "tier":  1 | 2 | 3,
        "model": "<model-id>",
        "score": <float>
      }
    """
    try:
        client = await get_cascader_client()
        resp = await client.post("/route", json={"prompt": prompt})

        if resp.status_code == 403:
            log.error("[CASCADER] Auth failed — check CASCADER_API_KEY")
            return None

        resp.raise_for_status()
        data = resp.json()

        return {
            "tier":  data.get("tier"),
            "model": data.get("model"),
            "score": data.get("score"),
        }

    except httpx.TimeoutException:
        log.warning("[CASCADER] Timeout after %.1fs", CASCADER_TIMEOUT)
        return None

    except httpx.ConnectError:
        log.warning("[CASCADER] Connection refused — is the cascader running?")
        return None

    except httpx.HTTPStatusError as exc:
        log.warning("[CASCADER] HTTP %s", exc.response.status_code)
        return None

    except Exception as exc:  # noqa: BLE001
        log.error("[CASCADER] Unexpected error: %s", exc, exc_info=True)
        return None


async def ping_cascader() -> bool:
    """Returns True if the cascader health endpoint responds."""
    try:
        client = await get_cascader_client()
        resp = await client.get("/health")
        return resp.status_code == 200
    except Exception:
        return False
