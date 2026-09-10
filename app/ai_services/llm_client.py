"""
llm_client.py
-------------
Routes LLM inference calls to the correct provider based on the tier
returned by the Model Cascader.

Tier -> Provider mapping (mirrors cascade_config.yaml):
  Tier 1 -- Groq         llama-3.3-70b-versatile
  Tier 2 -- Google GenAI gemini-2.5-flash     (GOOGLE_API_KEY_TIER2)
  Tier 3 -- Google GenAI gemini-3.5-flash     (GOOGLE_API_KEY_TIER3)

All errors are caught and logged -- never raises to the caller.
Returns None on failure; the router treats None as a graceful miss.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# =========================================
# CONFIG
# =========================================

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

GOOGLE_API_KEY_TIER2: Optional[str] = os.getenv("GOOGLE_API_KEY_TIER2")
GOOGLE_API_KEY_TIER3: Optional[str] = os.getenv("GOOGLE_API_KEY_TIER3")

DEFAULT_SYSTEM_PROMPT: str = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    "You are CEREBRUS, a helpful AI assistant.",
)

# Model IDs per tier — change these in .env without touching code.
TIER1_MODEL: str = os.getenv("TIER1_MODEL", "llama-3.3-70b-versatile")
TIER2_MODEL: str = os.getenv("TIER2_MODEL", "gemini-2.5-flash")
TIER3_MODEL: str = os.getenv("TIER3_MODEL", "gemini-3.5-flash")

FALLBACK_TIER: int = int(os.getenv("FALLBACK_TIER", "1"))

LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))


def _model_for_tier(tier: int) -> str:
    """Resolve model ID from env var based on tier number."""
    return {1: TIER1_MODEL, 2: TIER2_MODEL, 3: TIER3_MODEL}.get(tier, TIER1_MODEL)


# =========================================
# LAZY SINGLETONS -- instantiated once on first use
# =========================================

_groq_client = None
_google_client_tier2 = None
_google_client_tier3 = None


def _get_groq_client():
    """Return (or lazily create) the async Groq client."""
    global _groq_client
    if _groq_client is None:
        try:
            from groq import AsyncGroq
            _groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        except ImportError:
            log.error("[LLM] groq package not installed -- run: pip install groq")
            raise
    return _groq_client


def _get_google_client(tier: int):
    """Return (or lazily create) the Google GenAI client for the given tier."""
    global _google_client_tier2, _google_client_tier3

    if tier == 2:
        if _google_client_tier2 is None:
            try:
                from google import genai
                _google_client_tier2 = genai.Client(api_key=GOOGLE_API_KEY_TIER2)
            except ImportError:
                log.error("[LLM] google-genai package not installed -- run: pip install google-genai")
                raise
        return _google_client_tier2

    # tier == 3
    if _google_client_tier3 is None:
        try:
            from google import genai
            _google_client_tier3 = genai.Client(api_key=GOOGLE_API_KEY_TIER3)
        except ImportError:
            log.error("[LLM] google-genai package not installed -- run: pip install google-genai")
            raise
    return _google_client_tier3


# =========================================
# PROMPT BUILDER
# =========================================

def _build_system_prompt(summary: Optional[str]) -> str:
    """
    Build the system prompt.

    If a context summary is provided (needs_context=True), prepend it so the
    LLM is aware of conversation history.
    """
    if summary:
        return (
            f"{DEFAULT_SYSTEM_PROMPT}\n\n"
            f"The following is a summary of the conversation so far. "
            f"Use it to inform your response:\n\n{summary}"
        )
    return DEFAULT_SYSTEM_PROMPT


# =========================================
# TIER 1 -- Groq (llama-3.3-70b-versatile)
# =========================================

async def _call_groq(model: str, prompt: str, system_prompt: str) -> tuple[Optional[str], Optional[dict]]:
    """Call Groq chat completions API (async). Returns (text, token_usage)."""
    if not GROQ_API_KEY:
        log.error("[LLM][Groq] GROQ_API_KEY is not set")
        return None, None
    try:
        client = _get_groq_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=LLM_MAX_TOKENS,
        )
        text = response.choices[0].message.content
        usage = response.usage
        token_usage = {
            "prompt_tokens":     usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens":      usage.total_tokens,
            "provider":          "Groq",
            "model":             model,
        } if usage else None
        return text, token_usage
    except Exception as exc:
        log.error("[LLM][Groq] Error: %s", exc, exc_info=True)
        return None, None


# =========================================
# TIER 2 & 3 -- Google GenAI (Gemini)
# =========================================

async def _call_google(model: str, tier: int, prompt: str, system_prompt: str) -> tuple[Optional[str], Optional[dict]]:
    """Call Google GenAI generate_content (async). Returns (text, token_usage)."""
    key = GOOGLE_API_KEY_TIER2 if tier == 2 else GOOGLE_API_KEY_TIER3
    if not key:
        log.error("[LLM][Google] GOOGLE_API_KEY_TIER%d is not set", tier)
        return None, None
    try:
        from google.genai import types
        client = _get_google_client(tier)
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )
        text = response.text
        meta = response.usage_metadata
        token_usage = {
            "prompt_tokens":     meta.prompt_token_count,
            "completion_tokens": meta.candidates_token_count,
            "total_tokens":      meta.total_token_count,
            "provider":          "Google",
            "model":             model,
        } if meta else None
        return text, token_usage
    except Exception as exc:
        log.error("[LLM][Google Tier %d] Error: %s", tier, exc, exc_info=True)
        return None, None


# =========================================
# PUBLIC INTERFACE
# =========================================

async def call_llm(
    tier: int,
    prompt: str,
    summary: Optional[str] = None,
) -> tuple[Optional[str], Optional[dict]]:
    """
    Route the prompt to the correct LLM provider based on tier.

    Model IDs are read from env vars (TIER1_MODEL / TIER2_MODEL / TIER3_MODEL)
    so they can be swapped without touching code.

    Args:
        tier:    1 | 2 | 3 -- determines provider and model.
        prompt:  Original user prompt text.
        summary: Optional rolling context summary from the Context Classifier.
                 If provided, it is injected into the system prompt.

    Returns:
        A tuple of (text, token_usage) where:
          - text is the LLM response string, or None on failure.
          - token_usage is a dict with prompt_tokens, completion_tokens,
            total_tokens, provider, and model — or None on failure.
        Never raises.
    """
    model = _model_for_tier(tier)
    system_prompt = _build_system_prompt(summary)
    log.info("[LLM] Calling tier=%d model=%s (context=%s)", tier, model, "yes" if summary else "no")

    if tier == 1:
        return await _call_groq(model=model, prompt=prompt, system_prompt=system_prompt)
    elif tier in (2, 3):
        return await _call_google(model=model, tier=tier, prompt=prompt, system_prompt=system_prompt)
    else:
        fallback_model = _model_for_tier(FALLBACK_TIER)
        log.warning("[LLM] Unknown tier=%d, falling back to tier %d (%s)", tier, FALLBACK_TIER, fallback_model)
        if FALLBACK_TIER == 1:
            return await _call_groq(model=fallback_model, prompt=prompt, system_prompt=system_prompt)
        else:
            return await _call_google(model=fallback_model, tier=FALLBACK_TIER, prompt=prompt, system_prompt=system_prompt)
