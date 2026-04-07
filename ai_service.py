import json
import os
import re
from typing import Iterable

import requests

from config import GEMINI_MODEL, GENAI_PROVIDER, OPENAI_MODEL

_OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
_GEMINI_GENERATE_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\u2022|\u00e2\u20ac\u00a2)\s*")


def _strip_bullet_prefix(line: str) -> str:
    return _BULLET_PREFIX_RE.sub("", line).strip()


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
        return value if value > 0 else default
    except Exception:
        return default


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _provider() -> str:
    value = os.getenv("GENAI_PROVIDER", GENAI_PROVIDER).strip().lower()
    if value in {"google", "gemini"}:
        return "gemini"
    if value in {"openai"}:
        return "openai"
    return value


def _request_timeout_seconds() -> int:
    return _env_int("GENAI_TIMEOUT_SECONDS", _env_int("OPENAI_TIMEOUT_SECONDS", 20))


def _looks_like_jwt(token: str) -> bool:
    token = token.strip()
    return token.startswith("eyJ") and token.count(".") == 2


def _openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def _openai_model() -> str:
    return os.getenv("GENAI_MODEL", os.getenv("OPENAI_MODEL", OPENAI_MODEL)).strip() or OPENAI_MODEL


def _openai_url() -> str:
    base = os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE_URL", "")).strip().rstrip("/")
    if base:
        return f"{base}/chat/completions"
    return _OPENAI_CHAT_COMPLETIONS_URL


def _gemini_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()


def _gemini_model() -> str:
    return os.getenv("GENAI_MODEL", os.getenv("GEMINI_MODEL", GEMINI_MODEL)).strip() or GEMINI_MODEL


def _normalize_bullets(bullets: Iterable[str] | None) -> list[str]:
    if not bullets:
        return []
    cleaned = [str(b).strip() for b in bullets]
    return [b for b in cleaned if b]


def _rewrite_prompt(bullets_list: list[str], jd: str) -> str:
    return (
        "Rewrite the resume bullets to be ATS-friendly. "
        "Keep them truthful, concise, and impact-focused. "
        "Do not invent metrics.\n\n"
        f"JD:\n{jd}\n\n"
        "Bullets:\n"
        + "\n".join(bullets_list)
    )


def _clean_generated_bullets(text: str, fallback: list[str]) -> list[str]:
    cleaned = [_strip_bullet_prefix(line) for line in (text or "").splitlines()]
    cleaned = [line for line in cleaned if line]
    return cleaned or fallback


def _rewrite_with_openai(bullets_list: list[str], jd: str) -> str:
    payload = {
        "model": _openai_model(),
        "messages": [
            {
                "role": "system",
                "content": "Rewrite bullets in ATS-friendly style. No fake metrics.",
            },
            {"role": "user", "content": _rewrite_prompt(bullets_list, jd)},
        ],
    }
    res = requests.post(
        _openai_url(),
        headers={
            "Authorization": f"Bearer {_openai_key()}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=_request_timeout_seconds(),
    )
    res.raise_for_status()
    data = res.json()
    return str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


def _rewrite_with_gemini(bullets_list: list[str], jd: str) -> str:
    payload = {
        "contents": [{"parts": [{"text": _rewrite_prompt(bullets_list, jd)}]}],
        "generationConfig": {"temperature": 0.2},
    }
    res = requests.post(
        _GEMINI_GENERATE_URL_TEMPLATE.format(model=_gemini_model(), api_key=_gemini_key()),
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=_request_timeout_seconds(),
    )
    res.raise_for_status()
    data = res.json()
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    return "\n".join(str(part.get("text", "")).strip() for part in parts if isinstance(part, dict)).strip()


def get_genai_status() -> tuple[bool, str]:
    if not _env_truthy("ENABLE_GEN_AI", default=True):
        return False, "disabled (ENABLE_GEN_AI=0)"

    provider = _provider()
    if provider == "openai":
        key = _openai_key()
        if not key:
            return False, "disabled (missing OPENAI_API_KEY)"
        if _looks_like_jwt(key):
            return False, "disabled (OPENAI_API_KEY looks like a login JWT, not an OpenAI key)"
        return True, f"enabled ({provider}:{_openai_model()})"

    if provider == "gemini":
        key = _gemini_key()
        if not key:
            return False, "disabled (missing GEMINI_API_KEY)"
        return True, f"enabled ({provider}:{_gemini_model()})"

    return False, f"disabled (unsupported GENAI_PROVIDER='{provider}')"


def rewrite_bullets(bullets: Iterable[str] | None, jd: str) -> list[str]:
    bullets_list = _normalize_bullets(bullets)
    if not bullets_list:
        return []

    enabled, _ = get_genai_status()
    if not enabled:
        return bullets_list

    try:
        provider = _provider()
        if provider == "gemini":
            text = _rewrite_with_gemini(bullets_list, jd)
        else:
            text = _rewrite_with_openai(bullets_list, jd)
    except Exception:
        return bullets_list

    return _clean_generated_bullets(text, bullets_list)
