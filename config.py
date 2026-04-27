import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_env(path: Path = BASE_DIR / ".env") -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


load_env()

GENAI_PROVIDER = os.getenv("GENAI_PROVIDER", "openai").strip().lower()
GENAI_MODEL = os.getenv("GENAI_MODEL", "").strip()

OPENAI_MODEL = (GENAI_MODEL or os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip() or "gpt-4o-mini"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip() or "gemini-1.5-flash"

RECAPTCHA_SITE_KEY = os.getenv("RECAPTCHA_SITE_KEY", "").strip()
RECAPTCHA_SECRET_KEY = os.getenv("RECAPTCHA_SECRET_KEY", "").strip()
RECAPTCHA_MODE = os.getenv("RECAPTCHA_MODE", "local").strip().lower()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "").strip()
GOOGLE_REQUIRE_VERIFIED_EMAIL = os.getenv("GOOGLE_REQUIRE_VERIFIED_EMAIL", "false").strip().lower() in {"1", "true", "yes"}

JWT_SECRET = os.getenv("JWT_SECRET", "dev-jwt-secret")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER).strip()
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").strip().lower() not in {"0", "false", "no"}

UPLOAD_DIR = BASE_DIR / "uploads"
EXPORT_DIR = BASE_DIR / "exports"

UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)
