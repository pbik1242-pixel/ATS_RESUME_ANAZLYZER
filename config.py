import os
from pathlib import Path

GENAI_PROVIDER = os.getenv("GENAI_PROVIDER", "openai").strip().lower()
GENAI_MODEL = os.getenv("GENAI_MODEL", "").strip()

OPENAI_MODEL = (GENAI_MODEL or os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip() or "gpt-4o-mini"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip() or "gemini-1.5-flash"

JWT_SECRET = os.getenv("JWT_SECRET", "dev-jwt-secret")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
EXPORT_DIR = BASE_DIR / "exports"

UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)
