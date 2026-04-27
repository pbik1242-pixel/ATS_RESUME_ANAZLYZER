from __future__ import annotations

import base64
import csv
import io
import hashlib
import hmac
import json
import os
import re
import smtplib
import sqlite3
from datetime import datetime, timedelta
from email.message import EmailMessage
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
import auth
import database as db_mod
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from ai_service import get_genai_status, rewrite_bullets
from config import (
    EXPORT_DIR,
    JWT_EXPIRE_MINUTES,
    JWT_SECRET,
    SMTP_FROM,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USE_TLS,
    SMTP_USER,
    UPLOAD_DIR,
)
from resume_utils import (
    clean,
    export_docx,
    extract_bullets,
    extract_contact_details,
    extract_keywords,
    extract_tables,
    extract_text,
    formatting_score,
    keyword_score,
    semantic_score,
    tables_to_text,
)


def load_env(path: Path = Path(".env")) -> None:
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
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database.db"
DATASET_CSV = BASE_DIR / "resumesss" / "Resume" / "Resume.py.csv"
TEMPLATES_DIR = BASE_DIR / "templates"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
PAGE_SIZE = 24
PIPELINE_STATUSES = (
    "new",
    "shortlisted",
    "interview_scheduled",
    "interviewed",
    "rejected",
    "hired",
)
STATUS_LABELS = {
    "new": "New",
    "shortlisted": "Shortlisted",
    "interview_scheduled": "Interview Scheduled",
    "interviewed": "Interviewed",
    "rejected": "Rejected",
    "hired": "Hired",
}
ADMIN_EMAILS = {
    email.lower()
    for email in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", os.getenv("ADMIN_EMAILS", ""))
}

UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ATS Resume Analyzer")
app.add_middleware(
    SessionMiddleware,
    secret_key=JWT_SECRET,
    same_site="lax",
    https_only=False,
    max_age=60 * 60 * 24 * 14,
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
if TEMPLATES_DIR.exists():
    app.mount("/templates", StaticFiles(directory=str(TEMPLATES_DIR)), name="templates")

app.include_router(auth.router)

@app.on_event("startup")
def startup() -> None:
    db_mod.ensure_schema()


def normalize_upload_name(filename: str | None) -> tuple[str, str]:
    raw = str(filename or "").replace("\\", "/").split("/")[-1]
    safe = secure_filename(raw) or "resume.txt"
    return safe, Path(safe).suffix.lower()


def fallback_resume_points(text: str, limit: int = 12) -> list[str]:
    src = str(text or "")
    if not src.strip():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in src.splitlines():
        line = re.sub(r"\s+", " ", raw).strip().lstrip("-* ").strip()
        if 35 <= len(line) <= 260 and line.lower() not in seen:
            seen.add(line.lower())
            out.append(line)
            if len(out) >= limit:
                return out
    for part in re.split(r"(?<=[.!?])\s+", clean(src)):
        part = part.strip()
        if 35 <= len(part) <= 260 and part.lower() not in seen:
            seen.add(part.lower())
            out.append(part)
            if len(out) >= limit:
                return out
    return out[:limit]


ACTION_VERBS = (
    "analyzed", "built", "created", "delivered", "designed", "developed", "drove", "executed",
    "implemented", "improved", "led", "managed", "optimized", "resolved", "streamlined", "supported",
    "worked", "owned", "launched", "collaborated", "mentored", "coordinated",
)

SKILL_LABELS = {
    "c": "C",
    "cpp": "C++",
    "cplus": "C++",
    "csharp": "C#",
    "java": "Java",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "python": "Python",
    "ruby": "Ruby",
    "php": "PHP",
    "go": "Go",
    "golang": "Go",
    "rust": "Rust",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "scala": "Scala",
    "r": "R",
    "sql": "SQL",
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "oracle": "Oracle",
    "sqlite": "SQLite",
    "dotnet": ".NET",
    "net": ".NET",
    "node": "Node.js",
    "react": "React",
    "angular": "Angular",
    "vue": "Vue",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "spring": "Spring",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "terraform": "Terraform",
    "git": "Git",
    "linux": "Linux",
    "powershell": "PowerShell",
    "ssis": "SSIS",
    "ssrs": "SSRS",
    "ssms": "SSMS",
}

SKILL_ALLOWLIST = set(SKILL_LABELS.keys())
SKILL_PHRASES = {
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "data analysis": "Data Analysis",
    "data analytics": "Data Analytics",
    "data engineering": "Data Engineering",
    "data science": "Data Science",
    "power bi": "Power BI",
    "tableau": "Tableau",
    "excel": "Excel",
    "etl": "ETL",
    "api": "API",
    "rest api": "REST API",
    "microservices": "Microservices",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scikit learn": "Scikit-learn",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "html": "HTML",
    "css": "CSS",
    "bootstrap": "Bootstrap",
}


def _normalize_point(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    return cleaned.rstrip(";:,. ") + ("" if not cleaned else ".")


def combine_points(points: list[str]) -> list[str]:
    seen: set[str] = set()
    combined: list[str] = []
    for raw in points or []:
        cleaned = _normalize_point(raw)
        key = cleaned.lower()
        if len(cleaned) < 8 or key in seen:
            continue
        seen.add(key)
        combined.append(cleaned)
    return combined


def build_suggestions(points: list[str], missing: list[str]) -> list[str]:
    suggestions: list[str] = []
    missing_clean = [m.strip() for m in (missing or []) if m and m.strip() and m.strip() != "-"]
    if missing_clean:
        suggestions.append(
            "Add these missing keywords where they are true: "
            + ", ".join(missing_clean[:8])
            + ("..." if len(missing_clean) > 8 else "")
        )
    if points and any(not re.search(r"\d", p) for p in points):
        suggestions.append("Quantify impact with numbers (time saved, volume processed, % improvements).")
    if points and any(len(p) > 170 for p in points):
        suggestions.append("Shorten long bullets to 1 line and move extra detail into tools or scope lines.")
    if points and any(not re.match(r"(?i)^(" + "|".join(ACTION_VERBS) + r")\b", p) for p in points):
        suggestions.append("Start each bullet with a strong action verb to show ownership and outcomes.")
    if not points:
        suggestions.append("Add 4-6 focused bullets that describe impact, tools, and measurable results.")
    return suggestions


def _normalize_skill_token(token: str) -> str:
    key = re.sub(r"[^a-z0-9+.#]", "", token.lower())
    if key in SKILL_LABELS:
        return SKILL_LABELS[key]
    return token.strip()


def filter_skill_terms(terms: list[str]) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for raw in terms or []:
        token = str(raw or "").strip()
        if not token:
            continue
        key = re.sub(r"[^a-z0-9+.#]", "", token.lower())
        # keep experience years like "3 years", "5 yrs", "8+ years"
        if re.search(r"\b\d+\s*\+?\s*(years?|yrs?)\b", token, re.I):
            label = re.sub(r"\s+", " ", token)
        elif key in SKILL_ALLOWLIST:
            label = _normalize_skill_token(key)
        else:
            continue
        out_key = label.lower()
        if out_key in seen:
            continue
        seen.add(out_key)
        filtered.append(label)
    return filtered


def extract_profile_skills(text: str, limit: int = 20) -> list[str]:
    source = str(text or "")
    if not source.strip():
        return []
    lowered = source.lower()
    found: list[str] = []
    seen: set[str] = set()

    for phrase, label in SKILL_PHRASES.items():
        pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            key = label.lower()
            if key not in seen:
                seen.add(key)
                found.append(label)
                if len(found) >= limit:
                    return found

    for raw_key, label in SKILL_LABELS.items():
        pattern = r"(?<![a-z0-9])" + re.escape(raw_key) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            key = label.lower()
            if key not in seen:
                seen.add(key)
                found.append(label)
                if len(found) >= limit:
                    return found

    return found


def build_table_preview(
    tables: list[list[list[str]]],
    max_tables: int = 2,
    max_rows: int = 6,
    max_cols: int = 6,
    max_cell_len: int = 80,
) -> list[list[list[str]]]:
    preview: list[list[list[str]]] = []
    for table in tables[:max_tables]:
        rows: list[list[str]] = []
        for row in table[:max_rows]:
            trimmed: list[str] = []
            for cell in row[:max_cols]:
                value = (cell or "").strip()
                if max_cell_len and len(value) > max_cell_len:
                    value = value[: max_cell_len - 3] + "..."
                trimmed.append(value)
            if any(trimmed):
                rows.append(trimmed)
        if rows:
            preview.append(rows)
    return preview


def analyze_resume_text(
    display_name: str,
    raw_resume_text: str,
    job_text: str,
    job_skills: list[str],
    include_rewrite: bool,
    ext: str,
    table_text: str = "",
    tables: list[list[list[str]]] | None = None,
) -> dict[str, Any]:
    combined_raw = f"{raw_resume_text}\n{table_text}".strip()
    resume_text = clean(combined_raw)
    if not resume_text:
        raise ValueError(
            f"{display_name}: unable to extract text or tables. The file may be image-only, password-protected, or corrupted."
        )
    bullets = extract_bullets(raw_resume_text) or fallback_resume_points(raw_resume_text) or fallback_resume_points(resume_text)
    k_score, found, missing = keyword_score(resume_text, job_skills)
    display_found = filter_skill_terms(found)
    display_missing = filter_skill_terms(missing)
    display_job_skills = filter_skill_terms(job_skills) or extract_profile_skills(job_text) or job_skills[:10]
    resume_skills = extract_profile_skills(combined_raw) or display_found
    s_score = semantic_score(resume_text, job_text)
    f_score = formatting_score(resume_text)
    score = round((((k_score * 0.5) + (f_score * 0.2)) / 0.7 if s_score is None else (k_score * 0.5 + s_score * 0.3 + f_score * 0.2)) * 100, 1)
    rewritten = rewrite_bullets(bullets, job_text) if include_rewrite else []
    tables = tables or []
    return {
        "filename": display_name,
        "ext": ext,
        "score": score,
        "found": found,
        "missing": missing,
        "found_text": ", ".join(display_found) or "-",
        "missing_text": ", ".join(display_missing) or "-",
        "found_display": display_found,
        "missing_display": display_missing,
        "job_skills_display": display_job_skills,
        "resume_skills_display": resume_skills,
        "rewritten": rewritten,
        "skills_csv": ",".join(display_job_skills),
        "job_skills_csv": ",".join(display_job_skills),
        "resume_skills_csv": ",".join(resume_skills),
        "table_count": len(tables),
        "table_preview": build_table_preview(tables),
    }


async def analyze_uploaded_file(uploaded_file: UploadFile, job_text: str, job_skills: list[str], include_rewrite: bool = True) -> dict[str, Any]:
    safe, ext = normalize_upload_name(uploaded_file.filename)
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS)).upper().replace(".", "")
        raise ValueError(f"{safe}: unsupported file type. Please upload {supported} files.")
    unique = f"{Path(safe).stem}_{uuid4().hex[:10]}{ext}"
    path = UPLOAD_DIR / unique
    data = await uploaded_file.read()
    if not data:
        raise ValueError(f"{safe}: empty or unreadable file. Please upload a non-empty document.")
    path.write_bytes(data)
    raw_text = extract_text(path)
    tables = extract_tables(path)
    table_text = tables_to_text(tables)
    return analyze_resume_text(safe, raw_text, job_text, job_skills, include_rewrite, ext, table_text, tables)


def parse_saved_bullets(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            cleaned = []
            for x in parsed:
                text = str(x or "").strip()
                if not text:
                    continue
                # drop stray brackets/commas
                text = text.strip("[] ,")
                if len(text) < 3:
                    continue
                cleaned.append(text)
            return cleaned
    except Exception:
        # handle python-style list strings like "['a', 'b']"
        try:
            import ast

            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                cleaned = []
                for x in parsed:
                    text = str(x or "").strip().strip("[] ,")
                    if len(text) < 3:
                        continue
                    cleaned.append(text)
                return cleaned
        except Exception:
            pass
    return [line.strip().strip("[] ,") for line in str(raw).splitlines() if line.strip().strip("[] ,")]


def find_uploaded_resume_path(filename: str | None) -> Path | None:
    safe, ext = normalize_upload_name(filename)
    direct = UPLOAD_DIR / safe
    if direct.exists():
        return direct
    matches = list(UPLOAD_DIR.glob(f"{Path(safe).stem}_*{ext}"))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


@lru_cache(maxsize=256)
def resume_contact_snapshot(filename: str | None) -> dict[str, str]:
    """Return candidate name, contact, and email pulled from the resume text."""
    fallback_name = Path(str(filename or "Candidate")).stem or "Candidate"
    try:
        path = find_uploaded_resume_path(filename)
        if not path:
            return {"candidate_name": fallback_name, "contact": "", "email": ""}
        raw = extract_text(path)
        info = extract_contact_details(raw, fallback_name)
        return {
            "candidate_name": info.get("candidate_name") or fallback_name,
            "contact": info.get("contact") or "",
            "email": info.get("email") or "",
        }
    except Exception:
        return {"candidate_name": fallback_name, "contact": "", "email": ""}


def rehydrate_history_rewrite(history_id: int, user_id: int, filename: str | None, skills_csv: str | None) -> list[str]:
    path = find_uploaded_resume_path(filename)
    if not path:
        return []
    raw = extract_text(path)
    bullets = extract_bullets(raw) or fallback_resume_points(raw)
    if not bullets:
        return []
    job_text = "Target skills: " + ", ".join([s.strip() for s in (skills_csv or "").split(",") if s.strip()])
    rewritten = rewrite_bullets(bullets, job_text) or bullets
    with db_mod.db() as conn:
        conn.execute("UPDATE resume_history SET rewritten=? WHERE id=? AND user_id=?", (json.dumps(rewritten), history_id, user_id))
    return rewritten


def list_job_descriptions(user_id: int) -> list[sqlite3.Row]:
    with db_mod.db() as conn:
        return conn.execute(
            "SELECT id,title,description,created_at,updated_at FROM job_descriptions WHERE user_id=? ORDER BY updated_at DESC, id DESC",
            (user_id,),
        ).fetchall()


def get_job_description(user_id: int, job_id: int) -> sqlite3.Row | None:
    with db_mod.db() as conn:
        return conn.execute(
            "SELECT id,title,description,created_at,updated_at FROM job_descriptions WHERE id=? AND user_id=?",
            (job_id, user_id),
        ).fetchone()


def save_job_description(user_id: int, title: str, description: str, job_id: int | None = None) -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    clean_title = clean(title) or "Untitled Role"
    clean_description = clean(description)
    with db_mod.db() as conn:
        if job_id:
            conn.execute(
                "UPDATE job_descriptions SET title=?, description=?, updated_at=? WHERE id=? AND user_id=?",
                (clean_title, clean_description, timestamp, job_id, user_id),
            )
            return job_id
        inserted = conn.execute(
            "INSERT INTO job_descriptions(user_id,title,description,created_at,updated_at) VALUES (?,?,?,?,?)",
            (user_id, clean_title, clean_description, timestamp, timestamp),
        )
        return int(inserted.lastrowid)


def delete_job_description(user_id: int, job_id: int) -> bool:
    with db_mod.db() as conn:
        deleted = conn.execute("DELETE FROM job_descriptions WHERE id=? AND user_id=?", (job_id, user_id)).rowcount
    return bool(deleted)


def infer_job_title(job_text: str) -> str:
    lines = [line.strip(" -:\t") for line in str(job_text or "").splitlines() if line.strip()]
    if not lines:
        return "Target Role"
    first = lines[0]
    match = re.search(r"(?i)role\s*[:\-]\s*(.+)", first)
    if match:
        return clean(match.group(1))[:120] or "Target Role"
    return clean(first)[:120] or "Target Role"

def is_admin_user(user: sqlite3.Row) -> bool:
    role = str(user["role"] or "").strip().lower()
    return role == "admin" or str(user["email"] or "").strip().lower() in ADMIN_EMAILS or int(user["id"]) == 1


def is_hr_user(user: sqlite3.Row) -> bool:
    try:
        return str(user["role"] or "").lower() == "hr"
    except Exception:
        return False


def normalize_status(value: str | None, default: str = "new") -> str:
    key = str(value or "").strip().lower()
    return key if key in PIPELINE_STATUSES else default


def status_label(value: str | None) -> str:
    return STATUS_LABELS.get(normalize_status(value), STATUS_LABELS["new"])


def status_badge_class(value: str | None) -> str:
    status = normalize_status(value)
    return {
        "new": "bg-slate-100 text-slate-700",
        "shortlisted": "bg-emerald-100 text-emerald-700",
        "interview_scheduled": "bg-sky-100 text-sky-700",
        "interviewed": "bg-amber-100 text-amber-700",
        "rejected": "bg-rose-100 text-rose-700",
        "hired": "bg-violet-100 text-violet-700",
    }.get(status, "bg-slate-100 text-slate-700")


def smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_PORT and SMTP_FROM)


def smtp_status_message() -> str:
    missing: list[str] = []
    if not SMTP_HOST:
        missing.append("SMTP_HOST")
    if not SMTP_PORT:
        missing.append("SMTP_PORT")
    if not SMTP_FROM:
        missing.append("SMTP_FROM")
    if not SMTP_USER:
        missing.append("SMTP_USER")
    if not SMTP_PASSWORD:
        missing.append("SMTP_PASSWORD")
    if missing:
        return "Missing SMTP settings: " + ", ".join(missing)
    host = SMTP_HOST.strip().lower()
    if host in {"bloody_beast", "localhost", "127.0.0.1"} or "_" in host:
        return (
            "SMTP_HOST looks invalid. Set it to a real mail server such as smtp.gmail.com "
            "and keep SMTP_PORT=587 with SMTP_USE_TLS=true."
        )
    if SMTP_FROM and SMTP_USER and SMTP_FROM.lower() != SMTP_USER.lower():
        return (
            "SMTP_FROM should usually match SMTP_USER for Gmail/Outlook accounts. "
            "Use the same email address unless your provider allows a different sender."
        )
    return "SMTP settings look complete."


def send_interview_selection_email(
    to_email: str,
    candidate_name: str,
    interview_date: str,
    contact: str = "",
    notes: str = "",
    recruiter_name: str = "",
) -> bool:
    recipient = str(to_email or "").strip()
    if not recipient or not smtp_configured():
        return False

    subject = "You have been selected for the interview"
    body_lines = [
        f"Hello {candidate_name or 'Candidate'},",
        "",
        "We are pleased to inform you that you have been selected for the next stage of the interview process.",
        "",
        f"Interview Date: {interview_date or 'To be confirmed'}",
        f"Contact: {contact or 'N/A'}",
    ]
    if notes.strip():
        body_lines.extend(["", f"Additional Notes: {notes.strip()}"])
    body_lines.extend(
        [
            "",
            "Please reply to this email if you have any questions or if you need any clarification.",
            "",
            f"Best regards,",
            recruiter_name or (SMTP_FROM or "HR Team"),
        ]
    )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = SMTP_FROM
    message["To"] = recipient
    message.set_content("\n".join(body_lines))

    try:
        if SMTP_USE_TLS:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.ehlo()
                server.starttls()
                if SMTP_USER and SMTP_PASSWORD:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                if SMTP_USER and SMTP_PASSWORD:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
        return True
    except Exception:
        return False

@lru_cache(maxsize=1)
def load_dataset() -> dict[str, Any]:
    if not DATASET_CSV.exists():
        return {"rows": [], "categories": []}
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    try:
        with DATASET_CSV.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            for idx, row in enumerate(csv.DictReader(handle), start=1):
                text = clean(str(row.get("Resume_str") or ""))
                if not text:
                    continue
                category = clean(str(row.get("Category") or "")) or "Uncategorized"
                rid = clean(str(row.get("ID") or idx)) or str(idx)
                counts[category] = counts.get(category, 0) + 1
                rows.append(
                    {
                        "id": rid,
                        "category": category,
                        "preview": text[:380] + ("..." if len(text) > 380 else ""),
                        "char_count": len(text),
                        "search_blob": f"{rid} {category} {text}".lower(),
                    }
                )
    except Exception:
        return {"rows": [], "categories": []}
    cats = [{"name": n, "count": c} for n, c in sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))]
    return {"rows": rows, "categories": cats}


def dataset_context(category: str = "", query: str = "", page: int = 1) -> dict[str, Any]:
    data = load_dataset()
    rows = data["rows"]
    category = clean(category)
    query = clean(query).lower()
    if category:
        rows = [r for r in rows if r["category"].lower() == category.lower()]
    if query:
        rows = [r for r in rows if query in r["search_blob"]]
    total = len(rows)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(1, page), total_pages)
    start = (page - 1) * PAGE_SIZE
    return {
        "dataset_total": len(data["rows"]),
        "dataset_categories": data["categories"],
        "results": rows[start : start + PAGE_SIZE],
        "total_results": total,
        "page": page,
        "total_pages": total_pages,
    }


def get_history_row(user_id: int, history_id: int) -> sqlite3.Row | None:
    with db_mod.db() as conn:
        return conn.execute(
            "SELECT id,filename,filetype,score,date,found,missing,rewritten,skills,job_skills,resume_skills,batch_id,folder_name,job_id,job_title,candidate_status,recruiter_notes FROM resume_history WHERE id=? AND user_id=?",
            (history_id, user_id),
        ).fetchone()


def save_single_history(user_id: int, result: dict[str, Any], job_id: int | None = None, job_title: str = "") -> int:
    with db_mod.db() as conn:
        inserted = conn.execute(
            "INSERT INTO resume_history(user_id,score,filename,filetype,date,found,missing,rewritten,skills,job_skills,resume_skills,batch_id,folder_name,job_id,job_title,candidate_status,recruiter_notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                user_id,
                result["score"],
                result["filename"],
                result["ext"].lstrip(".").upper(),
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                result["found_text"],
                result["missing_text"],
                json.dumps(result["rewritten"]),
                result["job_skills_csv"],
                result["job_skills_csv"],
                result["resume_skills_csv"],
                None,
                None,
                job_id,
                job_title,
                "new",
                "",
            ),
        )
        return int(inserted.lastrowid)


def save_bulk_history(user_id: int, results: list[dict[str, Any]], folder_name: str, job_id: int | None = None, job_title: str = "") -> str:
    batch_id = f"bulk-{uuid4().hex}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    label = folder_name.strip() or "Folder Upload"
    with db_mod.db() as conn:
        for row in results:
            inserted = conn.execute(
                "INSERT INTO resume_history(user_id,score,filename,filetype,date,found,missing,rewritten,skills,job_skills,resume_skills,batch_id,folder_name,job_id,job_title,candidate_status,recruiter_notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    user_id,
                    row["score"],
                    row["filename"],
                    row["ext"].lstrip(".").upper(),
                    timestamp,
                    row["found_text"],
                    row["missing_text"],
                    json.dumps(row["rewritten"]),
                    row["job_skills_csv"],
                    row["job_skills_csv"],
                    row["resume_skills_csv"],
                    batch_id,
                    label,
                    job_id,
                    job_title,
                    "new",
                    "",
                ),
            )
            row["history_id"] = int(inserted.lastrowid)
    return batch_id


def base_context(request: Request, **extra: Any) -> dict[str, Any]:
    user = current_user(request)
    is_logged_in = bool(user)
    is_hr = is_hr_user(user) if user else False
    is_admin = is_admin_user(user) if user else False

    primary_nav_items: list[dict[str, str]] = [{"label": "ATS RESUME ANALYZER", "href": "/"}]
    if is_hr:
        primary_nav_items.extend(
            [
                {"label": "Analyzed Resumes", "href": "/analyzed-resumes"},
                {"label": "Interview List", "href": "/interview-list"},
            ]
        )
    primary_nav_items.append({"label": "About", "href": "/about"})

    if is_logged_in:
        profile_menu_items = [
            {"label": "My Profile", "href": "/profile"},
            {"label": "Logout", "href": "/logout"},
        ]
    else:
        profile_menu_items = [
            {"label": "About", "href": "/about"},
            {"label": "Sign In", "href": "/login"},
            {"label": "Sign Up", "href": "/register"},
        ]

    return {
        "request": request,
        "current_user": user,
        "is_logged_in": is_logged_in,
        "is_hr": is_hr,
        "is_admin": is_admin,
        "primary_nav_items": primary_nav_items,
        "profile_menu_items": profile_menu_items,
        "role_label": "HR Organization" if is_hr else ("Administrator" if is_admin else "Candidate"),
        "pipeline_statuses": [{"value": item, "label": STATUS_LABELS[item]} for item in PIPELINE_STATUSES],
        **extra,
    }


def redirect(url: str, status_code: int = 303) -> RedirectResponse:
    return RedirectResponse(url=url, status_code=status_code)


def set_session_user(request: Request, user: sqlite3.Row, remember_me: bool = False) -> None:
    request.session["user_id"] = int(user["id"])
    request.session["user_name"] = user["name"] or ""
    request.session["user_email"] = user["email"] or ""
    request.session["user_role"] = (user["role"] or "candidate").lower()
    request.session["remember_me"] = bool(remember_me)
    # default consent to False until explicitly granted
    if "resume_consent" not in request.session:
        request.session["resume_consent"] = False


def clear_session(request: Request) -> None:
    request.session.clear()


def current_user(request: Request) -> sqlite3.Row | None:
    uid = request.session.get("user_id")
    if not uid:
        return None
    user = db_mod.get_user_by_id(int(uid))
    if not user:
        request.session.clear()
    return user


def require_user(request: Request) -> sqlite3.Row:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=303, detail="/login")
    return user


def require_consent(request: Request) -> None:
    if not request.session.get("resume_consent"):
        raise HTTPException(status_code=303, detail="/consent")


def require_hr(request: Request) -> sqlite3.Row:
    user = require_user(request)
    if not is_hr_user(user):
        raise HTTPException(status_code=403, detail="HR role required.")
    return user


def render_error(request: Request, code: int, message: str, status_code: int | None = None) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "error.html",
        base_context(request, code=code, message=message),
        status_code=status_code or code,
    )


def recent_history_for(user_id: int, limit: int = 6) -> list[sqlite3.Row]:
    with db_mod.db() as conn:
        return conn.execute("SELECT id,filename,score,date FROM resume_history WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()


def get_visible_pages(current_page: int, total_pages: int, window: int = 2) -> list[int]:
    start = max(1, current_page - window)
    end = min(total_pages, current_page + window)
    return list(range(start, end + 1))


def build_history_groups(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    batch_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        batch_id = str(row["batch_id"] or "").strip()
        if batch_id:
            group = batch_map.get(batch_id)
            if not group:
                group = {
                    "group_type": "folder",
                    "title": row["folder_name"] or "Folder Upload",
                    "latest_date": row["date"] or "-",
                    "items": [],
                }
                batch_map[batch_id] = group
                groups.append(group)
            group["items"].append(row)
        else:
            groups.append(
                {
                    "group_type": "single",
                    "title": row["filename"] or "Untitled Resume",
                    "latest_date": row["date"] or "-",
                    "items": [row],
                }
            )
    for group in groups:
        group["count"] = len(group["items"])
    return groups


def _clean_list(raw: str | None, limit: int = 4) -> list[str]:
    return [s.strip() for s in str(raw or "").split(",") if s.strip() and s.strip() != "-"][:limit]


def _score_display(score: Any) -> str:
    try:
        return f"{float(score):.1f}%"
    except Exception:
        return "-"


def _resume_summary(name: str, score_display: str, found_list: list[str], missing_list: list[str]) -> str:
    lead = f"{name} scored {score_display}".strip()
    strengths = f" after matching {', '.join(found_list[:4])}" if found_list else ""
    experience = " Shows strong programming background and solid experience." if found_list else " Shows solid overall experience."
    gaps = ""
    if missing_list:
        gaps = f" Consider adding {', '.join(missing_list[:3])} to strengthen alignment."
    return (lead + strengths + "." + experience + gaps).strip()


def analyzed_resumes_for(user_id: int) -> list[dict[str, Any]]:
    with db_mod.db() as conn:
        rows = conn.execute(
            "SELECT id,filename,filetype,score,date,found,missing FROM resume_history WHERE user_id=? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
    scores_sorted = sorted({float(row["score"]) for row in rows if row["score"] is not None}, reverse=True)
    payload: list[dict[str, Any]] = []
    for row in rows:
        found_list = _clean_list(row["found"], 4)
        missing_list = _clean_list(row["missing"], 3)
        contact_snapshot = resume_contact_snapshot(row["filename"])
        candidate_name = contact_snapshot["candidate_name"]
        score_val = float(row["score"]) if row["score"] is not None else None
        rank = scores_sorted.index(score_val) + 1 if score_val is not None else None
        reason_bits: list[str] = []
        if row["score"] is not None:
            reason_bits.append(f"ATS score {_score_display(row['score'])}")
        if found_list:
            reason_bits.append(f"matched {', '.join(found_list)}")
        if missing_list:
            reason_bits.append(f"could improve {', '.join(missing_list)}")
        if not reason_bits:
            reason_bits.append("Selected from your analyzed resumes.")
        summary = _resume_summary(candidate_name, _score_display(row["score"]), found_list, missing_list)
        payload.append(
            {
                "id": int(row["id"]),
                "filename": row["filename"],
                "filetype": (row["filetype"] or "").upper(),
                "score": row["score"],
                "score_display": _score_display(row["score"]),
                "rank": rank,
                "date": row["date"],
                "found_preview": ", ".join(found_list) or "-",
                "missing_preview": ", ".join(missing_list) or "-",
                "selection_reason": " • ".join(reason_bits),
                "candidate_name": candidate_name,
                "contact": contact_snapshot["contact"],
                "email": contact_snapshot["email"],
                "summary": summary,
            }
        )
    return payload


def save_interview_entry(
    user_id: int,
    history_id: int,
    candidate_name: str,
    contact: str,
    email: str,
    interview_date: str,
    notes: str,
) -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with db_mod.db() as conn:
        existing = conn.execute(
            "SELECT id FROM interview_list WHERE user_id=? AND history_id=?", (user_id, history_id)
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE interview_list
                SET candidate_name=?, contact=?, email=?, interview_date=?, notes=?, updated_at=?
                WHERE id=?
                """,
                (candidate_name, contact, email, interview_date, notes, timestamp, int(existing["id"])),
            )
            return int(existing["id"])
        else:
            inserted = conn.execute(
                """
                INSERT INTO interview_list(user_id,history_id,candidate_name,contact,email,interview_date,notes,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (user_id, history_id, candidate_name, contact, email, interview_date, notes, timestamp, timestamp),
            )
            return int(inserted.lastrowid)


def interview_entries_for(user_id: int) -> list[dict[str, Any]]:
    with db_mod.db() as conn:
        rows = conn.execute(
            """
            SELECT il.id, il.history_id, il.candidate_name, il.contact, il.email, il.interview_date, il.notes,
                   il.created_at, il.updated_at, il.email_sent_at,
                   rh.filename, rh.filetype, rh.score, rh.found, rh.missing, rh.date AS analyzed_date
            FROM interview_list il
            LEFT JOIN resume_history rh ON rh.id = il.history_id
            WHERE il.user_id=?
            ORDER BY COALESCE(il.interview_date, il.updated_at, il.created_at) ASC, il.id DESC
            """,
            (user_id,),
        ).fetchall()
    entries: list[dict[str, Any]] = []
    for row in rows:
        found_list = _clean_list(row["found"], 4)
        missing_list = _clean_list(row["missing"], 3)
        reason_bits: list[str] = []
        if row["score"] is not None:
            reason_bits.append(f"ATS score {_score_display(row['score'])}")
        if found_list:
            reason_bits.append(f"matched {', '.join(found_list)}")
        if missing_list:
            reason_bits.append(f"could improve {', '.join(missing_list)}")
        if not reason_bits:
            reason_bits.append("Selected from your analyzed resumes.")
        entries.append(
            {
                "id": int(row["id"]),
                "history_id": int(row["history_id"]) if row["history_id"] is not None else None,
                "candidate_name": row["candidate_name"] or "Candidate",
                "contact": row["contact"] or "-",
                "email": row["email"] or "-",
                "interview_date": row["interview_date"] or "TBD",
                "notes": row["notes"] or "",
                "created_at": row["created_at"] or "",
                "updated_at": row["updated_at"] or "",
                "email_sent_at": row["email_sent_at"] or "",
                "filename": row["filename"] or "Resume",
                "filetype": (row["filetype"] or "").upper(),
                "score_display": _score_display(row["score"]),
                "analyzed_date": row["analyzed_date"] or "-",
                "found_preview": ", ".join(found_list) or "-",
                "missing_preview": ", ".join(missing_list) or "-",
                "selection_reason": " • ".join(reason_bits),
            }
        )
    return entries


def token_secret() -> bytes:
    return JWT_SECRET.encode("utf-8")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def create_bearer_token(user: sqlite3.Row) -> str:
    payload = {
        "sub": int(user["id"]),
        "email": user["email"],
        "name": user["name"],
        "exp": int((datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)).timestamp()),
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(token_secret(), raw, hashlib.sha256).digest()
    return f"{_b64url(raw)}.{_b64url(sig)}"


def parse_bearer_token(token: str) -> dict[str, Any]:
    try:
        payload_part, sig_part = token.split(".", 1)
        raw = _b64url_decode(payload_part)
        expected = hmac.new(token_secret(), raw, hashlib.sha256).digest()
        actual = _b64url_decode(sig_part)
        if not hmac.compare_digest(expected, actual):
            raise ValueError("Invalid token signature")
        payload = json.loads(raw.decode("utf-8"))
        if int(payload.get("exp", 0)) < int(datetime.utcnow().timestamp()):
            raise ValueError("Token expired")
        return payload
    except Exception as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def api_user_from_request(request: Request) -> sqlite3.Row:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    payload = parse_bearer_token(auth.split(" ", 1)[1].strip())
    user = db_mod.get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> Response:
    if exc.status_code == 303 and exc.detail == "/login":
        return redirect("/login")
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    return render_error(request, exc.status_code, str(exc.detail), exc.status_code)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> Response:
    user = current_user(request)
    if not user:
        return redirect("/login")
    enabled, status_message = get_genai_status()
    return templates.TemplateResponse(
        request,
        "index.html",
        base_context(
            request,
            is_admin=is_admin_user(user),
            can_bulk=is_hr_user(user),
            genai_enabled=enabled,
            genai_status_message=status_message,
            recent_history=recent_history_for(int(user["id"])),
        ),
    )



@app.get("/consent", response_class=HTMLResponse)
async def consent_page(request: Request) -> Response:
    user = current_user(request)
    if not user:
        return redirect("/login")
    if request.session.get("resume_consent"):
        return redirect("/")
    return templates.TemplateResponse(request, "consent.html", base_context(request))


@app.post("/consent", response_class=HTMLResponse)
async def consent_submit(request: Request, allow: str = Form("")) -> Response:
    user = current_user(request)
    if not user:
        return redirect("/login")
    if allow == "yes":
        request.session["resume_consent"] = True
        return redirect("/")
    clear_session(request)
    request.session["flash_message"] = "You must consent to proceed."
    return redirect("/login")


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "about.html", base_context(request))


@app.post("/result", response_class=HTMLResponse)
async def analyze_result(
    request: Request,
    upload_mode: str = Form("single"),
    folder_name: str = Form(""),
    job: str = Form(""),
    resume: UploadFile | None = File(None),
    resumes: list[UploadFile] | None = File(None),
) -> Response:
    user = require_user(request)
    require_consent(request)
    job_text = clean(job)
    if not job_text:
        return render_error(request, 400, "Job description is required.", 400)

    uploads: list[UploadFile] = []
    if upload_mode == "bulk":
        uploads = [item for item in (resumes or []) if item and item.filename]
    elif resume and resume.filename:
        uploads = [resume]

    if not uploads:
        return render_error(request, 400, "Please upload at least one resume file.", 400)

    job_skills = extract_profile_skills(job_text)
    if not job_skills:
        job_skills = filter_skill_terms(extract_keywords(job_text))
    if not job_skills:
        job_skills = extract_keywords(job_text)

    if upload_mode == "bulk" and not is_hr_user(user):
        return render_error(request, 403, "Bulk folder uploads are only available to HR accounts.", 403)

    if upload_mode != "bulk":
        try:
            result = await analyze_uploaded_file(uploads[0], job_text, job_skills, True)
        except ValueError as exc:
            return render_error(request, 400, str(exc), 400)
        history_id = save_single_history(int(user["id"]), result)
        enabled, status_message = get_genai_status()
        combined_points = combine_points(result["rewritten"])
        suggestions = build_suggestions(combined_points, result["missing_display"])
        return templates.TemplateResponse(
            request,
            "result.html",
            base_context(
                request,
                resume=True,
                bulk=False,
                score=result["score"],
                found=result["found_text"],
                missing=result["missing_text"],
                found_display=result["found_display"],
                missing_display=result["missing_display"],
                rewritten=result["rewritten"],
                combined_points=combined_points,
                suggestions=suggestions,
                skills=result["resume_skills_csv"],
                job_skills=result["job_skills_csv"],
                resume_skills=result["resume_skills_display"],
                history_filename=result["filename"],
                history_filetype=result["ext"].lstrip(".").upper(),
                history_id=history_id,
                table_count=result.get("table_count", 0),
                table_preview=result.get("table_preview", []),
                genai_enabled=enabled,
                genai_status_message=status_message,
                is_hr=is_hr_user(user),
            ),
        )

    results: list[dict[str, Any]] = []
    skipped: list[str] = []
    for upload in uploads:
        try:
            results.append(await analyze_uploaded_file(upload, job_text, job_skills, True))
        except ValueError as exc:
            skipped.append(str(exc))
    if not results:
        return render_error(request, 400, "No supported resumes were found in the upload.", 400)

    batch_id = save_bulk_history(int(user["id"]), results, folder_name)
    leaderboard = sorted(results, key=lambda item: item["score"], reverse=True)
    for idx, row in enumerate(leaderboard, start=1):
        row["rank"] = idx
        row["found_count"] = len(row.get("found", []))
        row["missing_count"] = len(row.get("missing", []))
    table_resume_count = sum(1 for row in leaderboard if row.get("table_count", 0) > 0)
    return templates.TemplateResponse(
        request,
        "result.html",
        base_context(
            request,
            bulk=True,
            resume=False,
            processed_count=len(leaderboard),
            batch_total=len(uploads),
            top_score=leaderboard[0]["score"],
            avg_score=round(sum(x["score"] for x in leaderboard) / len(leaderboard), 1),
            skipped_count=len(skipped),
            skipped_files=skipped[:8],
            job_skill_count=len(job_skills),
            leaderboard=leaderboard,
            table_resume_count=table_resume_count,
            batch_id=batch_id,
            is_hr=is_hr_user(user),
        ),
    )


@app.get("/history/{history_id}", response_class=HTMLResponse)
async def open_history_result(request: Request, history_id: int) -> HTMLResponse:
    user = require_user(request)
    row = get_history_row(int(user["id"]), history_id)
    if not row:
        raise HTTPException(status_code=404, detail="History record not found.")
    rewritten = parse_saved_bullets(row["rewritten"]) or rehydrate_history_rewrite(history_id, int(user["id"]), row["filename"], row["job_skills"] or row["skills"])
    raw_missing_list = [s.strip() for s in str(row["missing"] or "").split(",") if s.strip() and s.strip() != "-"]
    display_missing_list = filter_skill_terms(raw_missing_list)
    display_found_list = filter_skill_terms([s.strip() for s in str(row["found"] or "").split(",") if s.strip() and s.strip() != "-"])
    display_resume_skills = filter_skill_terms([s.strip() for s in str(row["resume_skills"] or "").split(",") if s.strip()]) or display_found_list
    display_job_skills = filter_skill_terms([s.strip() for s in str(row["job_skills"] or row["skills"] or "").split(",") if s.strip()])
    combined_points = combine_points(rewritten)
    suggestions = build_suggestions(combined_points, display_missing_list)
    table_preview: list[list[list[str]]] = []
    table_count = 0
    history_path = find_uploaded_resume_path(row["filename"])
    if history_path:
        tables = extract_tables(history_path)
        table_preview = build_table_preview(tables)
        table_count = len(tables)
    enabled, status_message = get_genai_status()
    return templates.TemplateResponse(
        request,
        "result.html",
        base_context(
            request,
            resume=True,
            bulk=False,
            score=row["score"],
            found=", ".join(display_found_list) or "-",
            missing=", ".join(display_missing_list) or "-",
            found_display=display_found_list,
            missing_display=display_missing_list,
            rewritten=rewritten,
            combined_points=combined_points,
            suggestions=suggestions,
            skills=row["resume_skills"] or row["found"] or "",
            job_skills=row["job_skills"] or row["skills"] or "",
            resume_skills=display_resume_skills,
            history_id=history_id,
            history_filename=row["filename"],
            history_filetype=row["filetype"],
            history_date=row["date"],
            table_preview=table_preview,
            table_count=table_count,
            genai_enabled=enabled,
            genai_status_message=status_message,
            is_hr=is_hr_user(user),
        ),
    )


@app.post("/export")
async def export_resume(history_id: str = Form(""), skills: str = Form(""), bullets: str = Form("[]"), layout: str = Form("classic")) -> FileResponse:
    history_row = None
    rewritten = parse_saved_bullets(bullets)
    original_text = ""
    if not rewritten and history_id.strip().isdigit():
        with db_mod.db() as conn:
            history_row = conn.execute(
                "SELECT filename,rewritten,skills,job_skills,resume_skills,found,missing,user_id FROM resume_history WHERE id=?", (int(history_id),)
            ).fetchone()
        if history_row:
            rewritten = parse_saved_bullets(history_row["rewritten"])
            if not rewritten:
                # Attempt to regenerate and persist rewritten bullets from stored upload
                try:
                    regenerated = rehydrate_history_rewrite(
                        int(history_id),
                        int(history_row["user_id"]) if history_row["user_id"] is not None else 0,
                        history_row["filename"],
                        history_row["job_skills"] or history_row["skills"],
                    )
                    if regenerated:
                        rewritten = regenerated
                except Exception:
                    pass
            if not skills:
                # prefer resume skills for export; fall back to matched skills for older rows
                found = str(history_row["found"] or "").strip()
                skills = history_row["resume_skills"] or found
            # pull original resume text for context in export
            try:
                resume_path = find_uploaded_resume_path(history_row["filename"])
                if resume_path:
                    original_text = extract_text(resume_path)
            except Exception:
                original_text = ""

    # extra fallbacks to avoid empty bullets
    if not rewritten and original_text:
        try:
            rewritten = fallback_resume_points(original_text, limit=15)
        except Exception:
            rewritten = []

    display_name = "Candidate"
    if history_row and history_row["filename"]:
        display_name = Path(history_row["filename"]).stem or display_name
    out = export_docx(
        display_name,
        [s.strip() for s in skills.split(",") if s.strip()],
        rewritten,
        EXPORT_DIR,
        original_text=original_text,
        layout=layout or "classic",
    )
    if not out or not out.exists():
        raise HTTPException(status_code=500, detail="Unable to export DOCX.")
    return FileResponse(
        path=str(out),
        filename="ATS_Resume.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


def _export_points_response(history_id: str, bullets_json: str) -> Response:
    rewritten = parse_saved_bullets(bullets_json)
    if not rewritten and history_id.strip().isdigit():
        with db_mod.db() as conn:
            row = conn.execute("SELECT rewritten FROM resume_history WHERE id=?", (int(history_id),)).fetchone()
        if row:
            rewritten = parse_saved_bullets(row["rewritten"])
    text = "\n".join(f"{idx+1}. {point}" for idx, point in enumerate(rewritten or []))
    if not text.strip():
        raise HTTPException(status_code=400, detail="No improved points available to export.")
    headers = {"Content-Disposition": 'attachment; filename="AI_Improved_Points.txt"'}
    return Response(content=text, media_type="text/plain", headers=headers)


@app.post("/export/points")
async def export_points(history_id: str = Form(""), bullets: str = Form("[]")) -> Response:
    return _export_points_response(history_id, bullets)


@app.get("/export/points")
async def export_points_get(history_id: str = "", bullets: str = "[]") -> Response:
    return _export_points_response(history_id, bullets)


@app.get("/export/bulk/{batch_id}")
async def export_bulk(batch_id: str, request: Request) -> Response:
    user = require_user(request)
    batch_id = batch_id.strip()
    if not batch_id:
        return render_error(request, 400, "Missing batch identifier.", 400)
    with db_mod.db() as conn:
        rows = conn.execute(
            "SELECT filename,filetype,score,found,missing,skills,job_skills,resume_skills,rewritten FROM resume_history WHERE user_id=? AND batch_id=? ORDER BY score DESC",
            (int(user["id"]), batch_id),
        ).fetchall()
    if not rows:
        return render_error(request, 404, "Bulk batch not found.", 404)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Rank", "Filename", "File Type", "Score", "Resume Skills", "Matched Skills", "Missing Skills", "Improved Points"])
    for idx, row in enumerate(rows, start=1):
        matched_skills = str(row["found"] or "").strip() or str(row["resume_skills"] or "").strip() or "-"
        missing_skills = str(row["missing"] or "").strip() or "-"
        improved_points = " | ".join(combine_points(parse_saved_bullets(row["rewritten"]))) or "-"
        writer.writerow(
            [
                idx,
                row["filename"] or "",
                row["filetype"] or "",
                row["score"],
                row["resume_skills"] or matched_skills,
                matched_skills,
                missing_skills,
                improved_points,
            ]
        )
    filename = f"bulk-results-{batch_id}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=output.getvalue(), media_type="text/csv", headers=headers)


@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request, delete_id: int | None = None, delete_all: int | None = None) -> Response:
    user = require_user(request)
    notice = ""
    tone = "success"
    with db_mod.db() as conn:
        if delete_all:
            deleted = conn.execute("DELETE FROM resume_history WHERE user_id=?", (user["id"],)).rowcount
            notice = f"Deleted {int(deleted or 0)} history record(s)."
        elif delete_id:
            deleted = conn.execute("DELETE FROM resume_history WHERE id=? AND user_id=?", (delete_id, user["id"])).rowcount
            if deleted:
                notice = "History record deleted."
            else:
                notice = "History record not found."
                tone = "error"
        history = conn.execute("SELECT id,filename,filetype,score,date,batch_id,folder_name FROM resume_history WHERE user_id=? ORDER BY id DESC", (user["id"],)).fetchall()
        stats = conn.execute("SELECT COUNT(*) AS analyses_count, ROUND(AVG(score),1) AS avg_score, MAX(date) AS last_analysis FROM resume_history WHERE user_id=?", (user["id"],)).fetchone()
    return templates.TemplateResponse(
        request,
        "profile.html",
        base_context(
            request,
            user_name=user["name"] or "-",
            user_email=user["email"] or "-",
            user_created_at=user["created_at"] or "-",
            analysis_count=stats["analyses_count"] or 0,
            avg_score="-" if stats["avg_score"] is None else f'{float(stats["avg_score"]):.1f}%',
            last_analysis=stats["last_analysis"] or "-",
            history_groups=build_history_groups(list(history)),
            history_notice=notice,
            history_notice_tone=tone,
            is_admin=is_admin_user(user),
        ),
    )


@app.get("/analyzed-resumes", response_class=HTMLResponse)
async def analyzed_resumes_page(request: Request) -> Response:
    user = require_hr(request)
    notice = request.session.pop("flash_message", "")
    tone = request.session.pop("flash_tone", "success")
    return templates.TemplateResponse(
        request,
        "analyzed_resumes.html",
        base_context(
            request,
            resumes=analyzed_resumes_for(int(user["id"])),
            notice=notice,
            notice_tone=tone,
            today=datetime.now().strftime("%Y-%m-%d"),
            is_admin=is_admin_user(user),
        ),
    )


@app.post("/interview-list")
async def add_to_interview_list(
    request: Request,
    history_id: str = Form(""),
    candidate_name: str = Form(""),
    contact: str = Form(""),
    email: str = Form(""),
    interview_date: str = Form(""),
    notes: str = Form(""),
) -> Response:
    user = require_hr(request)
    history_id = history_id.strip()
    if not history_id or not history_id.isdigit():
        return render_error(request, 400, "Invalid resume reference.", 400)
    with db_mod.db() as conn:
        resume_row = conn.execute(
            "SELECT id,filename FROM resume_history WHERE id=? AND user_id=?", (int(history_id), int(user["id"]))
        ).fetchone()
    if not resume_row:
        return render_error(request, 404, "Resume not found for this account.", 404)
    contact_snapshot = resume_contact_snapshot(resume_row["filename"])
    candidate_name = candidate_name.strip() or contact_snapshot["candidate_name"] or Path(resume_row["filename"] or "Candidate").stem or "Candidate"
    contact = contact.strip() or contact_snapshot["contact"]
    email = email.strip() or contact_snapshot["email"]
    interview_date = interview_date.strip() or datetime.now().strftime("%Y-%m-%d")
    notes = notes.strip()
    if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return render_error(request, 400, "Please enter a valid email address.", 400)
    entry_id = save_interview_entry(
        int(user["id"]),
        int(history_id),
        candidate_name,
        contact,
        email,
        interview_date,
        notes,
    )
    email_sent = False
    if email:
        email_sent = send_interview_selection_email(
            email,
            candidate_name,
            interview_date,
            contact=contact,
            notes=notes,
            recruiter_name=str(user["name"] or ""),
        )
        if email_sent:
            with db_mod.db() as conn:
                conn.execute(
                    "UPDATE interview_list SET email_sent_at=?, updated_at=? WHERE id=? AND user_id=?",
                    (
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        entry_id,
                        int(user["id"]),
                    ),
                    )
    if email and email_sent:
        request.session["flash_message"] = f"Interview email sent to {candidate_name}."
        request.session["flash_tone"] = "success"
    elif email:
        request.session["flash_message"] = f"{candidate_name} was added, but the email could not be sent. {smtp_status_message()}"
        request.session["flash_tone"] = "error"
    else:
        request.session["flash_message"] = f"{candidate_name} added to the interview list."
        request.session["flash_tone"] = "success"
    return redirect("/analyzed-resumes")


@app.post("/interview-list/send-email")
async def send_interview_email(request: Request, entry_id: str = Form("")) -> Response:
    user = require_hr(request)
    entry_id = entry_id.strip()
    if not entry_id.isdigit():
        request.session["flash_message"] = "Invalid interview entry."
        request.session["flash_tone"] = "error"
        return redirect("/interview-list")

    with db_mod.db() as conn:
        entry = conn.execute(
            """
            SELECT il.id, il.candidate_name, il.contact, il.email, il.interview_date, il.notes,
                   il.email_sent_at, rh.filename
            FROM interview_list il
            LEFT JOIN resume_history rh ON rh.id = il.history_id
            WHERE il.id=? AND il.user_id=?
            """,
            (int(entry_id), int(user["id"])),
        ).fetchone()

    if not entry:
        request.session["flash_message"] = "Interview entry not found."
        request.session["flash_tone"] = "error"
        return redirect("/interview-list")

    if not entry["email"]:
        request.session["flash_message"] = "No email address is saved for this candidate."
        request.session["flash_tone"] = "error"
        return redirect("/interview-list")

    sent = send_interview_selection_email(
        entry["email"],
        entry["candidate_name"] or "Candidate",
        entry["interview_date"] or "To be confirmed",
        contact=entry["contact"] or "",
        notes=entry["notes"] or "",
        recruiter_name=str(user["name"] or ""),
    )
    if sent:
        with db_mod.db() as conn:
            conn.execute(
                "UPDATE interview_list SET email_sent_at=?, updated_at=? WHERE id=? AND user_id=?",
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    int(entry["id"]),
                    int(user["id"]),
                ),
            )
        request.session["flash_message"] = f"Interview email sent to {entry['candidate_name'] or 'Candidate'}."
        request.session["flash_tone"] = "success"
    else:
        request.session["flash_message"] = f"Email could not be sent. {smtp_status_message()}"
        request.session["flash_tone"] = "error"
    return redirect("/interview-list")


@app.post("/interview-list/update-date")
async def update_interview_date(request: Request, entry_id: str = Form(""), interview_date: str = Form("")) -> Response:
    user = require_hr(request)
    entry_id = entry_id.strip()
    interview_date = interview_date.strip()
    if not entry_id.isdigit():
        request.session["flash_message"] = "Invalid interview entry."
        request.session["flash_tone"] = "error"
        return redirect("/interview-list")
    if not interview_date:
        request.session["flash_message"] = "Please pick a new interview date."
        request.session["flash_tone"] = "error"
        return redirect("/interview-list")
    with db_mod.db() as conn:
        existing = conn.execute(
            "SELECT id FROM interview_list WHERE id=? AND user_id=?",
            (int(entry_id), int(user["id"])),
        ).fetchone()
        if not existing:
            request.session["flash_message"] = "Interview entry not found."
            request.session["flash_tone"] = "error"
            return redirect("/interview-list")
        conn.execute(
            "UPDATE interview_list SET interview_date=?, updated_at=? WHERE id=? AND user_id=?",
            (
                interview_date,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                int(entry_id),
                int(user["id"]),
            ),
        )
    request.session["flash_message"] = "Interview date updated."
    request.session["flash_tone"] = "success"
    return redirect("/interview-list")


@app.get("/interview-list", response_class=HTMLResponse)
async def interview_list_page(request: Request, delete_id: int | None = None) -> Response:
    user = require_hr(request)
    notice = request.session.pop("flash_message", "")
    tone = request.session.pop("flash_tone", "success")
    if delete_id:
        with db_mod.db() as conn:
            removed = conn.execute("DELETE FROM interview_list WHERE id=? AND user_id=?", (delete_id, user["id"])).rowcount
        request.session["flash_message"] = "Interview entry removed." if removed else "Entry not found."
        request.session["flash_tone"] = "success" if removed else "error"
        return redirect("/interview-list")
    return templates.TemplateResponse(
        request,
        "interview_list.html",
        base_context(
            request,
            interviews=interview_entries_for(int(user["id"])),
            notice=notice,
            notice_tone=tone,
            today=datetime.now().strftime("%Y-%m-%d"),
            is_admin=is_admin_user(user),
        ),
    )


@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request) -> HTMLResponse:
    user = require_user(request)
    if not is_admin_user(user):
        raise HTTPException(status_code=403, detail="Admin access required.")
    today = datetime.now().strftime("%Y-%m-%d")
    with db_mod.db() as conn:
        users = conn.execute(
            """
            SELECT u.id,u.name,u.email,u.created_at,
                   COALESCE(ls.total_logins,0) AS total_logins, ls.last_login_time,
                   COALESCE(rs.resume_count,0) AS resume_count, ROUND(COALESCE(rs.avg_score,0),1) AS avg_score,
                   rs.last_resume_time
            FROM users u
            LEFT JOIN (SELECT user_id,COUNT(*) AS total_logins,MAX(login_time) AS last_login_time FROM login_history GROUP BY user_id) ls ON ls.user_id=u.id
            LEFT JOIN (SELECT user_id,COUNT(*) AS resume_count,AVG(score) AS avg_score,MAX(date) AS last_resume_time FROM resume_history GROUP BY user_id) rs ON rs.user_id=u.id
            ORDER BY COALESCE(ls.last_login_time,u.created_at) DESC,u.id DESC
            """
        ).fetchall()
        recent = conn.execute("SELECT lh.login_time,lh.email,lh.ip_address,lh.user_agent,u.name FROM login_history lh LEFT JOIN users u ON u.id=lh.user_id ORDER BY lh.id DESC LIMIT 50").fetchall()
        total_users = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"]
        total_events = conn.execute("SELECT COUNT(*) AS count FROM login_history").fetchone()["count"]
        today_logins = conn.execute("SELECT COUNT(*) AS count FROM login_history WHERE login_time LIKE ?", (f"{today}%",)).fetchone()["count"]
    return templates.TemplateResponse(
        request,
        "admin_dashboard.html",
        base_context(
            request,
            admin_name=user["name"] or "Admin",
            total_users=total_users,
            total_login_events=total_events,
            today_logins=today_logins,
            users=users,
            recent_logins=recent,
        ),
    )


@app.get("/hr/dashboard", response_class=HTMLResponse)
async def hr_dashboard(request: Request) -> HTMLResponse:
    user = require_hr(request)
    steps = [
        "Create Job - paste JD or upload PDF.",
        "Upload Resumes (single or bulk).",
        "ATS Engine scores + extracts tables.",
        "Rank / filter by score, tables, missing skills.",
        "Shortlist, reject, and add notes per candidate.",
        "Export shortlist as CSV or improved bullets as TXT/DOCX.",
    ]
    html = """<html><head><title>HR Dashboard</title></head><body>
    <h2>Welcome, {name} (HR)</h2>
    <p>Use the standard upload page to run the ATS engine, or integrate via the API token.</p>
    <ol>{steps}</ol>
    <p><a href="/about">Back to app</a></p>
    </body></html>""".format(name=user["name"] or "HR", steps="".join(f"<li>{s}</li>" for s in steps))
    return HTMLResponse(content=html)


@app.get("/api-token", response_class=HTMLResponse)
async def api_token(request: Request) -> HTMLResponse:
    user = require_user(request)
    token = create_bearer_token(user)
    return templates.TemplateResponse(
        request,
        "token.html",
        base_context(
            request,
            token=token,
            expires_minutes=JWT_EXPIRE_MINUTES,
        ),
    )


@app.get("/api/me")
async def api_me(request: Request) -> JSONResponse:
    user = api_user_from_request(request)
    return JSONResponse(
        {
            "id": int(user["id"]),
            "name": user["name"],
            "email": user["email"],
            "created_at": user["created_at"],
            "role": user["role"] or "candidate",
            "is_admin": is_admin_user(user),
        }
    )


@app.get("/api/history")
async def api_history(request: Request) -> JSONResponse:
    user = api_user_from_request(request)
    with db_mod.db() as conn:
        rows = conn.execute(
            "SELECT id,filename,filetype,score,date,batch_id,folder_name FROM resume_history WHERE user_id=? ORDER BY id DESC",
            (user["id"],),
        ).fetchall()
    return JSONResponse(
        [
            {
                "id": int(row["id"]),
                "filename": row["filename"],
                "filetype": row["filetype"],
                "score": row["score"],
                "date": row["date"],
                "batch_id": row["batch_id"],
                "folder_name": row["folder_name"],
            }
            for row in rows
        ]
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
