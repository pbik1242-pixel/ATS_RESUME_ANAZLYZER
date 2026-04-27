from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from werkzeug.security import generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, type_def: str) -> None:
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_def}")


def ensure_schema() -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing schema file: {SCHEMA_PATH}")

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with db() as conn:
        conn.executescript(schema_sql)

        # Backward compatibility for databases created before this schema file.
        for col, dtype in [("created_at", "TEXT"), ("role", "TEXT"), ("session_version", "INTEGER")]:
            _add_column_if_missing(conn, "users", col, dtype)

        for name in (
            "found",
            "missing",
            "rewritten",
            "skills",
            "job_skills",
            "resume_skills",
            "batch_id",
            "folder_name",
            "job_id",
            "job_title",
            "candidate_status",
            "recruiter_notes",
        ):
            _add_column_if_missing(conn, "resume_history", name, "TEXT")

        for name in (
            "candidate_name",
            "contact",
            "email",
            "interview_date",
            "notes",
            "status",
            "email_sent_at",
            "created_at",
            "updated_at",
            "history_id",
            "user_id",
        ):
            _add_column_if_missing(conn, "interview_list", name, "TEXT")

        for name in ("created_at", "updated_at"):
            _add_column_if_missing(conn, "job_descriptions", name, "TEXT")

        for name in ("login_time", "ip_address", "user_agent"):
            _add_column_if_missing(conn, "login_history", name, "TEXT")

        conn.execute("UPDATE users SET session_version=COALESCE(session_version, 0)")
        conn.execute("UPDATE resume_history SET candidate_status='new' WHERE candidate_status IS NULL OR TRIM(candidate_status)=''")
        conn.execute("UPDATE interview_list SET status='interview_scheduled' WHERE status IS NULL OR TRIM(status)=''")


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    with db() as conn:
        return conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()


def get_user_by_email(email: str) -> sqlite3.Row | None:
    with db() as conn:
        return conn.execute("SELECT * FROM users WHERE email=?", (email.strip(),)).fetchone()


def register_user(name: str, email: str, password: str, confirm_password: str, role: str = "candidate") -> tuple[bool, str]:
    if not name or not email or not password or not confirm_password:
        return False, "Name, email, and password are required."
    if password != confirm_password:
        return False, "Passwords do not match."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    role = role.strip().lower() or "candidate"
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO users(name,email,password,created_at,role,session_version) VALUES (?,?,?,?,?,?)",
                (
                    name.strip(),
                    email.strip(),
                    generate_password_hash(password),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    role,
                    0,
                ),
            )
        return True, "Account created. Please sign in."
    except sqlite3.IntegrityError:
        return False, "That email is already registered."
    except Exception:
        return False, "Registration failed."


def ensure_google_user(email: str, name: str, role: str = "candidate") -> sqlite3.Row | None:
    email = email.strip()
    if not email:
        return None

    safe_role = role.strip().lower() or "candidate"
    display_name = name.strip() or email.split("@", 1)[0]
    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user:
            if display_name and not str(user["name"] or "").strip():
                conn.execute("UPDATE users SET name=? WHERE id=?", (display_name, user["id"]))
                user = conn.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
            return user

        conn.execute(
            "INSERT INTO users(name,email,password,created_at,role,session_version) VALUES (?,?,?,?,?,?)",
            (
                display_name,
                email,
                generate_password_hash(uuid4().hex),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                safe_role,
                0,
            ),
        )
        return conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()


def record_login_success(user_id: int, email: str, ip: str, user_agent: str) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO login_history(user_id,email,login_time,ip_address,user_agent) VALUES (?,?,?,?,?)",
            (user_id, email, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ip, user_agent),
        )


def save_single_history(user_id: int, result: dict) -> int:
    with db() as conn:
        inserted = conn.execute(
            """
            INSERT INTO resume_history(
                user_id,score,filename,filetype,date,found,missing,rewritten,
                skills,job_skills,resume_skills,candidate_status
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                user_id,
                result["score"],
                result["filename"],
                result["ext"].lstrip(".").upper(),
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                result["found_text"],
                result["missing_text"],
                json.dumps(result["rewritten"]),
                result["skills_csv"],
                result["job_skills_csv"],
                result["resume_skills_csv"],
                "new",
            ),
        )
        return int(inserted.lastrowid)
