from __future__ import annotations

from pathlib import Path

import app
import database
import auth


def _configure_temp_db(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(database, "SCHEMA_PATH", Path(__file__).resolve().parents[1] / "schema.sql")
    monkeypatch.setattr(app.db_mod, "DB_PATH", db_path)
    monkeypatch.setattr(app.db_mod, "SCHEMA_PATH", Path(__file__).resolve().parents[1] / "schema.sql")
    monkeypatch.setattr(app, "DB_PATH", db_path)
    database.ensure_schema()


def _insert_user(role: str = "candidate") -> int:
    with database.db() as conn:
        inserted = conn.execute(
            "INSERT INTO users(name,email,password,created_at,role,session_version) VALUES (?,?,?,?,?,?)",
            ("Test User", f"{role}@example.com", "hash", "2026-04-30 10:00:00", role, 0),
        )
        return int(inserted.lastrowid)


def _insert_history(user_id: int, filename: str, score: float, date: str, candidate_status: str = "new") -> int:
    with database.db() as conn:
        inserted = conn.execute(
            """
            INSERT INTO resume_history(
                user_id,score,filename,filetype,date,found,missing,rewritten,
                skills,job_skills,resume_skills,batch_id,folder_name,job_id,job_title,candidate_status,recruiter_notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                user_id,
                score,
                filename,
                "PDF",
                date,
                "SQL",
                "",
                "[]",
                "SQL",
                "SQL",
                "SQL",
                None,
                None,
                None,
                None,
                candidate_status,
                "",
            ),
        )
        return int(inserted.lastrowid)


def test_register_user_normalizes_invalid_role_to_candidate(monkeypatch, tmp_path) -> None:
    _configure_temp_db(monkeypatch, tmp_path)

    ok, message = database.register_user("Alice", "alice@example.com", "password123", "password123", role="admin")
    assert ok, message

    user = database.get_user_by_email("alice@example.com")
    assert user is not None
    assert user["role"] == "candidate"


def test_analyzed_resumes_for_keeps_newest_duplicate(monkeypatch, tmp_path) -> None:
    _configure_temp_db(monkeypatch, tmp_path)
    user_id = _insert_user("hr")
    older_id = _insert_history(user_id, "ArogyaSwamy-BA.pdf", 61.4, "2026-04-30 18:26")
    newer_id = _insert_history(user_id, "ArogyaSwamy-BA.PDF", 72.0, "2026-04-30 18:33")
    _insert_history(user_id, "Keerthana_Krishnaraj.pdf", 100.0, "2026-04-30 18:31")

    resumes = app.analyzed_resumes_for(user_id)
    duplicate_ids = [row["id"] for row in resumes if row["filename"].lower().startswith("arogyaswamy-ba")]

    assert older_id not in duplicate_ids
    assert len(duplicate_ids) == 1
    assert duplicate_ids[0] == newer_id


def test_interview_entry_lifecycle_updates_resume_status(monkeypatch, tmp_path) -> None:
    _configure_temp_db(monkeypatch, tmp_path)
    user_id = _insert_user("hr")
    history_id = _insert_history(user_id, "Candidate.pdf", 84.0, "2026-04-30 18:31")

    entry_id = app.save_interview_entry(
        user_id,
        history_id,
        "Candidate",
        "+1 555 000 1111",
        "candidate@example.com",
        "2026-05-01",
        "Interview scheduled",
    )

    with database.db() as conn:
        status = conn.execute("SELECT candidate_status FROM resume_history WHERE id=?", (history_id,)).fetchone()["candidate_status"]
        interview_row = conn.execute("SELECT id FROM interview_list WHERE id=?", (entry_id,)).fetchone()

    assert status == "interview_scheduled"
    assert interview_row is not None

    removed = app.remove_interview_entry(user_id, entry_id)
    assert removed

    with database.db() as conn:
        status = conn.execute("SELECT candidate_status FROM resume_history WHERE id=?", (history_id,)).fetchone()["candidate_status"]
        interview_row = conn.execute("SELECT id FROM interview_list WHERE id=?", (entry_id,)).fetchone()

    assert status == "new"
    assert interview_row is None


def test_local_captcha_mode_accepts_checkbox_token(monkeypatch) -> None:
    monkeypatch.setattr(auth, "RECAPTCHA_MODE", "local")
    monkeypatch.setattr(auth, "RECAPTCHA_SITE_KEY", "site-key")
    monkeypatch.setattr(auth, "RECAPTCHA_SECRET_KEY", "secret-key")

    ok, error = auth._verify_recaptcha("on")

    assert ok is True
    assert error is None


def test_google_captcha_mode_still_verifies_token(monkeypatch) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, bool]:
            return {"success": True}

    def _post(url, data, timeout):
        assert url == auth.RECAPTCHA_VERIFY_URL
        assert data["secret"] == "secret-key"
        assert data["response"] == "token-value"
        return _Response()

    monkeypatch.setattr(auth, "RECAPTCHA_MODE", "google")
    monkeypatch.setattr(auth, "RECAPTCHA_SITE_KEY", "site-key")
    monkeypatch.setattr(auth, "RECAPTCHA_SECRET_KEY", "secret-key")
    monkeypatch.setattr(auth.requests, "post", _post)

    ok, error = auth._verify_recaptcha("token-value", "127.0.0.1")

    assert ok is True
    assert error is None
