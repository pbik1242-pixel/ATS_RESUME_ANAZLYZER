CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    role TEXT NOT NULL DEFAULT 'candidate',
    session_version INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS resume_history (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    score REAL,
    filename TEXT NOT NULL,
    filetype TEXT NOT NULL,
    date TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    found TEXT,
    missing TEXT,
    rewritten TEXT,
    skills TEXT,
    job_skills TEXT,
    resume_skills TEXT,
    batch_id TEXT,
    folder_name TEXT,
    job_id INTEGER,
    job_title TEXT,
    candidate_status TEXT NOT NULL DEFAULT 'new',
    recruiter_notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS interview_list (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    history_id INTEGER NOT NULL,
    candidate_name TEXT NOT NULL,
    contact TEXT,
    email TEXT,
    interview_date TEXT NOT NULL,
    notes TEXT,
    status TEXT NOT NULL DEFAULT 'interview_scheduled',
    email_sent_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (history_id) REFERENCES resume_history(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS job_descriptions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS login_history (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    email TEXT,
    login_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ip_address TEXT,
    user_agent TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_resume_history_user_id ON resume_history(user_id);
CREATE INDEX IF NOT EXISTS idx_resume_history_batch_id ON resume_history(batch_id);
CREATE INDEX IF NOT EXISTS idx_interview_list_user_id ON interview_list(user_id);
CREATE INDEX IF NOT EXISTS idx_interview_list_history_id ON interview_list(history_id);
CREATE INDEX IF NOT EXISTS idx_job_descriptions_user_id ON job_descriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_login_history_user_id ON login_history(user_id);
