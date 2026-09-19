"""
core/leads.py -- Inbound leads, customer inquiries, and subscription upgrade requests.
Persisted in portal_users.db so auth and subscriber leads are unified.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORTAL_DB_PATH = PROJECT_ROOT / "portal_users.db"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    conn = sqlite3.connect(str(PORTAL_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_leads_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inquiries (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            email      TEXT NOT NULL,
            message    TEXT NOT NULL,
            status     TEXT NOT NULL DEFAULT 'new',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS upgrade_requests (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER,
            name           TEXT NOT NULL,
            email          TEXT NOT NULL,
            requested_tier TEXT NOT NULL DEFAULT 'Pro Trader',
            status         TEXT NOT NULL DEFAULT 'pending',
            created_at     TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def submit_inquiry(name: str, email: str, message: str) -> int:
    conn = get_connection()
    cur = conn.execute("""
        INSERT INTO inquiries (name, email, message, status, created_at)
        VALUES (?, ?, ?, 'new', ?)
    """, (name.strip(), email.strip().lower(), message.strip(), _now_iso()))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_all_inquiries(limit: int = 50) -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM inquiries ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_inquiry_status(inquiry_id: int, status: str):
    conn = get_connection()
    conn.execute("""
        UPDATE inquiries SET status = ? WHERE id = ?
    """, (status, inquiry_id))
    conn.commit()
    conn.close()


def delete_inquiry(inquiry_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM inquiries WHERE id = ?", (inquiry_id,))
    conn.commit()
    conn.close()


def submit_upgrade_request(name: str, email: str, requested_tier: str = 'Pro Trader', user_id: int = None) -> int:
    conn = get_connection()
    existing = conn.execute("""
        SELECT id FROM upgrade_requests WHERE email = ? AND status = 'pending'
    """, (email.strip().lower(),)).fetchone()
    if existing:
        conn.close()
        return existing["id"]

    cur = conn.execute("""
        INSERT INTO upgrade_requests (user_id, name, email, requested_tier, status, created_at)
        VALUES (?, ?, ?, ?, 'pending', ?)
    """, (user_id, name.strip(), email.strip().lower(), requested_tier, _now_iso()))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_pending_upgrade_requests() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM upgrade_requests WHERE status = 'pending' ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_upgrade_requests(limit: int = 50) -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM upgrade_requests ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def resolve_upgrade_request(request_id: int, status: str = 'approved'):
    conn = get_connection()
    conn.execute("""
        UPDATE upgrade_requests SET status = ? WHERE id = ?
    """, (status, request_id))
    conn.commit()
    conn.close()


init_leads_db()
