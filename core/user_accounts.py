"""
User Accounts Database - Multi-user MT5 copy trading credential store.
Supports free 14-day trial → paid subscription lifecycle.
"""
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USER_DB_PATH = PROJECT_ROOT / "user_accounts.db"

TRIAL_DAYS = 14  # Free trial duration


def find_installed_terminals() -> dict[str, str]:
    """
    Search standard Windows locations for MetaTrader 5 / broker terminals.
    Returns a dict of {Label: full_path_to_terminal64.exe}.
    """
    found = {}
    known_paths = [
        ("FTMO MT5 Terminal", r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"),
        ("Standard MetaTrader 5", r"C:\Program Files\MetaTrader 5\terminal64.exe"),
        ("MetaTrader 5 (x86)", r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"),
    ]
    for label, p in known_paths:
        if Path(p).exists():
            found[label] = p

    for root in [r"C:\Program Files", r"C:\Program Files (x86)", os.path.expandvars(r"%LOCALAPPDATA%\Programs")]:
        if not os.path.exists(root):
            continue
        try:
            for item in os.listdir(root):
                full_dir = os.path.join(root, item)
                if os.path.isdir(full_dir):
                    t64 = os.path.join(full_dir, "terminal64.exe")
                    if os.path.exists(t64) and t64 not in found.values():
                        clean_label = item.replace("Terminal", "").replace("terminal", "").strip()
                        found[f"{clean_label} MT5"] = t64
        except Exception:
            pass

    return found


def get_connection():
    conn = sqlite3.connect(str(USER_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create/migrate the user_accounts table."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_accounts (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT NOT NULL,
            email            TEXT UNIQUE NOT NULL,
            mt5_login        TEXT NOT NULL,
            mt5_password     TEXT NOT NULL,
            mt5_server       TEXT NOT NULL,
            risk_type        TEXT NOT NULL DEFAULT 'fixed',
            risk_value       REAL NOT NULL DEFAULT 0.01,
            max_daily_trades INTEGER NOT NULL DEFAULT 10,

            -- Personalized terminal & broker settings
            terminal_path    TEXT NOT NULL DEFAULT '',
            account_type     TEXT NOT NULL DEFAULT 'standard',

            -- Subscription fields
            subscription_status TEXT NOT NULL DEFAULT 'trial',
            -- 'trial' | 'paid' | 'expired' | 'paused'
            trial_started_at    TEXT NOT NULL,
            trial_ends_at       TEXT NOT NULL,
            paid_until          TEXT,          -- NULL until manually set
            paid_note           TEXT,          -- e.g. "Stripe txn abc123"

            -- Runtime
            enabled          INTEGER NOT NULL DEFAULT 1,
            last_trade_at    TEXT,
            created_at       TEXT NOT NULL
        )
    """)
    # Migration: add subscription & custom terminal columns to existing DBs
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(user_accounts)").fetchall()
    }
    migrations = {
        "subscription_status": "TEXT NOT NULL DEFAULT 'trial'",
        "trial_started_at":    "TEXT NOT NULL DEFAULT ''",
        "trial_ends_at":       "TEXT NOT NULL DEFAULT ''",
        "paid_until":          "TEXT",
        "paid_note":           "TEXT",
        "telegram_chat_id":    "TEXT NOT NULL DEFAULT ''",
        "terminal_path":       "TEXT NOT NULL DEFAULT ''",
        "account_type":        "TEXT NOT NULL DEFAULT 'standard'",
    }
    for col, definition in migrations.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE user_accounts ADD COLUMN {col} {definition}")
    conn.commit()
    conn.close()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trial_end_iso() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=TRIAL_DAYS)).isoformat()


def subscription_label(user: dict) -> tuple[str, str]:
    """
    Returns (status_label, css_color) for display.
    Also auto-expires trial users whose trial_ends_at has passed.
    """
    status = user.get("subscription_status", "trial")

    if status == "paid":
        paid_until = user.get("paid_until")
        if paid_until:
            try:
                exp = datetime.fromisoformat(paid_until)
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) > exp:
                    return ("⛔ Paid Expired", "#FF4466")
            except Exception:
                pass
        return ("✅ Paid", "#00FF88")

    if status == "trial":
        ends = user.get("trial_ends_at", "")
        try:
            exp = datetime.fromisoformat(ends)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            remaining = (exp - datetime.now(timezone.utc)).days
            if remaining < 0:
                return ("⚠️ Trial Expired", "#FF4466")
            elif remaining <= 3:
                return (f"🕐 Trial ({remaining}d left)", "#FFB800")
            else:
                return (f"🆓 Trial ({remaining}d left)", "#00E5FF")
        except Exception:
            return ("🆓 Trial", "#00E5FF")

    if status == "expired":
        return ("⛔ Expired", "#FF4466")

    if status == "paused":
        return ("⏸️ Paused", "#888")

    return (status, "#888")


def is_subscription_active(user: dict) -> bool:
    """Returns True if this user should receive trades right now."""
    if not user.get("enabled"):
        return False

    status = user.get("subscription_status", "trial")

    if status == "paid":
        paid_until = user.get("paid_until")
        if paid_until:
            try:
                exp = datetime.fromisoformat(paid_until)
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                return datetime.now(timezone.utc) <= exp
            except Exception:
                pass
        return True  # paid with no expiry date = lifetime

    if status == "trial":
        ends = user.get("trial_ends_at", "")
        try:
            exp = datetime.fromisoformat(ends)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) <= exp
        except Exception:
            return False

    return False  # expired / paused


# ── CRUD ───────────────────────────────────────────────────────────────────────

def add_user(name: str, email: str, mt5_login: str, mt5_password: str,
             mt5_server: str, risk_type: str = "fixed",
             risk_value: float = 0.01, max_daily_trades: int = 10,
             terminal_path: str = "", account_type: str = "standard") -> int:
    """Register a new subscriber. Starts a 14-day trial automatically."""
    conn = get_connection()
    now = _now_iso()
    trial_end = _trial_end_iso()
    cur = conn.execute("""
        INSERT INTO user_accounts
            (name, email, mt5_login, mt5_password, mt5_server,
             risk_type, risk_value, max_daily_trades,
             terminal_path, account_type,
             subscription_status, trial_started_at, trial_ends_at,
             enabled, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'trial', ?, ?, 1, ?)
    """, (name, email, mt5_login, mt5_password, mt5_server,
          risk_type, risk_value, max_daily_trades,
          terminal_path, account_type,
          now, trial_end, now))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def update_user(user_id: int, **kwargs):
    """Update any fields of a user row by id."""
    allowed = {
        "name", "email", "mt5_login", "mt5_password", "mt5_server",
        "risk_type", "risk_value", "max_daily_trades", "enabled",
        "subscription_status", "paid_until", "paid_note",
        "trial_ends_at", "telegram_chat_id", "terminal_path", "account_type",
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    cols = ", ".join(f"{k} = ?" for k in updates)
    vals = list(updates.values()) + [user_id]
    conn = get_connection()
    conn.execute(f"UPDATE user_accounts SET {cols} WHERE id = ?", vals)
    conn.commit()
    conn.close()

# Auto-migrate on module load
try:
    init_db()
except Exception:
    pass


def mark_paid(user_id: int, paid_until_iso: str | None = None, note: str = ""):
    """Mark a user as paid. paid_until_iso = None means lifetime."""
    conn = get_connection()
    conn.execute("""
        UPDATE user_accounts
        SET subscription_status = 'paid',
            paid_until = ?,
            paid_note  = ?,
            enabled    = 1
        WHERE id = ?
    """, (paid_until_iso, note, user_id))
    conn.commit()
    conn.close()


def extend_trial(user_id: int, extra_days: int = 14):
    """Extend a user's trial by N more days from today."""
    new_end = (datetime.now(timezone.utc) + timedelta(days=extra_days)).isoformat()
    conn = get_connection()
    conn.execute("""
        UPDATE user_accounts
        SET trial_ends_at = ?, subscription_status = 'trial'
        WHERE id = ?
    """, (new_end, user_id))
    conn.commit()
    conn.close()


def delete_user(user_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM user_accounts WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def get_all_users() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM user_accounts ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_enabled_users() -> list[dict]:
    """Return only users whose subscription is currently active."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM user_accounts WHERE enabled = 1 ORDER BY id"
    ).fetchall()
    conn.close()
    # Filter by live subscription status (trial not expired, paid not expired)
    return [dict(r) for r in rows if is_subscription_active(dict(r))]


def get_user_by_email(email: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM user_accounts WHERE email = ?", (email,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def mark_last_trade(user_id: int):
    conn = get_connection()
    conn.execute(
        "UPDATE user_accounts SET last_trade_at = ? WHERE id = ?", (_now_iso(), user_id)
    )
    conn.commit()
    conn.close()


def count_todays_trades(user_id: int) -> int:
    """Count trades executed today for rate-limiting purposes."""
    from datetime import date
    today = date.today().isoformat()
    conn = get_connection()
    row = conn.execute("""
        SELECT COUNT(*) as cnt FROM user_accounts
        WHERE id = ? AND last_trade_at LIKE ?
    """, (user_id, today + "%")).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def get_subscribers_with_telegram() -> list[dict]:
    """Return enabled subscribers who have a telegram_chat_id set."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM user_accounts WHERE enabled = 1 AND telegram_chat_id != '' ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows if is_subscription_active(dict(r))]


# Auto-initialise on import
init_db()
