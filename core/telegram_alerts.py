"""
core/telegram_alerts.py
Sends per-subscriber Telegram signal alerts.
Bot token lives in config.yaml under telegram.subscriber_bot_token.
Each subscriber stores their chat_id in user_accounts.db.
"""
import logging
import urllib.request
import urllib.parse
import json
from pathlib import Path

logger = logging.getLogger("TelegramAlerts")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_bot_token() -> str | None:
    """Read the subscriber bot token from config.yaml."""
    try:
        import yaml
        with open(PROJECT_ROOT / "config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("telegram", {}).get("subscriber_bot_token") or None
    except Exception as e:
        logger.warning(f"Could not load telegram config: {e}")
        return None


def _send(bot_token: str, chat_id: str, text: str) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        logger.error(f"Telegram send failed (chat {chat_id}): {e}")
        return False


def build_signal_message(signal_row: dict, user: dict) -> str:
    """Build a rich HTML Telegram message for a subscriber."""
    symbol      = signal_row.get("symbol", "?")
    sig_type    = signal_row.get("signal", "?")
    conf        = signal_row.get("confidence", 0)
    sl          = signal_row.get("sl_price", 0)
    tp          = signal_row.get("tp_price", 0)
    price       = signal_row.get("price_at_signal", 0)
    regime      = signal_row.get("regime", "")
    risk_val    = user.get("risk_value", 0.01)
    risk_type   = user.get("risk_type", "fixed")

    arrow       = "🟢 BUY" if sig_type == "BUY" else "🔴 SELL"
    risk_label  = f"{risk_val} lots" if risk_type == "fixed" else f"{risk_val}% risk"
    regime_str  = f" · {regime}" if regime else ""

    return (
        f"⚡ <b>ForexAlert Signal</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{arrow}  <b>{symbol}</b>{regime_str}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📍 Entry:  <code>{price:.5f}</code>\n"
        f"🛑 SL:     <code>{sl:.5f}</code>\n"
        f"🎯 TP:     <code>{tp:.5f}</code>\n"
        f"📊 Conf:   <b>{conf:.0%}</b>\n"
        f"⚖️  Risk:   {risk_label}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"<i>Trades execute automatically on your MT5 account.</i>"
    )


def build_trade_result_message(symbol: str, sig_type: str, result_code: str, user: dict) -> str:
    """Build a Telegram message confirming trade execution."""
    if result_code and result_code.isdigit():
        status = f"✅ Executed — Ticket #{result_code}"
    elif "BROKER_UNSUPPORTED" in str(result_code):
        status = f"⚠️ Broker Unsupported — {result_code.replace('BROKER_UNSUPPORTED:', '').strip()}"
    elif "SYMBOL_NOT_FOUND" in str(result_code):
        status = f"⚠️ Symbol not available on your broker server"
    elif result_code and result_code.startswith("FAILED"):
        status = f"❌ Failed — {result_code.replace('FAILED:', '')}"
    elif result_code == "CONNECTION_FAILED":
        status = "⚠️ MT5 connection failed — check your credentials"
    else:
        status = f"⚠️ {result_code}"

    arrow = "🟢 BUY" if sig_type == "BUY" else "🔴 SELL"
    return (
        f"⚡ <b>Trade Update</b>\n"
        f"{arrow} <b>{symbol}</b>\n"
        f"{status}"
    )


def notify_subscribers(signal_row: dict, execution_results: dict | None = None) -> int:
    """
    Send Telegram alerts to all active subscribers who have a chat_id set.
    execution_results: {user_name: ticket_or_error} from multi_executor (optional).
    Returns number of messages sent.
    """
    from core.user_accounts import get_enabled_users

    bot_token = _load_bot_token()
    if not bot_token:
        logger.warning("No subscriber_bot_token in config.yaml — Telegram alerts skipped.")
        return 0

    users = get_enabled_users()
    sent = 0

    for user in users:
        chat_id = user.get("telegram_chat_id", "")
        if not chat_id:
            continue  # subscriber hasn't set up Telegram yet

        # Build message: show execution result if available, else just signal
        user_name = user.get("name", "")
        if execution_results and user_name in execution_results:
            result_code = str(execution_results[user_name])
            text = build_trade_result_message(
                signal_row.get("symbol", "?"),
                signal_row.get("signal", "?"),
                result_code,
                user,
            )
        else:
            text = build_signal_message(signal_row, user)

        ok = _send(bot_token, chat_id, text)
        if ok:
            sent += 1
            logger.info(f"📲 Telegram sent to {user_name} (chat {chat_id})")
        else:
            logger.warning(f"📵 Telegram failed for {user_name} (chat {chat_id})")

    return sent


def send_test_message(chat_id: str) -> bool:
    """Send a test message to verify a subscriber's chat ID. Called from dashboard."""
    bot_token = _load_bot_token()
    if not bot_token:
        return False
    text = (
        "✅ <b>ForexAlert — Connected!</b>\n\n"
        "You will receive signal alerts here whenever a trade fires.\n\n"
        "<i>This is a test message.</i>"
    )
    return _send(bot_token, chat_id, text)


def get_bot_username(bot_token: str) -> str | None:
    """Get the bot's @username for displaying to subscribers."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        with urllib.request.urlopen(url, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                return "@" + result["result"]["username"]
    except Exception:
        pass
    return None
