import pytest
from datetime import datetime, timezone
from core.market_hours import is_weekend_halt, is_crypto
from core.guardrail import get_guardrail
from core.dynamic_model_whitelist import is_pair_whitelisted_for_model
from core.confluence_model import scan_and_execute_all_pairs

def test_crypto_is_weekend_exempt():
    # Test during weekend: Sunday 12:00 UTC
    sun_dt = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
    forex_halted, _ = is_weekend_halt(sun_dt, symbol="EURUSD")
    assert forex_halted is True

    # Crypto must be False (not halted)
    for sym in ["BTCUSD", "ETHUSD", "SOLUSD"]:
        halted, reason = is_weekend_halt(sun_dt, symbol=sym)
        assert halted is False
        assert "Crypto 24/7" in reason

def test_guardrail_crypto_safety():
    guard = get_guardrail()
    # General status during weekend may halt forex
    forex_status = guard.get_safety_status(symbol="EURUSD")
    assert forex_status["safe"] is False
    assert "WEEKEND_HALT" in forex_status["reason"]

    # Crypto must pass as safe
    crypto_status = guard.get_safety_status(symbol="BTCUSD")
    assert crypto_status["safe"] is True
    assert crypto_status["reason"] == "OK"

def test_crypto_dynamic_model_whitelists():
    # Crypto symbols with 0 past trades must not be deadlocked
    for model in ["confluence_ml_p60", "confluence_std_p25", "foundation_v1"]:
        for sym in ["BTCUSD", "ETHUSD", "SOLUSD"]:
            assert is_pair_whitelisted_for_model(model, sym) is True

def test_confluence_crypto_scan_runs_without_weekend_halt():
    # Dry run scan with auto_trade=False
    res = scan_and_execute_all_pairs(enabled_symbols=["BTCUSD", "ETHUSD", "SOLUSD"], force_auto_trade=False)
    assert res.get("scanned") == 3
    assert res.get("halted_reason") is None
