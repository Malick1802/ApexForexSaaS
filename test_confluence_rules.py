"""
Unit tests for Confluence 3-Candle Breakout & Reclamation rules.
Validates that:
1. False breakout where Candle 1 only wicks above the line but closes below does NOT trigger.
2. Floating candle where Candle 1 never touches the line (Low > Line) does NOT trigger.
3. True breakout & reclamation (C1 body crosses line, C2 body reclaims line) DOES trigger.
4. True BUY breakout & reclamation DOES trigger.
"""
from unittest.mock import MagicMock, patch
from core.confluence_model import evaluate_confluence_setup

def make_rates(c1, c2, c3):
    return [
        {"time": 100, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
        {"time": 1000, "open": c1["open"], "high": c1["high"], "low": c1["low"], "close": c1["close"]},
        {"time": 1900, "open": c2["open"], "high": c2["high"], "low": c2["low"], "close": c2["close"]},
        {"time": 2800, "open": c3["open"], "high": c3["high"], "low": c3["low"], "close": c3["close"]},
    ]

def setup_mock_mt5(rates):
    mock_mt5 = MagicMock()
    mock_mt5.TIMEFRAME_M15 = 15
    mock_mt5.copy_rates_from_pos.return_value = rates
    mock_mt5.account_info.return_value.balance = 10000.0
    mock_mt5.account_info.return_value.login = 1514884839
    mock_mt5.account_info.return_value.server = "FTMO-Demo"
    return mock_mt5

def test_order1_wick_only_rejected():
    # Order 1 scenario: Line at 111.99
    # C1: Open 110.65, High 112.04, Low 110.65, Close 111.92 (Closed BELOW 111.99)
    # C2: Open 111.96, High 112.04, Low 111.59, Close 111.90 (Closed BELOW 111.99)
    rates = make_rates(
        {"open": 110.65, "high": 112.04, "low": 110.65, "close": 111.92},
        {"open": 111.96, "high": 112.04, "low": 111.59, "close": 111.90},
        {"open": 111.90, "high": 111.95, "low": 111.80, "close": 111.85},
    )
    mock_mt5 = setup_mock_mt5(rates)
    with patch("core.confluence_model.get_mt5", return_value=mock_mt5):
        with patch("core.confluence_model.get_day_lines", return_value={"upper_day_line": None, "lower_day_line": None, "day_start_utc": None}):
            with patch("core.confluence_model.get_swing_lines", return_value={"upper_swing_line": 111.99, "lower_swing_line": 108.00}):
                res = evaluate_confluence_setup(symbol="SOLUSD", include_wicks=True)
                assert res["setup_detected"] is False, "Order 1 wick-only rejection should NOT trigger a setup!"

def test_order2_floating_candle_rejected():
    # Order 2 scenario: Line at 111.99
    # C1: Open 112.17, High 112.38, Low 112.10, Close 112.24 (Floating entirely above 111.99, Low > 111.99)
    # C2: Open 112.22, High 112.31, Low 111.66, Close 111.85
    rates = make_rates(
        {"open": 112.17, "high": 112.38, "low": 112.10, "close": 112.24},
        {"open": 112.22, "high": 112.31, "low": 111.66, "close": 111.85},
        {"open": 111.85, "high": 112.00, "low": 111.70, "close": 111.80},
    )
    mock_mt5 = setup_mock_mt5(rates)
    with patch("core.confluence_model.get_mt5", return_value=mock_mt5):
        with patch("core.confluence_model.get_day_lines", return_value={"upper_day_line": None, "lower_day_line": None, "day_start_utc": None}):
            with patch("core.confluence_model.get_swing_lines", return_value={"upper_swing_line": 111.99, "lower_swing_line": 108.00}):
                res = evaluate_confluence_setup(symbol="SOLUSD", include_wicks=True)
                assert res["setup_detected"] is False, "Order 2 floating candle should NOT trigger a setup!"

def test_valid_sell_breakout_and_reclamation():
    # Valid SELL: Line at 111.99
    # C1: Bullish breakout across 111.99 (Open 111.50, Close 112.20)
    # C2: Bearish reclamation below 111.99 (Open 112.20, Close 111.80)
    rates = make_rates(
        {"open": 111.50, "high": 112.30, "low": 111.40, "close": 112.20},
        {"open": 112.20, "high": 112.25, "low": 111.70, "close": 111.80},
        {"open": 111.80, "high": 111.90, "low": 111.60, "close": 111.75},
    )
    mock_mt5 = setup_mock_mt5(rates)
    with patch("core.confluence_model.get_mt5", return_value=mock_mt5):
        with patch("core.confluence_model.get_day_lines", return_value={"upper_day_line": None, "lower_day_line": None, "day_start_utc": None}):
            with patch("core.confluence_model.get_swing_lines", return_value={"upper_swing_line": 111.99, "lower_swing_line": 108.00}):
                with patch("core.confluence_model.calculate_lot_size", return_value={"lots": 0.1, "risk_usd": 50.0, "balance": 10000.0}):
                    res = evaluate_confluence_setup(symbol="SOLUSD", include_wicks=True)
                    assert res["setup_detected"] is True, "Valid SELL setup should trigger!"
                    assert res["direction"] == "SELL"
                    assert res["entry"] == 111.70
                    assert res["reclaimed_line"] == 111.99

def test_valid_buy_breakout_and_reclamation():
    # Valid BUY: Line at 108.00
    # C1: Bearish breakout below 108.00 (Open 108.50, Close 107.70)
    # C2: Bullish reclamation above 108.00 (Open 107.70, Close 108.30)
    rates = make_rates(
        {"open": 108.50, "high": 108.60, "low": 107.60, "close": 107.70},
        {"open": 107.70, "high": 108.40, "low": 107.65, "close": 108.30},
        {"open": 108.30, "high": 108.50, "low": 108.20, "close": 108.40},
    )
    mock_mt5 = setup_mock_mt5(rates)
    with patch("core.confluence_model.get_mt5", return_value=mock_mt5):
        with patch("core.confluence_model.get_day_lines", return_value={"upper_day_line": None, "lower_day_line": None, "day_start_utc": None}):
            with patch("core.confluence_model.get_swing_lines", return_value={"upper_swing_line": 111.99, "lower_swing_line": 108.00}):
                with patch("core.confluence_model.calculate_lot_size", return_value={"lots": 0.1, "risk_usd": 50.0, "balance": 10000.0}):
                    res = evaluate_confluence_setup(symbol="SOLUSD", include_wicks=True)
                    assert res["setup_detected"] is True, "Valid BUY setup should trigger!"
                    assert res["direction"] == "BUY"
                    assert res["entry"] == 108.40
                    assert res["reclaimed_line"] == 108.00

if __name__ == "__main__":
    test_order1_wick_only_rejected()
    test_order2_floating_candle_rejected()
    test_valid_sell_breakout_and_reclamation()
    test_valid_buy_breakout_and_reclamation()
    print("ALL CONFLUENCE GEOMETRY TESTS PASSED SUCCESSFULLY!")
