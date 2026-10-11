"""
Unit tests for Confluence 3-Candle Breakout & Reclamation rules (Wick Mode).
Validates that:
1. Order 1: Candle 1 wick sweeps the line, but Magic Candle closed as a doji (body ratio 13.3%) -> REJECTED (not bearish).
2. Order 1 variant: Magic Candle closed as a bullish doji -> REJECTED (not bearish).
3. Order 2: Candle 1 was floating above the line (Low 112.10 > Line 111.99) -> REJECTED (never touched line).
4. Valid Wick Sweep SELL: C1 sweeps line with upper wick (even closing below line), C2 is strong bearish reclamation -> TRIGGERS.
5. Valid Wick Sweep BUY: C1 sweeps line with lower wick (even closing above line), C2 is strong bullish reclamation -> TRIGGERS.
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

def test_order1_doji_magic_candle_rejected():
    # Order 1 scenario: Line at 111.99
    # C1: Open 110.65, High 112.04, Low 110.65, Close 111.92 (Wick sweeps 111.99)
    # C2 (Magic Candle): Open 111.96, High 112.04, Low 111.59, Close 111.90
    # Range = 0.45, Body = 0.06 -> Body Ratio = 13.3% (< 15% DOJI)
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
                assert res["setup_detected"] is False, "Order 1 with doji Magic Candle should NOT trigger a setup!"

def test_order1_bullish_doji_rejected():
    # Order 1 variant where C2 is literally a bullish doji (Open 111.89, Close 111.90)
    rates = make_rates(
        {"open": 110.65, "high": 112.04, "low": 110.65, "close": 111.92},
        {"open": 111.89, "high": 112.04, "low": 111.59, "close": 111.90},
        {"open": 111.90, "high": 111.95, "low": 111.80, "close": 111.85},
    )
    mock_mt5 = setup_mock_mt5(rates)
    with patch("core.confluence_model.get_mt5", return_value=mock_mt5):
        with patch("core.confluence_model.get_day_lines", return_value={"upper_day_line": None, "lower_day_line": None, "day_start_utc": None}):
            with patch("core.confluence_model.get_swing_lines", return_value={"upper_swing_line": 111.99, "lower_swing_line": 108.00}):
                res = evaluate_confluence_setup(symbol="SOLUSD", include_wicks=True)
                assert res["setup_detected"] is False, "Order 1 with bullish doji Magic Candle should NOT trigger a setup!"

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

def test_valid_wick_sweep_sell():
    # Valid SELL with WICK sweep: Line at 111.99
    # C1: Bullish candle sweeping 111.99 with upper wick to 112.20, body closes at 111.80 (below line)
    # C2: Strong bearish candle (Open 111.80, High 111.95, Low 111.20, Close 111.30)
    rates = make_rates(
        {"open": 110.50, "high": 112.20, "low": 110.40, "close": 111.80},
        {"open": 111.80, "high": 111.95, "low": 111.20, "close": 111.30},
        {"open": 111.30, "high": 111.40, "low": 111.10, "close": 111.25},
    )
    mock_mt5 = setup_mock_mt5(rates)
    with patch("core.confluence_model.get_mt5", return_value=mock_mt5):
        with patch("core.confluence_model.get_day_lines", return_value={"upper_day_line": None, "lower_day_line": None, "day_start_utc": None}):
            with patch("core.confluence_model.get_swing_lines", return_value={"upper_swing_line": 111.99, "lower_swing_line": 108.00}):
                with patch("core.confluence_model.calculate_lot_size", return_value={"lots": 0.1, "risk_usd": 50.0, "balance": 10000.0}):
                    res = evaluate_confluence_setup(symbol="SOLUSD", include_wicks=True)
                    assert res["setup_detected"] is True, "Valid Wick Sweep SELL setup should trigger!"
                    assert res["direction"] == "SELL"
                    assert res["entry"] == 111.20
                    assert res["reclaimed_line"] == 111.99

def test_valid_wick_sweep_buy():
    # Valid BUY with WICK sweep: Line at 108.00
    # C1: Bearish candle sweeping 108.00 with lower wick to 107.70, body closes at 108.20 (above line)
    # C2: Strong bullish candle (Open 108.20, High 108.90, Low 108.10, Close 108.80)
    rates = make_rates(
        {"open": 109.50, "high": 109.60, "low": 107.70, "close": 108.20},
        {"open": 108.20, "high": 108.90, "low": 108.10, "close": 108.80},
        {"open": 108.80, "high": 109.00, "low": 108.60, "close": 108.70},
    )
    mock_mt5 = setup_mock_mt5(rates)
    with patch("core.confluence_model.get_mt5", return_value=mock_mt5):
        with patch("core.confluence_model.get_day_lines", return_value={"upper_day_line": None, "lower_day_line": None, "day_start_utc": None}):
            with patch("core.confluence_model.get_swing_lines", return_value={"upper_swing_line": 111.99, "lower_swing_line": 108.00}):
                with patch("core.confluence_model.calculate_lot_size", return_value={"lots": 0.1, "risk_usd": 50.0, "balance": 10000.0}):
                    res = evaluate_confluence_setup(symbol="SOLUSD", include_wicks=True)
                    assert res["setup_detected"] is True, "Valid Wick Sweep BUY setup should trigger!"
                    assert res["direction"] == "BUY"
                    assert res["entry"] == 108.90
                    assert res["reclaimed_line"] == 108.00

if __name__ == "__main__":
    test_order1_doji_magic_candle_rejected()
    test_order1_bullish_doji_rejected()
    test_order2_floating_candle_rejected()
    test_valid_wick_sweep_sell()
    test_valid_wick_sweep_buy()
    print("ALL 5 WICK-BASED CONFLUENCE TESTS PASSED SUCCESSFULLY!")
