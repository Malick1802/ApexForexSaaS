"""
Integration & Unit test suite for Dual-Master Symbol-Aware Routing:
- MetaQuotes-Demo is Master exclusively for Forex, Metals, Commodities.
- FTMO is Master exclusively for Crypto (BTCUSD, ETHUSD, SOLUSD).
- Cross-account execution and copying prevention.
"""
import pytest
from core.user_accounts import get_master_account, get_enabled_users
from core.market_hours import is_crypto
from core.mt5_connector import get_mt5


def test_master_account_resolution():
    """Verify get_master_account resolves correctly by asset class."""
    # Forex -> MetaQuotes Demo (Account 10, Login 10012808167)
    fx_master = get_master_account("EURUSD")
    assert fx_master is not None
    assert int(fx_master["mt5_login"]) == 10012808167
    assert fx_master["is_master"] == 1
    assert fx_master["is_crypto_master"] == 0

    # Metal / Commodity -> MetaQuotes Demo
    gold_master = get_master_account("XAUUSD")
    assert int(gold_master["mt5_login"]) == 10012808167

    # Crypto -> FTMO Demo (Account 13, Login 1514884839)
    btc_master = get_master_account("BTCUSD")
    assert btc_master is not None
    assert int(btc_master["mt5_login"]) == 1514884839
    assert btc_master["is_crypto_master"] == 1
    assert btc_master["is_master"] == 0

    eth_master = get_master_account("ETHUSD")
    assert int(eth_master["mt5_login"]) == 1514884839


def test_copy_trader_isolation():
    """
    Verify FTMO never executes or copies Forex trades,
    and MetaQuotes never executes or copies Crypto trades.
    """
    # For Forex: FTMO must NOT be in enabled users
    fx_users = get_enabled_users("EURUSD")
    fx_logins = [int(u["mt5_login"]) for u in fx_users]
    assert 1514884839 not in fx_logins, "FTMO account must NOT copy or execute Forex trades!"

    # For Crypto: MetaQuotes must NOT be in enabled users
    crypto_users = get_enabled_users("BTCUSD")
    crypto_logins = [int(u["mt5_login"]) for u in crypto_users]
    assert 10012808167 not in crypto_logins, "MetaQuotes account must NOT copy or execute Crypto trades!"


def test_live_mt5_symbol_attachment():
    """Verify MT5 connector switches to the proper running terminal instance."""
    # Check EURUSD attaches to MetaQuotes
    mt5_fx = get_mt5("EURUSD")
    assert mt5_fx is not None
    acc_fx = mt5_fx.account_info()
    assert acc_fx is not None
    assert acc_fx.login == 10012808167
    assert "MetaQuotes" in acc_fx.server

    # Check BTCUSD attaches to FTMO
    mt5_crypto = get_mt5("BTCUSD")
    assert mt5_crypto is not None
    acc_crypto = mt5_crypto.account_info()
    assert acc_crypto is not None
    assert acc_crypto.login == 1514884839
    assert "FTMO" in acc_crypto.server


def test_guardrail_dual_master():
    """Verify guardrail independently checks safety without state collision."""
    from core.guardrail import get_guardrail
    guard = get_guardrail()

    crypto_safety = guard.get_safety_status(symbol="BTCUSD")
    assert crypto_safety["safe"] is True

    # Check that crypto is active even if weekend (forex halted)
    forex_safety = guard.get_safety_status(symbol="EURUSD")
    # EURUSD may or may not be weekend halted depending on test run time, but should not crash
    assert "safe" in forex_safety


def test_symbol_aware_lot_sizing():
    """Verify lot sizing queries the respective master account balance."""
    from core.executive import ExecutiveEngine
    import yaml

    exec_engine = ExecutiveEngine(config_path="config.yaml")
    btc_lots = exec_engine._calculate_lot_size("BTCUSD", sl_pips=100.0)
    assert btc_lots > 0
    assert btc_lots <= 1.0  # FTMO $10k account margin/risk cap prevents massive sizing

    eur_lots = exec_engine._calculate_lot_size("EURUSD", sl_pips=10.0)
    assert eur_lots > 0


def test_mt5_provider_symbol_routing():
    """Verify MT5Provider dynamically selects correct broker."""
    from data_pipeline.providers.mt5_provider import MT5Provider
    provider = MT5Provider()

    # Normalizing BTCUSD connects to FTMO
    norm_btc = provider.normalize_symbol("BTCUSD")
    assert norm_btc == "BTCUSD"
    acc_btc = provider.mt5.account_info()
    assert acc_btc.login == 1514884839

    # Normalizing EURUSD connects to MetaQuotes
    norm_eur = provider.normalize_symbol("EURUSD")
    assert norm_eur == "EURUSD"
    acc_eur = provider.mt5.account_info()
    assert acc_eur.login == 10012808167
