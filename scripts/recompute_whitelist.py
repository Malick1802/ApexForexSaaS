import sys
import os
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.performance_gate import get_performance_gate

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - WHITELIST - %(levelname)s - %(message)s')
logger = logging.getLogger("WhitelistTool")

def main():
    logger.info("🛡️  Recomputing Dynamic Performance Whitelist...")
    
    gate = get_performance_gate()
    
    # 1. Sync with the latest Static OOS Audit Baseline
    audit_json = PROJECT_ROOT / "logs" / "fleet_oos_results_static.json"
    if audit_json.exists():
        logger.info(f"Syncing with Audit Baseline: {audit_json}")
        gate.sync_with_audit_report(str(audit_json))
    else:
        logger.warning("Audit Baseline not found! Starting from empty list.")

    # 2. Recompute based on live results (Recency Rule: 14 days)
    logger.info("Applying Recency Rule (Last 14 days)...")
    gate.recompute_from_db(lookback_days=14)
    
    # 3. Final summary
    logger.info(f"✅ Performance Matrix Recomputed for {len(gate.performance_matrix)} symbols.")
    
    # Show highlights (Approved GOLD)
    for symbol, tiers in gate.performance_matrix.items():
        approved_tiers = [t for t, data in tiers.items() if data.get('status') == 'APPROVED']
        if approved_tiers:
            logger.info(f"  ⭐ {symbol} Approved Tiers: {', '.join(approved_tiers)}")

if __name__ == "__main__":
    main()
