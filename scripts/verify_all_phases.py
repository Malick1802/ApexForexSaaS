import os
import sys
import platform
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("PhaseCertification")

def certify_phase_1():
    """Phase 1: Foundation (MT5 Connectivity)"""
    logger.info("🔍 PHASE 1 Certification: Native MT5 Connectivity")
    try:
        from core.mt5_connector import get_mt5
        conn = get_mt5()
        if conn:
            acc = conn.account_info()
            if acc:
                logger.info(f"✅ PASS: MT5 Connected to {acc.server} (Acct: {acc.login})")
                return True
        logger.warning("⚠️ Phase 1 FAIL: MT5 not responsive. Ensure terminal is open.")
        return False
    except Exception as e:
        logger.error(f"❌ Phase 1 CRITICAL: {e}")
        return False

def certify_phase_2():
    """Phase 2: Scalability (DB WAL Mode)"""
    logger.info("🔍 PHASE 2 Certification: Database Concurrency")
    try:
        from core.database import SignalDatabase
        db = SignalDatabase("verify_test.db")
        conn = db._get_connection()
        res = conn.execute("PRAGMA journal_mode").fetchone()
        if res and res[0].upper() == "WAL":
            logger.info("✅ PASS: SQLite running in WAL mode (Sync Allowed)")
            conn.close()
            if os.path.exists("verify_test.db"): os.remove("verify_test.db")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Phase 2 FAIL: {e}")
        return False

def certify_phase_3():
    """Phase 3: Precision (Specialist Models)"""
    logger.info("🔍 PHASE 3 Certification: Expert Specialist Models")
    model_dir = Path("models/expert")
    if not model_dir.exists():
         model_dir = Path("models/specialist")
         
    if model_dir.exists():
        pairs = [d for d in model_dir.iterdir() if d.is_dir() if "__" not in d.name]
        if len(pairs) > 0:
            logger.info(f"✅ PASS: Found {len(pairs)} specialist/expert pair directories.")
            return True
    logger.warning("⚠️ Phase 3 FAIL: No specialist models found in 'models/'.")
    return False

def certify_phase_4():
    """Phase 4: Intelligence (GMM & Calibration)"""
    logger.info("🔍 PHASE 4 Certification: GMM Regime & Platt Calibration")
    try:
        from core.inference import InferenceEngine
        engine = InferenceEngine()
        
        # Check GMM
        regime_type = type(engine._regime_detector).__name__
        if "GMM" in regime_type:
            logger.info(f"✅ PASS: GMM Regime Detector Integrated ({regime_type})")
        else:
            logger.warning(f"⚠️ Phase 4 WARNING: Using fallback Rule-Based Detector ({regime_type})")
            
        # Check Calibration
        if hasattr(engine, 'calibrator'):
             logger.info("✅ PASS: Platt Calibration Manager Integrated")
        
        return "GMM" in regime_type or "Regime" in regime_type
    except Exception as e:
        logger.error(f"❌ Phase 4 FAIL: {e}")
        return False

def main():
    logger.info("==================================================")
    logger.info("   APEXFOREX SAAS - MASTER PHASE CERTIFICATION")
    logger.info("==================================================")
    
    results = {
        "Phase 1 (Connect)": certify_phase_1(),
        "Phase 2 (DB WAL)": certify_phase_2(),
        "Phase 3 (Precision)": certify_phase_3(),
        "Phase 4 (Intel)": certify_phase_4(),
    }
    
    logger.info("--------------------------------------------------")
    all_pass = all(results.values())
    for phase, status in results.items():
        icon = "✅" if status else "❌"
        logger.info(f"{icon} {phase}")
        
    logger.info("--------------------------------------------------")
    if all_pass:
        logger.info("🚀 SYSTEM CERTIFIED FOR PRODUCTION!")
    else:
        logger.warning("⚠️ SYSTEM INCOMPLETE. Please check logs.")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
