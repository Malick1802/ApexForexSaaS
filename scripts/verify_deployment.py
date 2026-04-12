import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyDeployment")

def check_env():
    """Verify .env file exists and contains essential keys."""
    env_path = Path(".env")
    if not env_path.exists():
        logger.error("❌ .env file missing! You must copy this from your local machine.")
        return False
    
    with open(env_path, 'r') as f:
        content = f.read()
        essential_keys = ["TELEGRAM_TOKEN", "CHAT_ID"]
        missing = [k for k in essential_keys if k not in content]
        if missing:
            logger.warning(f"⚠️ .env might be missing keys: {', '.join(missing)}")
    
    logger.info("✅ .env file detected.")
    return True

def check_models():
    """Verify models directory is not empty."""
    model_dir = Path("models")
    if not model_dir.exists():
        logger.error("❌ models/ directory missing!")
        return False
    
    # Check for subdirectories (each pair should have one)
    subdirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]
    if len(subdirs) < 5: # Expecting 31+, but 5 is a minimum sanity check
        logger.error(f"❌ models/ directory only contains {len(subdirs)} model folders. Did you copy the contents?")
        return False
    
    logger.info(f"✅ models/ directory contains {len(subdirs)} pair models.")
    return True

def check_mt5():
    """Verify MetaTrader 5 connectivity (Native on Windows, Bridge on Linux)."""
    import platform
    is_windows = platform.system() == "Windows"
    
    try:
        if is_windows:
            import MetaTrader5 as mt5
            conn = mt5 # Use module directly on Windows
            logger.info("🖥️ Checking native Windows MT5 connection...")
        else:
            from mt5linux import MetaTrader5 as mt5
            conn = mt5() # Instantiate for Bridge
            logger.info("🌉 Checking Linux/Wine Bridge connection...")
            
        if not conn.initialize():
            logger.warning("⚠️ MT5 Initialize failed. Ensure the MT5 Terminal is OPEN and logged in.")
            return False
        
        acc = conn.account_info()
        if not acc:
            logger.warning("⚠️ MT5 Connected but no account login found. Check MT5 Terminal.")
            conn.shutdown()
            return False
        
        logger.info(f"✅ MT5 Connected: {acc.server} (Account #{acc.login})")
        conn.shutdown()
        return True
    except ImportError:
        pkg = "MetaTrader5" if is_windows else "mt5linux"
        logger.error(f"❌ {pkg} package not installed. Run scripts/azure_setup.ps1")
        return False
    except Exception as e:
        logger.error(f"❌ MT5 Check Error: {e}")
        return False

def main():
    logger.info("==================================================")
    logger.info("   APEXFOREX SAAS - AZURE DEPLOYMENT VERIFIER")
    logger.info("==================================================")
    
    env_ok = check_env()
    models_ok = check_models()
    mt5_ok = check_mt5()
    
    logger.info("--------------------------------------------------")
    if env_ok and models_ok and mt5_ok:
        logger.info("🚀 SYSTEM READY FOR PRODUCTION!")
        logger.info("You can now run 'RESTART.bat' to start the services.")
    else:
        logger.error("🚫 DEPLOYMENT INCOMPLETE. Please fix the errors above.")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
