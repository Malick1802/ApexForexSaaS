import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import joblib
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.win_rate_trainer import WinRateFactory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/volume_boost.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_low_volume_targets(report_path):
    targets = []
    if not os.path.exists(report_path):
        return []
        
    with open(report_path, 'r') as f:
        for line in f:
            if line.startswith('|') and 'Pair' not in line and '---' not in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 8:
                    pair = parts[1]
                    sig_type = parts[2]
                    target_str = parts[3].replace('%', '')
                    volume_str = parts[6]
                    
                    try:
                        target = int(target_str)
                        volume = int(volume_str)
                        
                        # FILTER: Only 90% and 95% targets with < 1000 volume
                        if target in [90, 95] and volume < 1000:
                            targets.append({
                                "pair": pair,
                                "type": sig_type,
                                "target": target,
                                "volume": volume
                            })
                    except ValueError:
                        continue
    
    # Priority Sorting: Lower volume first (needs most help)
    targets.sort(key=lambda x: x['volume'])
    return targets

def process_volume_boost(model_info):
    symbol = model_info['pair']
    sig_type = model_info['type']
    target = model_info['target']
    current_vol = model_info['volume']
    
    # Re-instantiate factory in worker
    from models.win_rate_trainer import WinRateFactory
    factory = WinRateFactory()
    base_dir = Path("models")
    
    logger.info(f"🚀 BOOSTING: {symbol} {sig_type} {target}% (Current Vol: {current_vol})")
    
    try:
        trained_dir = base_dir / "specialist" / symbol / sig_type
        if not trained_dir.exists():
            return f"Error: Base dir missing for {symbol} {sig_type}"
            
        scaler = joblib.load(trained_dir / "scaler.joblib")
        X_val, y_val = factory._load_full_history(symbol, sig_type, scaler)
        
        if len(X_val) == 0:
            return f"Error: No history for {symbol}"
            
        # VOLUME BOOST: Set min_trades to 1000
        model_opt, scaler_opt, thresh_opt, wr_opt, trades_opt = factory.optimize_expert(
            symbol, sig_type, target, X_val, y_val, scaler, min_trades=1000
        )
        
        if model_opt:
            improvement = trades_opt - current_vol
            if improvement > 0:
                save_dir = base_dir / symbol / str(target) / sig_type
                save_dir.mkdir(parents=True, exist_ok=True)
                model_opt.save(save_dir / "model.keras")
                joblib.dump(scaler_opt, save_dir / "scaler.joblib")
                
                config = {
                    "symbol": symbol, "type": sig_type, "target_win_rate": target,
                    "threshold": thresh_opt, "trades": int(trades_opt), "win_rate": float(wr_opt),
                    "optimized": True, "volume_boost": True, "previous_volume": current_vol,
                    "updated_at": datetime.now().isoformat()
                }
                with open(save_dir / "config.json", "w") as f:
                    json.dump(config, f, indent=2)
                return f"✅ BOOST SUCCESS: {symbol} {sig_type} {target}% -> {trades_opt} trades (+{improvement})"
            else:
                return f"⚠️ No Volume Increase: {symbol} {sig_type} {target}% stuck at {trades_opt}"
        else:
            return f"❌ Optimization Failed: {symbol} {sig_type} {target}%"
            
    except Exception as e:
        return f"🔥 CRASH: {symbol} {sig_type} {target}%: {e}"

def run_volume_optimization():
    report_path = r'c:\Users\artem\Downloads\ApexForexSaaS\models\selective_accuracy_report.md'
    targets = parse_low_volume_targets(report_path)
    
    if not targets:
        logger.info("Values already satisfying conditions (All 90/95% models have > 1000 trades).")
        return

    logger.info(f"🏎️ Starting HIGH-VOLUME BOOST for {len(targets)} models...")
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    # REDUCED WORKERS FOR STABILITY (Windows Process Pool Fix)
    workers = 2 
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_model = {executor.submit(process_volume_boost, m): m for m in targets}
        
        for future in as_completed(future_to_model):
            res = future.result()
            logger.info(f"Result: {res}")
            
            # Regenerate report on success
            if "BOOST SUCCESS" in res:
                from models.win_rate_trainer import WinRateFactory
                factory = WinRateFactory()
                factory.generate_comprehensive_report()
            
    logger.info("✨ Volume Boost Campaign completed.")
    from models.win_rate_trainer import WinRateFactory
    factory = WinRateFactory()
    factory.generate_comprehensive_report()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_volume_optimization()
