
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.win_rate_trainer import WinRateFactory
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_low_volume_models(report_path, threshold=100):
    low_vol = []
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
                        if volume < threshold and target >= 90:
                            low_vol.append({
                                "pair": pair,
                                "type": sig_type,
                                "target": target,
                                "volume": volume
                            })
                    except ValueError:
                        continue
    return low_vol

def process_single_model(model_info):
    symbol = model_info['pair']
    sig_type = model_info['type']
    target = model_info['target']
    current_vol = model_info['volume']
    
    # Re-instantiate factory in worker
    from models.win_rate_trainer import WinRateFactory
    factory = WinRateFactory()
    base_dir = Path("models")
    
    logger.info(f"--- Processing {symbol} {sig_type} {target}% ---")
    
    try:
        trained_dir = base_dir / "specialist" / symbol / sig_type
        if not trained_dir.exists():
            return f"Error: Base dir missing for {symbol} {sig_type}"
            
        scaler = joblib.load(trained_dir / "scaler.joblib")
        X_val, y_val = factory._load_full_history(symbol, sig_type, scaler)
        
        if len(X_val) == 0:
            return f"Error: No history for {symbol}"
            
        model_opt, scaler_opt, thresh_opt, wr_opt, trades_opt = factory.optimize_expert(
            symbol, sig_type, target, X_val, y_val, scaler, min_trades=100
        )
        
        if model_opt and trades_opt > current_vol:
            save_dir = base_dir / symbol / str(target) / sig_type
            save_dir.mkdir(parents=True, exist_ok=True)
            model_opt.save(save_dir / "model.keras")
            joblib.dump(scaler_opt, save_dir / "scaler.joblib")
            
            config = {
                "symbol": symbol, "type": sig_type, "target_win_rate": target,
                "threshold": thresh_opt, "trades": int(trades_opt), "win_rate": float(wr_opt),
                "optimized": True, "previous_volume": current_vol,
                "created_at": datetime.now().isoformat()
            }
            with open(save_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            return f"SUCCESS: {symbol} {sig_type} {target}% -> {trades_opt} trades"
        else:
            return f"NO_IMPROVEMENT: {symbol} {sig_type} {target}%"
            
    except Exception as e:
        return f"FAILED: {symbol} {sig_type} {target}%: {e}"

def run_retraining():
    report_path = r'c:\Users\artem\Downloads\ApexForexSaaS\models\selective_accuracy_report.md'
    low_vol_models = parse_low_volume_models(report_path)
    
    if not low_vol_models:
        logger.info("No low-volume models found.")
        return

    logger.info(f"🚀 Starting PARALLEL Volume-Optimized Retraining for {len(low_vol_models)} models...")
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    workers = max(1, os.cpu_count() - 2)
    
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_model = {executor.submit(process_single_model, m): m for m in low_vol_models}
        
        for future in as_completed(future_to_model):
            res = future.result()
            logger.info(f"Result: {res}")
            results.append(res)
            
            # Regenerate report incrementally so we don't lose progress if cancelled
            if "SUCCESS" in res:
                logger.info("Regenerating incremental report...")
                from models.win_rate_trainer import WinRateFactory
                factory = WinRateFactory()
                factory.generate_comprehensive_report()
            
    logger.info("✨ All workers completed. Running final report...")
    from models.win_rate_trainer import WinRateFactory
    factory = WinRateFactory()
    factory.generate_comprehensive_report()

if __name__ == "__main__":
    from datetime import datetime
    import multiprocessing
    multiprocessing.freeze_support()
    run_retraining()
