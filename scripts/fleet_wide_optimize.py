
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_all_models(report_path):
    models = []
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
                        # We process ALL models for the fleet-wide iteration
                        models.append({
                            "pair": pair,
                            "type": sig_type,
                            "target": target,
                            "volume": volume
                        })
                    except ValueError:
                        continue
    
    # Priority Sorting: 90%+ targets with < 100 trades first
    def get_priority(m):
        if m['target'] >= 90 and m['volume'] < 100:
            return 0  # Highest Priority
        elif m['target'] >= 90:
            return 1  # High target but already hits volume
        else:
            return 2  # Lower targets
            
    models.sort(key=get_priority)
    return models

def process_single_model(model_info):
    symbol = model_info['pair']
    sig_type = model_info['type']
    target = model_info['target']
    current_vol = model_info['volume']
    
    # Re-instantiate factory in worker
    from models.win_rate_trainer import WinRateFactory
    factory = WinRateFactory()
    base_dir = Path("models")
    
    logger.info(f"--- FLEET OPTIMIZING: {symbol} {sig_type} {target}% ---")
    
    try:
        trained_dir = base_dir / "specialist" / symbol / sig_type
        if not trained_dir.exists():
            return f"Error: Base dir missing for {symbol} {sig_type}"
            
        scaler = joblib.load(trained_dir / "scaler.joblib")
        X_val, y_val = factory._load_full_history(symbol, sig_type, scaler)
        
        if len(X_val) == 0:
            return f"Error: No history for {symbol}"
            
        # Fleet-wide search targets MAXIMUM volume (min_trades=0 but we pick the max vol that hits the WR)
        # Actually optimize_expert has volume priority logic
        model_opt, scaler_opt, thresh_opt, wr_opt, trades_opt = factory.optimize_expert(
            symbol, sig_type, target, X_val, y_val, scaler, min_trades=200 # Higher bar for fleet-wide
        )
        
        if model_opt and trades_opt > current_vol:
            save_dir = base_dir / symbol / str(target) / sig_type
            save_dir.mkdir(parents=True, exist_ok=True)
            model_opt.save(save_dir / "model.keras")
            joblib.dump(scaler_opt, save_dir / "scaler.joblib")
            
            config = {
                "symbol": symbol, "type": sig_type, "target_win_rate": target,
                "threshold": thresh_opt, "trades": int(trades_opt), "win_rate": float(wr_opt),
                "optimized": True, "fleet_wide": True, "previous_volume": current_vol,
                "updated_at": datetime.now().isoformat()
            }
            with open(save_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            return f"IMPROVED: {symbol} {sig_type} {target}% -> {trades_opt} trades"
        else:
            return f"PEAK_ALREADY: {symbol} {sig_type} {target}%"
            
    except Exception as e:
        return f"FAILED: {symbol} {sig_type} {target}%: {e}"

def run_fleet_optimization():
    report_path = r'c:\Users\artem\Downloads\ApexForexSaaS\models\selective_accuracy_report.md'
    all_models = parse_all_models(report_path)
    
    if not all_models:
        logger.info("No models found in report.")
        return

    logger.info(f"🚀 Starting FLEET-WIDE Master Optimization for {len(all_models)} models...")
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    workers = max(1, os.cpu_count() - 2)
    
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_model = {executor.submit(process_single_model, m): m for m in all_models}
        
        for future in as_completed(future_to_model):
            res = future.result()
            logger.info(f"Result: {res}")
            results.append(res)
            
            # Regenerate report incrementally
            if "IMPROVED" in res:
                from models.win_rate_trainer import WinRateFactory
                factory = WinRateFactory()
                factory.generate_comprehensive_report()
            
    logger.info("✨ Fleet-Wide Optimization completed. Running final report...")
    from models.win_rate_trainer import WinRateFactory
    factory = WinRateFactory()
    factory.generate_comprehensive_report()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_fleet_optimization()
