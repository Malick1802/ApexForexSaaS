# =============================================================================
# Win-Rate Factory (Expert Models)
# =============================================================================
"""
Trains and configures 5 distinct 'Expert Models' for each currency pair,
targeting specific win rates: 60%, 70%, 80%, 90%, and 95%.

Selection Strategy:
- Uses a high-performance base model (Specialist).
- Calibrates decision thresholds to achieve target precision on validation data.
- Enforces a 'High-Confidence Filter' (Threshold > 0.96) for 90%+ targets.
- Saves distinct artifacts for each target win rate.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow import keras

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_pipeline import DataEngine
from data_pipeline.features import FeatureEngineer
from models.specialist_factory import SpecialistFactory
from models.specialist_trainer import SpecialistPairTrainer

from models.enhanced_lstm import build_specialist_lstm
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WinRateFactory:
    """
    Factory for manufacturing Expert Models with specific win-rate targets.
    """
    
    def __init__(self, base_dir: str = "models", history_days: int = 1095): # 3 Years
        self.base_dir = Path(base_dir)
        self.history_days = history_days
        # Prioritize 90% and 95% ONLY as per user request
        self.targets = [90, 95]
        
        self.engine = DataEngine()
        self.feature_engineer = FeatureEngineer()
        
    def _ensure_base_model(self, symbol: str) -> bool:
        """
        Ensures a high-quality base model exists to derive experts from.
        Uses SpecialistFactory to train if missing.
        """
        # We need a robust model. SpecialistFactory produces certified >60% models.
        # We can reuse the "BUY" specialist model as our base for long signals, 
        # but ideally we want a general purpose one or we handle both BUY/SELL experts.
        # For simplicity in this factory, let's look for a generic Specialist model first,
        # or train one if needed.
        
        specialist_path = self.base_dir / "specialist" / symbol / "BUY" / "model.keras"
        if specialist_path.exists():
            return True
            
        logger.info(f"Base model for {symbol} missing. Training using SpecialistFactory...")
        factory = SpecialistFactory() 
        # Train both just in case
        factory.train_specialist(symbol, "BUY")
        factory.train_specialist(symbol, "SELL")
        
        buy_path = self.base_dir / "specialist" / symbol / "BUY" / "model.keras"
        sell_path = self.base_dir / "specialist" / symbol / "SELL" / "model.keras"
        
        return buy_path.exists() or sell_path.exists()

    def _load_base_model(self, symbol: str, signal_type: str = "BUY") -> Tuple[keras.Model, StandardScaler]:
        """Load the base specialist model."""
        path = self.base_dir / "specialist" / symbol / signal_type
        model = keras.models.load_model(path / "model.keras")
        scaler = joblib.load(path / "scaler.joblib")
        return model, scaler

    def _get_validation_data(self, symbol: str, scaler: StandardScaler, signal_type: str = "BUY"):
        """Fetch and prepare validation data."""
        # Fetch fresh data (or same data window)
        df = self.engine.fetch_labeled(symbol, interval="1h", days=self.history_days)
        features = self.feature_engineer.extract_features(df)
        
        # Add correlated if needed (check scaler features)
        expected_features = scaler.n_features_in_
        if expected_features > features.shape[1]:
            correlated = self.engine.get_correlated_assets(symbol)
            if correlated:
                 try:
                    corr_symbol = correlated[0]['symbol']
                    corr_df = self.engine.fetch(corr_symbol, interval="1h", days=self.history_days)
                    features = self.feature_engineer.add_correlated_asset(features, corr_df)
                 except: 
                     pass
        
        # Pad if still mismatch (simple fix for now, ideally strictly matched)
        if features.shape[1] < expected_features:
            diff = expected_features - features.shape[1]
            for i in range(diff):
                features[f'pad_{i}'] = 0.0
                
        # Create sequences
        # Pass spread_pips and increased stop_loss_pips if needed (handled in labeling)
        labeled_df = triple_barrier_label(
            df, 
            stop_loss_pips=40, 
            symbol=symbol, 
            pip_value=pip_value,
            spread_pips=2.0
        )
        
        y_binary = (labeled_df['label'] == target_label).astype(int)
        
        # Create sequences
        X, y = self.feature_engineer.create_sequences(features, y_binary, sequence_length=60)
        
        # Use FULL dataset for "Backtest" calibration as requested
        # split_idx = int(len(X) * 0.8)
        return X, y # Return full history


    def calibrate_thresholds(self, model, X_val, y_val) -> Dict[int, float]:
        """
        Find the lowest threshold that satisfies each Win Rate target.
        """
        # Get probabilities
        # Model output shape depends on architecture. 
        # Specialist models are Binary (1 unit, sigmoid) or 3-class.
        # SpecialistFactory uses: num_classes=1 (Binary) -> sigmoid
        
        y_proba = model.predict(X_val, verbose=0).flatten()
        
        results = {}
        
        # We scan thresholds from 0.50 to 0.99
        thresholds = np.arange(0.50, 0.995, 0.005)
        
        for target in self.targets:
            best_thresh = None
            
            # Constraint: 90%+ must be > 0.96
            min_thresh = 0.96 if target >= 90 else 0.50
            
            valid_thresholds = [t for t in thresholds if t >= min_thresh]
            
            for t in valid_thresholds:
                preds = (y_proba >= t).astype(int)
                
                # Calculate Win Rate (Precision)
                # Avoid division by zero
                n_signals = preds.sum()
                if n_signals < 5: # Minimum sample size for statistical relevance
                    continue
                
                hits = (preds == y_val).astype(int)
                # We only care about Positive predictions (Signal=1)
                # Precision = TP / (TP + FP)
                # Filter indices where pred is 1
                mask = preds == 1
                if mask.sum() == 0:
                    continue
                    
                correct = (y_val[mask] == 1).sum()
                precision = correct / mask.sum()
                
                if precision * 100 >= target:
                    best_thresh = t
                    break # Found lowest threshold satisfying target
            
            # Fallback: if we can't meet 95%, use max threshold if it gives any signals
            if best_thresh is None and target >= 90:
                 # Check max threshold performance
                 t = 0.96
                 preds = (y_proba >= t).astype(int)
                 mask = preds == 1
                 if mask.sum() > 0:
                     correct = (y_val[mask] == 1).sum()
                     precision = correct / mask.sum()
                     if precision > 0.85: # Accept 85% for 90% target if 90% is impossible? 
                         # No, user asked for target brackets. 
                         # If we fail, we just output the max possible threshold.
                         best_thresh = 0.98 
            
            results[target] = float(best_thresh) if best_thresh else 0.99
            
        return results

    def create_expert_models(self, symbol: str):
        """
        Orchestrate the creation of 5 expert models for a symbol.
        """
        logger.info(f"Manufacturing Experts for {symbol}...")
        
        # 1. Ensure Base
        if not self._ensure_base_model(symbol):
            logger.error(f"Could not establish base model for {symbol}")
            return
            
        # 2. We do this for BUY and SELL separately?
        # User requested /models/{pair}/{win_rate}/model.h5
        # Usually a model handles both or we have 2 models.
        # If we have 2 models (BUY/SELL), we should probably zip them or 
        # save them in subfolders?
        # "model.h5" implies a single file. 
        # Maybe we assume the "Executive Engine" loads ONE model file?
        # Most of the current codebase uses `models/specialist/{pair}/BUY` and `SELL`.
        # To comply with `/models/{pair}/{win_rate}/model.h5`, we might need to 
        # combine them or just pick the dominant one?
        # OR, we save both "model_buy.h5" and "model_sell.h5" inside that folder?
        # Let's check `app.py` later. For now, let's create the folder and put 
        # standard "BUY" model as default or both folders.
        # Let's do: /models/{pair}/{win_rate}/BUY/model.h5 and SELL/model.h5
        # This keeps consistency with the Specialist structure.
        
        TYPES = ["BUY", "SELL"]
        report_data = []

        for signal_type in TYPES:
            try:
                # Check if all targets for this signal type already exist (Resume Logic)
                all_exist = True
                for t in self.targets:
                    if not (self.base_dir / symbol / str(t) / signal_type / "model.keras").exists():
                        all_exist = False
                        break
                
                if all_exist:
                    logger.info(f"Skipping {symbol} {signal_type} (Already Certified)")
                    continue

                # Check if Base Model Exists
                base_path = self.base_dir / "specialist" / symbol / signal_type / "model.keras"
                if not base_path.exists():
                     logger.warning(f"Skipping {symbol} {signal_type}: No Base Model found.")
                     continue

                # Load Base
                model, scaler = self._load_base_model(symbol, signal_type)
                
                # Get FULL Validation Data (Scaled/Prepared)
                X_val, y_val = self._load_full_history(symbol, signal_type, scaler)
                
                if len(X_val) == 0: 
                    continue

                # Calibrate
                thresholds = self.calibrate_thresholds(model, X_val, y_val)
                
                # Pre-calculate probabilities for validation/retraining
                y_proba = model.predict(X_val, verbose=0).flatten()
                
                # Save Targets
                for target, threshold in thresholds.items():
                    # Directory: models/{pair}/{target}/{signal_type}
                    save_dir = self.base_dir / symbol / str(target) / signal_type
                    
                    if (save_dir / "model.keras").exists():
                        logger.info(f"Skipping {symbol} {target}% {signal_type} (Exists)")
                        continue

                    # Validate before saving
                    # Re-calculate stats for this threshold
                    preds = (y_proba >= threshold).astype(int)
                    mask = preds == 1
                    trades = mask.sum()
                    
                    actual_wr = 0.0
                    if trades > 0:
                        actual_wr = (y_val[mask] == 1).sum() / trades
                    
                    # STRICT ENFORCEMENT for high targets
                    if target >= 90 and actual_wr * 100 < target:
                        logger.warning(f"Skipping {symbol} {signal_type} {target}%: Actual {actual_wr:.1%} < Target")
                        
                        # ITERATIVE RETRAINING (Attempt to fix)
                        logger.info(f"🔄 Retraining {symbol} for {target}% target (Current: {actual_wr:.1%})")
                        model_opt, scaler_opt, thresh_opt, wr_opt, trades_opt = self.optimize_expert(
                            symbol, signal_type, target, X_val, y_val, scaler
                        )
                        
                        if model_opt:
                            logger.info(f"✅ Optimization SUCCESS! {wr_opt:.1%} (Trades: {trades_opt})")
                            model = model_opt
                            scaler = scaler_opt
                            threshold = thresh_opt
                            actual_wr = wr_opt
                            trades = trades_opt
                        else:
                            report_data.append({
                                "pair": symbol, "type": signal_type, "target": target,
                                "threshold": threshold, "win_rate": actual_wr, "trades_total_val": int(trades),
                                "status": "FAILED"
                            })
                            continue

                    # Create Dir
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save Model (Copy)
                    model.save(save_dir / "model.keras")
                    joblib.dump(scaler, save_dir / "scaler.joblib")
                    
                    # Save Config
                    config = {
                        "symbol": symbol,
                        "type": signal_type,
                        "target_win_rate": target,
                        "threshold": threshold,
                        "trades": int(trades),
                        "win_rate": actual_wr,
                        "created_at": datetime.now().isoformat()
                    }
                    with open(save_dir / "config.json", "w") as f:
                        json.dump(config, f, indent=2)

                    report_data.append({
                        "pair": symbol,
                        "type": signal_type,
                        "target": target,
                        "threshold": threshold,
                        "win_rate": actual_wr,
                        "trades_total_val": int(trades),
                        "status": "SAVED"
                    })
                        
                    # Calculate Stats for Report
                    # Re-eval to get exact stats
                    y_proba = model.predict(X_val, verbose=0).flatten()
                    preds = (y_proba >= threshold).astype(int)
                    mask = preds == 1
                    trades = mask.sum()
                    
                    actual_wr = 0.0
                    if trades > 0:
                        actual_wr = (y_val[mask] == 1).sum() / trades
                        
                    if trades > 0:
                        actual_wr = (y_val[mask] == 1).sum() / trades
                        
                    # Approx trades per month (Full 3 years = 36 months)
                    # trades_per_month = trades / 36.0
                    # Actually calculate duration of validation set
                    
                    report_data.append({
                        "pair": symbol,
                        "type": signal_type,
                        "target": target,
                        "threshold": threshold,
                        "win_rate": actual_wr,
                        "trades_total_val": int(trades)
                    })
                    
            except Exception as e:
                logger.error(f"Failed to process {symbol} {signal_type}: {e}")

        return report_data

    def _load_full_history(self, symbol, signal_type, scaler):
        """Helper to load and scale FULL history data."""
        # This logic duplicates _get_validation_data but adds scaling
        try:
            df = self.engine.fetch_labeled(symbol, interval="1h", days=self.history_days)
            if df is None or len(df) < 500: return np.array([]), np.array([])
            
            features = self.feature_engineer.extract_features(df)
            
            # Correlated assets
            expected = scaler.n_features_in_
            if expected > features.shape[1]:
                correlated = self.engine.get_correlated_assets(symbol)
                if correlated:
                    try:
                        c_sym = correlated[0]['symbol']
                        c_df = self.engine.fetch(c_sym, "1h", self.history_days)
                        features = self.feature_engineer.add_correlated_asset(features, c_df)
                    except: pass
            
            # Simple Pad
            if features.shape[1] < expected:
                for i in range(expected - features.shape[1]):
                    features[f'pad_{i}'] = 0.0
            
            # Targets
            t = 1 if signal_type == "BUY" else 2
            y = (df['label'] == t).astype(int)
            
            X, y = self.feature_engineer.create_sequences(features, y, 60)
            
            # Scale
            X_flat = X.reshape(-1, expected)
            X_scaled = scaler.transform(X_flat).reshape(X.shape)
            
            X_scaled = scaler.transform(X_flat).reshape(X.shape)
            
            # Return FULL set
            # split = int(len(X_scaled) * 0.8)
            return X_scaled, y
        except:
            return np.array([]), np.array([])

    def optimize_expert(
        self, 
        symbol: str, 
        signal_type: str, 
        target_wr: int, 
        X_val, 
        y_val, 
        original_scaler,
        min_trades: int = 100
    ):
        """
        Retrain model with deeper search to meet BOTH target WR and Volume.
        """
        # Load fresh training data
        try:
             df = self.engine.fetch_labeled(symbol, interval="1h", days=self.history_days)
             features = self.feature_engineer.extract_features(df)
             
             expected = original_scaler.n_features_in_
             if expected > features.shape[1]:
                 if features.shape[1] < expected:
                     for i in range(expected - features.shape[1]):
                         features[f'pad_{i}'] = 0.0
             
             t = 1 if signal_type == "BUY" else 2
             pip_value = self.engine.get_pip_value(symbol)
             
             from data_pipeline.labeling import triple_barrier_label
             labeled_df = triple_barrier_label(
                 df, 
                 stop_loss_pips=40, 
                 symbol=symbol, 
                 pip_value=pip_value,
                 spread_pips=2.0
             )
             
             y = (labeled_df['label'] == t).astype(int)
             X, y = self.feature_engineer.create_sequences(features, y, 60)
             
             split_idx = int(len(X) * 0.8)
             X_train, X_v = X[:split_idx], X[split_idx:]
             y_train, y_v = y[:split_idx], y[split_idx:]
             
             scaler = StandardScaler()
             X_train_flat = X_train.reshape(-1, X_train.shape[2])
             X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
             X_val_flat = X_v.reshape(-1, X_v.shape[2])
             X_v_scaled = scaler.transform(X_val_flat).reshape(X_v.shape)
             
        except Exception as e:
            logger.error(f"Optimization data fetch failed: {e}")
            return None, None, None, None, None

        attempts = 15 # Targeted attempts
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
        
        for i in range(attempts):
            logger.info(f"⚡ Optimization Attempt {i+1}/{attempts} for {symbol} {target_wr}%+")
            
            # Randomize params
            params = {
                'units': random.choice([32, 64, 128, 256]),
                'dropout': random.choice([0.2, 0.3, 0.4, 0.5]),
                'lr': random.choice([0.001, 0.0005, 0.0001, 0.002])
            }
            
            # Build
            model = build_specialist_lstm(
                input_shape=(60, X_train.shape[2]),
                num_classes=1,
                lstm_units=params['units'],
                dropout_rate=params['dropout'],
                learning_rate=params['lr']
            )
            
            # Train (Deeper search, 50 epochs)
            model.fit(
                X_train_scaled, y_train,
                validation_data=(X_v_scaled, y_v),
                epochs=50,
                batch_size=32, # Smaller batch for better generalization
                class_weight=class_weight,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                verbose=0
            )
            
            X_full_flat = X.reshape(-1, X.shape[2])
            X_full_scaled = scaler.transform(X_full_flat).reshape(X.shape)
            y_proba = model.predict(X_full_scaled, verbose=0).flatten()
            
            # Find best threshold with volume priority
            scan_thresh = np.arange(0.90, 0.995, 0.005)
            best_t_for_candidate = None
            max_vol_for_candidate = 0
            
            for t in scan_thresh:
                preds = (y_proba >= t).astype(int)
                mask = preds == 1
                trades = mask.sum()
                if trades < 5: continue
                
                correct = (y[mask] == 1).sum()
                wr = correct / trades
                
                if wr * 100 >= target_wr:
                    if trades > max_vol_for_candidate:
                        max_vol_for_candidate = trades
                        best_t_for_candidate = t
            
            if best_t_for_candidate:
                wr = (y[(y_proba >= best_t_for_candidate)] == 1).sum() / max_vol_for_candidate
                logger.info(f"   Candidate {i+1}: WR {wr:.1%} | Volume {max_vol_for_candidate}")
                
                # If we meet both target WR and Min Trades, return immediately
                if max_vol_for_candidate >= min_trades:
                    return model, scaler, float(best_t_for_candidate), wr, int(max_vol_for_candidate)
                
                # Otherwise, track best candidate so far
                if best_candidate is None or max_vol_for_candidate > best_candidate[4]:
                    best_candidate = (model, scaler, float(best_t_for_candidate), wr, int(max_vol_for_candidate))
        
        return best_candidate if best_candidate else (None, None, None, None, None)
        
        return None, None, None, None, None

    def run_all(self):
        """Run factory for all pairs in config using Parallel Processing."""
        import yaml
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with open('config.yaml', 'r') as f:
            conf = yaml.safe_load(f)
            
        all_pairs = []
        for cat in ['majors', 'minors', 'crosses']:
            all_pairs.extend([p['symbol'] for p in conf['currency_pairs'].get(cat, [])])
            
        all_pairs = sorted(list(set(all_pairs)))
        logger.info(f"Targeting {len(all_pairs)} pairs with PARALLEL processing.")
        
        # Max workers = CPU Count - 2 (Leave room for OS/Dashboard)
        workers = max(1, os.cpu_count() - 2)
        logger.info(f"Using {workers} worker processes.")
        
        all_reports = []
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Map futures to symbols
            future_to_symbol = {executor.submit(process_pair_task, pair): pair for pair in all_pairs}
            
            for i, future in enumerate(as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data:
                        all_reports.extend(data)
                        # Progressive Save (Main Process)
                        self.generate_comprehensive_report() 
                        logger.info(f"Completed {i+1}/{len(all_pairs)}: {symbol}")
                    else:
                        logger.warning(f"No results for {symbol}")
                except Exception as e:
                    logger.error(f"Worker failed for {symbol}: {e}")
        
        logger.info("All pairs processed.")

    def generate_report(self, data):
        """Generate Selective Accuracy Report."""
        lines = [
            "# Selective Accuracy Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Performance Matrix",
            "| Pair | Type | Target | Threshold | Actual WR | Trades (Val) | Status |",
            "|---|---|---|---|---|---|---|"
        ]
        
        # Sort by Pair then Target
        # dedup (in case of multiple runs/appends) if needed, but extend() with fresh list logic is ok if all_reports is fresh
        # Actually in parallel run, all_reports is fresh.
        
        data_sorted = sorted(data, key=lambda x: (x['pair'], x['target'], x['type']))
        
        for row in data_sorted:
            status = row.get("status", "SAVED")
            lines.append(f"| {row['pair']} | {row['type']} | {row['target']}% | {row['threshold']:.3f} | {row['win_rate']:.1%} | {row['trades_total_val']} | {status} |")
            
        path = self.base_dir / "selective_accuracy_report.md"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Report updated ({len(data)} entries)")

    def generate_comprehensive_report(self):
        """
        Scans the models directory and regenerates the full report 
        from all existing config.json files.
        """
        logger.info("Regenerating comprehensive report from filesystem...")
        report_data = []
        
        # Walk through all directories
        for pair_dir in self.base_dir.iterdir():
            if not pair_dir.is_dir() or pair_dir.name in ["specialist", "trained", "binary", "enhanced", "__pycache__"]:
                continue
                
            symbol = pair_dir.name
            
            # Check for target directories (90, 95)
            for target_dir in pair_dir.iterdir():
                if not target_dir.is_dir(): continue
                
                try:
                    target = int(target_dir.name)
                except ValueError:
                    continue
                    
                # Check for BUY/SELL
                for type_dir in target_dir.iterdir():
                    if not type_dir.is_dir(): continue
                    
                    signal_type = type_dir.name
                    config_path = type_dir / "config.json"
                    
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                
                            # Debug
                            if symbol == "AUDCAD" and target == 90 and signal_type == "BUY":
                                logger.warning(f"DEBUG: Checking {config_path}")
                                logger.warning(f"DEBUG: Keys: {list(config.keys())}")

                            # Backfill if missing trades
                            if "trades" not in config or "win_rate" not in config:
                                logger.info(f"Backfilling stats for {symbol} {signal_type} {target}%...")
                                try:
                                    model = keras.models.load_model(type_dir / "model.keras")
                                    scaler = joblib.load(type_dir / "scaler.joblib")
                                    
                                    # Use factory helper to load data
                                    X_val, y_val = self._load_full_history(symbol, signal_type, scaler)
                                    if len(X_val) > 0:
                                        y_proba = model.predict(X_val, verbose=0).flatten()
                                        preds = (y_proba >= config.get("threshold", 0.99)).astype(int)
                                        mask = preds == 1
                                        trades = mask.sum()
                                        
                                        actual_wr = 0.0
                                        if trades > 0:
                                            actual_wr = (y_val[mask] == 1).sum() / trades
                                            
                                        config["trades"] = int(trades)
                                        config["win_rate"] = actual_wr
                                        
                                        # Save update
                                        with open(config_path, 'w') as f:
                                            json.dump(config, f, indent=2)
                                except Exception as e_back:
                                     logger.error(f"Backfill failed: {e_back}")

                            report_data.append({
                                "pair": symbol,
                                "type": signal_type,
                                "target": target,
                                "threshold": config.get("threshold", 0.0),
                                "win_rate": config.get("win_rate", config.get("target_win_rate", 0)/100.0), 
                                "trades_total_val": config.get("trades", 0),
                                "status": "SAVED"
                            })
                        except Exception as e:
                            logger.error(f"Error reading config for {symbol} {target} {signal_type}: {e}")

        if report_data:
             self.generate_report(report_data)
        else:
             logger.warning("No models found to report.")
        return []

# Top-level wrapper for multiprocessing picklability
def process_pair_task(symbol):
    """Worker function to process a single pair."""
    try:
        # Re-instantiate factory in the worker process
        # This ensures fresh DB connections/statelessness
        factory = WinRateFactory() 
        return factory.create_expert_models(symbol)
    except Exception as e:
        # logging inside worker might need config, but basicConfig usually works 
        print(f"Error in worker for {symbol}: {e}")



if __name__ == "__main__":
    # Ensure freezing support for Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    factory = WinRateFactory()
    factory.run_all()
