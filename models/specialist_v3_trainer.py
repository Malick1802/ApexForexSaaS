"""
Specialist v3 Factory
=====================
Inherits Macro intelligence from Global Brain v3 and 
fine-tunes for specific pair performance.
"""

import os, gc, sys, json, logging, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.mt5_connector import get_mt5
from models.foundation_trainer_v3 import (
    fetch_mt5_pair, fetch_yf_macro, build_base_features, 
    add_global_context_v3, rolling_zscore, triple_barrier_label,
    ForexDataGenerator
)
from models.global_brain import GatedResidualNetwork, VariableSelectionNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SpecialistV3")

class SpecialistV3Trainer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.foundation_dir = PROJECT_ROOT / "models" / "foundation_v3"
        self.output_dir = PROJECT_ROOT / "models" / "specialist" / symbol
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Config
        with open(self.foundation_dir / "config.json", 'r') as f:
            self.f_config = json.load(f)
            
        self.seq_len = self.f_config.get("seq_len", 48)
        self.units = self.f_config.get("units", 64)
        self.n_features = self.f_config.get("n_features")

    def run(self, epochs=15):
        logger.info(f"--- STARTING SPECIALIST ADAPTATION: {self.symbol} ---")
        
        # 1. Load Foundation Brain
        custom_objs = {
            'GatedResidualNetwork': GatedResidualNetwork,
            'VariableSelectionNetwork': VariableSelectionNetwork
        }
        logger.info("Loading Foundation Brain...")
        brain = keras.models.load_model(str(self.foundation_dir / "foundation_brain.keras"), custom_objects=custom_objs)
        
        # 2. Freeze Global Knowledge
        # We freeze the VSN and the LSTM layers to keep the macro intelligence intact.
        # We only train the Gated Residual Network and the Output layer.
        for layer in brain.layers:
            if any(x in layer.name for x in ['variable_selection', 'lstm', 'attention']):
                layer.trainable = False
                logger.debug(f"  Freezing: {layer.name}")
            else:
                layer.trainable = True
                logger.debug(f"  Thawing: {layer.name}")
        
        # 3. Fetch Pair-Specific Data
        mt5 = get_mt5()
        raw = {self.symbol: fetch_mt5_pair(mt5, self.symbol, 1825)} # 5 years
        
        macro_tickers = {
            "SP500": "^GSPC", "OIL": "CL=F", "NASDAQ": "^IXIC",
            "TNX": "^TNX", "IRX": "^IRX", "VIX": "^VIX",
            "DXY": "DX-Y.NYB", "COPPER": "HG=F", "BTC": "BTC-USD"
        }
        for k, v in macro_tickers.items():
            df = fetch_yf_macro(k, v, 1825 + 60)
            if not df.empty: raw[k] = df
            
        # 4. Process
        common = None
        for df in raw.values():
            common = df.index if common is None else common.intersection(df.index)
        aligned = {s: df.reindex(common).ffill().bfill() for s, df in raw.items()}
        
        feat = build_base_features(aligned[self.symbol])
        feat = add_global_context_v3(feat, aligned, feat.index)
        for col in feat.columns:
            feat[col] = rolling_zscore(feat[col])
        feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
        
        labels = triple_barrier_label(aligned[self.symbol].reindex(feat.index))
        
        features_dict = {self.symbol: feat.values.astype(np.float32)}
        labels_dict = {self.symbol: labels.values.astype(np.int32)}
        
        # 5. Generators
        train_gen = ForexDataGenerator(features_dict, labels_dict, [self.symbol], 'train', stride=2)
        val_gen   = ForexDataGenerator(features_dict, labels_dict, [self.symbol], 'val', stride=2)
        
        # 6. Fine-Tune
        brain.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-5), # Very low LR for fine-tuning
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Fine-tuning Specialist head for {self.symbol}...")
        brain.fit(
            train_gen, 
            validation_data=val_gen, 
            epochs=epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
                keras.callbacks.ModelCheckpoint(str(self.output_dir / "specialist_brain.keras"), save_best_only=True)
            ]
        )
        
        # 7. Evaluate and Save Specialist Package
        val_loss, val_acc = brain.evaluate(val_gen, verbose=0)
        
        config = {
            "symbol": self.symbol,
            "parent_foundation": "v3",
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "fine_tuned_at": datetime.now().isoformat(),
            "seq_len": self.seq_len,
            "features": self.n_features
        }
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"✅ Specialist Adaptation Complete: {self.symbol}")
        
        del brain, features_dict, labels_dict, aligned, raw
        keras.backend.clear_session()
        gc.collect()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python specialist_v3_trainer.py <SYMBOL>")
        sys.exit(1)
    
    symbol = sys.argv[1]
    SpecialistV3Trainer(symbol).run()
