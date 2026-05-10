import sys
import numpy as np
from pathlib import Path
from tensorflow import keras
import joblib

sys.path.insert(0, str(Path.cwd()))
from data_pipeline.engine import DataEngine
from data_pipeline.features import FeatureEngineer
from data_pipeline.global_features import GlobalFeatureEngineer
from data_pipeline.labeling import triple_barrier_label
from models.global_brain import VariableSelectionNetwork, GatedResidualNetwork
from scripts.wfa_expert_14d import align_features, FEATURE_COLS
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

symbol = "EURUSD"
engine = DataEngine()
df = engine.fetch(symbol, interval="1h", days=30)
df_labeled = triple_barrier_label(df, symbol=symbol)

fe = FeatureEngineer()
gfe = GlobalFeatureEngineer()

base_f = fe.extract_features(df_labeled)

global_data = {}
for g in ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "GOLD", "^TNX"]:
    try:
        gdf = engine.fetch(g, interval="1h", days=30)
        if gdf is not None: global_data[g] = gdf
    except Exception:
        pass

features = gfe.add_global_features(symbol, base_f, global_data)
features = align_features(features, df)

y = df_labeled['label'].astype(int)
common_idx = features.index.intersection(y.index)
X, _ = fe.create_sequences(features.loc[common_idx], y.loc[common_idx], sequence_length=60)

scaler = joblib.load("models/foundation/scaler.joblib")
X_flat = X.reshape(-1, X.shape[2])
X_scaled = scaler.transform(X_flat).reshape(len(X), 60, -1)

model = keras.models.load_model(
    f"models/expert/{symbol}/expert_model.keras",
    custom_objects={'VariableSelectionNetwork': VariableSelectionNetwork, 'GatedResidualNetwork': GatedResidualNetwork}
)

preds = model.predict(X_scaled, verbose=0)
pred_classes = np.argmax(preds, axis=1)

print("\n--- EURUSD RAW PREDICTIONS ---")
print("Total predictions:", len(pred_classes))
print("HOLD (0):", np.sum(pred_classes == 0))
print("BUY  (1):", np.sum(pred_classes == 1))
print("SELL (2):", np.sum(pred_classes == 2))

buy_conf = preds[:, 1]
print(f"Max BUY confidence: {np.max(buy_conf):.2f}")
print(f"Mean BUY confidence: {np.mean(buy_conf):.2f}")
