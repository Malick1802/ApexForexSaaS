import joblib
import numpy as np

scaler_path = r'c:\Users\artem\Downloads\ApexForexSaaS\models\foundation\scaler.joblib'
scaler = joblib.load(scaler_path)

print(f"Features in: {scaler.n_features_in_}")
if hasattr(scaler, 'feature_names_in_'):
    print(f"Feature names: {list(scaler.feature_names_in_)}")
else:
    print("No feature names found in scaler.")

print(f"Mean shape: {scaler.mean_.shape}")
print(f"Scale shape: {scaler.scale_.shape}")
