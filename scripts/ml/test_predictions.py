#!/usr/bin/env python3
"""Test that predictions are consistent."""
import json
import numpy as np
import pandas as pd
import xgboost as xgb

# Load test data
import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
df = pd.read_csv(os.path.join(base_dir, 'data/level_events.csv'))
X = df.drop('held', axis=1).values.astype(np.float32)

# Load model
model = xgb.XGBClassifier()
model.load_model(os.path.join(base_dir, 'data/level_model.json'))

# Load scaler
with open(os.path.join(base_dir, 'data/level_model.scaler.json'), 'r') as f:
    scaler_data = json.load(f)

mean = np.array(scaler_data['mean'])
scale = np.array(scaler_data['scale'])

# Scale and predict first 5 samples
X_scaled = (X[:5] - mean) / scale
probs = model.predict_proba(X_scaled)[:, 1]

print("First 5 predictions from Python:")
for i, (feat, prob) in enumerate(zip(X[:5], probs)):
    print(f"  Sample {i}: features[0:3]={feat[:3]}, prob={prob:.6f}")

print("\nRaw features for Rust comparison:")
for i in range(5):
    print(f"  {i}: {list(X[i])}")
