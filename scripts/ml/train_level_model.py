#!/usr/bin/env python3
"""
Train a model to predict whether a level will hold or break.

Usage: python train_level_model.py <input_csv> <output_onnx>
Example: python train_level_model.py data/level_events.csv data/level_model.onnx
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb

def main():
    if len(sys.argv) < 3:
        print("Usage: python train_level_model.py <input_csv> <output_onnx>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print("=" * 60)
    print("     Level Hold/Break Prediction Model Training")
    print("=" * 60)
    print()

    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} level approach events")

    # Separate features and labels
    X = df.drop('held', axis=1).values.astype(np.float32)
    y = df['held'].values.astype(np.int32)

    print(f"  Features: {X.shape[1]}")
    print(f"  Held: {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  Broke: {len(y) - y.sum()} ({100*(1-y.mean()):.1f}%)")
    print()

    # Chronological split (no shuffle for time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"  Train held ratio: {100*y_train.mean():.1f}%")
    print(f"  Val held ratio: {100*y_val.mean():.1f}%")
    print()

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Handle class imbalance by computing scale_pos_weight
    # scale_pos_weight = count(negative) / count(positive)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Class imbalance: scale_pos_weight = {scale_weight:.2f}")
    print()

    # Train XGBoost with class balancing
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_weight,  # Handle class imbalance
        objective='binary:logistic',
        eval_metric='auc',
        early_stopping_rounds=30,
        random_state=42,
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=True
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("                    RESULTS")
    print("=" * 60)

    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]

    accuracy = (y_pred == y_val).mean()
    auc = roc_auc_score(y_val, y_prob)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    print()

    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['BROKE', 'HELD']))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(f"  Predicted BROKE  HELD")
    print(f"  Actual")
    print(f"  BROKE    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"  HELD     {cm[1,0]:5d}  {cm[1,1]:5d}")
    print()

    # Feature importance
    print("Top 10 Important Features:")
    feature_names = df.drop('held', axis=1).columns.tolist()
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
    print()

    # Export to ONNX
    print(f"Exporting model to {output_path}...")
    onnx_exported = False
    try:
        # Try XGBoost's built-in ONNX export first (available in newer versions)
        import onnx
        model.save_model(output_path.replace('.onnx', '_xgb.json'))

        # Use onnxmltools with explicit target_opset
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType

        initial_type = [('float_input', OnnxFloatTensorType([None, X.shape[1]]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=12)

        # Save the ONNX model
        onnx.save_model(onnx_model, output_path)
        print("  ONNX model saved")
        onnx_exported = True
    except Exception as e:
        print(f"  ONNX export failed: {e}")

    # Always save XGBoost native format as backup
    json_path = output_path.replace('.onnx', '.json')
    model.save_model(json_path)
    print(f"  XGBoost JSON model saved to {json_path}")

    if not onnx_exported:
        print("  Note: Use the .json model with xgboost library for inference")

    # Save scaler
    scaler_path = output_path.replace('.onnx', '.scaler.json')
    scaler_data = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
    }
    with open(scaler_path, 'w') as f:
        json.dump(scaler_data, f)
    print(f"  Scaler saved to {scaler_path}")

    print("\nTraining complete!")

    # Print prediction distribution for validation set
    print("\nPrediction Distribution (validation set):")
    print(f"  Min: {y_prob.min():.3f}")
    print(f"  Max: {y_prob.max():.3f}")
    print(f"  Mean: {y_prob.mean():.3f}")
    print(f"  Median: {np.median(y_prob):.3f}")
    print(f"  > 0.6: {(y_prob > 0.6).sum()} ({100*(y_prob > 0.6).mean():.1f}%)")
    print(f"  > 0.7: {(y_prob > 0.7).sum()} ({100*(y_prob > 0.7).mean():.1f}%)")
    print(f"  < 0.4: {(y_prob < 0.4).sum()} ({100*(y_prob < 0.4).mean():.1f}%)")
    print(f"  < 0.3: {(y_prob < 0.3).sum()} ({100*(y_prob < 0.3).mean():.1f}%)")

    # Show precision at different confidence thresholds
    print("\nPrecision at different confidence thresholds:")
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        high_conf = y_prob >= threshold
        if high_conf.sum() > 0:
            precision = y_val[high_conf].mean()
            print(f"  >= {threshold:.2f}: {high_conf.sum():4d} predictions, {100*precision:.1f}% actually held")
        else:
            print(f"  >= {threshold:.2f}: No predictions")

    # Same for "will break" (low confidence)
    print("\nPrecision for 'will break' predictions:")
    for threshold in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25]:
        low_conf = y_prob <= threshold
        if low_conf.sum() > 0:
            break_precision = 1 - y_val[low_conf].mean()
            print(f"  <= {threshold:.2f}: {low_conf.sum():4d} predictions, {100*break_precision:.1f}% actually broke")

if __name__ == '__main__':
    main()
