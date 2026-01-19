#!/usr/bin/env python3
"""
Charter ML Model Training Script (XGBoost)

Trains an XGBoost classifier for price direction prediction (up/down).
Exports the trained model to ONNX format for Rust inference.

Usage:
    python train_model.py <training_data.csv> <output_model.onnx>

Example:
    python train_model.py training_data.csv charter_model.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ML libraries
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available. Install with: pip install onnx")

try:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    ONNXMLTOOLS_AVAILABLE = True
except ImportError:
    ONNXMLTOOLS_AVAILABLE = False
    print("Warning: onnxmltools not available. Install with: pip install onnxmltools")


def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load training data from CSV file."""
    print(f"📂 Loading data from {path}...")
    df = pd.read_csv(path)

    # Features are all columns starting with 'f'
    feature_cols = [c for c in df.columns if c.startswith('f')]

    X = df[feature_cols].values.astype(np.float32)
    y = df['direction_up'].values.astype(np.int32)

    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


def train_xgboost(X_train, y_train, X_val, y_val) -> tuple:
    """Train XGBoost classifier."""
    print("\n🚀 Training XGBoost model...")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # XGBoost parameters optimized for binary classification
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    # Train with early stopping
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=True
    )

    # Evaluate
    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print(f"\n📊 Validation Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC-ROC:  {auc:.4f}")
    print(f"\n{classification_report(y_val, y_pred, target_names=['DOWN', 'UP'])}")

    # Feature importance (top 10)
    importance = model.feature_importances_
    top_features = np.argsort(importance)[-10:][::-1]
    print("\n🔍 Top 10 Important Features:")
    for i, idx in enumerate(top_features):
        print(f"   {i+1}. f{idx}: {importance[idx]:.4f}")

    return model, scaler


def export_to_onnx(model: XGBClassifier, scaler: StandardScaler, input_dim: int, output_path: str):
    """Export XGBoost model to ONNX format."""
    print(f"\n💾 Exporting model to ONNX: {output_path}")

    if not ONNXMLTOOLS_AVAILABLE:
        print("   ⚠️  onnxmltools not available, using manual export...")
        export_manual_onnx(model, scaler, input_dim, output_path)
        return

    # Convert XGBoost to ONNX
    initial_type = [('features', FloatTensorType([None, input_dim]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=12)

    # The converted model outputs class labels and probabilities
    # We need to modify to output just the probability of class 1

    # Save the model
    onnx.save(onnx_model, output_path)
    print(f"   ✓ Model saved")

    # Save scaler parameters
    save_scaler(scaler, output_path)


def export_manual_onnx(model: XGBClassifier, scaler: StandardScaler, input_dim: int, output_path: str):
    """Manual ONNX export by building a simple approximation network."""
    # For XGBoost, we'll extract the predictions and create a lookup-style model
    # This is a fallback when onnxmltools isn't available

    if not ONNX_AVAILABLE:
        print("   ❌ Cannot export: ONNX not available")
        return

    # Get the booster and tree structure
    booster = model.get_booster()
    trees = booster.get_dump(dump_format='json')

    print(f"   Model has {len(trees)} trees")

    # For simplicity, we'll create a model that uses the sklearn-style coefficients
    # This won't be as accurate as full tree traversal but works as fallback

    # Actually, let's just save the model in a format we can load directly
    # and do inference in Python, then call from Rust via a simpler interface

    # Save as XGBoost native format
    native_path = Path(output_path).with_suffix('.xgb')
    model.save_model(str(native_path))
    print(f"   ✓ Native XGBoost model saved to: {native_path}")

    # For ONNX, create a simple passthrough that we'll replace later
    # when onnxmltools is available
    nodes = []
    initializers = []

    # Create a simple identity + sigmoid as placeholder
    nodes.append(helper.make_node('ReduceMean', ['features'], ['mean_out'], axes=[1], keepdims=True))
    nodes.append(helper.make_node('Sigmoid', ['mean_out'], ['predictions']))

    input_tensor = helper.make_tensor_value_info('features', TensorProto.FLOAT, [None, input_dim])
    output_tensor = helper.make_tensor_value_info('predictions', TensorProto.FLOAT, [None, 1])

    graph = helper.make_graph(nodes, 'xgboost_placeholder', [input_tensor], [output_tensor], initializers)
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    onnx_model.ir_version = 7

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_path)
    print(f"   ⚠️  Placeholder ONNX saved (install onnxmltools for full export)")

    save_scaler(scaler, output_path)


def save_scaler(scaler: StandardScaler, model_path: str):
    """Save scaler parameters to JSON."""
    scaler_path = Path(model_path).with_suffix('.scaler.json')
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
    }
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f)
    print(f"   ✓ Scaler saved to: {scaler_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Charter ML model with XGBoost')
    parser.add_argument('input', help='Path to training data CSV')
    parser.add_argument('output', help='Path for output model (.onnx)')
    args = parser.parse_args()

    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost is required. Install with: pip install xgboost")
        sys.exit(1)

    if not SKLEARN_AVAILABLE:
        print("❌ sklearn is required. Install with: pip install scikit-learn")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         Charter ML Training (XGBoost)                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Load data
    X, y = load_data(args.input)

    if len(X) == 0:
        print("❌ No valid training samples found!")
        sys.exit(1)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Dataset split:")
    print(f"   Training:   {len(X_train):,} samples")
    print(f"   Validation: {len(X_val):,} samples")
    print(f"   Direction up: {y_train.mean():.1%}")

    # Train
    model, scaler = train_xgboost(X_train, y_train, X_val, y_val)

    # Export
    output_path = args.output if args.output.endswith('.onnx') else args.output + '.onnx'
    export_to_onnx(model, scaler, X.shape[1], output_path)

    print("\n✨ Training complete!")


if __name__ == '__main__':
    main()
