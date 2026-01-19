#!/usr/bin/env python3
"""
Charter ML Hyperparameter Tuning Script (using Optuna)

Automatically searches for optimal XGBoost hyperparameters using Bayesian optimization.
Uses chronological (walk-forward) validation to prevent look-ahead bias.

Usage:
    python tune_hyperparameters.py <training_data.csv> <output_model.onnx> [--trials N]

Example:
    python tune_hyperparameters.py training_data.csv charter_model.onnx --trials 100
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
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

try:
    import onnx
    from onnx import helper, TensorProto
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

    # Handle NaN/Inf - use median imputation instead of zeros
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    return X, y


def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective function to maximize validation AUC."""

    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'early_stopping_rounds': 20,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train model with early stopping
    model = XGBClassifier(**params)

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False,
    )

    # Evaluate
    y_prob = model.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, y_prob)

    return auc


def train_best_model(best_params: dict, X_train, y_train, X_val, y_val) -> tuple:
    """Train final model with best hyperparameters."""
    print("\n🚀 Training final model with best hyperparameters...")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Update params for final training
    final_params = best_params.copy()
    final_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'early_stopping_rounds': 20,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1,
    })

    model = XGBClassifier(**final_params)

    # Train with early stopping
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=True
    )

    # Report early stopping results
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
    print(f"\n   Best iteration: {best_iteration} (stopped early from {model.n_estimators} max trees)")

    # Evaluate
    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print(f"\n📊 Final Validation Results:")
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
        print("   ⚠️  onnxmltools not available, skipping ONNX export")
        # Save as XGBoost native format instead
        native_path = Path(output_path).with_suffix('.xgb')
        model.save_model(str(native_path))
        print(f"   ✓ Native XGBoost model saved to: {native_path}")
        save_scaler(scaler, str(native_path))
        return

    # Convert XGBoost to ONNX
    initial_type = [('features', FloatTensorType([None, input_dim]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=12)

    # Save the model
    onnx.save(onnx_model, output_path)
    print(f"   ✓ Model saved")

    # Save scaler parameters
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
    parser = argparse.ArgumentParser(description='Tune Charter ML model hyperparameters with Optuna')
    parser.add_argument('input', help='Path to training data CSV')
    parser.add_argument('output', help='Path for output model (.onnx)')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials (default: 50)')
    args = parser.parse_args()

    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost is required. Install with: pip install xgboost")
        sys.exit(1)

    if not SKLEARN_AVAILABLE:
        print("❌ sklearn is required. Install with: pip install scikit-learn")
        sys.exit(1)

    if not OPTUNA_AVAILABLE:
        print("❌ Optuna is required. Install with: pip install optuna")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║      Charter ML Hyperparameter Tuning (Optuna)              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Load data
    X, y = load_data(args.input)

    if len(X) == 0:
        print("❌ No valid training samples found!")
        sys.exit(1)

    # Walk-forward validation: Use chronological split (no random shuffle)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    print(f"\n📊 Dataset split (chronological/walk-forward):")
    print(f"   Training:   {len(X_train):,} samples (first 80%)")
    print(f"   Validation: {len(X_val):,} samples (last 20%)")
    print(f"   Train direction up: {y_train.mean():.1%}")
    print(f"   Val direction up:   {y_val.mean():.1%}")

    # Run Optuna optimization
    print(f"\n🔍 Starting hyperparameter search ({args.trials} trials)...")
    print("   This may take a while...\n")

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=args.trials,
        show_progress_bar=True
    )

    # Print results
    print("\n✨ Optimization complete!")
    print(f"\n📈 Best trial:")
    print(f"   Value (AUC): {study.best_trial.value:.4f}")
    print(f"   Params:")
    for key, value in study.best_trial.params.items():
        print(f"      {key}: {value}")

    # Train final model with best parameters
    best_params = study.best_trial.params
    model, scaler = train_best_model(best_params, X_train, y_train, X_val, y_val)

    # Export
    output_path = args.output if args.output.endswith('.onnx') else args.output + '.onnx'
    export_to_onnx(model, scaler, X.shape[1], output_path)

    # Save best hyperparameters to JSON
    params_path = Path(output_path).with_suffix('.params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"   ✓ Best hyperparameters saved to: {params_path}")

    print("\n✨ Training complete!")


if __name__ == '__main__':
    main()
