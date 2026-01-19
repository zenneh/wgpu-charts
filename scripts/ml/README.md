# Charter ML Training Scripts

Machine learning scripts for training cryptocurrency price direction prediction models.

## Overview

This directory contains two training scripts:

1. **`train_model.py`** - Standard training with default or custom hyperparameters
2. **`tune_hyperparameters.py`** - Automated hyperparameter tuning using Optuna

## Prerequisites

```bash
pip install xgboost scikit-learn onnx onnxmltools optuna
```

## Workflow

### Step 1: Generate Training Data

First, export training data from the Rust codebase:

```bash
cargo run --bin ml_export -- data/btcusdt_1m.csv training_data.csv --max-candles 200000
```

This generates a CSV with:
- **302 features**: 4 timeframes × 74 features + 6 global features (including RSI-14)
- **Labels**: direction_up, price_change_pct, level_broke

### Step 2: Choose Your Training Method

#### Option A: Quick Training (Default Hyperparameters)

Fast training with reasonable defaults - good for experimentation:

```bash
python scripts/ml/train_model.py training_data.csv charter_model.onnx
```

**Default hyperparameters:**
- n_estimators: 200
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- min_child_weight: 1
- gamma: 0
- reg_alpha: 0.1
- reg_lambda: 1.0

**Features:**
- Walk-forward validation (chronological 80/20 split)
- Early stopping (20 rounds)
- StandardScaler normalization
- Exports to ONNX for Rust inference

#### Option B: Optimized Training (Recommended for Production)

**One-step training with automatic hyperparameter optimization:**

```bash
python scripts/ml/tune_hyperparameters.py training_data.csv charter_model.onnx --trials 100
```

**What it does:**
1. **Search Phase** (90% of time): Runs 100 trials of Bayesian optimization to find best hyperparameters
2. **Training Phase** (10% of time): Automatically trains final model with best parameters
3. **Export**: Saves trained model + scaler + best hyperparameters

**Search space:**
- n_estimators: 100-500
- max_depth: 3-10
- learning_rate: 0.01-0.3 (log scale)
- subsample: 0.6-1.0
- colsample_bytree: 0.6-1.0
- min_child_weight: 1-10
- gamma: 0-5
- reg_alpha: 0-2
- reg_lambda: 0-2

**Output:**
- `charter_model.onnx` - Trained model ready to use
- `charter_model.onnx.scaler.json` - Feature normalization parameters
- `charter_model.params.json` - Best hyperparameters (for future reuse)

**Typical runtime:** 30-60 minutes for 100 trials

**Note:** No retraining needed! The tuning script outputs a production-ready model.

### When to Use `train_model.py --params`

The `--params` flag is useful for specific scenarios:

```bash
python scripts/ml/train_model.py new_data.csv model_v2.onnx --params best_params.json
```

**Use cases:**
- **Retraining on new data** with proven hyperparameters (e.g., weekly model updates)
- **Experimenting** with specific hyperparameter combinations without full tuning
- **CI/CD pipelines** where tuning happens once, then params are versioned and reused

## Output Files

Both training scripts produce:
- `charter_model.onnx` - Trained ONNX model ready for inference
- `charter_model.onnx.scaler.json` - StandardScaler parameters (mean/scale)

Additionally, `tune_hyperparameters.py` creates:
- `charter_model.params.json` - Best hyperparameters (for future reuse)

**Important:** The `.onnx` file from `tune_hyperparameters.py` is already trained and ready to use - no additional training step required!

## Model Architecture

**Type:** XGBoost Gradient Boosting Classifier

**Task:** Binary classification (price goes UP or DOWN)

**Features (302 total):**

### Global Features (6):
1. current_volume_normalized - Volume relative to 100-candle average
2. price_change_normalized - Price change from previous candle
3. body_ratio - Candle body size / total range
4. is_bullish - Binary (1.0 if close > open)
5. num_timeframes - Number of timeframes (always 4)
6. **rsi_14** - RSI-14 normalized to 0-1 range *(NEW)*

### Per-Timeframe Features (74 × 4 timeframes = 296):

**Timeframes:** 5m, 1h, 1d, 1w

For each timeframe:
- **60 level features**: 4 categories × 3 levels × 5 features
  - Bullish/Bearish Hold levels (support/resistance)
  - Bullish/Bearish Greedy levels
  - Per level: distance, hit_count, respected_hits, is_active, age
- **12 trend features**: 2 trends × 6 features
  - Latest bullish/bearish trendlines
  - Per trend: slope, distance, hit_count, is_active, age, exists
- **2 aggregate features**: active_level_count, active_trend_count

## Validation Strategy

**Walk-Forward Validation:**
- Chronological 80/20 split (first 80% train, last 20% validation)
- Prevents look-ahead bias (no training on future data)
- Realistic evaluation for time-series predictions

**Early Stopping:**
- Monitors validation AUC-ROC
- Stops if no improvement for 20 consecutive rounds
- Prevents overfitting

## Metrics

Models are evaluated on:
- **Accuracy**: Overall correctness
- **AUC-ROC**: Area under ROC curve (primary metric for optimization)
- **Precision/Recall/F1**: Per-class performance
- **Feature Importance**: Top 10 most influential features

## Improvements Implemented

✅ **Early Stopping** - Automatic overfitting prevention
✅ **RSI Feature** - Momentum indicator (14-period RSI)
✅ **Walk-Forward Validation** - Chronological split (no look-ahead bias)
✅ **Hyperparameter Tuning** - Automated optimization with Optuna

## Usage Examples

### Quick Start (Default Params)
```bash
# 1. Export data
cargo run --bin ml_export -- data/btcusdt_1m.csv training.csv

# 2. Train model with defaults
python scripts/ml/train_model.py training.csv model.onnx

# 3. Deploy
cp model.onnx data/charter_model.onnx
cp model.onnx.scaler.json data/charter_model.onnx.scaler.json
```

### Production Workflow (Optimized - Recommended)
```bash
# 1. Export data
cargo run --bin ml_export -- data/btcusdt_1m.csv training.csv --max-candles 300000

# 2. One-step optimization + training (takes ~1 hour)
python scripts/ml/tune_hyperparameters.py training.csv model.onnx --trials 100

# 3. Deploy (model is already trained!)
cp model.onnx data/charter_model.onnx
cp model.onnx.scaler.json data/charter_model.onnx.scaler.json
cp model.params.json data/charter_model.params.json
```

### Retraining with Saved Hyperparameters
```bash
# Week 1: Initial tuning
python scripts/ml/tune_hyperparameters.py week1_data.csv model.onnx --trials 100
# Saves: model.onnx + model.params.json

# Week 2: New data arrives, reuse proven hyperparameters
cargo run --bin ml_export -- data/btcusdt_1m.csv week2_data.csv
python scripts/ml/train_model.py week2_data.csv model_v2.onnx --params model.params.json

# Deploy updated model
cp model_v2.onnx data/charter_model.onnx
```

## Troubleshooting

**Import errors:**
```bash
pip install --upgrade xgboost scikit-learn onnx onnxmltools optuna
```

**ONNX export fails:**
- Fallback: Model saves as `.xgb` format (native XGBoost)
- Install: `pip install onnxmltools`

**Feature count mismatch:**
- Ensure ml_export was run with latest code (302 features)
- Regenerate training data if feature schema changed

**Low AUC-ROC (<0.55):**
- Increase training data size (--max-candles)
- Run hyperparameter tuning
- Check for class imbalance in labels
- Verify temporal consistency of features

## Next Steps

Potential future improvements:
- Multi-target prediction (direction + magnitude)
- LSTM/GRU for temporal sequences
- Additional features (MACD, Bollinger Bands, ATR)
- Ensemble methods (combine multiple models)
- Online learning (incremental model updates)
