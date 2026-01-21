#!/usr/bin/env python3
"""
Export XGBoost model to text dump format for use with gbdt Rust crate.

Usage: python export_xgboost_dump.py <model_json> <output_dump>
"""

import sys
import xgboost as xgb

def main():
    if len(sys.argv) < 3:
        print("Usage: python export_xgboost_dump.py <model_json> <output_dump>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    print(f"Exporting to dump format: {output_path}...")
    # Get the booster and dump to text format
    booster = model.get_booster()
    booster.dump_model(output_path, dump_format='text')

    print("Done!")
    print(f"\nTo use in Rust backtest:")
    print(f"  --model {output_path}")

if __name__ == '__main__':
    main()
