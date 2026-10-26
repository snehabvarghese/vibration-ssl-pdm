"""
Evaluation and stress-testing utility for Motor Current & Temperature models.

Allows testing model accuracy across:
  1. Multiple random seeds / bearing splits (mean +/- std)
  2. Varying sensor noise levels (stress testing)
  3. Operating temperature sweeps (thermal sensitivity & anomaly triggering)
  4. Classifier comparison (Random Forest vs Gradient Boosting vs SVM)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from core.motor_temperature import analyse_motor_signal
from models.motor_temperature_model import MotorPdMModel
from tests.fixtures.paderborn_synthetic import generate_synthetic_motor_current
from training.train_motor_models import prepare_dataset, split_by_bearing


def evaluate_across_seeds(
    n_seeds: int = 5,
    clf_type: str = "rf",
) -> pd.DataFrame:
    """Evaluate accuracy and Macro-F1 variability across multiple random bearing splits."""
    print(f"\n=======================================================")
    print(f" 1. MULTI-SEED REPEATABILITY EVALUATION ({n_seeds} RUNS)")
    print(f"=======================================================")

    results = []
    base_seed = 42

    for i in range(n_seeds):
        seed = base_seed + i * 13
        X, y, bearing_ids, record_ids, feature_names = prepare_dataset()
        train_mask, test_mask = split_by_bearing(
            bearing_ids,
            record_ids=record_ids,
            labels=y,
            test_ratio=0.33,
            random_seed=seed,
        )

        for mode in ["current_only", "combined"]:
            model = MotorPdMModel(mode=mode, clf_type=clf_type, random_state=seed)
            model.fit(X[train_mask], y[train_mask], feature_names)
            metrics = model.evaluate(X[test_mask], y[test_mask], feature_names)

            results.append({
                "Seed": seed,
                "Model": mode,
                "Accuracy": metrics.accuracy,
                "Macro_F1": metrics.macro_f1,
                "Anomaly_AUC": metrics.anomaly_roc_auc or 0.0,
            })

    df = pd.DataFrame(results)
    summary = df.groupby("Model").agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_Std=("Accuracy", "std"),
        F1_Mean=("Macro_F1", "mean"),
        F1_Std=("Macro_F1", "std"),
        AUC_Mean=("Anomaly_AUC", "mean"),
        AUC_Std=("Anomaly_AUC", "std"),
    ).reset_index()

    print(summary.to_string(index=False))
    return summary


def evaluate_noise_robustness(
    noise_sigmas: List[float] = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20],
) -> pd.DataFrame:
    """Stress-test accuracy under increasing electrical / sensor noise."""
    print(f"\n=======================================================")
    print(f" 2. NOISE ROBUSTNESS STRESS TEST")
    print(f"=======================================================")

    conditions = ["Normal", "Outer_Race", "Inner_Race"]
    rows = []

    for sigma in noise_sigmas:
        correct_preds = 0
        total_preds = 0
        correct_status = 0

        for cond in conditions:
            for rep in range(15):
                # Generate signal with specific noise level
                i1, i2, temp_val = generate_synthetic_motor_current(
                    cond, duration_seconds=1.0, seed=1000 + rep
                )
                # Add extra Gaussian noise
                rng = np.random.default_rng(2000 + rep)
                i1 += rng.normal(0, sigma, size=len(i1))
                i2 += rng.normal(0, sigma, size=len(i2))

                out = analyse_motor_signal(
                    i1, i2, temp_val, sampling_rate=64000, model_mode="combined"
                )

                if out["predicted_condition"] == cond:
                    correct_preds += 1
                expected_status = "HEALTHY" if cond == "Normal" else "ANOMALOUS"
                if out["status"] == expected_status:
                    correct_status += 1
                total_preds += 1

        acc = correct_preds / total_preds
        status_acc = correct_status / total_preds
        rows.append({
            "Noise_Sigma": sigma,
            "Classification_Accuracy": round(acc, 4),
            "Health_Status_Accuracy": round(status_acc, 4),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def evaluate_temperature_sweep(
    temperatures: List[float] = [20.0, 35.0, 45.0, 50.0, 65.0, 80.0, 95.0],
) -> pd.DataFrame:
    """Evaluate how varying ambient / operating temperatures affect anomaly detection."""
    print(f"\n=======================================================")
    print(f" 3. TEMPERATURE SENSITIVITY SWEEP")
    print(f"=======================================================")

    rows = []
    # Test on Normal motor current across different temperature values
    i1_norm, i2_norm, _ = generate_synthetic_motor_current("Normal", duration_seconds=1.0, seed=42)

    for temp in temperatures:
        out = analyse_motor_signal(
            i1_norm, i2_norm, temperature=temp, sampling_rate=64000, model_mode="combined"
        )
        rows.append({
            "Temperature_Celsius": temp,
            "Predicted_Condition": out["predicted_condition"],
            "Anomaly_Score": out["anomaly_score"],
            "Status": out["status"],
            "Confidence": out["confidence"],
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Motor Current & Temperature accuracy across different values")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds for repeatability")
    parser.add_argument("--noise-test", action="store_true", help="Run noise robustness test")
    parser.add_argument("--temp-sweep", action="store_true", help="Run temperature sensitivity sweep")
    parser.add_argument("--all", action="store_true", help="Run all evaluation sweeps")
    args = parser.parse_args()

    run_all = args.all or (not args.noise_test and not args.temp_sweep)

    if run_all or args.seeds:
        evaluate_across_seeds(n_seeds=args.seeds)
    if run_all or args.noise_test:
        evaluate_noise_robustness()
    if run_all or args.temp_sweep:
        evaluate_temperature_sweep()
