"""
Training and Evaluation Pipeline for Motor Current & Temperature Models.

Conducts leak-free, bearing-level split training for:
  1. Current-Only Baseline
  2. Temperature-Only Baseline
  3. Combined Current-Plus-Temperature Model

Saves versioned checkpoints to checkpoints/motor_temp/ and machine-readable
metrics to results/motor_temp/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd

from models.motor_temperature_model import MotorPdMModel
from preprocessing.motor_features import extract_multimodal_features
from preprocessing.paderborn_loader import load_paderborn_mat, segment_record
from tests.fixtures.paderborn_synthetic import (
    create_synthetic_dataset_manifest,
    create_synthetic_paderborn_mat,
)

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints" / "motor_temp"
RESULTS_DIR = ROOT_DIR / "results" / "motor_temp"


def prepare_dataset(
    data_dir: Optional[Path] = None,
    use_synthetic_if_empty: bool = True,
    window_seconds: float = 0.5,
    max_files_per_bearing: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load and extract features for Paderborn records with bearing-level grouping.

    Returns
    -------
    X : np.ndarray
        Feature matrix (N, D).
    y : np.ndarray
        Condition labels (N,).
    bearing_ids : np.ndarray
        Bearing identifier for each segment (N,).
    feature_names : list of str
        Names of all extracted features.
    """
    records = []
    
    # 1. Search for real Paderborn .mat files if data_dir is provided or exists
    mat_files: List[Path] = []
    if data_dir is not None and data_dir.exists():
        mat_files = sorted(list(data_dir.rglob("*.mat")))
    elif (ROOT_DIR / "data" / "paderborn").exists():
        mat_files = sorted(list((ROOT_DIR / "data" / "paderborn").rglob("*.mat")))

    is_synthetic = False
    if mat_files:
        print(f"Found {len(mat_files)} real Paderborn .mat files on disk.")
        if max_files_per_bearing is not None:
            from collections import defaultdict
            b_groups = defaultdict(list)
            for p in mat_files:
                parts = p.stem.split("_")
                b_id = parts[3] if len(parts) >= 4 else "unknown"
                b_groups[b_id].append(p)
            selected = []
            for b_id, files in b_groups.items():
                selected.extend(files[:max_files_per_bearing])
            mat_files = selected
            print(f"Subsampled to {len(mat_files)} files ({max_files_per_bearing} per bearing category).")

        for p in mat_files:
            try:
                rec = load_paderborn_mat(p)
                records.append(rec)
            except Exception as e:
                print(f"Skipping unreadable file {p.name}: {e}")
    elif use_synthetic_if_empty:
        is_synthetic = True
        print("Note: Real multi-GB Paderborn dataset not found on local disk.")
        print("Loading synthetic Paderborn benchmark fixture for software & model evaluation...")
        manifest = create_synthetic_dataset_manifest()
        for item in manifest:
            buf = create_synthetic_paderborn_mat(
                filename=item["filename"],
                condition=item["condition"],
                duration_seconds=2.0,
                seed=item["seed"],
            )
            rec = load_paderborn_mat(buf, filename_hint=item["filename"])
            records.append(rec)

    if not records:
        raise RuntimeError("No records available to train motor current & temperature models.")

    # 2. Segment records and extract features
    all_features: List[np.ndarray] = []
    all_labels: List[str] = []
    all_bearing_ids: List[str] = []
    feature_names: List[str] = []

    all_record_ids: List[str] = []

    for rec in records:
        segs = segment_record(rec, window_seconds=window_seconds, overlap=0.5)
        for s in segs:
            vec, names, _ = extract_multimodal_features(
                current_phase1=s["phase_current_1"],
                current_phase2=s["phase_current_2"],
                temperature=s["temperature_celsius"],
                sampling_rate=s["sampling_rate"],
            )
            if not feature_names:
                feature_names = names
            all_features.append(vec)
            all_labels.append(s["condition_label"])
            all_bearing_ids.append(s["bearing_id"])
            all_record_ids.append(s["record_filename"])

    X = np.array(all_features, dtype=np.float64)
    y = np.array(all_labels, dtype=object)
    bearing_ids = np.array(all_bearing_ids, dtype=object)
    record_ids = np.array(all_record_ids, dtype=object)

    print(f"Dataset prepared: {len(X)} segments from {len(records)} recordings.")
    print(f"Data source: {'SYNTHETIC_BENCHMARK_FIXTURE' if is_synthetic else 'REAL_PADERBORN_FILES'}")
    print(f"Classes: {dict(pd.Series(y).value_counts())}")
    return X, y, bearing_ids, record_ids, feature_names


def split_by_bearing(
    bearing_ids: np.ndarray,
    record_ids: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    test_ratio: float = 0.33,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform leak-free split.

    If each class has multiple bearings, splits at the physical bearing level.
    If each class has only 1 bearing (e.g. subset), splits at the recording run level
    so that every class is represented in train/test with zero cross-recording window leakage.
    """
    rng = np.random.default_rng(random_seed)
    unique_bearings = np.unique(bearing_ids)
    
    # Check if there are enough bearings to split by bearing
    can_split_bearing = False
    if labels is not None:
        from collections import defaultdict
        b_per_class = defaultdict(set)
        for b, l in zip(bearing_ids, labels):
            b_per_class[l].add(b)
        can_split_bearing = all(len(s) >= 2 for s in b_per_class.values())
    elif len(unique_bearings) >= 6:
        can_split_bearing = True

    if can_split_bearing:
        shuffled = rng.permutation(unique_bearings)
        n_test = max(1, int(round(len(unique_bearings) * test_ratio)))
        test_bearings = set(shuffled[:n_test])
        train_bearings = set(shuffled[n_test:])

        train_mask = np.isin(bearing_ids, list(train_bearings))
        test_mask = np.isin(bearing_ids, list(test_bearings))
        overlap = set(bearing_ids[train_mask]).intersection(set(bearing_ids[test_mask]))
        assert len(overlap) == 0, f"Bearing leakage detected: {overlap}"
        return train_mask, test_mask

    # Fallback to record-level leak-free split (grouped by record_filename)
    recs = record_ids if record_ids is not None else bearing_ids
    unique_recs = np.unique(recs)
    
    if labels is not None:
        # Stratified by label across unique recordings
        rec_to_label = dict(zip(recs, labels))
        test_recs = set()
        for l in np.unique(labels):
            class_recs = [r for r, cl in rec_to_label.items() if cl == l]
            shuffled_c = rng.permutation(class_recs)
            n_t = max(1, int(round(len(class_recs) * test_ratio)))
            test_recs.update(shuffled_c[:n_t])
    else:
        shuffled_r = rng.permutation(unique_recs)
        n_test_r = max(1, int(round(len(unique_recs) * test_ratio)))
        test_recs = set(shuffled_r[:n_test_r])

    test_mask = np.isin(recs, list(test_recs))
    train_mask = ~test_mask

    # Assert zero recording leakage
    overlap = set(recs[train_mask]).intersection(set(recs[test_mask]))
    assert len(overlap) == 0, f"Record leakage detected: {overlap}"
    return train_mask, test_mask


def run_experiment(
    data_dir: Optional[Path] = None,
    clf_type: str = "rf",
    random_seed: int = 42,
    max_files_per_bearing: Optional[int] = None,
) -> Dict[str, Any]:
    """Train and evaluate Current-Only, Temperature-Only, and Combined models."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, bearing_ids, record_ids, feature_names = prepare_dataset(
        data_dir=data_dir,
        max_files_per_bearing=max_files_per_bearing,
    )
    train_mask, test_mask = split_by_bearing(
        bearing_ids,
        record_ids=record_ids,
        labels=y,
        test_ratio=0.33,
        random_seed=random_seed,
    )

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\nTrain set: {len(X_train)} segments | Test set: {len(X_test)} segments")
    print(f"Train bearings: {sorted(list(set(bearing_ids[train_mask])))}")
    print(f"Test bearings:  {sorted(list(set(bearing_ids[test_mask])))}")

    models = {
        "current_only": MotorPdMModel(mode="current_only", clf_type=clf_type, random_state=random_seed),
        "temperature_only": MotorPdMModel(mode="temperature_only", clf_type=clf_type, random_state=random_seed),
        "combined": MotorPdMModel(mode="combined", clf_type=clf_type, random_state=random_seed),
    }

    results: Dict[str, Any] = {}
    summary_rows = []

    for name, model in models.items():
        print(f"\n--- Training {name.upper()} model ({clf_type}) ---")
        model.fit(X_train, y_train, feature_names)
        metrics = model.evaluate(X_test, y_test, feature_names)

        results[name] = {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
            "anomaly_roc_auc": metrics.anomaly_roc_auc,
            "confusion_matrix": metrics.confusion_matrix,
            "class_report": metrics.class_report,
            "classes": metrics.classes,
        }

        summary_rows.append({
            "Model": name,
            "Accuracy": metrics.accuracy,
            "Macro F1": metrics.macro_f1,
            "Macro Precision": metrics.macro_precision,
            "Macro Recall": metrics.macro_recall,
            "Anomaly ROC-AUC": metrics.anomaly_roc_auc if metrics.anomaly_roc_auc is not None else 0.0,
        })

        # Save model checkpoint
        ckpt_path = CHECKPOINTS_DIR / f"{name}_model.joblib"
        joblib.dump(model, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path.relative_to(ROOT_DIR)}")

    # Summary table
    df_summary = pd.DataFrame(summary_rows)
    print("\n=== EXPERIMENT RESULTS COMPARISON TABLE ===")
    print(df_summary.to_string(index=False))

    # Save comparison CSV and JSON metrics
    df_summary.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save feature names and metadata
    with open(CHECKPOINTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "classes": models["combined"].classes_,
                "clf_type": clf_type,
                "random_seed": random_seed,
            },
            f,
            indent=2,
        )

    print(f"\nMetrics written to {RESULTS_DIR.relative_to(ROOT_DIR)}/metrics.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train motor current and temperature models")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing Paderborn .mat files")
    parser.add_argument("--clf", type=str, default="rf", choices=["rf", "gb", "svm"], help="Classifier type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-files", type=int, default=None, help="Max files per bearing to process")
    args = parser.parse_args()

    data_path = Path(args.data_dir) if args.data_dir else None
    run_experiment(
        data_dir=data_path,
        clf_type=args.clf,
        random_seed=args.seed,
        max_files_per_bearing=args.max_files,
    )
