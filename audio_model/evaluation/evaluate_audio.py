"""
Evaluate acoustic anomaly detection models (Dense Autoencoder) on MIMII DG test sets.

Methodology & Standards:
  * Strict No-Leakage: Models are evaluated purely on held-out TEST clips.
    Decision thresholds are loaded directly from the training checkpoint (derived on
    normal training clips, e.g. 95th percentile). No tuning on test data is performed.
  * Metric Definitions:
    - AUC: Area under Receiver Operating Characteristic curve.
    - Partial AUC (pAUC, max_fpr=0.1): scikit-learn's `roc_auc_score(..., max_fpr=0.1)`.
      NOTE ON pAUC: scikit-learn uses the McClish (1989) standardisation formula:
          pAUC_McClish = 0.5 * (1.0 + (pAUC_raw - min_area) / (max_area - min_area))
      where min_area = max_fpr^2 / 2 and max_area = max_fpr. This normalises the partial
      AUC to the range [0.5, 1.0] so that a random guesser scores 0.5 and a perfect
      detector scores 1.0, rather than reporting raw integral area in [0, max_fpr].
    - Classification metrics: accuracy, precision, recall, F1 score, false-alarm rate (FAR),
      and confusion matrix computed at the train-derived threshold.
    - DCASE Metric: Harmonic mean of AUC and pAUC across sections and domains per machine,
      as standardized in the DCASE Challenge Task 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import hmean
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from audio_model.config_audio import (
    CHECKPOINTS_AUDIO_DIR,
    FIGURES_AUDIO_DIR,
    METRICS_AUDIO_DIR,
    PROCESSED_AUDIO_DIR,
    get_device,
)
from audio_model.models.autoencoder import DenseAutoencoder


def compute_clip_scores(
    model: torch.nn.Module,
    stacked_clips: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Compute clip anomaly scores as the mean frame reconstruction MSE error."""
    model.eval()
    n_clips = stacked_clips.shape[0]
    scores = np.empty(n_clips, dtype=np.float32)

    with torch.no_grad():
        for i in range(n_clips):
            clip_frames = torch.from_numpy(stacked_clips[i]).to(device)
            recon = model(clip_frames)
            frame_mse = torch.mean((clip_frames - recon) ** 2, dim=-1)
            scores[i] = frame_mse.mean().item()

    return scores


def compute_auc_and_pauc(
    y_true: np.ndarray,
    scores: np.ndarray,
    max_fpr: float = 0.1,
) -> Tuple[float, float]:
    """Compute ROC-AUC and McClish-standardised partial AUC (pAUC).

    Note on pAUC:
      scikit-learn's `roc_auc_score(..., max_fpr=max_fpr)` standardizes the partial
      AUC using the McClish (1989) formula:
          pAUC_McClish = 0.5 * (1.0 + (pAUC_raw - min_area) / (max_area - min_area))
      where min_area = max_fpr^2 / 2 and max_area = max_fpr.
      This maps the partial area from [min_area, max_area] into [0.5, 1.0], ensuring
      that random guessing achieves 0.50 and perfect separation achieves 1.00.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    auc = float(roc_auc_score(y_true, scores))
    pauc = float(roc_auc_score(y_true, scores, max_fpr=max_fpr))
    return auc, pauc


def compute_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute binary classification metrics at a fixed decision threshold.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (0=normal, 1=anomaly).
    scores : np.ndarray
        Clip reconstruction error scores.
    threshold : float
        Decision threshold derived strictly from training normal clips.

    Returns
    -------
    Dict[str, Any]
        Dictionary with accuracy, precision, recall, f1, false_alarm_rate, and confusion matrix.
    """
    y_pred = (scores > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = int(len(y_true))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_alarm_rate": far,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        },
        "total_clips": total,
        "normal_clips": int((y_true == 0).sum()),
        "anomaly_clips": int((y_true == 1).sum()),
    }


def evaluate_single_machine(
    machine: str,
    processed_dir: Path,
    checkpoints_dir: Path,
    device: torch.device,
    threshold_key: str = "p95",
) -> Dict[str, Any]:
    """Evaluate one machine's autoencoder on test clips across all sections and domains."""
    machine_dir = processed_dir / machine
    checkpoint_path = checkpoints_dir / f"{machine}_ae.pt"

    if not machine_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {machine_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim = ckpt.get("input_dim", 640)
    hidden_dim = ckpt.get("hidden_dim", 128)
    bottleneck_dim = ckpt.get("bottleneck_dim", 8)
    thresholds = ckpt.get("thresholds", {})
    threshold = float(thresholds.get(threshold_key, thresholds.get("p95", 0.5)))

    # Instantiate model
    model = DenseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load test data
    test_stacked = np.load(machine_dir / "test_stacked.npy")  # (N_clips, 309, 640)
    test_labels = np.load(machine_dir / "test_labels.npy")    # (N_clips,)
    test_sections = np.load(machine_dir / "test_sections.npy")
    test_domains = np.load(machine_dir / "test_domains.npy")
    test_filenames = np.load(machine_dir / "test_filenames.npy")

    # Score all test clips
    scores = compute_clip_scores(model, test_stacked, device)

    sections = sorted(list(np.unique(test_sections)))
    domains = ["source", "target"]

    section_domain_results: List[Dict[str, Any]] = []
    auc_values: List[float] = []
    pauc_values: List[float] = []

    # 1. Evaluate per (section, domain)
    for sec in sections:
        for dom in domains:
            mask = (test_sections == sec) & (test_domains == dom)
            if not np.any(mask):
                continue

            sub_y = test_labels[mask]
            sub_scores = scores[mask]

            auc, pauc = compute_auc_and_pauc(sub_y, sub_scores, max_fpr=0.1)
            clf_metrics = compute_binary_metrics(sub_y, sub_scores, threshold)

            auc_values.append(auc)
            pauc_values.append(pauc)

            res_entry = {
                "machine": machine,
                "section": sec,
                "domain": dom,
                "auc": auc,
                "pauc": pauc,
                "threshold": threshold,
                "threshold_key": threshold_key,
                "normal_mean_score": float(sub_scores[sub_y == 0].mean()) if (sub_y == 0).any() else None,
                "anomaly_mean_score": float(sub_scores[sub_y == 1].mean()) if (sub_y == 1).any() else None,
                **clf_metrics,
            }
            section_domain_results.append(res_entry)

    # 2. Evaluate per domain aggregated
    domain_results: Dict[str, Dict[str, Any]] = {}
    for dom in domains:
        dmask = test_domains == dom
        if np.any(dmask):
            d_y = test_labels[dmask]
            d_scores = scores[dmask]
            d_auc, d_pauc = compute_auc_and_pauc(d_y, d_scores, max_fpr=0.1)
            d_clf = compute_binary_metrics(d_y, d_scores, threshold)
            domain_results[dom] = {
                "auc": d_auc,
                "pauc": d_pauc,
                "threshold": threshold,
                "normal_mean_score": float(d_scores[d_y == 0].mean()),
                "anomaly_mean_score": float(d_scores[d_y == 1].mean()),
                **d_clf,
            }

    # 3. Overall machine evaluation
    overall_auc, overall_pauc = compute_auc_and_pauc(test_labels, scores, max_fpr=0.1)
    overall_clf = compute_binary_metrics(test_labels, scores, threshold)

    # DCASE harmonic mean aggregations across all section-domain pairs
    hmean_auc = float(hmean(auc_values)) if auc_values else 0.0
    hmean_pauc = float(hmean(pauc_values)) if pauc_values else 0.0
    hmean_official = float(hmean(auc_values + pauc_values)) if (auc_values and pauc_values) else 0.0

    return {
        "machine": machine,
        "threshold": threshold,
        "threshold_key": threshold_key,
        "all_thresholds": thresholds,
        "scores": scores,
        "test_labels": test_labels,
        "test_sections": test_sections,
        "test_domains": test_domains,
        "test_filenames": test_filenames,
        "section_domain_results": section_domain_results,
        "domain_results": domain_results,
        "overall": {
            "auc": overall_auc,
            "pauc": overall_pauc,
            "threshold": threshold,
            "normal_mean_score": float(scores[test_labels == 0].mean()),
            "anomaly_mean_score": float(scores[test_labels == 1].mean()),
            **overall_clf,
        },
        "dcase_metrics": {
            "hmean_auc": hmean_auc,
            "hmean_pauc": hmean_pauc,
            "hmean_all_official": hmean_official,
            "n_evaluated_conditions": len(section_domain_results),
        },
    }


def plot_roc_curves(
    machine_evals: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate ROC curves comparing Source vs Target domain, highlighting pAUC range [0, 0.1]."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_machines = len(machine_evals)

    fig, axes = plt.subplots(1, n_machines, figsize=(7 * n_machines, 6), constrained_layout=True)
    if n_machines == 1:
        axes = [axes]

    palette = {
        "source": "#1f77b4",  # Blue
        "target": "#d62728",  # Red
    }

    for ax, (m, data) in zip(axes, machine_evals.items()):
        labels = data["test_labels"]
        scores = data["scores"]
        doms = data["test_domains"]

        # Plot source domain ROC
        for dom in ["source", "target"]:
            mask = doms == dom
            if np.any(mask):
                fpr, tpr, _ = roc_curve(labels[mask], scores[mask])
                dom_auc = data["domain_results"][dom]["auc"]
                dom_pauc = data["domain_results"][dom]["pauc"]
                ax.plot(
                    fpr,
                    tpr,
                    color=palette[dom],
                    lw=2.5,
                    label=f"{dom.capitalize()} (AUC: {dom_auc:.3f}, pAUC: {dom_pauc:.3f})",
                )

        # Plot overall ROC
        fpr_all, tpr_all, _ = roc_curve(labels, scores)
        ax.plot(
            fpr_all,
            tpr_all,
            color="#2ca02c",
            lw=1.8,
            linestyle="--",
            label=f"Overall (AUC: {data['overall']['auc']:.3f})",
        )

        # Highlight pAUC region (FPR in [0, 0.1])
        ax.axvspan(0.0, 0.1, color="#f39c12", alpha=0.15, label="pAUC Region (FPR ≤ 0.1)")
        ax.axvline(0.1, color="#e67e22", linestyle=":", lw=1.2)

        # Chance diagonal
        ax.plot([0, 1], [0, 1], color="#7f8c8d", linestyle=":", lw=1.2, label="Random (0.50)")

        ax.set_title(f"ROC Curves: {m.upper()} (Source vs Target)", fontsize=13, fontweight="bold")
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_score_histograms(
    machine_evals: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot anomaly score distributions for Normal vs Anomaly clips with train threshold line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_machines = len(machine_evals)

    fig, axes = plt.subplots(1, n_machines, figsize=(7 * n_machines, 5.5), constrained_layout=True)
    if n_machines == 1:
        axes = [axes]

    for ax, (m, data) in zip(axes, machine_evals.items()):
        labels = data["test_labels"]
        scores = data["scores"]
        threshold = data["threshold"]
        threshold_key = data["threshold_key"]

        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        # Determine bin edges
        all_vals = np.concatenate([normal_scores, anomaly_scores])
        bins = np.linspace(float(all_vals.min()), float(all_vals.max()), 35)

        ax.hist(
            normal_scores,
            bins=bins,
            alpha=0.65,
            color="#3498db",
            density=True,
            edgecolor="#2980b9",
            label=f"Normal (N={len(normal_scores)}, Mean={normal_scores.mean():.3f})",
        )
        ax.hist(
            anomaly_scores,
            bins=bins,
            alpha=0.65,
            color="#e74c3c",
            density=True,
            edgecolor="#c0392b",
            label=f"Anomaly (N={len(anomaly_scores)}, Mean={anomaly_scores.mean():.3f})",
        )

        ax.axvline(
            threshold,
            color="#2c3e50",
            linestyle="--",
            lw=2.5,
            label=f"Train Threshold ({threshold_key.upper()}={threshold:.3f})",
        )

        ax.set_title(f"Score Distribution: {m.upper()} (Normal vs Anomaly)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Reconstruction MSE Anomaly Score", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(
    machine_evals: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot confusion matrices for Source, Target, and Overall test clips per machine."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    machines = list(machine_evals.keys())
    n_rows = len(machines)
    cols = ["Source Domain", "Target Domain", "Combined Overall"]

    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 4.5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, m in enumerate(machines):
        data = machine_evals[m]
        sub_items = [
            (data["domain_results"]["source"]["confusion_matrix"]["matrix"], "Source Domain"),
            (data["domain_results"]["target"]["confusion_matrix"]["matrix"], "Target Domain"),
            (data["overall"]["confusion_matrix"]["matrix"], "Combined Overall"),
        ]

        for col_idx, (cm_list, col_title) in enumerate(sub_items):
            ax = axes[row_idx, col_idx]
            cm = np.array(cm_list)
            im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

            tn, fp, fn, tp = cm.ravel()
            total = cm.sum()
            total_norm = tn + fp
            total_anom = fn + tp

            # Annotate numbers and rates
            cell_texts = [
                [f"TN: {tn}\n({tn/max(1,total_norm):.1%})", f"FP: {fp}\n({fp/max(1,total_norm):.1%})"],
                [f"FN: {fn}\n({fn/max(1,total_anom):.1%})", f"TP: {tp}\n({tp/max(1,total_anom):.1%})"],
            ]

            for i in range(2):
                for j in range(2):
                    color = "white" if cm[i, j] > (cm.max() * 0.6) else "black"
                    ax.text(j, i, cell_texts[i][j], ha="center", va="center", color=color, fontsize=11, fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred Normal (0)", "Pred Anomaly (1)"], fontsize=10)
            ax.set_yticklabels(["True Normal (0)", "True Anomaly (1)"], fontsize=10)
            ax.set_title(f"{m.upper()}: {col_title}\n(Total: {total})", fontsize=11, fontweight="bold")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_auc_pauc_bars(
    machine_evals: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Create grouped bar chart of AUC and pAUC per machine and domain."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    categories: List[str] = []
    auc_bars: List[float] = []
    pauc_bars: List[float] = []

    for m, data in machine_evals.items():
        # Source
        categories.append(f"{m.capitalize()}\nSource")
        auc_bars.append(data["domain_results"]["source"]["auc"])
        pauc_bars.append(data["domain_results"]["source"]["pauc"])

        # Target
        categories.append(f"{m.capitalize()}\nTarget")
        auc_bars.append(data["domain_results"]["target"]["auc"])
        pauc_bars.append(data["domain_results"]["target"]["pauc"])

        # Harmonic Mean across sections & domains
        categories.append(f"{m.capitalize()}\nH-Mean (All)")
        auc_bars.append(data["dcase_metrics"]["hmean_auc"])
        pauc_bars.append(data["dcase_metrics"]["hmean_pauc"])

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width / 2, auc_bars, width, label="ROC-AUC", color="#2980b9", edgecolor="#1a5276")
    rects2 = ax.bar(x + width / 2, pauc_bars, width, label="pAUC (max_fpr=0.1, McClish)", color="#e67e22", edgecolor="#b96417")

    # Add text labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Chance level reference line at 0.50
    ax.axhline(0.50, color="#7f8c8d", linestyle="--", lw=1.5, label="Chance Level (0.50)")

    ax.set_title("AUC and Partial AUC (pAUC) Across Machines and Domains", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_evaluation(
    machines: List[str],
    processed_dir: Path,
    checkpoints_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    threshold_key: str = "p95",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute evaluation for all machines, compute metrics, export JSON/CSV and plots."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device_str = get_device()
    device = torch.device(device_str)

    machine_evals: Dict[str, Dict[str, Any]] = {}
    csv_rows: List[Dict[str, Any]] = []

    for m in machines:
        eval_data = evaluate_single_machine(
            machine=m,
            processed_dir=processed_dir,
            checkpoints_dir=checkpoints_dir,
            device=device,
            threshold_key=threshold_key,
        )
        machine_evals[m] = eval_data

        # 1. Section & Domain rows
        for sub in eval_data["section_domain_results"]:
            cm = sub["confusion_matrix"]
            csv_rows.append({
                "machine": sub["machine"],
                "section": sub["section"],
                "domain": sub["domain"],
                "n_normal": sub["normal_clips"],
                "n_anomaly": sub["anomaly_clips"],
                "n_total": sub["total_clips"],
                "auc": round(sub["auc"], 5),
                "pauc_0.1": round(sub["pauc"], 5),
                "threshold": round(sub["threshold"], 5),
                "threshold_key": sub["threshold_key"],
                "accuracy": round(sub["accuracy"], 5),
                "precision": round(sub["precision"], 5),
                "recall": round(sub["recall"], 5),
                "f1": round(sub["f1"], 5),
                "false_alarm_rate": round(sub["false_alarm_rate"], 5),
                "tn": cm["tn"],
                "fp": cm["fp"],
                "fn": cm["fn"],
                "tp": cm["tp"],
            })

        # 2. Aggregated domain rows (Source vs Target)
        for dom, d_data in eval_data["domain_results"].items():
            cm = d_data["confusion_matrix"]
            csv_rows.append({
                "machine": m,
                "section": "ALL",
                "domain": dom,
                "n_normal": d_data["normal_clips"],
                "n_anomaly": d_data["anomaly_clips"],
                "n_total": d_data["total_clips"],
                "auc": round(d_data["auc"], 5),
                "pauc_0.1": round(d_data["pauc"], 5),
                "threshold": round(d_data["threshold"], 5),
                "threshold_key": threshold_key,
                "accuracy": round(d_data["accuracy"], 5),
                "precision": round(d_data["precision"], 5),
                "recall": round(d_data["recall"], 5),
                "f1": round(d_data["f1"], 5),
                "false_alarm_rate": round(d_data["false_alarm_rate"], 5),
                "tn": cm["tn"],
                "fp": cm["fp"],
                "fn": cm["fn"],
                "tp": cm["tp"],
            })

        # 3. Overall machine row
        ov = eval_data["overall"]
        cm_ov = ov["confusion_matrix"]
        csv_rows.append({
            "machine": m,
            "section": "ALL",
            "domain": "ALL",
            "n_normal": ov["normal_clips"],
            "n_anomaly": ov["anomaly_clips"],
            "n_total": ov["total_clips"],
            "auc": round(ov["auc"], 5),
            "pauc_0.1": round(ov["pauc"], 5),
            "threshold": round(ov["threshold"], 5),
            "threshold_key": threshold_key,
            "accuracy": round(ov["accuracy"], 5),
            "precision": round(ov["precision"], 5),
            "recall": round(ov["recall"], 5),
            "f1": round(ov["f1"], 5),
            "false_alarm_rate": round(ov["false_alarm_rate"], 5),
            "tn": cm_ov["tn"],
            "fp": cm_ov["fp"],
            "fn": cm_ov["fn"],
            "tp": cm_ov["tp"],
        })

    # Convert to DataFrame
    df_eval = pd.DataFrame(csv_rows)

    # Export CSV
    csv_path = metrics_dir / "audio_eval.csv"
    df_eval.to_csv(csv_path, index=False)

    # Build JSON structure
    json_export: Dict[str, Any] = {
        "metadata": {
            "evaluation_type": "Acoustic Anomaly Detection (Dense Autoencoder Baseline)",
            "pauc_note": (
                "Partial AUC with max_fpr=0.1 computed via sklearn.metrics.roc_auc_score, "
                "which implements McClish (1989) standardization to interval [0.5, 1.0]."
            ),
            "threshold_rule": (
                f"Strictly derived from training normal clips using {threshold_key.upper()} percentile. "
                "No tuning on test data."
            ),
        },
        "machines": {},
    }

    for m, d in machine_evals.items():
        json_export["machines"][m] = {
            "thresholds": d["all_thresholds"],
            "applied_threshold": {
                "key": d["threshold_key"],
                "value": d["threshold"],
            },
            "dcase_metrics": d["dcase_metrics"],
            "domain_results": {
                dom: {
                    k: v for k, v in d_res.items()
                }
                for dom, d_res in d["domain_results"].items()
            },
            "overall_results": d["overall"],
            "conditions": d["section_domain_results"],
        }

    json_path = metrics_dir / "audio_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)

    # Generate Figures
    plot_roc_curves(machine_evals, figures_dir / "audio_roc_curves.png")
    plot_roc_curves(machine_evals, figures_dir / "roc_curves.png")

    plot_score_histograms(machine_evals, figures_dir / "audio_score_histograms.png")
    plot_score_histograms(machine_evals, figures_dir / "score_histograms.png")

    plot_confusion_matrices(machine_evals, figures_dir / "audio_confusion_matrices.png")
    plot_confusion_matrices(machine_evals, figures_dir / "confusion_matrices.png")

    plot_auc_pauc_bars(machine_evals, figures_dir / "audio_auc_pauc_comparison.png")
    plot_auc_pauc_bars(machine_evals, figures_dir / "auc_pauc_bars.png")

    return json_export, df_eval


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Acoustic Autoencoder Anomaly Detection")
    parser.add_argument("--machines", nargs="+", default=["fan", "bearing"], help="Machines to evaluate")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_AUDIO_DIR, help="Path to processed data")
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_AUDIO_DIR, help="Path to checkpoints")
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_AUDIO_DIR, help="Path to save metrics")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_AUDIO_DIR, help="Path to save figures")
    parser.add_argument("--threshold-key", type=str, default="p95", help="Threshold key to use (p90, p95, p99)")
    args = parser.parse_args(argv)

    json_res, df_res = run_evaluation(
        machines=args.machines,
        processed_dir=args.processed_dir,
        checkpoints_dir=args.checkpoints_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        threshold_key=args.threshold_key,
    )

    # Print Clean Summary Tables
    print("\n" + "=" * 105)
    print("DCASE ACOUSTIC ANOMALY DETECTION EVALUATION SUMMARY (MIMII DG TEST SET)")
    print("=" * 105)
    print(f"{'Machine':9s} | {'Section':10s} | {'Domain':7s} | {'AUC':7s} | {'pAUC (0.1)':10s} | {'Acc':6s} | {'Prec':6s} | {'Recall':6s} | {'FAR':6s} | {'F1':6s}")
    print("-" * 105)

    for idx, row in df_res.iterrows():
        is_sub = row["section"] != "ALL"
        prefix = "  " if is_sub else ""
        highlight = "*" if not is_sub and row["domain"] == "ALL" else ""
        print(
            f"{prefix}{row['machine']:7s} | {row['section']:10s} | {row['domain']:7s} | "
            f"{row['auc']:.4f}  | {row['pauc_0.1']:.4f}     | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['false_alarm_rate']:.4f} | {row['f1']:.4f} {highlight}"
        )

    print("=" * 105)
    print("\nDCASE CHALLENGE HARMONIC MEAN AGGREGATES:")
    print("-" * 75)
    print(f"{'Machine':10s} | {'Harmonic Mean AUC':20s} | {'Harmonic Mean pAUC':20s} | {'Official Score (H-Mean All)':25s}")
    print("-" * 75)
    for m in args.machines:
        dm = json_res["machines"][m]["dcase_metrics"]
        print(f"{m:10s} | {dm['hmean_auc']:.4f}{' '*14} | {dm['hmean_pauc']:.4f}{' '*14} | {dm['hmean_all_official']:.4f}")
    print("=" * 75 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
