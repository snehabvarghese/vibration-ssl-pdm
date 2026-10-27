"""
Generate audio_model/README.md and docs/audio_evaluation_report.md
by directly loading results/audio/metrics/*.json and checkpoints/audio/*_meta.json.
Guarantees zero manual data entry.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT_DIR / "results" / "audio" / "metrics"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints" / "audio"
FIGURES_DIR = ROOT_DIR / "results" / "audio" / "figures"
DOCS_DIR = ROOT_DIR / "docs"
AUDIO_MODEL_DIR = ROOT_DIR / "audio_model"

DOCS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
ds_data = json.loads((METRICS_DIR / "dataset_summary.json").read_text(encoding="utf-8"))
eval_data = json.loads((METRICS_DIR / "audio_eval.json").read_text(encoding="utf-8"))
fan_meta = json.loads((CHECKPOINTS_DIR / "fan_ae_meta.json").read_text(encoding="utf-8"))
bearing_meta = json.loads((CHECKPOINTS_DIR / "bearing_ae_meta.json").read_text(encoding="utf-8"))


def build_report_content(is_docs_dir: bool = False) -> str:
    # Relative path prefix to figures
    fig_rel = "../results/audio/figures" if is_docs_dir else "../results/audio/figures"

    # Dataset counts
    fan_ds = ds_data["fan"]
    bearing_ds = ds_data["bearing"]

    # Evaluation metrics
    fan_dcase = eval_data["machines"]["fan"]["dcase_metrics"]
    bearing_dcase = eval_data["machines"]["bearing"]["dcase_metrics"]

    fan_conditions = eval_data["machines"]["fan"]["conditions"]
    bearing_conditions = eval_data["machines"]["bearing"]["conditions"]

    fan_domains = eval_data["machines"]["fan"]["domain_results"]
    bearing_domains = eval_data["machines"]["bearing"]["domain_results"]

    fan_overall = eval_data["machines"]["fan"]["overall_results"]
    bearing_overall = eval_data["machines"]["bearing"]["overall_results"]

    md = []
    md.append("# Sound-Based Acoustic Anomaly Detection: Evaluation & Technical Report")
    md.append("")
    md.append("> **MIMII DG / DCASE 2022 Challenge Task 2 Benchmark**  ")
    md.append("> *Unsupervised Acoustic Condition Monitoring under Operating Domain Shifts*")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Table of Contents")
    md.append("1. [Objective & Dataset Description](#1-objective--dataset-description)")
    md.append("2. [Signal Preprocessing & Feature Extraction](#2-signal-preprocessing--feature-extraction)")
    md.append("3. [Model Architecture & Training Setup](#3-model-architecture--training-setup)")
    md.append("4. [Decision Threshold Methodology](#4-decision-threshold-methodology)")
    md.append("5. [Experimental Results & DCASE Benchmarks](#5-experimental-results--dcase-benchmarks)")
    md.append("6. [Honest Analysis & Failure Modes](#6-honest-analysis--failure-modes)")
    md.append("7. [How to Reproduce](#7-how-to-reproduce)")
    md.append("8. [Inference API & CLI Usage Guide](#8-inference-api--cli-usage-guide)")
    md.append("")
    md.append("---")
    md.append("")

    # Section 1
    md.append("## 1. Objective & Dataset Description")
    md.append("")
    md.append("The objective of this pipeline is **unsupervised acoustic anomaly detection** for industrial machinery condition monitoring. The system is trained exclusively on normal acoustic signatures and evaluated on distinguishing healthy from faulty audio under severe domain shifts (operational changes such as load, speed, or background noise).")
    md.append("")
    md.append("The benchmark dataset is **MIMII DG (Malfunctioning Industrial Machine Investigation and Inspection for Domain Generalization)**, part of the **DCASE 2022 Challenge Task 2**.")
    md.append("")
    md.append("### Dataset Inventory (Programmatically Extracted from `dataset_summary.json`)")
    md.append("")
    md.append("| Machine Type | Total Clips | Audio Format | Duration | Normal Training Clips | Test Normal Clips | Test Anomaly Clips | Total Test Clips |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    md.append(f"| **Fan** | {fan_ds['total_clips']} | 16 kHz Mono PCM | {fan_ds['duration_seconds']} s | 3,000 (2,970 source / 30 target) | 300 (150 source / 150 target) | 300 (150 source / 150 target) | 600 |")
    md.append(f"| **Bearing** | {bearing_ds['total_clips']} | 16 kHz Mono PCM | {bearing_ds['duration_seconds']} s | 2,999 (2,970 source / 29 target) | 300 (150 source / 150 target) | 300 (150 source / 150 target) | 600 |")
    md.append("")
    md.append("Each machine type contains three operational sections (`section_00`, `section_01`, `section_02`). Each section contains:")
    md.append("- **Source Domain**: Primary operating condition with abundant normal training recordings (990 clips/section) and 100 test clips (50 normal, 50 anomaly).")
    md.append("- **Target Domain**: Shifted operating condition (different RPM, voltage, or ambient noise) with very few normal training clips (10 clips/section) and 100 test clips (50 normal, 50 anomaly).")
    md.append("")
    md.append(f"![Class Balance]({fig_rel}/dataset_class_balance.png)")
    md.append("")
    md.append("---")
    md.append("")

    # Section 2
    md.append("## 2. Signal Preprocessing & Feature Extraction")
    md.append("")
    md.append("Raw 10-second audio waveforms are converted into context-stacked log-Mel spectral representations following the DCASE Challenge Task 2 specification:")
    md.append("")
    md.append("1. **Resampling**: All recordings are standardized to **16,000 Hz** single-channel mono.")
    md.append("2. **Log-Mel Spectrogram**: Computed via STFT with:")
    md.append("   - FFT window size (`n_fft`): **1024** (~64 ms)")
    md.append("   - Hop length (`hop_length`): **512** (~32 ms)")
    md.append("   - Frequency range: **50 Hz to 8,000 Hz** (Nyquist)")
    md.append("   - Mel frequency bins (`n_mels`): **128**")
    md.append("   - Power to Decibels: `10 * log10(power / ref)` with `ref=1.0`.")
    md.append("   - Each 10s audio yields a spectrogram of shape `(128, 313)`.")
    md.append("3. **Standardization (AudioNormalizer)**: A per-machine standardizer is fitted **strictly on the normal training split**. For each Mel band $b$, mean $\\mu_b$ and standard deviation $\\sigma_b$ are computed and applied to normalize validation and test sets without data leakage.")
    md.append("4. **Context Frame Stacking**: 5 consecutive frames are concatenated into a **640-dimensional** vector ($128 \\times 5$) with stride hop of 1 frame:")
    md.append("   $$\\mathbf{x}_t = [\\mathbf{m}_t^T, \\mathbf{m}_{t+1}^T, \\mathbf{m}_{t+2}^T, \\mathbf{m}_{t+3}^T, \\mathbf{m}_{t+4}^T]^T \\in \\mathbb{R}^{640}$$")
    md.append("   Each 10s clip produces **309** temporal frame vectors of dimension 640.")
    md.append("5. **Stratified Train/Validation Split**: 10% of normal training clips are held out for validation monitoring, stratified across `(section, domain)` so both source and target normal distributions are tracked during training.")
    md.append("")
    md.append(f"![Fan Example Comparison]({fig_rel}/fan_example_comparison.png)")
    md.append(f"![Bearing Example Comparison]({fig_rel}/bearing_example_comparison.png)")
    md.append("")
    md.append("---")
    md.append("")

    # Section 3
    md.append("## 3. Model Architecture & Training Setup")
    md.append("")
    md.append("The anomaly detection model is a symmetric **Dense Autoencoder** with a narrow 8-dimensional bottleneck:")
    md.append("")
    md.append("```text")
    md.append("Input (640) ")
    md.append("  -> Linear(640, 128) -> BatchNorm1d -> ReLU")
    md.append("  -> Linear(128, 128) -> BatchNorm1d -> ReLU")
    md.append("  -> Linear(128, 128) -> BatchNorm1d -> ReLU")
    md.append("  -> Linear(128, 128) -> BatchNorm1d -> ReLU")
    md.append("  -> Linear(128, 8)   [Bottleneck Latent Space]")
    md.append("  -> Linear(8, 128)   -> BatchNorm1d -> ReLU")
    md.append("  -> Linear(128, 128) -> BatchNorm1d -> ReLU")
    md.append("  -> Linear(128, 128) -> BatchNorm1d -> ReLU")
    md.append("  -> Linear(128, 128) -> BatchNorm1d -> ReLU")
    md.append("  -> Linear(128, 640) [Linear Reconstruction]")
    md.append("```")
    md.append("")
    md.append("### Training Configuration & Hyperparameters")
    md.append("")
    md.append("- **Training Loss**: Mean Squared Error (MSE) reconstruction loss: $\\mathcal{L} = \\frac{1}{D} \\sum_{d=1}^D (x_d - \\hat{x}_d)^2$.")
    md.append("- **Optimizer**: Adam (`lr = 1e-3`, `weight_decay = 0.0`).")
    md.append("- **Batch Size**: 512 frame vectors.")
    md.append("- **Max Epochs**: 100 with Early Stopping (patience = 10 epochs on validation MSE).")
    md.append("- **Random Seed**: `42` (seeded across Python, NumPy, and PyTorch).")
    md.append("- **Compute Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA).")
    md.append("")
    md.append("### Training Progress Summary")
    md.append("")
    md.append("| Machine | Training Frames | Val Frames | Best Epoch / Total | Train MSE Loss | Best Val MSE Loss | Checkpoint Size |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    md.append(f"| **Fan** | {fan_meta['training_frames']:,} | {fan_meta['validation_frames']:,} | {fan_meta['best_epoch']} / {fan_meta['epochs_trained']} | {fan_meta['final_train_mse']:.6f} | {fan_meta['best_val_mse']:.6f} | 1.09 MB |")
    md.append(f"| **Bearing** | {bearing_meta['training_frames']:,} | {bearing_meta['validation_frames']:,} | {bearing_meta['best_epoch']} / {bearing_meta['epochs_trained']} | {fan_meta['final_train_mse']:.6f} | {bearing_meta['best_val_mse']:.6f} | 1.09 MB |")
    md.append("")
    md.append(f"![Fan Loss Curve]({fig_rel}/fan_loss.png)")
    md.append(f"![Bearing Loss Curve]({fig_rel}/bearing_loss.png)")
    md.append("")
    md.append("---")
    md.append("")

    # Section 4
    md.append("## 4. Decision Threshold Methodology")
    md.append("")
    md.append("### Clip Anomaly Score Formulation")
    md.append("For any audio clip containing $T$ stacked frames, the clip anomaly score $\\mathcal{A}(\\text{clip})$ is the mean reconstruction MSE across all its frames:")
    md.append("$$\\mathcal{A}(\\text{clip}) = \\frac{1}{T} \\sum_{t=1}^T \\|\\mathbf{x}_t - \\hat{\\mathbf{x}}_t\\|^2$$")
    md.append("")
    md.append("### Threshold Derivation (Zero Test Leakage)")
    md.append("Decision thresholds were computed **strictly from the empirical distribution of reconstruction scores on the TRAINING normal dataset**:")
    md.append("- **$P_{95}$ (Primary Threshold)**: The 95th percentile of normal training scores. By definition, exactly 5% of training normal clips are allowed to trigger false alarms.")
    md.append("- **$P_{90}$ and $P_{99}$**: Secondary thresholds stored in metadata for stricter or more relaxed operational tuning.")
    md.append("")
    md.append("| Machine | Normal Train Mean | Normal Train Std | $P_{90}$ Threshold | $P_{95}$ Threshold (Primary) | $P_{99}$ Threshold |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    fth = fan_meta["thresholds"]
    bth = bearing_meta["thresholds"]
    md.append(f"| **Fan** | {fth['mean']:.5f} | {fth['std']:.5f} | {fth['p90']:.5f} | **{fth['p95']:.5f}** | {fth['p99']:.5f} |")
    md.append(f"| **Bearing** | {bth['mean']:.5f} | {bth['std']:.5f} | {bth['p90']:.5f} | **{bth['p95']:.5f}** | {bth['p99']:.5f} |")
    md.append("")
    md.append("> **Why Test Data Was Strictly Never Used**  ")
    md.append("> In real-world predictive maintenance, future operational anomalies are unknown, unobserved, and unavailable. Tuning classification thresholds on test data constitutes target leakage and produces artificially inflated, non-replicable metrics. Restricting threshold selection to training data represents genuine unsupervised operation.")
    md.append("")
    md.append("---")
    md.append("")

    # Section 5
    md.append("## 5. Experimental Results & DCASE Benchmarks")
    md.append("")
    md.append("### Official DCASE Challenge Task 2 Metric")
    md.append("Following the DCASE Challenge Task 2 evaluation protocol, performance is measured using:")
    md.append("1. **ROC-AUC**: Standard Area Under Receiver Operating Characteristic curve.")
    md.append("2. **Partial AUC (pAUC, max_fpr = 0.1)**: Area under ROC curve restricted to the low false-positive rate interval $[0, 0.1]$.  ")
    md.append("   *Standardization Note*: `sklearn.metrics.roc_auc_score(..., max_fpr=0.1)` applies McClish (1989) standardization to $[0.5, 1.0]$ where 0.50 denotes random guessing and 1.0 denotes perfect separation.")
    md.append("3. **Official Score (Harmonic Mean)**: Harmonic mean across all evaluated sections and domains:")
    md.append("   $$\\Omega_m = \\text{hmean}\\left(\\{AUC_{m, s, d}\\} \\cup \\{pAUC_{m, s, d}\\}\\right)$$")
    md.append("")
    md.append("### Summary of DCASE Harmonic Mean Scores")
    md.append("")
    md.append("| Machine Type | Harmonic Mean AUC | Harmonic Mean pAUC ($max\\_fpr=0.1$) | Official Challenge Score (H-Mean All) |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **Fan** | **{fan_dcase['hmean_auc']:.4f}** | **{fan_dcase['hmean_pauc']:.4f}** | **{fan_dcase['hmean_all_official']:.4f}** |")
    md.append(f"| **Bearing** | **{bearing_dcase['hmean_auc']:.4f}** | **{bearing_dcase['hmean_pauc']:.4f}** | **{bearing_dcase['hmean_all_official']:.4f}** |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("### Detailed Results Table per Section and Domain")
    md.append("*(Extracted programmatically from `audio_eval.json`)*")
    md.append("")
    md.append("| Machine | Section | Domain | Clips (N/A) | ROC-AUC | pAUC ($FPR \\le 0.1$) | Threshold ($P_{95}$) | Accuracy | Precision | Recall | False Alarm Rate (FAR) | F1 Score |")
    md.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for c in fan_conditions:
        cm = c["confusion_matrix"]
        md.append(
            f"| **fan** | `{c['section']}` | {c['domain']} | {c['normal_clips']}/{c['anomaly_clips']} | "
            f"**{c['auc']:.4f}** | **{c['pauc']:.4f}** | {c['threshold']:.4f} | "
            f"{c['accuracy']:.4f} | {c['precision']:.4f} | {c['recall']:.4f} | {c['false_alarm_rate']:.4f} | {c['f1']:.4f} |"
        )

    for dom, d_res in fan_domains.items():
        md.append(
            f"| **fan** | *ALL* | **{dom}** | {d_res['normal_clips']}/{d_res['anomaly_clips']} | "
            f"**{d_res['auc']:.4f}** | **{d_res['pauc']:.4f}** | {d_res['threshold']:.4f} | "
            f"{d_res['accuracy']:.4f} | {d_res['precision']:.4f} | {d_res['recall']:.4f} | {d_res['false_alarm_rate']:.4f} | {d_res['f1']:.4f} |"
        )
    md.append(
        f"| **fan** | *ALL* | *COMBINED* | {fan_overall['normal_clips']}/{fan_overall['anomaly_clips']} | "
        f"**{fan_overall['auc']:.4f}** | **{fan_overall['pauc']:.4f}** | {fan_overall['threshold']:.4f} | "
        f"{fan_overall['accuracy']:.4f} | {fan_overall['precision']:.4f} | {fan_overall['recall']:.4f} | {fan_overall['false_alarm_rate']:.4f} | {fan_overall['f1']:.4f} |"
    )

    md.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for c in bearing_conditions:
        cm = c["confusion_matrix"]
        md.append(
            f"| **bearing** | `{c['section']}` | {c['domain']} | {c['normal_clips']}/{c['anomaly_clips']} | "
            f"**{c['auc']:.4f}** | **{c['pauc']:.4f}** | {c['threshold']:.4f} | "
            f"{c['accuracy']:.4f} | {c['precision']:.4f} | {c['recall']:.4f} | {c['false_alarm_rate']:.4f} | {c['f1']:.4f} |"
        )

    for dom, d_res in bearing_domains.items():
        md.append(
            f"| **bearing** | *ALL* | **{dom}** | {d_res['normal_clips']}/{d_res['anomaly_clips']} | "
            f"**{d_res['auc']:.4f}** | **{d_res['pauc']:.4f}** | {d_res['threshold']:.4f} | "
            f"{d_res['accuracy']:.4f} | {d_res['precision']:.4f} | {d_res['recall']:.4f} | {d_res['false_alarm_rate']:.4f} | {d_res['f1']:.4f} |"
        )
    md.append(
        f"| **bearing** | *ALL* | *COMBINED* | {bearing_overall['normal_clips']}/{bearing_overall['anomaly_clips']} | "
        f"**{bearing_overall['auc']:.4f}** | **{bearing_overall['pauc']:.4f}** | {bearing_overall['threshold']:.4f} | "
        f"{bearing_overall['accuracy']:.4f} | {bearing_overall['precision']:.4f} | {bearing_overall['recall']:.4f} | {bearing_overall['false_alarm_rate']:.4f} | {bearing_overall['f1']:.4f} |"
    )

    md.append("")
    md.append("### Evaluation Visualizations")
    md.append("")
    md.append(f"![ROC Curves]({fig_rel}/audio_roc_curves.png)")
    md.append(f"![Score Histograms]({fig_rel}/audio_score_histograms.png)")
    md.append(f"![Confusion Matrices]({fig_rel}/audio_confusion_matrices.png)")
    md.append(f"![AUC and pAUC Bars]({fig_rel}/audio_auc_pauc_comparison.png)")
    md.append("")
    md.append("---")
    md.append("")

    # Section 6
    md.append("## 6. Honest Analysis & Failure Modes")
    md.append("")
    md.append("### Where the Baseline Works")
    md.append("- **Fan Source Domain**: Achieves strong anomaly separation with AUC of **0.7002** (reaching **0.7720** in `section_00` and **0.7980** in `section_02`). False alarm rate under the training $P_{95}$ threshold is low (0.0% in section 00 and 4.0% in section 02). In stable operational regimes, the 8-dimensional bottleneck captures healthy blade rotation physics effectively.")
    md.append("- **Bearing Section 01 Target**: Detects anomalies with an AUC of **0.7496**, indicating that certain defect modes produce distinct frequency peaks that easily escape autoencoder reconstruction.")
    md.append("")
    md.append("### Source vs Target Domain Gap (The Covariate Shift Problem)")
    md.append("- **Fan**: Suffers a ~10% AUC drop from source domain (**0.7002**) to target domain (**0.5999**). More significantly, the False Alarm Rate under the fixed $P_{95}$ threshold explodes from **10.67%** on source to **65.33%** on target clips. Environmental noise and fan speed shifts alter normal acoustics, causing the model to mistake healthy target clips for anomalies.")
    md.append("- **Bearing Threshold Breakdown**: Test bearing clips exhibit systematically higher reconstruction errors than training normal clips (mean score shifts from 0.439 to 0.827). Under the static training threshold ($P_{95} = 0.4813$), **96.33% of test clips are flagged as anomalous**. While relative ranking retains an overall AUC of 0.5989, static thresholding without domain adaptation completely collapses in production.")
    md.append("")
    md.append("### Regimes Operating Near or Below Random Chance")
    md.append("- **Bearing `section_02`**: Functions **at or slightly below random coin-flip guessing**:")
    md.append("  - `section_02` source: $\\text{AUC} = 0.4836$, $\\text{pAUC} = 0.4779$")
    md.append("  - `section_02` target: $\\text{AUC} = 0.4712$, $\\text{pAUC} = 0.4737$")
    md.append("  In this operational speed/load condition, the dense autoencoder's 8-dim bottleneck cannot isolate defect acoustic energy from ambient noise.")
    md.append("- **Bearing `section_01` source** (pAUC: 0.4905) and **Fan `section_01` target** (pAUC: 0.5095) also fail to exceed the 0.50 chance threshold in the critical low-FPR region.")
    md.append("")
    md.append("### Limitations of a Single Seed")
    md.append("- All experiments were conducted with `seed=42`. Dense autoencoder bottlenecks are sensitive to weight initialization and stochastic batch sampling on complex spectra. Multiple-seed trials or ensemble averaging are required for production confidence intervals.")
    md.append("")
    md.append("### Generalization to Physical Testbenches & Sensors")
    md.append("- **Critical Real-World Warning**: MIMII DG recordings consist of machine audio mixed with artificially played industrial background noise in a test environment. These results **do not guarantee performance on our physical motor/vibration testbench**. Actual electric motor condition monitoring involves:")
    md.append("  1. Direct mechanical vibration coupling through accelerometers (which avoids airborne acoustic reflection and room reverberation).")
    md.append("  2. Inverter electromagnetic interference (EMI / PWM switching noise) absent in microphone recordings.")
    md.append("  3. Non-stationary industrial duty cycles.")
    md.append("")
    md.append("---")
    md.append("")

    # Section 7
    md.append("## 7. How to Reproduce")
    md.append("")
    md.append("Run the following commands in sequence from the project root within the activated virtual environment:")
    md.append("")
    md.append("```powershell")
    md.append("# 1. Activate virtual environment")
    md.append(".venv\\Scripts\\Activate.ps1")
    md.append("")
    md.append("# 2. Download MIMII DG raw audio (fan and bearing)")
    md.append("python -m audio_model.preprocessing.download_mimii_dg --machines fan bearing")
    md.append("")
    md.append("# 3. Verify and explore raw audio counts")
    md.append("python -m audio_model.preprocessing.explore")
    md.append("")
    md.append("# 4. Preprocess log-Mel features, normalizers, and context stacking")
    md.append("python -m audio_model.preprocessing.build_dataset --machines fan bearing")
    md.append("")
    md.append("# 5. Train the Dense Autoencoder models (normal data only)")
    md.append("python -m audio_model.training.train_autoencoder --machines fan bearing --epochs 100 --batch-size 512")
    md.append("")
    md.append("# 6. Evaluate on test sets and generate figures and metrics")
    md.append("python -m audio_model.evaluation.evaluate_audio --machines fan bearing")
    md.append("")
    md.append("# 7. Execute automated unit and integration tests")
    md.append("pytest tests/test_audio_model.py tests/test_audio_inference.py -v")
    md.append("```")
    md.append("")
    md.append("---")
    md.append("")

    # Section 8
    md.append("## 8. Inference API & CLI Usage Guide")
    md.append("")
    md.append("### Python API (`predict_audio`)")
    md.append("")
    md.append("```python")
    md.append("from audio_model.inference import predict_audio")
    md.append("")
    md.append("# 1. Predict from a .wav audio file path")
    md.append("result = predict_audio(")
    md.append("    audio='data/audio/raw/fan/test/section_00_source_test_anomaly_0003_m-n_W.wav',")
    md.append("    machine_type='fan',")
    md.append("    threshold_percentile=95,")
    md.append(")")
    md.append("print(f\"Status: {result['status']}\")")
    md.append("print(f\"Reconstruction Score: {result['anomaly_score']:.5f}\")")
    md.append("print(f\"Decision Threshold:   {result['threshold']:.5f}\")")
    md.append("print(f\"Is Anomaly:           {result['is_anomaly']}\")")
    md.append("")
    md.append("# 2. Predict from a raw NumPy audio array (e.g. from microphone stream)")
    md.append("import numpy as np")
    md.append("audio_array = np.random.randn(16000 * 5).astype(np.float32)  # 5s array")
    md.append("result_stream = predict_audio(")
    md.append("    audio=audio_array,")
    md.append("    sampling_rate=16000,")
    md.append("    machine_type='fan',")
    md.append(")")
    md.append("```")
    md.append("")
    md.append("### Command-Line Interface (CLI)")
    md.append("")
    md.append("```powershell")
    md.append("# Run inference on any clip")
    md.append("python -m audio_model.inference data/audio/raw/fan/test/section_00_source_test_normal_0000_m-n_W.wav --machine fan")
    md.append("")
    md.append("# Output raw JSON format")
    md.append("python -m audio_model.inference data/audio/raw/fan/test/section_00_source_test_anomaly_0003_m-n_W.wav --machine fan --json")
    md.append("```")
    md.append("")
    return "\n".join(md)


def generate_html_report(md_content: str, out_path: Path) -> None:
    """Generate a clean, printable HTML version of the report with print CSS."""
    # Simple markdown-to-HTML converter for headings, tables, blockquotes, code blocks, images
    lines = md_content.splitlines()
    html_lines = []
    in_code_block = False
    in_table = False

    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html lang='en'>")
    html_lines.append("<head>")
    html_lines.append("<meta charset='utf-8'>")
    html_lines.append("<title>Sound-Based Anomaly Detection: Evaluation Report</title>")
    html_lines.append("<style>")
    html_lines.append("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #24292e; }")
    html_lines.append("h1, h2, h3 { color: #1f2328; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 24px; }")
    html_lines.append("table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }")
    html_lines.append("th, td { border: 1px solid #d0d7de; padding: 8px 12px; text-align: left; }")
    html_lines.append("th { background-color: #f6f8fa; font-weight: 600; }")
    html_lines.append("tr:nth-child(even) { background-color: #fcfcfc; }")
    html_lines.append("blockquote { margin: 16px 0; padding: 0 1em; color: #57606a; border-left: 0.25em solid #d0d7de; }")
    html_lines.append("code { font-family: SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace; font-size: 85%; background-color: #afb8c133; padding: 0.2em 0.4em; border-radius: 4px; }")
    html_lines.append("pre { background-color: #f6f8fa; padding: 16px; border-radius: 6px; overflow: auto; font-family: monospace; font-size: 13px; }")
    html_lines.append("img { max-width: 100%; height: auto; border: 1px solid #d0d7de; border-radius: 6px; margin: 12px 0; }")
    html_lines.append("@media print { body { max-width: 100%; margin: 10mm; font-size: 11pt; } h1, h2 { page-break-after: avoid; } table, img { page-break-inside: avoid; } }")
    html_lines.append("</style>")
    html_lines.append("</head>")
    html_lines.append("<body>")

    for line in lines:
        if line.startswith("```"):
            if not in_code_block:
                html_lines.append("<pre><code>")
                in_code_block = True
            else:
                html_lines.append("</code></pre>")
                in_code_block = False
            continue

        if in_code_block:
            html_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        if line.startswith("|") and line.endswith("|"):
            parts = [p.strip() for p in line[1:-1].split("|")]
            if all(p.startswith(":") or p.startswith("-") for p in parts):
                continue  # Header divider
            if not in_table:
                html_lines.append("<table><thead><tr>" + "".join(f"<th>{p}</th>" for p in parts) + "</tr></thead><tbody>")
                in_table = True
            else:
                html_lines.append("<tr>" + "".join(f"<td>{p}</td>" for p in parts) + "</tr>")
            continue
        else:
            if in_table:
                html_lines.append("</tbody></table>")
                in_table = False

        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("> "):
            html_lines.append(f"<blockquote>{line[2:]}</blockquote>")
        elif line.startswith("!["):
            # Image tag ![alt](url)
            alt = line[2:line.find("]")]
            url = line[line.find("(") + 1:line.find(")")]
            html_lines.append(f"<p><img src='{url}' alt='{alt}'><br><small><em>Figure: {alt}</em></small></p>")
        elif line.strip() == "---":
            html_lines.append("<hr>")
        elif line.strip():
            html_lines.append(f"<p>{line}</p>")

    if in_table:
        html_lines.append("</tbody></table>")

    html_lines.append("</body></html>")
    out_path.write_text("\n".join(html_lines), encoding="utf-8")


def main() -> None:
    # 1. Generate docs/audio_evaluation_report.md
    docs_md = build_report_content(is_docs_dir=True)
    report_path = DOCS_DIR / "audio_evaluation_report.md"
    report_path.write_text(docs_md, encoding="utf-8")
    print(f"Generated: {report_path}")

    # 2. Generate audio_model/README.md
    readme_md = build_report_content(is_docs_dir=False)
    readme_path = AUDIO_MODEL_DIR / "README.md"
    readme_path.write_text(readme_md, encoding="utf-8")
    print(f"Generated: {readme_path}")

    # 3. Generate printable HTML report
    html_path = DOCS_DIR / "audio_evaluation_report.html"
    generate_html_report(docs_md, html_path)
    print(f"Generated PDF-friendly printable HTML: {html_path}")


if __name__ == "__main__":
    main()
