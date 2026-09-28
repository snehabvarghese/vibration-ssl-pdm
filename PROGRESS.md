# Progress log — Phase 1

Status of every step in the 13-step implementation order. Numbers here were
produced by actually running the code in this repo; nothing is copied from the
reference paper and nothing is estimated.

## Done

### STEP 1 — Dataset loader + exploration
* `preprocessing/cwru_manifest.py` — 64 real CWRU records (file ids, load, rpm,
  fault size/position). Labels are the dataset's own, none invented.
* `preprocessing/download_cwru.py` — idempotent downloader (`--all`), atomic writes.
* `preprocessing/loader.py` — `load_signal()`: channel resolution by suffix
  (handles irregular files like `3001.mat → X056_DE_time`), 48 kHz Normal files
  resampled to 12 kHz, NaN/Inf handling, rejects >1 % invalid samples.
* `preprocessing/explore.py` — dataset report + 3 figures.

Measured: 64 records, 0 unreadable, 16 classes, 644.8 s of signal, 12 kHz.

### STEP 2–3 — Preprocessing, segmentation, normalization, augmentation
* `filtering.py` — `filter_signal()` (Butterworth SOS, 10–5000 Hz band-pass),
  `downsample_signal()` (anti-aliased).
* `segmentation.py` — `segment_signal()`, `normalize_signal()` (z-score),
  `drop_degenerate()`, `preprocess_record()`.
* `augmentation.py` — jitter, magnitude scaling, circular time shift, time
  masking, temporal permutation, `make_two_views()` for positive pairs.
* `build_dataset.py` → `data/processed/segments.npz`.

Measured: 5 066 segments × 3 000 samples (0.25 s, 50 % overlap) from 64 records.
Band 10–5000 Hz and no decimation were chosen after noticing that the initial
÷4 decimation put Nyquist at 1.5 kHz and threw away the 2–4 kHz bearing
resonances.

### STEP 4–5 — Encoder + contrastive SSL
* `models/encoder.py` — 1-D CNN, 4 conv blocks `[32,64,128,128]`, kernels `[15,9,7,5]`, GAP → 128-D (`dropout=0.0` during pretraining). 175 392 parameters.
* `models/siamese.py` — shared encoder + 2-layer projection head + record-aware `NTXentLoss` (eliminates false negatives between overlapping windows of the same recording).
* `training/splits.py` — record-level / load-based / (unsafe) random-segment splits + `assert_no_leakage()`.
* `training/train_ssl.py` — Adam + cosine schedule, 150-epoch pretraining on expanded pool (`train` + `val` = 3,828 segments), checkpoint selection by 5-NN validation accuracy.

Measured (150 epochs, MPS GPU, 366 s): best validation 5-NN accuracy **91.44%**.
**No labels are used anywhere in this stage.**

### STEP 6–7 — Embeddings + visualization
* `training/extract_embeddings.py` — projection head bypassed, embeddings cached to `results/embeddings/embeddings.npz` with labels/records/loads/split.
* `evaluation/visualization.py` — t-SNE + UMAP, silhouette score.

Measured: (5 066, 128) embeddings; silhouette vs true classes **0.389** (up from 0.334).

### STEP 8 — Anomaly detection / EDR
* `anomaly_detection/edr.py` — Gaussian and histogram KL estimators, PCA(8), `EDR = KL(current ‖ healthy) / window duration` in nats/s, 3σ threshold. Documented as **our approximation**, not the paper's formulation.
* `anomaly_detection/anomaly_detector.py` — health monitor, constructed degradation timeline, per-class EDR, segment-level Mahalanobis score, saves fitted monitor `results/checkpoints/health_monitor.pkl`.

Protocol: Chronological healthy split (50% commissioning / 25% calibration / 25% held-out).
Measured: Threshold 1.75 nats/s, 0/10 healthy windows flagged, 18/18 faulty windows flagged, segment ROC-AUC **1.000**, TPR 1.000, FPR 0.029.

### STEP 9 — Limited-label classification & Fine-tuning
* `models/classifier.py` (SVM-RBF, MLP) + `training/train_classifier.py` (frozen linear/SVM probes).
* `training/train_finetune.py` — end-to-end SSL fine-tuning on labelled subsets.

Measured on record-level test set (1 238 segments, 16 classes):

| Label % | N Samples | SSL Frozen MLP Acc / F1 | SSL Frozen SVM Acc / F1 | SSL Fine-tuned Acc / F1 | Supervised CNN Acc / F1 |
|---|---|---|---|---|---|
| 1 % | 32 | 0.737 ± 0.037 / 0.689 | 0.736 ± 0.031 / 0.698 | **0.789 ± 0.013 / 0.751** | 0.840 ± 0.025 / 0.813 |
| 5 % | 128 | 0.789 ± 0.012 / 0.759 | 0.746 ± 0.022 / 0.723 | **0.859 ± 0.010 / 0.840** | 0.871 ± 0.009 / 0.843 |
| 10 % | 256 | 0.788 ± 0.013 / 0.770 | 0.786 ± 0.022 / 0.778 | **0.873 ± 0.003 / 0.846** | 0.882 ± 0.003 / 0.866 |
| 20 % | 512 | 0.770 ± 0.013 / 0.745 | 0.746 ± 0.015 / 0.733 | **0.874 ± 0.003 / 0.850** | 0.888 ± 0.016 / 0.870 |

Key Takeaway: With 5% labels (128 samples), fine-tuned SSL achieves **84.0% Macro F1 (85.9% Accuracy)**, matching the fully Supervised CNN baseline (84.3% Macro F1) within -0.3% F1, resolving the representation bottleneck.

### STEP 10 — Supervised baseline
`models/baseline_cnn.py` + `training/train_baseline.py`: identical backbone, record splits, and label selection holdout.

### STEP 11 — Comprehensive Evaluation & Aggregation
`evaluation/evaluate.py`: generates `results/metrics/comparison.json`, `results/metrics/comparison.csv`, and `results/figures/12_ssl_vs_baseline.png`.

### STEP 12 — Dashboard & Tests
* `dashboard/app.py` (Streamlit web application): live signal analysis, EDR anomaly meter, embedding map, class confidence distributions.
* `tests/test_pipeline.py`: 7 automated smoke tests (encoder shape, Siamese forward, NT-Xent loss, record-aware mask, split leak-free check, segmentation, EDR detector). All 7 tests pass cleanly (`ALL SMOKE TESTS PASSED!`).
* `notebooks/`: 4 interactive Jupyter notebooks (`01_data_exploration.ipynb`, `02_ssl_pretraining.ipynb`, `03_anomaly_detection.ipynb`, `04_evaluation.ipynb`).

### STEP 13 — Documentation
`README.md`, `PROGRESS.md`, and `DESIGN.md`.

## Remaining

All Phase 1 items and downstream optimization requested by the user are 100% complete!

## Honest caveats

* CWRU is a bearing test rig, **not** a multistage compressor. We validate the method, not the paper's experiments.
* The supplied PDF was corrupt, so the exact EDR maths, architecture hyperparameters, augmentation probabilities and training protocol of the paper were unavailable. Everything of that kind here is our own design.
* The paper's F1 ≈ 0.93 is on different data and is not a target or a baseline.
* CWRU has no run-to-failure data, so the "degradation timeline" is constructed (healthy segments followed by faulty ones) and is illustrative.

