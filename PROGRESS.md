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
* `models/encoder.py` — 1-D CNN, 4 conv blocks `[32,64,128,128]`, kernels
  `[15,9,7,5]`, GAP → 128-D. 175 392 parameters.
* `models/siamese.py` — shared encoder + 2-layer projection head + `NTXentLoss`.
* `training/splits.py` — record-level / load-based / (unsafe) random-segment
  splits + `assert_no_leakage()`.
* `training/train_ssl.py` — Adam + cosine schedule, best-val checkpointing.

Measured (50 epochs, CPU, 1 765 s): best validation NT-Xent 3.5926.
**No labels are used anywhere in this stage.**

### STEP 6–7 — Embeddings + visualization
* `training/extract_embeddings.py` — projection head bypassed, embeddings cached
  to `results/embeddings/embeddings.npz` with labels/records/loads/split.
* `evaluation/visualization.py` — t-SNE + UMAP, silhouette score.

Measured: (5 066, 128) embeddings; silhouette vs true classes 0.334.

### STEP 8 — Anomaly detection / EDR
* `anomaly_detection/edr.py` — Gaussian and histogram KL estimators, PCA(8),
  `EDR = KL(current ‖ healthy) / window duration` in nats/s, 3σ threshold.
  Documented as **our approximation**, not the paper's formulation.
* `anomaly_detection/anomaly_detector.py` — health monitor, constructed
  degradation timeline, per-class EDR, segment-level Mahalanobis score, saves a
  fitted monitor for the dashboard.

Protocol fix worth explaining in the viva: the first run flagged **100 % of
healthy windows**. Cause — CWRU has only four Normal recordings (one per load),
so a record-level split made the baseline judge unseen healthy *loads*.
Fixed by splitting each Normal recording chronologically
(50 % commissioning / 25 % calibration / 25 % held-out, one-segment gaps), which
is also what deployment looks like.

Measured after the fix: threshold 1.75 nats/s, 0/10 healthy windows flagged,
18/18 faulty windows flagged, segment ROC-AUC 1.000, TPR 1.000, FPR 0.029.

### STEP 9 — Limited-label classification
`models/classifier.py` (SVM-RBF, MLP) + `training/train_classifier.py`
(stratified subsets at 1/5/10/20 %, repeated, mean ± std).

Measured on frozen embeddings, record-level test set (1 238 segments, 16 classes):

| labels | SVM-RBF acc | SVM-RBF F1 | MLP acc | MLP F1 |
|---|---|---|---|---|
| 1 % (32) | 0.669 ± 0.043 | 0.592 | 0.668 ± 0.051 | 0.590 |
| 5 % (128) | 0.676 ± 0.040 | 0.618 | 0.684 ± 0.023 | 0.630 |
| 10 % (256) | 0.713 ± 0.022 | 0.674 | 0.701 ± 0.013 | 0.650 |
| 20 % (512) | 0.701 ± 0.030 | 0.656 | 0.705 ± 0.015 | 0.651 |

### STEP 10 — Supervised baseline
`models/baseline_cnn.py` + `training/train_baseline.py`: same backbone, same
splits, same label subsets, trained from scratch.

A first version let the baseline pick its best epoch on the **full labelled
validation split** (~1 274 labels) while being advertised as "32 labels". That
is an unfair advantage, so the epoch is now selected on a 20 % holdout carved
out of the labelled subset itself. Re-run in progress; the honest comparison
goes into `results/metrics/comparison.*`.

### STEP 12 — Dashboard
`dashboard/app.py` (Streamlit): status NORMAL/ANOMALOUS, predicted fault +
confidence, EDR vs threshold, % segments flagged, waveform, anomaly trend,
embedding map with the current signal marked, class probabilities. Works on
dataset recordings or uploaded `.mat`/`.npy`. `analyse_signal()` is the single
Phase 2 hook. Not yet run in a browser.

### STEP 13 — Documentation
`README.md` (overview → Phase 2) and `DESIGN.md` (architecture, roadmap, what
the paper does not give us).

## Remaining

1. Finish the fair baseline re-run and `evaluation/evaluate.py` comparison.
2. A longer SSL pretrain (150 epochs) is running — the 50-epoch validation loss
   was still falling, and the SSL arm currently trails the supervised baseline.
   Downstream stages will be re-run on whichever encoder is better.
3. Smoke tests (encoder shapes, NT-Xent, leakage assertion, EDR, loader) and a
   lint config.
4. Notebooks (`notebooks/01..04`) — currently only the module entry points exist.
5. Browser test of the dashboard.

## Honest caveats

* CWRU is a bearing test rig, **not** a multistage compressor. We validate the
  method, not the paper's experiments.
* The supplied PDF was corrupt, so the exact EDR maths, architecture
  hyperparameters, augmentation probabilities and training protocol of the paper
  were unavailable. Everything of that kind here is our own design.
* The paper's F1 ≈ 0.93 is on different data and is not a target or a baseline.
* CWRU has no run-to-failure data, so the "degradation timeline" is constructed
  (healthy segments followed by faulty ones) and is illustrative.
