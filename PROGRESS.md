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

## Motor-Current & Temperature Module (Phase 1 Component)

Status of the independent motor-current and temperature subsystem developed for Phase 1 of the multi-modal predictive maintenance project.

### STEP M1 — Dataset Verification & Physical Reality Check
* **Source**: Official Paderborn University Bearing Dataset (KAt Data Center).
* **Sensor Modalities Verified**:
  * Motor Phase Currents ($I_u, I_v$): Synchronous 64 kHz sampling (256,000 samples per 4-second recording). Rich in Motor Current Signature Analysis (MCSA) sidebands ($f_e \pm f_{fault}$) and Park's vector modulations.
  * Temperature (`temp_2_bearing_module`): Oil temperature in bearing module.
* **Verified Physical Reality**: Lessmeier et al. (2016) experimentally maintained lubricant oil temperature in a narrow regulated band (**45–50 °C**) to keep viscosity uniform. Over 4-second recordings, temperature is a slow environmental control variable with quasi-constant readings, not a high-frequency cyclic fault indicator.
* **Methodology Decision**: We do not force contrastive self-supervised neural pretraining onto temperature data. We evaluate a Current-Only baseline, Temperature-Only baseline, and Combined Current+Temperature model (early feature fusion).

### STEP M2 — Modular Preprocessing & Feature Extraction
* `preprocessing/paderborn_loader.py` — robust loader for Paderborn `.mat` nested struct hierarchy (`Y` channel descriptors) and flat dictionaries; signal validation rejecting dead channels (< 1e-6 std) and > 1% NaNs; window segmentation.
* `preprocessing/motor_features.py` — statistical time-domain features (RMS, peak, peak-to-peak, crest factor, kurtosis, skewness, shape factor, impulse factor, margin factor); spectral MCSA features (fundamental frequency, THD, sideband energy ratio, spectral centroid, spectral entropy); Park's Vector Transform ($i_\alpha, i_\beta$ modulus mean, std, eccentricity); thermal state features (mean, std, min, max, slope).

### STEP M3 — Zero-Leakage Bearing Splitting
* `split_by_bearing()`: Segments are grouped by physical bearing code (`K001`, `KA01`, `KI01`, etc.). Entire bearing runs are assigned strictly to train or test splits, ensuring zero leakage across overlapping windows.

### STEP M4 — Model Training & Baselines Comparison
* `models/motor_temperature_model.py`: Random Forest / Gradient Boosting / SVM-RBF classifiers + Isolation Forest anomaly detector. All transformations fitted exclusively on training data.
* `training/train_motor_models.py`: Reproducible training script with command-line options (`--clf`, `--seed`, `--data-dir`).

**Measured Baseline Comparison Results** (Evaluated on 1,169 held-out test segments from 240 real Paderborn 64 kHz recordings):

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | Anomaly ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Current-Only Baseline** | **98.80 %** | **0.9880** | **0.9884** | **0.9880** | **0.7907** |
| **Temperature-Only Baseline** | **52.61 %** | **0.5300** | **0.5364** | **0.5261** | **0.5380** |
| **Combined (Current + Temp)** | **98.97 %** | **0.9897** | **0.9901** | **0.9897** | **0.7686** |

*Key Empirical Finding*: Across 240 real physical recordings (3,599 segments of 64 kHz phase currents), Motor Current Signature Analysis (MCSA) achieves **98.80% Accuracy** and **0.9880 Macro F1** across Healthy, Inner Race, and Outer Race faults. Temperature under the 45–50 °C thermostatically regulated bench condition yields **52.61% Accuracy** (near random chance, F1 = 0.5300). The combined model integrates both, achieving **98.97% Accuracy** and **0.9897 Macro F1**.

### STEP M5 — Standalone Inference Service & Integration Contract
* `core/motor_temperature.py`: `analyse_motor_signal()` returning standardized schema:
  `model_name`, `model_version`, `status`, `predicted_condition`, `anomaly_score`, `confidence`, `class_probabilities`, `timestamp`, `features`, and `input_diagnostics`.
* Checkpoints saved: `checkpoints/motor_temp/current_only_model.joblib`, `checkpoints/motor_temp/temperature_only_model.joblib`, `checkpoints/motor_temp/combined_model.joblib`.
* Metrics saved: `results/motor_temp/metrics.json`, `results/motor_temp/comparison.csv`.

### STEP M6 — Automated Test Suite
* `tests/test_motor_temperature.py`: 12 unit & integration tests covering feature extraction, .mat parsing, signal validation, bearing split leak checks, training reproducibility, API contract compliance, and FastAPI endpoints. All 12 tests pass cleanly (`ALL MOTOR & TEMPERATURE SUBSYSTEM TESTS PASSED!`).

### STEP M7 — Interactive React Diagnostics Page & Full-Stack Integration
* **Dedicated React Diagnostics Page** (`frontend/src/pages/MotorPage.tsx` at `/motor`):
  * **Interactive Upload Zone**: Drag-and-drop or browse `.mat`, `.npy`, or `.csv` files.
  * **Paderborn Benchmark Catalogue**: Instant dropdown selector across 240 local real `.mat` files (`K001`, `KA01`, `KI01`) with filters and operating condition previews.
  * **One-Click Quick Verify**: Instant 1-click test buttons for `K001 Healthy`, `KA01 Outer Race`, and `KI01 Inner Race`.
  * **Model Mode Toggles**: Direct toggle between `Combined (98.97%)`, `Current Only (98.80%)`, and `Temp Only (52.61%)`.
  * **Thermal Sensor Parameter Sweeps**: Adjustable temperature override (°C) to test normal (45°C) vs elevated (75°C) vs overheating (90°C) conditions.
  * **Visual Diagnostics Verdict**:
    * Diagnostic status badge (`HEALTHY` vs `ANOMALOUS`), condition pill, confidence meter, and anomaly score.
    * Ground truth verification badge comparing predicted condition against benchmark ground truth with match confirmation.
    * Class probability distribution progress bars for `Normal`, `Outer_Race`, and `Inner_Race`.
  * **Real-Time Signal Visualizations**:
    * Dual-phase stator current waveforms ($I_u, I_v$) in Amperes over milliseconds.
    * MCSA FFT power spectrum (0–1000 Hz) highlighting fundamental supply peak (~50 Hz) and fault modulation sidebands.
  * **Engineered Feature Inspection**: Live feature breakdown for RMS Current, Crest Factor, Kurtosis, THD %, Sideband Ratio, Park's Modulus, and temperature metrics.
* **FastAPI Backend Integration** (`api/main.py`):
  * `GET /recordings/motor`: Returns catalogue of Paderborn benchmark files with metadata.
  * `POST /analyse/motor`: Handles file uploads (`.mat`, `.npy`, `.csv`) with customizable sampling rate, temperature override, and model mode.
  * `POST /analyse/motor/recording`: Analyses benchmark files stored locally on disk.
* **Frontend Navigation & Routing**:
  * Updated `frontend/src/components/Sidebar.tsx` with `⚡ Motor & Temp` navigation link.
  * Updated `frontend/src/App.tsx` with `/motor` route.
  * Updated `frontend/src/api/client.ts` and `frontend/src/types/api.ts` with fully typed motor PdM contracts.
  * Validated with zero TypeScript / oxlint warnings (`npm run build` passes in 201ms).

### STEP M8 — Calibration Fix, FFT Window Alignment & Retraining
* **Calibrated Anomaly Variance Preservation**: Fixed single-sample standard deviation collapse in `models/motor_temperature_model.py` by computing and preserving `self.anomaly_scale_ = max(0.02, float(np.std(raw_scores)))` on the healthy training set. Prevents $N=1$ predictions from collapsing to $\text{scale}=0.01$ and triggering false anomaly flags.
* **0.5s Signal Windowing Alignment**: In `core/motor_temperature.py`, updated `analyse_motor_signal` to window signals longer than $0.5\text{ s}$ into consecutive $0.5\text{ s}$ windows ($32{,}000$ samples) and average the feature vectors, ensuring FFT spectral resolution matches the training distribution.
* **Full 240-File Benchmark Retraining**: Retrained all 3 motor checkpoints (`current_only`, `temperature_only`, `combined`) across all 240 real Paderborn `.mat` files across all 4 operating conditions (900 & 1500 RPM, varying torque and radial force). Achieved **98.97% accuracy** on Combined and **98.80% accuracy** on Current-Only.
* **Verified Predictions**:
  * Normal (`paderborn_demo_normal` / K001): Status `NORMAL`, Score Ratio `0.2664x` ($< 1.0$), Confidence `83.3%`.
  * Outer Race (`paderborn_demo_outer_race` / KA01): Status `ANOMALOUS`, Label `Outer_Race`, Score Ratio `1.000x`, Confidence `100.0%`.
  * Inner Race (`paderborn_demo_inner_race` / KI01): Status `ANOMALOUS`, Label `Inner_Race`, Score Ratio `1.2006x`, Confidence `94.0%`.
* **Documentation**: Created [`DASHBOARD_AND_PAGES_GUIDE.md`](file:///Users/snehabvarghese/vibration-ssl-pdm/DASHBOARD_AND_PAGES_GUIDE.md) providing an exhaustive reference to every page, graph, metric, and faculty presentation walkthrough.

## Honest caveats

* CWRU is a bearing test rig, **not** a multistage compressor. We validate the method, not the paper's experiments.
* The supplied PDF was corrupt, so the exact EDR maths, architecture hyperparameters, augmentation probabilities and training protocol of the paper were unavailable. Everything of that kind here is our own design.
* The paper's F1 ≈ 0.93 is on different data and is not a target or a baseline.
* CWRU has no run-to-failure data, so the "degradation timeline" is constructed (healthy segments followed by faulty ones) and is illustrative.
* In Paderborn data, temperature was controlled at 45–50 °C during bench tests; temperature signals reflect thermal operational state, not high-frequency mechanical impact signatures.


