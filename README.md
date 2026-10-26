# Self-Supervised Vibration Analytics for Predictive Maintenance — Phase 1

Final-year B.Tech project. Phase 1 builds and validates the **AI/ML software pipeline**
on a public vibration dataset. Physical sensors, ESP32, MQTT and real-time acquisition
are **Phase 2** and are deliberately not implemented here.

Inspired by *"Self-Supervised Vibration Analytics for Predictive Maintenance of
Multistage Compressors"*. We adapt its ideas (contrastive Siamese 1-D CNN, temporal
augmentations, learned embeddings, an entropy-divergence anomaly metric, a lightweight
downstream classifier); we do **not** reproduce it — see
[What we could not reproduce](#what-we-could-not-reproduce).

---

## 1. Overview

Industry has lots of unlabelled vibration data and very few labelled faults. So we

1. learn representations from vibration segments **without labels** (contrastive SSL),
2. detect abnormal behaviour by comparing embedding **distributions** to a healthy
   baseline (our EDR approximation),
3. classify faults from those frozen embeddings with **1–20 % of the labels**,
4. compare against a supervised CNN trained on the same label budget,
5. serve everything through a REST API, a React web app and the original Streamlit dashboard.

## 2. Architecture

```
CWRU .mat recording
   ↓  loader.py           channel select, 48 kHz → 12 kHz, NaN handling
   ↓  filtering.py        Butterworth band-pass 10–5000 Hz (+ optional decimation)
   ↓  segmentation.py     0.25 s windows, 50 % overlap, per-segment z-score
   ↓  augmentation.py     jitter / scaling / time-shift / masking / permutation
   ↓                      → two views of the same segment = a positive pair
   ↓  encoder.py          1-D CNN  (4 conv blocks, 175 k params) → 128-D embedding
   ↓  siamese.py          shared encoder + projection head, NT-Xent loss   ← NO LABELS
   ↓
   ├─ edr.py / anomaly_detector.py   KL(current ‖ healthy) per second → NORMAL/ANOMALOUS
   ├─ classifier.py                  SVM-RBF / MLP on frozen embeddings (few labels)
   ├─ baseline_cnn.py                supervised CNN from scratch (same label budget)
   └─ core/analysis.py               Streamlit-free inference: raw signal → verdict
        ├─ api/main.py               FastAPI service (JSON)
        │    └─ frontend/            React + TypeScript web app
        └─ dashboard/app.py          Streamlit monitor (original UI)
```

The projection head exists only during pretraining; embeddings always come from the
encoder.

## 3. Dataset

**Case Western Reserve University bearing dataset** — chosen because the reference
paper uses it as its public benchmark and it is the standard vibration PdM benchmark.
Note it is a *bearing test rig*, **not** a multistage compressor: we validate the
*method*, not the paper's compressor results.

What is actually loaded (measured, from `results/metrics/dataset_summary.json`):

| | |
|---|---|
| Records | 64 (0 unreadable) |
| Classes | 16 — real CWRU labels, nothing invented |
| Total signal | 644.8 s |
| Sampling rate | 12 kHz (Normal files are natively 48 kHz → resampled) |
| Segments | 5 066 × 3 000 samples (0.25 s) |

Classes: `Normal`, `IR007/014/021/028`, `B007/014/021/028`,
`OR007@3/@6/@12`, `OR014@6`, `OR021@3/@6/@12` (drive-end channel, loads 0–3 hp).

Download (≈ 100 MB, not committed):

```bash
python -m preprocessing.download_cwru --all
```

Swapping datasets means replacing `preprocessing/cwru_manifest.py` + `loader.py`;
everything downstream only sees `(segments, labels, record_ids)`.

## 4. Installation

## 4. Installation

Requirements: **Python 3.9–3.11** (developed on 3.10; 3.12 is not supported by the
pinned ML packages) and **Node.js 18+** for the web app.

```bash
# Linux / macOS
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows PowerShell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CUDA/MPS are used automatically when available (`config.get_device()`); CPU is enough.

The raw dataset (≈ 100 MB) is not committed. Download it once:

```bash
python -m preprocessing.download_cwru --all
```

The trained model (`checkpoints/ssl_encoder.pt`, `checkpoints/health_monitor.joblib`)
and `results/embeddings/embeddings.npz` **are committed**, so the web app runs
without retraining.

## 5. Running the web app

Two terminals, both from the repo root.

**Backend (FastAPI)**

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
python -m uvicorn api.main:app --reload --port 8000
```

```bash
# Linux / macOS
source .venv/bin/activate
export PYTHONPATH=.
python -m uvicorn api.main:app --reload --port 8000
```

Health check: http://127.0.0.1:8000/health (should report `"model_loaded": true`);
interactive API docs: http://127.0.0.1:8000/docs.

**Frontend (React + Vite)**

```bash
cd frontend
npm install        # first time only
npm run dev
```

Open http://localhost:5173. If the backend runs on another port, set
`VITE_API_URL` in `frontend/.env`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | service and model status |
| `POST /analyse` | upload `.mat` / `.npy` / `.csv` + sampling rate → verdict, waveform, EDR trend, embedding points |
| `GET /recordings` | CWRU recordings available in `data/raw/` |
| `POST /analyse/recording/{file_id}` | analyse a dataset recording |
| `GET /map` | background embedding map (PCA) |

| Page | Purpose |
|---|---|
| `/analyse` | pick a dataset recording or upload a file; status, predicted fault, EDR, charts |
| `/history` | analyses from this browser session (stored in localStorage) |
| `/live` | Phase 2 placeholder (no streaming yet) |

Run the tests with `pytest -q`.

## 6. Running the pipeline

```bash
export PYTHONPATH=.

python -m preprocessing.download_cwru --all     # 1. raw data
python -m preprocessing.explore                 # 1. dataset report + figures
python -m preprocessing.demo                    # 2-3. preprocessing/augmentation demo
python -m preprocessing.build_dataset           # 2-3. → data/processed/segments.npz

python -m training.train_ssl                    # 5. contrastive pretraining (no labels)
python -m training.extract_embeddings           # 6. → results/embeddings/embeddings.npz
python -m evaluation.visualization              # 7. t-SNE + UMAP
python -m anomaly_detection.anomaly_detector    # 8. EDR anomaly detection
python -m training.train_classifier             # 9. SVM/MLP at 1/5/10/20 % labels
python -m training.train_baseline               # 10. supervised CNN baseline
python -m evaluation.evaluate                   # 11. comparison table + figure

streamlit run dashboard/app.py                  # 12. dashboard
```
On Windows PowerShell, replace `export PYTHONPATH=.` with
`$env:PYTHONPATH = (Get-Location).Path`. Retraining is optional because the trained model
is committed; run it only if you change the data or model.

**Reproducibility:** results from retraining vary slightly between runs and hardware
(GPU vs CPU). The committed `checkpoints/` and `results/` come from one canonical
run, so the numbers below match the committed model. If you retrain, expect small
differences and do not mix new metrics with the committed checkpoints.

Every stage caches its artefacts, so stages can be re-run independently.
All randomness goes through `config.set_seed(42)`.

## 7. Avoiding leakage

Windows overlap by 50 %, so a random segment split would put near-duplicates in both
train and test and inflate every number. Instead:

* **Classification** uses a **record-level** split (`training/splits.py`): all segments
  of a `.mat` file stay in one split, so test records — and their motor loads — are
  unseen. `assert_no_leakage()` enforces this.
* **Anomaly detection** cannot use that split for its healthy baseline: CWRU has only
  four Normal recordings, one per load, so a record-level split would force the
  baseline to call unseen *healthy* loads anomalous (our first run did exactly that —
  100 % false alarms). Instead each Normal recording is split **chronologically** into
  50 % commissioning (fit) / 25 % calibration / 25 % held-out, with a one-segment gap
  at each boundary. This mirrors deployment: the baseline is learned from the machine's
  own healthy commissioning period. Faults still come from unseen test records.

## 8. Results (measured on this repo — not copied from the paper)

Recreate with `./run_phase1.sh`; raw numbers live in `results/metrics/`.

**Anomaly detection** (`results/metrics/anomaly_detection.json`)

| Metric | Value |
|---|---|
| EDR threshold (healthy mean + 3σ) | 1.20 nats/s |
| Held-out healthy windows flagged | 2 / 10 |
| Faulty windows flagged | 18 / 18 |
| Segment-level ROC-AUC (fault vs healthy) | 0.9998 |
| Segment-level TPR / FPR | 0.988 / 0.000 |

Healthy baseline: 132 segments fit, 66 calibration, 70 held out; 1,199 faulty segments
evaluated. Mean EDR of the unseen Normal test class is 1.06 nats/s, below the
threshold, while every fault class scores between 6.2 and 55.6 nats/s.

**Fault classification** (`results/metrics/comparison.csv`), accuracy (mean ± std), record-level test set, 16 classes:

| Label budget | N labelled | SSL frozen MLP | SSL frozen SVM | SSL fine-tuned | Supervised CNN |
|---|---|---|---|---|---|
| 1 % | 32 | 63.4 ± 8.8 % | 64.0 ± 1.7 % | 78.9 ± 1.3 % | 83.8 ± 2.6 % |
| 5 % | 128 | 64.6 ± 3.3 % | 61.5 ± 0.7 % | 85.9 ± 1.0 % | 87.1 ± 1.0 % |
| 10 % | 256 | 66.5 ± 0.6 % | 61.9 ± 1.3 % | 87.3 ± 0.3 % | 87.9 ± 0.3 % |
| 20 % | 512 | 68.3 ± 2.0 % | 63.7 ± 1.3 % | 87.4 ± 0.3 % | 88.8 ± 1.5 % |

**Figures** (`results/figures/`): class distribution, waveforms, spectra, preprocessing stages, augmented views, SSL loss curve, t-SNE, UMAP, EDR trend, EDR per class, label efficiency, confusion matrices, SSL-vs-baseline.

## 9. EDR — what it is here

For an embedding window `W` and the healthy reference `H`, both projected to 8 PCA
components and modelled as regularised Gaussians:

```
EDR(W) = KL( N(μ_W, Σ_W) ‖ N(μ_H, Σ_H) ) / duration(W)      [nats / second]
```

Dividing by the window duration makes it a *rate*, so it is comparable across window
lengths. The threshold is the mean + 3σ of EDR over healthy calibration windows. A
histogram estimator is available (`--estimator histogram`), and a per-segment squared
Mahalanobis distance gives the dashboard a single-window score.

This is **our approximation** of the paper's metric. The paper's exact derivation was
not recoverable (below), so no equivalence is claimed.

## 10. Dashboard


The main UI is the React app (section 5). The original Streamlit monitor still works:

```bash
streamlit run dashboard/app.py
```

Both call the same inference core, `core/analysis.py`.

Pick a dataset recording or upload a `.mat`/`.npy` file; it shows status
(NORMAL/ANOMALOUS), predicted fault + confidence, EDR vs threshold, the fraction of
flagged segments, the waveform, the anomaly trend, the embedding map with the current
signal marked, and class probabilities.

`analyse_signal(signal, sampling_rate)` in `core/analysis.py` is the single Phase 2
hook: an MQTT/serial reader just calls it with a live buffer.

## 11. What we could not reproduce

The PDF supplied to us was corrupt (no extractor or renderer recovered its body text),
so only the public abstract-level description was available. Unavailable, therefore
designed by us and labelled as ours:

* the exact EDR formulation, estimator and thresholding rule;
* encoder depth/widths, projection dimensions, optimiser schedule;
* augmentation types and probabilities;
* the in-house multistage-compressor dataset (not public — we use CWRU only);
* the paper's exact train/test protocol, so its F1 ≈ 0.93 is not comparable to ours.

## 12. Phase 2 (not implemented)

```
Machine → accelerometer → ESP32 → Wi-Fi/MQTT → broker → inference service
        → Phase 1 encoder + EDR monitor + classifier → dashboard → alerts
```

The Phase 1 boundary is `analyse_signal()` plus `preprocessing/loader.py`: replace the
file reader with a stream reader and the rest is unchanged. The React `/live` page and
the `LiveReading` TypeScript interface (`frontend/src/types/api.ts`) define the intended
message contract; streaming (WebSocket) is not implemented yet.

**Known limits for real deployment:**

* The encoder expects 12 kHz data. A sensor with lower bandwidth will not capture the
  fault signatures, and upsampling cannot recover them.
* The classifier knows only the 16 CWRU classes. On a new machine, the EDR anomaly
  score (with a healthy baseline fitted on that machine) is the part expected to
  transfer; fault labels are not validated outside CWRU.

## 13. Repository layout

```
config.py                 all hyperparameters + seeds + device selection
preprocessing/            manifest, download, loader, filtering, segmentation,
                          augmentation, build_dataset, explore, demo,
                          paderborn_loader, motor_features
models/                   encoder, siamese (+NT-Xent), classifier, baseline_cnn,
                          motor_temperature_model
training/                 splits, datasets, train_ssl, extract_embeddings,
                          train_classifier, train_baseline, train_motor_models
anomaly_detection/        edr, anomaly_detector
evaluation/               metrics, visualization, evaluate
core/analysis.py          framework-free vibration inference
core/motor_temperature.py framework-free motor current & temp inference
api/main.py               FastAPI service
frontend/                 React + TypeScript web app
dashboard/app.py          Streamlit monitor
tests/                    test_pipeline.py, test_api_and_core.py, test_motor_temperature.py
checkpoints/              trained vibration & motor checkpoints
results/                  metrics and comparison tables
DESIGN.md                 architecture, roadmap, paper gaps
```

---

## 14. Motor-Current & Temperature Component (Phase 1)

This module develops, evaluates, and packages the motor-current and temperature predictive maintenance subsystem for Phase 1. It operates independently from the vibration pipeline and teammate sound model, ready for final integration.

### Dataset Verification & Preprocessing
* **Benchmark Source**: [Paderborn University Bearing Dataset](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download) (Chair of Design and Drive Technology - KAt).
* **Sensor Modalities**:
  * Motor Currents: Phase currents $I_u$ (`phase_current_1`) and $I_v$ (`phase_current_2`) recorded via LEM CKSR transducers at **64 kHz** (256,000 samples per 4-second recording).
  * Temperature: Bearing module oil temperature (`temp_2_bearing_module`).
* **Verified Physical Reality**: Lessmeier et al. (2016) experimentally kept oil temperature within a narrow thermostatically regulated band (**45–50 °C**) to eliminate lubricant viscosity drift. Over 4-second benchmark recordings, temperature is a quasi-constant operational context variable, not a high-frequency cyclic fault indicator. We therefore do not force contrastive self-supervised neural learning onto temperature data.
* **Feature Extraction** (`preprocessing/motor_features.py`):
  * **Time-domain**: RMS, peak, peak-to-peak, crest factor, kurtosis, skewness, shape factor, impulse factor, margin factor.
  * **Spectral MCSA**: Fundamental electrical frequency tracking (~50 Hz), THD (2nd–5th harmonics), fault sideband energy ratio ($f_e \pm f_{fault}$), spectral centroid, spectral entropy.
  * **Park's Vector Transform**: $i_\alpha, i_\beta$ modulus mean, standard deviation, peak-to-peak, and eccentricity.
  * **Thermal State**: Mean temperature, min, max, range, linear drift slope (°C/s).

### Experiments & Measured Baselines Comparison
All models are evaluated on 1,169 held-out test segments from 240 real Paderborn recordings (3,599 total segments) across all motor operating conditions:

| Model | Input Modality | Accuracy | Macro F1 | Macro Precision | Macro Recall | Anomaly ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Current-Only Baseline** | $I_u, I_v$ MCSA features | **98.80 %** | **0.9880** | **0.9884** | **0.9880** | **0.7907** |
| **Temperature-Only Baseline** | Thermal state (°C) | **52.61 %** | **0.5300** | **0.5364** | **0.5261** | **0.5380** |
| **Combined Model** | Early Feature Fusion | **98.97 %** | **0.9897** | **0.9901** | **0.9897** | **0.7686** |

*Takeaway*: Across 240 real Paderborn University 64 kHz recordings, Motor Current Signature Analysis captures mechanical fault dynamics with **98.80% Accuracy**. Temperature under controlled laboratory conditions provides a baseline health check (52.61% accuracy, near-random chance for distinguishing localized micro-flaking) and serves as an operating context gate in the combined model, boosting overall accuracy to **98.97%**.

### Team Integration Contract (`core/motor_temperature.py`)
For the teammate responsible for the final application and API integration, calling the motor model requires a single Python import:

```python
from core.motor_temperature import analyse_motor_signal

result = analyse_motor_signal(
    current_phase1=current_u,          # 1-D numpy array (e.g. 64 kHz or resampled)
    current_phase2=current_v,          # Optional 1-D array
    temperature=47.5,                  # Optional Celsius float (defaults to 45.0)
    sampling_rate=64_000,
    model_mode="combined",             # "combined" | "current_only" | "temperature_only"
    timestamp="2026-10-26T20:00:00Z",  # Source timestamp if present (never fabricated)
)
```

**Returned Output Schema**:
```json
{
  "model_name": "motor_combined_pdm",
  "model_version": "1.0.0",
  "status": "HEALTHY",
  "predicted_condition": "Normal",
  "anomaly_score": 0.0821,
  "confidence": 0.9850,
  "class_probabilities": {
    "Inner_Race": 0.005,
    "Normal": 0.985,
    "Outer_Race": 0.010
  },
  "timestamp": "2026-10-26T20:00:00Z",
  "features": { ... },
  "input_diagnostics": {
    "samples": 32000,
    "duration_seconds": 0.5,
    "sampling_rate_hz": 64000,
    "two_phase_available": true,
    "temperature_input_available": true
  }
}
```

### Web Application & Diagnostics UI
A dedicated **Motor Current & Temperature PdM** page is integrated directly into the React frontend at `/motor`:
* **Upload Custom Signals**: Drag & drop or browse `.mat`, `.npy`, or `.csv` current waveforms with thermal sensor parameters.
* **Instant Benchmark Selection**: Dropdown across 240 local Paderborn recordings (`K001 Healthy`, `KA01 Outer Race`, `KI01 Inner Race`) with condition filters.
* **Quick 1-Click Verification**: Quick-test buttons to immediately diagnose healthy and damaged bearing benchmarks.
* **Live Visualizations**: Dual-trace stator current waveforms ($I_u, I_v$), MCSA power spectrum (0–1000 Hz), class probability distributions, and engineered MCSA feature table.
* **Model Mode Toggles**: Live comparison between Combined (98.97%), Current-Only (98.80%), and Temperature-Only (52.61%).

### Reproducible Commands
* **Run Test Suite** (12 unit/integration tests including FastAPI endpoints):
  ```bash
  ./venv/bin/python tests/test_motor_temperature.py
  ```
* **Run FastAPI Backend Server**:
  ```bash
  ./venv/bin/uvicorn api.main:app --reload --port 8000
  ```
* **Run React Frontend Dev Server**:
  ```bash
  cd frontend && npm run dev
  ```
* **Build Frontend for Production**:
  ```bash
  cd frontend && npm run build
  ```
* **Train and Compare Baselines**:
  ```bash
  ./venv/bin/python training/train_motor_models.py --clf rf --seed 42
  ```
* **Multi-Value Stress Testing & Sweeps**:
  ```bash
  ./venv/bin/python evaluation/evaluate_motor.py --all
  ```
* **Download Real Paderborn Benchmark Archives** (subset of K001, KA01, KI01):
  ```bash
  ./venv/bin/python preprocessing/download_paderborn.py --subset
  ```
* Checkpoints: `checkpoints/motor_temp/` (`combined_model.joblib`, `current_only_model.joblib`, `temperature_only_model.joblib`).
* Metrics & CSVs: `results/motor_temp/metrics.json` and `results/motor_temp/comparison.csv`.