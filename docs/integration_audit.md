# Multimodal Predictive Maintenance System: Integration Audit

> **Repository**: `vibration-ssl-pdm`  
> **Date**: October 2026  
> **Scope**: Pre-integration audit of the three diagnostic models, FastAPI backend, and React frontend.  
> **Status**: Read-only audit (no application code modified).

---

## Executive Summary

The repository integrates three distinct predictive maintenance (PdM) subsystems designed for rotating machinery condition monitoring:
1. **Vibration Subsystem (CWRU)**: Self-Supervised Learning (SimCLR 1D-CNN) encoder + EDR divergence & Mahalanobis distance anomaly detector + 10-class bearing fault classifier.
2. **Sound Subsystem (MIMII DG / DCASE 2022 Task 2)**: 9-layer symmetric Dense Autoencoder on context-stacked log-Mel spectrograms for unsupervised acoustic anomaly detection.
3. **Motor Subsystem (Paderborn / Teammate)**: Motor Current Signature Analysis (MCSA) with Park's Vector Transform + thermal feature extraction coupled with an IsolationForest anomaly detector and Random Forest fault classifier.

This audit details the state of checkpoints, inference interfaces, data formats, thresholds, existing API endpoints, frontend pages, and architectural blockers prior to building the unified backend fusion engine and multimodal dashboard.

---

## 1. Subsystem Audit: Vibration (CWRU)

### 1.1 Checkpoints & Storage
| File | Size | Tracked in Git? | Status / Description |
| :--- | :---: | :---: | :--- |
| `checkpoints/ssl_encoder.pt` | ~924 KB | **Yes** | Frozen 1D-CNN feature extractor pretrained on CWRU via self-supervised contrastive learning. |
| `checkpoints/health_monitor.joblib` | ~13 KB | **Yes** | Serialized `HealthMonitor` instance containing PCA projection matrix and healthy baseline covariance. |
| `results/embeddings/embeddings.npz` | ~5.7 MB | **Yes** | Precomputed latent embeddings (128-dim) across CWRU splits used to fit downstream classifiers (`svm_rbf`). |
| `checkpoints/baseline_cnn.pt` | ~724 KB | No (gitignored) | Supervised benchmark model (optional baseline, not required for production inference). |

### 1.2 Exact Inference Entry Point
- **Python Interface**: [`core/analysis.py`](file:///c:/Main%20Project%20S7/MainProject/vibration-ssl-pdm/core/analysis.py) -> `analyse_signal(signal: np.ndarray, sampling_rate: int, clf_name: str = "svm_rbf") -> Optional[Dict[str, object]]`
- **API Endpoints**: `POST /analyse` (file upload) and `POST /analyse/recording/{file_id}` (preset CWRU record).

### 1.3 Expected Input Specifications
- **Data Shape**: 1-D numpy array of raw vibration acceleration samples.
- **Sampling Rate**: Native CWRU rate is `12,000 Hz` (or `48,000 Hz`). If another rate is uploaded to `/analyse`, it is automatically resampled via Fourier-method interpolation (`_resample_to`) to 12 kHz.
- **Units**: Acceleration in standard gravity ($g$) or raw piezoelectric accelerometer voltage proportional to $g$.
- **File Formats**: MATLAB `.mat` (extracts Drive End key `*_DE_time`), NumPy `.npy`, or tabular `.csv`.
- **Minimum Duration**: At least one analysis window of 0.25 s ($3,000$ samples at 12 kHz). Shorter signals return `None` (or HTTP 400).

### 1.4 Exact Output Schema & Types
```typescript
{
  n_segments: number;                  // Number of 0.25s segments extracted (50% overlap)
  embeddings: number[][];              // Shape (n_segments, 128) float32 SSL feature vectors
  predicted_class: string;             // Diagnostic class: "Normal", "IR007", "B007", "OR007@6", etc.
  confidence: number;                  // Highest class posterior probability [0.0, 1.0]
  class_probabilities: Record<string, number>; // Mapping of all 10 fault classes to probabilities
  edr_scores: number[];                // Sliding window Energy Dissipation Ratio scores (nats/s)
  edr_mean: number;                    // Mean EDR divergence over windows
  edr_threshold: number;               // Calibrated EDR threshold (healthy mean + 3*sigma)
  anomalous: boolean;                  // edr_mean > edr_threshold
  segment_scores: number[];            // Per-segment squared Mahalanobis distances
  segment_threshold: number;           // 99th percentile of healthy validation Mahalanobis distances
  fraction_segments_flagged: number;   // Ratio of segments exceeding segment_threshold [0.0, 1.0]
  // Appended by API layer:
  waveform?: { samples: number[]; sampling_rate: number }; // Downsampled to <=2000 points
  map_xy?: number[][];                 // 2D PCA projection coordinates for visualization
}
```

### 1.5 Anomaly Score Definition
- **Windowed Trend Metric**: Energy Dissipation Ratio (EDR) in $\text{nats/s}$, computing Kullback-Leibler / distribution divergence between windowed segment embeddings and healthy baseline.
- **Segment-Level Metric**: Squared Mahalanobis distance $D_M^2(\mathbf{z}) = (\mathbf{z} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})$ on PCA-reduced SSL embeddings against the healthy training centroid.

### 1.6 Threshold Location
- Calibrated during the commissioning split (first 50% healthy training, next 25% calibration):
  - `edr_threshold = mean(val_edr) + 3.0 * std(val_edr)`
  - `mahalanobis_threshold = percentile(val_mahalanobis, 99)`
- Stored directly inside `checkpoints/health_monitor.joblib` and mirrored in `results/metrics/anomaly_detection.json`.

### 1.7 Missing Pieces / Gaps for Integration
- Does not output a normalized $[0.0, 1.0]$ unified anomaly score (EDR is in unbounded nats/s).
- Labels are strictly CWRU 10-class bearing fault codes; need mapping to a common taxonomy for fusion.

---

## 2. Subsystem Audit: Sound / Acoustic (MIMII DG)

### 2.1 Checkpoints & Storage
| File | Size | Tracked in Git? | Status / Description |
| :--- | :---: | :---: | :--- |
| `checkpoints/audio/fan_ae.pt` | ~1.09 MB | No (gitignored) | PyTorch weights, normalizer stats (`scaler_mean`, `scaler_std`), and thresholds for Fan autoencoder. |
| `checkpoints/audio/bearing_ae.pt` | ~1.09 MB | No (gitignored) | PyTorch weights, normalizer stats, and thresholds for Bearing autoencoder. |
| `checkpoints/audio/fan_ae_meta.json` | ~525 B | No (gitignored) | Training metadata, best epoch, loss, and percentile thresholds. |
| `checkpoints/audio/bearing_ae_meta.json` | ~536 B | No (gitignored) | Training metadata, best epoch, loss, and percentile thresholds. |
| `checkpoints/audio/.gitkeep` | 0 B | **Yes** | Placeholder keeping the folder structure in Git. |

*Note*: Both checkpoint files exist on local disk and can be re-generated on demand via `python -m audio_model.training.train_autoencoder`.

### 2.2 Exact Inference Entry Point
- **Python Interface**: [`audio_model/inference.py`](file:///c:/Main%20Project%20S7/MainProject/vibration-ssl-pdm/audio_model/inference.py) -> `predict_audio(audio: Union[str, Path, np.ndarray], sampling_rate: Optional[int] = None, machine_type: str = "fan", threshold_percentile: int = 95) -> Dict[str, Any]`
- **CLI**: `python -m audio_model.inference <path/to/clip.wav> --machine fan --threshold-percentile 95`

### 2.3 Expected Input Specifications
- **Data Shape**: 1-D audio waveform array or `.wav` file path. Shorter clips ($<10\text{ s}$) are automatically padded with zeros to 160,000 samples. Longer clips ($>10\text{ s}$) are windowed into contiguous 10s chunks and averaged.
- **Sampling Rate**: `16,000 Hz` mono. Signals with different rates are automatically resampled via `librosa.resample`.
- **Units**: Floating-point acoustic amplitude normalized in $[-1.0, 1.0]$.
- **File Formats**: `.wav` files or 1-D NumPy float arrays.
- **Validation**: Rejects empty inputs, non-finite values (NaN/Inf), and silent signals ($\text{peak} < 10^{-5}$ or $\text{std} < 10^{-6}$).

### 2.4 Exact Output Schema & Types
```typescript
{
  machine_type: "fan" | "bearing";     // Machine identifier
  anomaly_score: number;               // Mean frame reconstruction MSE across all context vectors
  threshold: number;                   // Applied training normal percentile threshold (P95 default)
  is_anomaly: boolean;                 // anomaly_score > threshold
  status: "NORMAL" | "ANOMALOUS";      // Binary operational verdict
  frame_scores: number[];              // Per-frame reconstruction MSE array (309 values for 10s clip)
  sampling_rate_used: number;          // Always 16000 Hz
}
```

### 2.5 Anomaly Score Definition
- Mean Squared Error (MSE) of autoencoder reconstruction over $T$ stacked context frames:
  $$\mathcal{A} = \frac{1}{T} \sum_{t=1}^T \|\mathbf{x}_t - \hat{\mathbf{x}}_t\|^2$$
  where $\mathbf{x}_t \in \mathbb{R}^{640}$ ($128 \text{ Mel bins} \times 5 \text{ context frames}$).

### 2.6 Threshold Location
- Primary threshold is the 95th percentile ($P_{95}$) of normal training scores:
  - Fan $P_{95} = 0.2545$
  - Bearing $P_{95} = 0.4813$
- Stored inside `checkpoints/audio/<machine>_ae.pt` (`ckpt["thresholds"]`), `checkpoints/audio/<machine>_ae_meta.json`, and `results/audio/metrics/audio_eval.json`.

### 2.7 Missing Pieces / Gaps for Integration
- **No FastAPI Endpoints**: `api/main.py` currently has no routes to accept audio uploads or run audio presets.
- **No Frontend Page**: `frontend/src/` has no audio diagnostic views or charts.
- **Score Scaling**: Anomaly score is raw reconstruction MSE; needs min-max or sigmoid scaling to combine cleanly with vibration and motor scores.

---

## 3. Subsystem Audit: Motor Current & Temperature (Paderborn / Teammate)

### 3.1 Checkpoints & Storage
| File | Size | Tracked in Git? | Status / Description |
| :--- | :---: | :---: | :--- |
| `checkpoints/motor_temp/combined_model.joblib` | ~347 KB | **Yes** | Trained `MotorPdMModel` (Random Forest + Isolation Forest) on MCSA + thermal features. |
| `checkpoints/motor_temp/current_only_model.joblib` | ~341 KB | **Yes** | Trained `MotorPdMModel` on current features only. |
| `checkpoints/motor_temp/temperature_only_model.joblib` | ~216 KB | **Yes** | Trained `MotorPdMModel` on thermal features only. |
| `checkpoints/motor_temp/metadata.json` | ~1 KB | **Yes** | Benchmark metrics (accuracy, F1, ROC-AUC) and feature lists. |

### 3.2 Exact Inference Entry Point
- **Python Interface**: [`core/motor_temperature.py`](file:///c:/Main%20Project%20S7/MainProject/vibration-ssl-pdm/core/motor_temperature.py) -> `analyse_motor_signal(current_phase1: np.ndarray, current_phase2: Optional[np.ndarray] = None, temperature: Optional[Union[float, np.ndarray]] = None, sampling_rate: int = 64_000, model_mode: str = "combined", timestamp: Optional[str] = None) -> Dict[str, Any]`
- **Teammate Core Files**:
  - [`models/motor_temperature_model.py`](file:///c:/Main%20Project%20S7/MainProject/vibration-ssl-pdm/models/motor_temperature_model.py) -> `MotorPdMModel`
  - [`preprocessing/motor_features.py`](file:///c:/Main%20Project%20S7/MainProject/vibration-ssl-pdm/preprocessing/motor_features.py) -> `extract_multimodal_features(...)`
- **Rule**: **Do NOT modify or delete teammate's files**. Wrap with adapters instead.

### 3.3 Expected Input Specifications
- **Data Shape**:
  - `current_phase1`: 1-D array of stator current samples.
  - `current_phase2`: Optional 1-D array of phase 2 current (enables Park's vector transform radius calculation).
  - `temperature`: Scalar float or 1-D array of bearing housing temperature in Celsius ($^\circ\text{C}$). If omitted, defaults to $45.0^\circ\text{C}$ (nominal Paderborn operating temperature).
- **Sampling Rate**: `64,000 Hz` native Paderborn rate.
- **Units**: Current in Amperes (A), Temperature in Celsius ($^\circ\text{C}$).
- **File Formats**: MATLAB `.mat` (Paderborn format), NumPy `.npy`, CSV `.csv`.

### 3.4 Exact Output Schema & Types
```typescript
{
  model_name: string;                  // e.g. "motor_combined_pdm"
  model_version: string;               // "1.0.0"
  status: "HEALTHY" | "ANOMALOUS";     // Verdict based on anomaly_score >= 0.5 or pred != "Normal"
  predicted_condition: string;         // "Normal" | "Outer_Race" | "Inner_Race"
  anomaly_score: number;               // Calibrated score in [0.0, 1.0] (higher = more anomalous)
  confidence: number;                  // Maximum class posterior probability [0.0, 1.0]
  class_probabilities: {               // Class probability distribution
    Normal: number;
    Outer_Race: number;
    Inner_Race: number;
  };
  timestamp: string | null;            // Metadata timestamp if provided
  features: Record<string, number>;    // Extracted MCSA (RMS, crest, harmonics, Park's) & thermal features
  input_diagnostics: {
    samples: number;
    duration_seconds: number;
    sampling_rate_hz: number;
    two_phase_available: boolean;
    temperature_input_available: boolean;
  };
  // Appended by API layer:
  waveform_p1?: number[];              // Downsampled phase 1 current
  waveform_p2?: number[];              // Downsampled phase 2 current
  spectrum?: { freqs: number[]; magnitudes: number[] }; // FFT spectrum up to 1 kHz
  metadata?: Record<string, any>;
}
```

### 3.5 Anomaly Score Definition
- Calibrated sigmoid of the negative IsolationForest score:
  $$s = \frac{1}{1 + \exp\left(-\frac{\text{raw} - \tau_{\text{anom}}}{\sigma_{\text{raw}}}\right)} \in [0.0, 1.0]$$
  where $\text{raw} = -\text{score\_samples}(\mathbf{x}_{\text{scaled}})$.

### 3.6 Threshold Location
- Primary IsolationForest raw threshold $\tau_{\text{anom}}$ is fitted on 95th percentile of normal training scores.
- Calibrated decision threshold is fixed at $0.50$ (or non-normal class prediction).
- Stored inside each `.joblib` model file (`model.anomaly_threshold_`) and summarized in `checkpoints/motor_temp/metadata.json`.

### 3.7 Missing Pieces / Gaps for Integration
- Teammate code is self-contained and already wired into `api/main.py` via `/analyse/motor` and `/analyse/motor/recording`.
- Ready for integration into the multimodal fusion pipeline via adapter.

---

## 4. Existing FastAPI Endpoints

| Method | Path | Description | Input / Parameters | Response Schema |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | System health & hardware | None | `{"status": "healthy", "device": "cuda"\|"cpu", "model_loaded": true}` |
| `POST` | `/analyse` | Upload vibration file | `file` (.mat, .npy, .csv), `sampling_rate`, `channel`, `clf_name` | `AnalysisResult` (CWRU vibration diagnosis) |
| `GET` | `/map` | Background PCA embedding map | `max_points` (default 1500) | `EmbeddingMapResponse` (points with x, y, class_label) |
| `GET` | `/recordings` | List CWRU recordings | None | `Recording[]` (file_id, label, load_hp, rpm, etc.) |
| `POST` | `/analyse/recording/{file_id}` | Analyse CWRU recording preset | `file_id` path param, `clf_name` | `AnalysisResult` (CWRU vibration diagnosis + metadata) |
| `GET` | `/recordings/motor` | List Paderborn motor recordings | None | `MotorRecording[]` (filename, bearing_id, condition, rpm, torque) |
| `POST` | `/analyse/motor` | Upload motor current/temp file | `file`, `temperature`, `sampling_rate`, `model_mode` | `MotorAnalysisResult` (MCSA + thermal diagnosis) |
| `POST` | `/analyse/motor/recording` | Analyse Paderborn motor preset | `filename`, `temperature`, `model_mode` | `MotorAnalysisResult` (MCSA + thermal diagnosis + metadata) |

*Gaps in API*:
- Missing `GET /recordings/audio` to list MIMII DG test recordings.
- Missing `POST /analyse/audio` to analyze uploaded sound clips (.wav).
- Missing `POST /analyse/audio/recording` to score preset audio clips.
- Missing `POST /analyse/multimodal` (or `/fusion`) to accept synchronized or composite multimodal data and return fused health verdicts.

---

## 5. Existing Frontend Pages & UI Architecture

### 5.1 Technology Stack
- **Framework**: React 18 + Vite + TypeScript
- **Styling**: TailwindCSS (custom theme colors in `src/index.css`)
- **Routing**: `react-router-dom` v6
- **Charts**: `recharts` (Waveform line charts, PCA scatter plots, FFT spectrum, Radar charts)
- **API Client**: `src/api/client.ts` using Axios with `VITE_API_URL` (default `http://localhost:8000`)

### 5.2 Page Inventory
1. **`/analyse` -> `AnalysePage.tsx`**:
   - Primary Vibration analysis interface for CWRU data.
   - Upload dropzone (.mat, .npy, .csv) and dropdown of pre-loaded CWRU benchmark recordings.
   - Displays prediction badge, confidence score, EDR trend chart, Mahalanobis bar, waveform line chart, and 2D PCA embedding space.
2. **`/motor` -> `MotorPage.tsx`**:
   - Motor Current & Temperature diagnostics for Paderborn data.
   - Supports file upload or preset selection.
   - Displays MCSA phase current waveforms, FFT frequency spectrum (up to 1 kHz), Park's Vector orbital plot, temperature gauge, and condition classification.
3. **`/history` -> `HistoryPage.tsx`**:
   - Chronological table of past analyses run during the session.
4. **`/live` -> `LivePage.tsx`**:
   - Simulated streaming vibration telemetry feed with rolling anomaly detector.

*Gaps in Frontend*:
- No page or navigation tab for **Sound / Acoustic Anomaly Detection**.
- No page or overview dashboard for **Multimodal Sensor Fusion**.
- No visual indicator or badge marking preset records as **"REPLAY"**.

---

## 6. Comprehensive Integration Blockers & Action Plan

### 6.1 Identified Architectural Blockers

| # | Blocker | Impact | Solution Strategy |
| :-: | :--- | :--- | :--- |
| **1** | **No Audio API Endpoints** | The frontend and external clients have no way to access the trained audio models. | Add `POST /analyse/audio`, `GET /recordings/audio`, and `POST /analyse/audio/recording` to `api/main.py`. |
| **2** | **Heterogeneous Anomaly Scales** | Vibration EDR is unbounded $[\text{nats/s}]$, Sound MSE is $[0.1, 1.5]$, Motor IsolationForest is calibrated $[0.0, 1.0]$. Raw weighted averaging is mathematically invalid. | Build a dedicated `FusionEngine` that normalizes each modality into a calibrated risk probability $P(\text{anomaly} \mid \text{signal}) \in [0.0, 1.0]$ before fusing. |
| **3** | **Disparate Sampling Frequencies & Durations** | Vibration is 12 kHz (0.25s windows), Sound is 16 kHz (10s clips), Motor is 64 kHz (continuous). Raw sample concatenation is impossible. | Implement **Late Decision / Score Fusion** rather than early raw feature concatenation. Each modality extracts domain-specific features independently; the fusion layer combines anomaly scores and class probabilities. |
| **4** | **Replay Label Transparency** | Preset recordings could be mistaken for live telemetry. | Add explicit `is_replay: true` and `data_source: "replay"` metadata to all preset endpoints and render amber `REPLAY DATASET` badges in the UI. |
| **5** | **Teammate Code Protection** | Teammate's files (`models/motor_temperature_model.py`, `preprocessing/motor_features.py`) must remain untouched. | Create adapter wrappers in a separate module (`core/adapters/` or `core/fusion/`) without editing teammate code. |
| **6** | **Audio Checkpoints in Git** | `checkpoints/audio/*` is in `.gitignore`, whereas other checkpoints are committed. | Ensure clear reproduction documentation and verify local checkpoint presence. |

---

## 7. Next Steps for Implementation

1. **Step 1: Audio Endpoints in FastAPI**:
   - Expose `predict_audio` via `POST /analyse/audio` and `GET /recordings/audio` in `api/main.py`.
   - Add unit tests in `tests/test_api_audio.py`.
2. **Step 2: Multimodal Fusion Engine**:
   - Create `core/fusion.py` implementing calibrated late fusion across Vibration, Sound, and Motor.
   - Define unified status: `HEALTHY`, `WARNING`, `CRITICAL`.
   - Add comprehensive tests in `tests/test_fusion.py`.
3. **Step 3: Frontend Integration**:
   - Add `AudioPage.tsx` and `FusionPage.tsx` (or unified overview) to `frontend/src/pages/`.
   - Add navigation links in `Sidebar.tsx`.
   - Add "REPLAY" indicator badges across preset views.
