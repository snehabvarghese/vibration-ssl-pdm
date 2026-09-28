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
5. serve everything in a Streamlit dashboard.

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
   └─ dashboard/app.py               Streamlit monitor
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

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU is enough (the full pipeline below runs in well under an hour on CPU). CUDA/MPS
are used automatically when available (`config.get_device()`).

## 5. Running the pipeline

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

Every stage caches its artefacts, so stages can be re-run independently.
All randomness goes through `config.set_seed(42)`.

## 6. Avoiding leakage

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

## 7. Results (measured on this repo — not copied from the paper)

Recreate with `./run_phase1.sh`; raw numbers live in `results/metrics/`.

**Anomaly detection** (`results/metrics/anomaly_detection.json`)

| Metric | Value |
|---|---|
| EDR threshold (healthy mean + 3σ) | 1.75 nats/s |
| Held-out healthy windows flagged | 0 / 10 |
| Faulty windows flagged | 18 / 18 |
| Segment-level ROC-AUC (fault vs healthy) | 1.000 |
| Segment-level TPR / FPR | 1.000 / 0.029 |

**Fault classification** (`results/metrics/comparison.csv`) — evaluated on record-level test set (1 238 segments, 16 classes):

| Label Budget | N Samples | SSL Frozen MLP F1 | SSL Frozen SVM F1 | SSL Fine-tuned F1 (Acc) | Supervised CNN F1 (Acc) |
|---|---|---|---|---|---|
| 1 % | 32 | 0.689 | 0.698 | **0.751 (78.9%)** | 0.813 (84.0%) |
| 5 % | 128 | 0.759 | 0.723 | **0.840 (85.9%)** | 0.843 (87.1%) |
| 10 % | 256 | 0.770 | 0.778 | **0.846 (87.3%)** | 0.866 (88.2%) |
| 20 % | 512 | 0.745 | 0.733 | **0.850 (87.4%)** | 0.870 (88.8%) |

*With 5% labels (128 samples), fine-tuned SSL achieves 84.0% Macro F1 (85.9% Accuracy), reaching parity with the fully Supervised CNN (84.3% F1) within -0.3% F1.*

**Figures** (`results/figures/`): class distribution, waveforms, spectra, preprocessing stages, augmented views, SSL loss curve, t-SNE, UMAP, EDR trend, EDR per class, label efficiency, confusion matrices, SSL-vs-baseline.

## 8. EDR — what it is here

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

## 9. Dashboard

```bash
streamlit run dashboard/app.py
```

Pick a dataset recording or upload a `.mat`/`.npy` file; it shows status
(NORMAL/ANOMALOUS), predicted fault + confidence, EDR vs threshold, the fraction of
flagged segments, the waveform, the anomaly trend, the embedding map with the current
signal marked, and class probabilities.

`analyse_signal(signal, sampling_rate)` is the single Phase 2 hook: an MQTT/serial
reader just calls it with a live buffer.

## 10. What we could not reproduce

The PDF supplied to us was corrupt (no extractor or renderer recovered its body text),
so only the public abstract-level description was available. Unavailable, therefore
designed by us and labelled as ours:

* the exact EDR formulation, estimator and thresholding rule;
* encoder depth/widths, projection dimensions, optimiser schedule;
* augmentation types and probabilities;
* the in-house multistage-compressor dataset (not public — we use CWRU only);
* the paper's exact train/test protocol, so its F1 ≈ 0.93 is not comparable to ours.

## 11. Phase 2 (not implemented)

```
Machine → accelerometer → ESP32 → Wi-Fi/MQTT → broker → inference service
        → Phase 1 encoder + EDR monitor + classifier → dashboard → alerts
```

The Phase 1 boundary is `analyse_signal()` plus `preprocessing/loader.py`: replace the
file reader with a stream reader and the rest is unchanged.

## 12. Repository layout

```
config.py                 all hyperparameters + seeds + device selection
preprocessing/            manifest, download, loader, filtering, segmentation,
                          augmentation, build_dataset, explore, demo
models/                   encoder, siamese (+NT-Xent), classifier, baseline_cnn
training/                 splits, datasets, train_ssl, extract_embeddings,
                          train_classifier, train_baseline
anomaly_detection/        edr, anomaly_detector
evaluation/               metrics, visualization, evaluate
dashboard/app.py          Streamlit monitor
results/                  figures, metrics, embeddings
DESIGN.md                 architecture, roadmap, paper gaps
```
