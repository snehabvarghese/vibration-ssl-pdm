# Self-Supervised Vibration Analytics for Predictive Maintenance of Multistage Compressors

## Bibliographic Details
- **Journal:** Journal of Vibration Engineering & Technologies (2026) 14:227 (Springer Nature Singapore)
- **DOI:** https://doi.org/10.1007/s42417-026-02455-2
- **Article type:** Original Paper (23 pages)
- **Authors:**
  - Rajagopal M (corresponding; mrajagopal1973@gmail.com), Dept. of Mechanical Engineering, PERI Institute of Technology, Mannivakkam, Chennai, Tamil Nadu, India
  - Shashank R, Dept. of Artificial Intelligence and Data Science, Saveetha Engineering College, Sriperumpudur, Kancheepuram, Tamil Nadu, India
  - Shreeshanth R, Dept. of Computer Science and Engineering, Saveetha Engineering College, Sriperumpudur, Kancheepuram, Tamil Nadu, India
- **Dates:** Received 30 Dec 2025 / Revised 7 Mar 2026 / Accepted 31 Mar 2026 / Published online 10 Apr 2026
- **Keywords:** Self-supervised learning, Predictive maintenance, Vibration analysis, Multistage compressors, Fault detection, Condition monitoring
- **Competing interests:** None declared.
- **Author contributions:** Rajagopal M: conceptualization, methodology, supervision, formal analysis, writing (original draft, review & editing). Shashank R: investigation, software, validation, writing (review & editing). Shreeshanth R: data curation, visualization, writing (review & editing).
- **Data availability:** Datasets are available from the corresponding author on reasonable request.

---

## 1. Abstract
- **Purpose:** Multistage compressors are vital industrial assets, so timely fault detection is crucial. Traditional vibration-based PdM depends on supervised learning, which needs extensive labeled fault data that is rarely available. The study presents a self-supervised learning (SSL) framework that learns fault-relevant features from unlabeled data.
- **Methods:** Contrastive Siamese convolutional architecture with domain-specific temporal augmentations, followed by a lightweight supervised classifier trained on few labeled samples. An **Entropy Divergence Rate (EDR)** metric captures distributional shifts in the learned feature space for early anomaly detection. Validated on a six-month in-house multistage compressor dataset plus an adapted CWRU bearing dataset.
- **Results:** F1-score of **0.93** and a **28% improvement in early fault detection** vs FFT-based and fully supervised deep learning approaches.
- **Conclusion:** Substantially reduces reliance on labeled data while improving early fault recognition; suited to scalable, practical industrial PdM.

---

## 2. Introduction
- Multistage compressors are used in oil and gas processing, chemical manufacturing, refrigeration and air separation. Complex multi-stage rotating assemblies operating continuously under high mechanical and thermal stress are prone to progressive degradation: **bearing wear, rotor imbalance, shaft misalignment**. Undetected incipient faults can cause catastrophic failure, economic loss and safety hazards.
- **Reactive maintenance** causes unplanned shutdowns; **preventive maintenance** may replace healthy components prematurely. **Predictive maintenance (PdM)** uses continuous condition monitoring. Vibration analysis is among the most widely adopted PdM techniques because of its sensitivity to mechanical defects and early-stage fault signatures.
- ML-based vibration diagnosis is an active area, but most approaches are supervised and need large labeled fault sets, which are scarce, costly and incomplete in industry. CNNs and RNNs improved feature extraction, but remain limited by label availability.
- SSL learns representations from unlabeled data via pretext or contrastive objectives. It has succeeded in vision and speech, but is little used for complex industrial assets like multistage compressors. Existing SSL studies mostly use generic bearing datasets, prioritize representation accuracy, and overlook early fault detection, domain-specific degradation and deployment.
- **Proposed:** SSL framework combining contrastive Siamese learning with compressor-specific temporal augmentations and auxiliary pretext tasks, plus an EDR-based anomaly detection mechanism.

### Novelty and Contributions
1. A **domain-specific SSL framework** for multistage compressors, with compressor-specific temporal augmentations and multi-task pretext objectives reflecting compressor dynamics and degradation, unlike generic time-series encoders.
2. **EDR-based anomaly metric** embedded in the learned feature space to identify incipient faults through distributional change without prior fault knowledge or labels.
3. **Label efficiency:** high classification performance with few labeled samples.
4. Validation on a **six-month in-house industrial dataset** and an **adapted benchmark dataset** (practical relevance, reproducibility, fair comparison).
5. **Deployment-oriented design:** low inference latency, edge-computing compatibility, extensibility to federated learning.

---

## 3. Literature Review (Summarized)

### Deep Learning for Machinery Fault Diagnosis
CNNs automatically extract features from raw vibration signals, reducing manual feature engineering. Zhao et al. [1] showed the efficacy of deep learning in machine health monitoring, with considerable gains in fault detection precision on rotating machinery datasets.

### SSL for Industrial Time-Series
Models train on pretext tasks (forecasting temporal sequences, evaluating similarity between signal segments), then are refined with minimal labels. Contrastive methods **SimCLR** (positive/negative pairs via augmentation [2]) and **MoCo** (momentum encoder plus dynamic dictionary [3]) have been adapted for vibration analysis, often with time cropping, jittering and permutation.

### Visualization and Benchmark Datasets
- **UMAP** [4] is widely used to inspect learned representations and cluster separability.
- Common benchmarks: **CWRU bearing dataset** and **IMS bearing dataset** (Univ. of Cincinnati) [5–7].
- **ISO 10816** [8] provides vibration severity criteria.

### Evolution of PdM and PHM
Mobley [9] laid the groundwork (vibration analysis, thermography, oil analysis, condition monitoring). PdM evolved into Prognostics and Health Management (PHM) with degradation modeling and Remaining Useful Life (RUL) estimation (Goriveau et al. [10]). IIoT boosted scalable PdM; the Industrial Internet Consortium [11] showed real-time PdM via interoperable IIoT frameworks (sensors, cloud analytics, edge computing).

### ML and Hybrid Models for PdM
- Ensembles and deep networks beat conventional statistical methods on complex industrial datasets [12].
- Costa et al. (2024) [13 in reference list]: hybrid clustering (unsupervised clustering + supervised classification) for industrial compressors.
- Chevtchenko et al. [13/25]: anomaly detection for induction motors using real-time IoT data; semi-/unsupervised methods are efficient with scarce labels.
- Habib and Mohamed [14]: hybrid CNN–LSTM for RUL, combining spatial features and temporal sequence learning.

### SSL for PdM
Zhang et al. [15] classified SSL frameworks for time series into generative, contrastive and predictive. Tran et al. [16] showed SSL anomaly detection greatly improves PdM in Industrial IoT.

### Signal Processing and Feature Extraction
- Li and Wang [17]: Variable Filtered-Waveform VMD (VFW-VMD) for broadband vibrations and bearing fault features.
- Wang et al. [18]: wavelet-guided neural network with dynamic frequency decomposition.

### Physics-Based and Digital Twin Approaches
Liu et al. [19] (digital-twin diagnostics with cross-device transfer learning), He et al. [20] (PMSM partial demagnetization via air-gap flux density), Xu et al. [21] (open-phase faults in five-phase PMSM), Ren et al. [22] (mechanism-guided early insulation deterioration detection).

### Vibration Dynamics and Structural Monitoring
Xu et al. [23] (inerter-based dampers for bridge vortex-induced vibration), Zheng et al. [24] (HTS maglev electromechanical coupling).

### IoT and Edge
Chevtchenko et al. [25] (IoT-based PdM for induction motors), Ringler et al. [26] (edge-centric ML: quicker response, lower bandwidth, better reliability than cloud-only).

### Nonlinear Dynamics and Reviews
Cui et al. [27] (generalized Van der Pol model of vortex-induced vibrations). Govindaswamy & Ramadoss [28] (review: hybrid deep learning/ensembles handle non-stationary signals). Liao & Wang [29] (data-driven health indicators).

### Industry 4.0 Architectures
Calabrese et al. [30] (SOPHIA event-driven PdM), Paolanti et al. [31] (ML anomaly detection in mechatronics).

### Comparative Studies of ML Algorithms
Ünal et al. [32] (ensemble and deep models beat regression on telemetry), Ingole et al. [33] (nonlinear regression better for aircraft engine PdM).

### Federated Learning and Advanced Methods
Kairouz et al. [34], Ahn et al. [35] (federated PdM under time-series distribution shifts), Zhang et al. [36] (AMB-rotor disturbance-observer control), Tian et al. [37] (adaptive tempered reversible-jump MCMC), He et al. [38] (neurodynamic predictive control for omnidirectional robots).

### Structural Vibration and Modal Identification
Zhou et al. [39, 40] (spectral element methods for Mindlin plates, free vibration and bandgaps), Yuan et al. [41] (optimal sparse sensing for full-field vibration reconstruction), Qin & Tang [42] (improved empirical wavelet transform + Hilbert transform for modal ID), Tran et al. [43] (thermoelastic model of CNT-reinforced plates).

### Weak Fault Detection and Signal Modeling
Gardner [44] (Fraction-of-Time probability framework), Lu et al. [45] (stochastic resonance to amplify weak fault signals), Matsubara et al. [46] (dynamic micro X-ray CT + dynamic mechanical analysis for rubber).

---

## 4. Methodology

### 4.1 Overview
Conventional SSL for time series uses generic pretext tasks or dataset-agnostic contrastive frameworks. This method integrates:
- domain-informed temporal augmentations,
- multi-task SSL objectives (contrastive learning, temporal context prediction, signal reconstruction),
- EDR-based anomaly scoring.

**Six workflow stages (Fig. 1):**
1. Sensor placement and data collection
2. Preprocessing of vibration signals
3. Self-supervised representation learning
4. Dimensionality reduction and embedding visualization
5. Anomaly detection
6. Decision support for maintenance actions

**Fig. 2 pipeline:** Sensors → (signal conditioning, segmentation, normalization) → Self-Supervised Learning (augmentation, contrastive loss) → Embedding Visualization → Anomaly Detection (autoencoder + EDR) → Decision Support; also SSL → Fault Classification → Maintenance Support.

### 4.2 Sensor Deployment and Data Acquisition
- **Tri-axial accelerometers** at: inlet shaft, inter-stage bearings and couplings, outlet casing, motor mount.
- Sampling frequencies **10–50 kHz**, sufficient for bearing impacts, shaft misalignment and early structural degradation.
- DAQ synchronized with compressor operating cycles for accurate time stamping.
- Supplementary logged parameters: rotor speed, load, ambient temperature, maintenance logs.
- **Continuous six-month collection** covering normal operation, natural wear, injected test faults and scheduled maintenance. The dataset reflects class imbalance, non-stationary behavior and progressive fault evolution.
- Augmentation parameters standardized with **dimensionless time-frequency ratios** relative to rotational speed and valve timing (physical consistency across compressors with different stage counts, speeds, capacities).
- Startup and shutdown transients treated as distinct temporal contexts, not mixed with steady state.
- All augmentations bounded in magnitude and duration by compressor operating frequencies and valve actuation timescales, to preserve impulsive fault signatures.

### 4.3 Signal Preprocessing
1. **Butterworth band-pass filter, 10 Hz – 20 kHz** (suppresses low-frequency drift and structural noise).
2. **Sliding windows of 1–3 s, 50% overlap.**
3. **Downsampling 2:1.**
4. **Z-score normalization** per channel.
5. During SSL: augmented views via random jittering, magnitude scaling and temporal permutation.

### 4.4 Self-Supervised Feature Representation Learning
- **Siamese architecture, twin convolutional encoders.**
- **Positive pairs:** different augmented views of the same segment (minor temporal transforms). **Negative pairs:** segments from different time windows or different machines.
- Signals from different stages and sensor locations are treated as correlated views of the same operating state. Inter-stage coupling (pressure pulsations, valve impact transients, torsional transmission across couplings) yields structured temporal dependencies and localized high-frequency components.
- Invariance is enforced only to augmentation perturbations; physically meaningful impulsive events stay consistent across positive pairs. Valve-related transients form stable latent attractors; broadband steady rotational noise is deemphasized.
- **Loss: NT-Xent (InfoNCE).**
- **Encoder:** stack of 1D conv layers + ReLU, batch normalization, global average pooling → projection head → embeddings of **128–256 dimensions**.
- **Auxiliary pretext tasks:** temporal context prediction; signal reconstruction with denoising auto-encoders; identification of synthetic signal transformations applied during augmentation. Joint optimization captures invariant and discriminative fault-related structure.

### 4.5 Dimensionality Reduction and Visualization
- Embeddings projected to 2D with **t-SNE** and **UMAP**.
- Well-separated clusters for normal, rotor imbalance, shaft misalignment, bearing wear indicate strong discriminative capability.
- EDR reflects distributional shifts in vibration energy across latent frequency-temporal structures; early fault progression appears as a monotonic increase in divergence from the healthy baseline.

### 4.6 Anomaly Detection with EDR
- Definition: **EDR(t) = D_KL(P_t || P_baseline)**, where D_KL is Kullback–Leibler divergence, P_t the current feature distribution and P_baseline the distribution under healthy operation. Rising EDR = growing deviation from normal.
- Complementary detectors: **One-Class SVM** and **Isolation Forest** trained on healthy embeddings.
- A segment is flagged anomalous if EDR exceeds a **dynamically defined threshold** or is classified as an outlier by the ML models.
- **Weekly EDR trend monitoring** identifies gradual degradation.
- Unlike amplitude indicators, EDR detects distributional change in latent space before notable amplitude increases. KL-based, so computationally efficient for real-time use.
- **EDR calculation process (Fig. 3):**
  1. Raw multi-sensor input
  2. Pre-processing (de-trending, band-pass, normalization, optional VMD/wavelet)
  3. Temporal segmentation (sliding window)
  4. Siamese feature encoder (twin CNN/TCN branches, shared weights → embeddings z1, z2)
  5. Probability density estimation (histogram/KDE; reference vs test window)
  6. Entropy computation (H_ref = −Σ p_ref log p_ref; H_test = −Σ p_test log p_test)
  7. Divergence calculation (D = |H_test − H_ref|, or KL/JSD variant)
  8. Temporal rate estimation (EDR = dD/dt, finite difference across consecutive windows)
  9. Thresholding and decision (adaptive threshold, load/temperature aware, false-alarm suppression)
  10. Health state output (Normal / Incipient fault / Developed fault / Alarm trigger)
- Decision logic includes persistence and confidence thresholds; alarms need **multi-window persistence** to meet industrial safety norms and avoid false shutdowns.
- Rationale: EDR represents the evolution of uncertainty over time (suited to non-stationary fault development); resilient to slow load or ambient-temperature changes; consistent thresholds under healthy conditions.
- **Drift handling:** periodic baseline recalibration during confirmed healthy operation, latent-space normalization, and a dynamic reference distribution with exponential forgetting so slow sensor aging is not mistaken for failure. Sudden sensor failures show distinct divergence patterns.

### 4.7 Decision Support and Maintenance Recommendation
- Dashboard for operators and maintenance engineers: health score trends, embedding visualizations, anomaly progression (RMS peaks, EDR trajectories).
- Automated alerts with messages like "Rotor imbalance detected", "Inspect inter-stage bearings", derived from mappings between anomaly signatures and fault mechanisms, validated with maintenance records and expert knowledge.
- Supports federated deployment. Moves maintenance from fixed schedules to condition-based recommendations.

### 4.8 Deployment and Scalability
- Inference latency of downstream classifier on edge platforms (e.g., **NVIDIA Jetson Xavier**) **< 50 ms** per segment. Pretraining is offline on central servers; updated parameters are pushed to edge devices.
- Combined encoder–classifier pipeline uses **< 100 MB** memory, stable under streaming data.
- Federated learning: only model weights or latent statistics exchanged; raw data stays on-site (privacy, bandwidth).
- **Challenges:** synchronization across heterogeneous sensor networks, firmware/model update management, robustness to sampling-rate and noise variation across plants.
- A small labeled subset is needed for calibration; the encoder can be adapted to similar systems by adjusting fine-tuning. The representation layer can be retrained for different sensor networks while keeping the anomaly detection process.

### 4.9 Data Governance
Industrial data anonymized (asset identifiers removed, statistical normalization at edge node). Federated SSL keeps raw signals on-site.

### 4.10 Mechanical Coupling Awareness
- Compressor vibration arises from inter-stage pressure fluctuations, valve opening/seating impacts, torsional shaft interactions and flow excitation.
- High-frequency valve impulses have robust time-frequency localization and form stable attractors; steady-state rotational noise lacks temporal structure and sits in low-discriminative latent regions.
- Augmentations (jitter, slight time-warping, window cropping, amplitude scaling) are limited via dimensionless ratios normalized by shaft speed and primary valve actuation frequencies.
- **SKRgram** demodulation is noted as a supportive front-end for hybrid future work.

---

## 5. Experimental Framework

### 5.1 Instrumentation
- **3 industrial multistage compressors** instrumented at inlet shaft housing, inter-stage bearings/couplings, outlet casing, motor mount.
- **PCB Piezotronics 356A32** tri-axial accelerometers (100 mV/g, 0.5 Hz–10 kHz), synchronized sampling at **48 kHz**.
- **NI PXI-4472** DAQ (48 kHz per channel, 24-bit).
- Load levels: **50%, 75%, 100%**. Ambient temperature 25–40 °C.
- Sensor placement validated by modal testing. RPM, motor current and background noise recorded.
- (Also stated elsewhere in the paper: tri-axial piezoelectric accelerometers attached to cylinder head and crankcase, sampled at 25.6 kHz with 24-bit DAQ and anti-aliasing filters.)
- **Compute:** Intel Core i7-12700H CPU, 32 GB RAM, NVIDIA RTX 3060 (6 GB), PyTorch 2.1. Edge: **NVIDIA Jetson Xavier NX** (384 CUDA cores, 8 GB RAM), Siamese encoder real-time with **18 ms average latency per window**.
- VFW-VMD used only for comparative analysis, not needed in the proposed framework.

### 5.2 Datasets
- **In-house:** over **3,000 hours** of vibration data from multistage compressors across six months. Includes natural progression (bearing wear, rotor imbalance) and deliberately introduced faults. Maintenance logs (component replacements, repair timestamps) served as ground-truth labels.
- **CWRU bearing dataset (adapted):** segmented into **2 s windows**, normalized, resampled to match in-house statistics. Original: 12 kHz and 48 kHz, inner race / outer race / ball faults, loads up to 3 hp.

### 5.3 Data Splitting
Per compressor: **70%** unlabeled for SSL pretraining, **10%** labeled for fine-tuning, **20%** labeled held-out test. **Cross-machine validation** (test on unseen machines). **SMOTE** applied only to supervised baselines for class imbalance; **no oversampling in SSL workflow**. At inference only one encoder branch is used; augmentations only during training. Only embeddings or EDR metrics need to be sent in edge–cloud setups.

### 5.4 Training Configuration
**Pretraining:**
- Siamese CNN encoder: **3 conv blocks** (conv + batch norm + ReLU)
- MLP projection head **256 → 128**
- NT-Xent loss, **temperature 0.5**
- Augmentations: time warping, signal permutation, amplitude scaling, additive jitter
- **200 epochs, batch size 256, Adam, weight decay 1e-4**
- Cosine annealing LR from **1e-3 to 1e-5**
- NVIDIA RTX 3080, CUDA 11.7, PyTorch 1.12

**Downstream classifiers:** shallow MLP (2 hidden layers) and SVM with RBF kernel. Training < 20 min. Hyperparameters by grid search with 5-fold cross-validation.

### 5.5 Baselines
1. **FFT + SVM:** spectral centroid, dominant peak frequency, band energy ratios; RBF-SVM.
2. **Supervised CNN:** 1D CNN on raw vibration, fully supervised.
3. **LSTM:** temporal dependency model; overfits with limited labels.
4. **Autoencoder + Clustering:** unsupervised, reconstruction-error based, latent clustering.
All use identical preprocessing, segmentation and splits.

### 5.6 Evaluation Metrics
Precision, recall, F1, overall accuracy; **Early Detection Rate (EDR)**: the temporal lead by which a fault is identified relative to documented onset in maintenance logs (higher = earlier). Embedding quality: silhouette score, t-SNE/UMAP. Imbalance robustness: AUC-ROC and PR curves. Each experiment repeated **5 times**, reported as mean ± SD.

### 5.7 Sensitivity Analysis of Design Choices
Varied: (i) augmentation strategy, (ii) embedding dimension, (iii) labeled subset size.

**Temporal augmentations** (time masking, Gaussian jitter, amplitude scaling, segment permutation):
- **Time masking + jittering** best: F1 **+6.2%** over single-augmentation training.
- Time masking helps with transient dropouts; jittering improves noise tolerance.
- Amplitude scaling: marginal (< 1.5%).
- Excessive permutation degrades separability (destroys temporal structure of impulsive signatures, esp. valve and bearing defects).

**Embedding dimension** (32, 64, 128, 256): **128 best trade-off**. 32→128 improved F1; 256 gave < 1% gain with ~18% more inference time. t-SNE confirmed compactness at 128.

**Labeled subset size** (1%, 5%, 10%, 20%): F1 > **0.88 with only 1%** labels; stable at **5%** (< 1% variance); 10–20% no significant gain. Labeling requirements reduced by ~**92%** vs fully supervised CNN with comparable accuracy.

**Computational:** selected config (time masking + jittering, 128-dim, 5% labels) cut inference time **38%** vs supervised CNN.

---

## 6. Results and Discussion

### 6.1 Classification Performance
- Average **F1 = 0.93**, vs supervised CNN **0.84** and LSTM **0.86**.
- **Early Detection Rate improved ~28%** vs supervised baselines.
- Overall accuracy **94.5%**; per-class F1 > 0.90 in most cases.
- **Confusion matrix (Fig. 4)** over Normal, Imbalance, Bearing Wear, Misalignment. Text: 48/50 normal, 47/50 imbalance, 46/50 bearing wear, 48/50 misalignment correct; misclassifications mostly between bearing wear and misalignment. (The plotted figure shows counts: Normal row 273/1/3/2; Imbalance 28/247/4/7; "Insocal" [Bearing Fault] 13/18/94/15; Misalignment 13/18/6/217.)
- Framework is "self-supervised with minimal supervised calibration" rather than fully unsupervised; stable with 5% labels (92% annotation reduction).

### 6.2 Failure Case Analysis
Bearing wear vs shaft misalignment confusion: outer-race wear gives high-frequency impulsive events on shaft harmonics; slight misalignment raises 1× and 2× rotational components with moderate sidebands (often axial). At low severity the time/frequency features overlap. Fix may need supplementary sensors, higher-resolution data or hybrid feature fusion.

### 6.3 EDR (Early Detection Rate) Trends over 12 Weeks (Fig. 5, Table 1)

| Week | FFT+SVM | Supervised CNN | SSL (Proposed) |
|---|---|---|---|
| 1 | 0.40 ± 0.02 | 0.45 ± 0.03 | 0.50 ± 0.02 |
| 2 | 0.43 ± 0.03 | 0.48 ± 0.02 | 0.57 ± 0.03 |
| 4 | 0.50 ± 0.03 | 0.55 ± 0.02 | 0.70 ± 0.03 |
| 6 | 0.56 ± 0.02 | 0.60 ± 0.03 | 0.80 ± 0.02 |
| 8 | 0.60 ± 0.03 | 0.66 ± 0.03 | 0.86 ± 0.03 |
| 10 | 0.63 ± 0.02 | 0.69 ± 0.02 | 0.89 ± 0.02 |
| 12 | 0.66 ± 0.03 | 0.73 ± 0.02 | 0.91 ± 0.02 |

By Week 6 the SSL model already surpasses the supervised CNN's Week-12 value. SSL also has narrower error bars (more stable across runs).

### 6.4 Comparative Evaluation (Table 2)
- **FFT+SVM:** EDR ~0.40→0.66; limited adaptability, depends on feature engineering, weak early-stage detection.
- **Supervised CNN:** EDR ~0.45→0.73; better trend tracking but needs large annotated datasets, less robust with scarce labels.
- **SSL (proposed):** EDR ~0.50→0.91; highest growth rate, earliest sensitivity, least label dependence.

### 6.5 Noise Robustness
Synthetic broadband noise added, SNR 20 dB → 0 dB. Detection performance consistent at **5 dB SNR** with < 4% F1 decline (latent distribution comparison stays sensitive to structural shifts even when amplitude is hidden).

### 6.6 Computational Benchmarking and Transfer (Table 3)

| Model | Paradigm | F1 | Inference (ms) | Params (M) | Labels | Edge suitability |
|---|---|---|---|---|---|---|
| Traditional feature-based + SVM (FFT, RMS, kurtosis) | Fully supervised | 0.84 | 4.10 | 0.02 | 100% | High |
| Supervised 1D CNN (baseline) | Fully supervised | 0.93 | 12.8 | 1.25 | 100% | Moderate |
| Lightweight CNN | Fully supervised | 0.90 | 8.60 | 0.48 | 100% | Moderate–High |
| Hybrid physics-informed CNN | Supervised + domain features | 0.92 | 14.5 | 1.40 | 80–100% | Moderate |
| Proposed SSL encoder + linear head | SSL + 5% fine-tuning | 0.92 | 7.90 | 0.52 | 5% | High |
| Proposed transfer test (Industrial → CWRU) | Zero-shot encoder + 5% target fine-tuning | 0.89 | 7.90 | 0.52 | 5% (target) | High |
| EDR-based threshold monitoring | Unsupervised statistical monitoring | fault trend detection (no class labels) | 3.70 | 0.01 | 0% | Very High |

- Encoder achieves **38% inference-time reduction** vs standard supervised CNN with comparable F1; model size **0.52 M parameters**.
- **Cross-domain transfer:** pretrained on industrial compressor data, evaluated on CWRU; linear head fine-tuned with 5% target labels gives **F1 0.89** (~**96%** of source-domain performance), indicating domain-invariant vibration structures.

### 6.7 Statistical Significance
Paired two-tailed t-tests and Wilcoxon signed-rank tests over 5 runs (F1 and EDR): SSL gains over CNN and FFT+SVM are significant, **p < 0.01 (F1)** and **p < 0.005 (EDR trends)**.

**EDR threshold sensitivity:** ±10% threshold change → F1 variation within **1.8%**. Load- and temperature-normalized entropy baselines reduced false positives during regulated load increments.

### 6.8 Comparison with Transformer and Recent SSL Methods
- **CPC:** F1 **0.88**, EDR **0.80** (Week 12) – better than FFT+SVM and LSTM, below proposed.
- **Transformer + TS2Vec:** F1 **0.90**, EDR **0.85**; ~3× training time and larger memory footprint, less practical for edge.
- **Proposed Siamese CNN SSL:** F1 **0.93**, EDR **0.91**, low training cost, fast inference.

### 6.9 Ablation Study (baseline F1 0.93, EDR 0.91 @ Week 12)
1. **No temporal augmentations:** F1 0.87, EDR 0.79.
2. **No auxiliary pretext tasks:** F1 0.89, EDR 0.82.
3. **Shallow MLP instead of Siamese CNN encoder:** F1 0.84, EDR 0.74 (largest drop).
Temporal augmentations and Siamese CNN encoder contribute most; benefits are synergistic.

### 6.10 Embedding Analysis
t-SNE/UMAP show well-separated clusters for Normal, Imbalance, Misalignment, Bearing Wear. SSL embeddings have higher silhouette scores than supervised CNN embeddings.

### 6.11 ROC and Robustness (Fig. 6, Table 4)

| Fault category | AUC (mean ± SD) |
|---|---|
| Normal | 0.90 ± 0.01 |
| Imbalance | 0.91 ± 0.01 |
| Bearing Wear | 0.93 ± 0.01 |
| Misalignment | 0.93 ± 0.01 |

PR curves stable even for underrepresented classes (Bearing Wear).

### 6.12 Key Insights
1. High accuracy with minimal labeled data.
2. Earlier fault recognition (anticipates faults before escalation).
3. Robust, interpretable embedding space.
4. Practical deployment potential (edge devices, multi-site).

### 6.13 Physical Interpretation of Learned Representations
- **Low-frequency** modes ↔ rotational asymmetry and load transfer across stages.
- **Mid-frequency** ↔ pressure oscillations and inter-stage flow interactions.
- **High-frequency transients** ↔ valve slap, piston ring wear, clearance effects.
- Noted for wobble-plate and multistage setups: cyclical radial force variations and kinematic limits affect mid-band embeddings.
- Performance dropped moving from controlled datasets to industrial data (lower SNR), but fault separability persisted since it relied on temporal consistency rather than precise amplitude.

### 6.14 Ethics
No human participants; machine data from standard operational monitoring with organizational consent. The system is decision support only: it does not independently initiate safety-critical shutdowns; humans make final decisions.

---

## 7. Additional Discussion Sections
- **Digital twins and multi-physics sensing:** can act as a data-informed diagnostic component in digital twins; extend to motor current, air-gap flux, temperature, pressure, acoustics.
- **Nonlinear vibration and entropy dynamics:** compressor vibrations are nonlinear, intermittent, mode-coupled; EDR assesses dynamical complexity without assuming linearity or stationarity.
- **Economics:** training offline, on-site inference needs minimal hardware. Cost–benefit: **preventing a single unexpected shutdown justifies the edge unit cost**. Encoder feasible on embedded GPU systems with **< 15 W** power. Can feed maintenance optimization models with dependent competing failures.
- **Safety, compliance, false alarms:** EDR alerts screened by persistence and confidence before escalation.
- **Sensor degradation and data drift:** handled via shifting healthy reference.
- **Privacy-preserving federated deployment:** only embeddings or model updates shared.
- **Relation to advanced signal processing:** chirplet transforms, SKRgram, wavelet-guided networks as front-ends; physics-driven augmentation and diffusion-based generative models as future pathways.
- **Generalization:** applicable to bearings, gearboxes, pumps and motor-side electrical faults, especially with multiple sensors.

---

## 8. Conclusion
The framework combines contrastive SSL, domain-specific augmentations, auxiliary pretext tasks and EDR for label-efficient, early fault detection in multistage compressors. On a six-month industrial dataset plus adapted CWRU it achieves **F1 0.93** and **28% better early detection** than FFT-based and supervised deep learning baselines. It is scalable, label-efficient and deployment-ready.

## 9. Future Scope
1. **Domain adaptation** across compressor types, conditions and sites.
2. **Multi-modal integration** (temperature, pressure, acoustic, etc.).
3. **Online and continual learning** without full retraining.
4. **Real-time edge deployment** and full-scale plant trials.
Also: integrate SKRgram-based filtering for fault localization/interpretability; hybrid signal-processing + representation-learning pipelines.

## 10. Abbreviations
SSL (Self-Supervised Learning), EDR (Entropy Divergence Rate), FFT, CPC (Contrastive Predictive Coding), PdM, CNN, LSTM, t-SNE, UMAP, SVM, MLP, KL Divergence, FL (Federated Learning), SMOTE.

---

## 11. Internal Inconsistencies Worth Noting (observations from reading the paper)
- **"EDR" is used for two different things:** *Entropy Divergence Rate* (anomaly metric, defined as KL divergence in one place and as d|ΔH|/dt in the Fig. 3 flowchart) and *Early Detection Rate* (evaluation metric).
- **Sampling rate** is given as 10–50 kHz, 48 kHz (PXI-4472), and 25.6 kHz in different places; band-pass upper edge is 20 kHz while the accelerometers are rated to 10 kHz.
- **Encoder size:** 0.52 M parameters (Table 3 / text), 0.42 M (Conclusion), "less than X million" (placeholder left in Data Splitting section).
- **Software/hardware:** PyTorch 2.1 / RTX 3060 (experimental setup) vs PyTorch 1.12 / RTX 3080 (pretraining config).
- **Proposed model's F1:** 0.93 in text and ablation; 0.92 in Table 3 (SSL + linear head). Supervised CNN F1 is 0.84 in text but 0.93 in Table 3.
- **Projection head/embedding:** encoders described with 128–256-dim embeddings; projection head 256→128.
- **Confusion matrix (Fig. 4):** text reports 48/50, 47/50, 46/50, 48/50, but the plotted matrix has different counts and class label typos ("Insocal").
- **Figure references:** the text cites Fig. 4 for EDR trends (it is Fig. 5) and Fig. 5 for AUC values (it is Fig. 6); Fig. 1 labels "Signal Preprocessing" twice.
- **Augmentations:** listed as jitter/scaling/permutation in preprocessing, but permutation is shown to hurt separability in sensitivity analysis; the ablation lists "jittering, scaling, and permutation".
- Reference numbering for Chevtchenko et al. appears as [13] in one place and [25] in another; Costa et al. is cited as "(2024)" without a number.

---

## 12. References (as listed)
1. Zhao R, Yan R, Chen Z, Mao K, Wang P, Gao RX (2019) Deep learning and its applications to machine health monitoring. Mech Syst Signal Process 115:213–237.
2. Chen T, Kornblith S, Norouzi M, Hinton G (2020) A simple framework for contrastive learning of visual representations. ICML 119:1597–1607.
3. He K, Fan H, Wu Y, Xie S, Girshick R (2020) Momentum contrast for unsupervised visual representation learning. CVPR 9729–9738.
4. McInnes L, Healy J, Melville J (2018) UMAP. J Open Source Softw 3(29):861.
5. Smith W, Lee S, Kurfess TR (2015) Analysis of the CWRU bearing dataset. Case Western Reserve University.
6. Lee J, Qiu H, Yu G, Lin J (2007) IMS bearing dataset. IMS Center, Univ. of Cincinnati.
7. Case Western Reserve University (2015) Bearing data center dataset.
8. ISO 10816-3 (2009) Mechanical vibration – Evaluation of machine vibration by measurements on non-rotating parts – Part 3.
9. Mobley RK (2002) An Introduction to Predictive Maintenance, 2nd edn. Butterworth-Heinemann.
10. Goriveau R, Medjaher K, Zerhouni N (2016) From prognostics and health systems management to predictive maintenance. PHM Soc 8:1–12.
11. Industrial Internet Consortium (2015) Industrial Internet Reference Architecture (IIRA), v1.7.
12. Levin S (2024) Enhancing predictive maintenance in the industrial sector: a comparative analysis of ML models. AIP Conf Proc 3243:020035.
13. Costa A, Mastriani E, Incardona F, Munari K, Spinello S (2024) Hybrid clustering approaches for predictive maintenance of high-pressure industrial compressors. arXiv:2411.13919.
14. Habib MK, Mohamed K (2023) ML-based predictive maintenance using CNN–LSTM network. IEEE ICMA 2224–2229.
15. Zhang K, Wen Q, Zhang C, et al. (2024) Self-supervised learning for time series analysis: taxonomy, progress, and prospects. IEEE TPAMI 46(10):6775–6794.
16. Tran DH, Nguyen VL, Nguyen H, Jang YM (2022) Self-supervised anomaly detection in industrial IoT. Electronics 11(14).
17. Li N, Wang H (2025) Variable filtered-waveform VMD and its application in rolling bearing fault feature extraction. Entropy 27(3):277.
18. Wang H, Li YF, Men T, Li L (2024) Physically interpretable wavelet-guided networks with dynamic frequency decomposition for machine intelligence fault prediction. IEEE Trans SMC: Systems 54(8):4863–4875.
19. Liu Y, Huo M, Li M, He L, Qi N (2025) Establishing a digital twin diagnostic model based on cross-device transfer learning. IEEE Trans Instrum Meas 74:3533610.
20. He W, Hang J, Ding S, Sun L, Hua W (2024) Robust diagnosis of partial demagnetization fault in PMSMs using radial air-gap flux density. IEEE Trans Ind Electron 71(10):12001–12010.
21. Xu X, Hang J, Cao K, Ding S, Wang W (2025) Unified open-phase fault diagnosis of five-phase PMSM system. IEEE Trans Transp Electrific 11(5):12063–12075.
22. Ren Y, Zheng D, Yang Y, et al. (2025) A mechanism-guided intelligent enhancement method for early aging information of winding insulation. IEEE Trans Ind Inf 21(7):5440–5450.
23. Xu K, Bi K, Ge Y, Zhao L, Han Q, Du X (2020) Performance evaluation of inerter-based dampers for vortex-induced vibration control of long-span bridges. Struct Control Health Monit 27:e2529.
24. Zheng J et al. (2024) Potential and electro-mechanical coupling analysis of a novel HTS maglev system. IEEE Trans Intell Transp Syst 25(10):13573–13583.
25. Chevtchenko SF, Santos MCM, Vieira DM, Mota RL, Rocha E, Cruz B, Araújo D, Andrade E (2023) Predictive maintenance model based on anomaly detection in induction motors using real-time IoT data. arXiv:2310.14949.
26. Ringler N, Knittel D, Ponsart J-C, Nouari M, Yakob A, Romani D (2023) ML-based real-time predictive maintenance at the edge for manufacturing systems. IEEE GlobConET.
27. Cui W, Zhao L, Ge Y, Xu K (2024) A generalized van der Pol nonlinear model of vortex-induced vibrations of bridge decks with multistability. Nonlinear Dyn 112:259–272.
28. Govindaswamy TR, Ramadoss R (2026) Fault diagnosis and RUL estimation in rotating machinery: a review. J Vib Eng Technol (in press).
29. Liao W, Wang Y (2013) Data-driven machinery prognostics approach using a predictive maintenance model. J Comput 8(1):225–231.
30. Calabrese M, Levialdi N, Tibaldi L (2020) SOPHIA: an event-based IoT architecture for predictive maintenance. Information 11(4):202.
31. Paolanti M, Romeo L, Felicetti A, Mancini A, Frontoni E, Loncarski J (2018) ML approach for predictive maintenance in Industry 4.0. IEEE/ASME MESA 1–6.
32. Ünal AAF, Kaleli AY, Ummak E, Albayrak Ö (2021) A comparison of state-of-the-art ML algorithms on fault indication and RUL determination by telemetry data. FiCloud 79–85.
33. Ingole O, Pande A, Dongre A, Jadhav D, Dhamecha D, Daspute H (2022) Investigation of different regression models for predictive maintenance of aircraft engines. ICERECT 1–6.
34. Kairouz P, McMahan HB et al. (2021) Advances and open problems in federated learning. Found Trends Mach Learn 14(1–2):1–210.
35. Ahn J, Lee Y, Lee Y, Kim N, Park C, Jeong J (2023) Federated learning for predictive maintenance and anomaly detection using time-series distribution shifts in manufacturing. Sensors 23(17):7331.
36. Zhang Y, Xu Y, Zhou J, Zhou Y, Mahfoud J (2025) Vibration control of AMB-rotor system under base motions based on disturbance observer. IEEE/ASME Trans Mechatron 30(6):5398–5407.
37. Tian Z, Lee A, Zhou S (2024) Adaptive tempered reversible jump algorithm for Bayesian curve fitting. Inverse Probl 40(4):045024.
38. He D, Lin Y, Dai Z, Yang SX (2025) Neurodynamics-based visual servo predictive control for logistics omnidirectional robots. IEEE Trans Ind Electron 72(12):14646–14655.
39. Zhou Y, Yao F, Wang Y, Bai C, Liu S, Abdel Wahab M (2025) Free vibration of orthotropic rectangular Mindlin plates via spectral element model. Thin-Walled Struct 216:113679.
40. Zhou Y, Yao F, Bai C, Li K, Zhu S, Abdel Wahab M (2024) Bandgap characteristics of periodic Mindlin plates via the spectral element method. Thin-Walled Struct 205:112370.
41. Yuan J, Szydlowski M, Wang X (2024) An optimal sparse sensing approach for scanning point selection and response reconstruction in full-field structural vibration testing. Mech Syst Signal Process 212:111298.
42. Qin S, Tang J (2023) Modal parameter identification in civil structures via Hilbert transform ensemble with improved empirical wavelet transform. J Vib Control 30(7–8):1621–1634.
43. Tran KQ, Duong TV, Hoang TD, Abdel Wahab M, Hackl K, Nguyen-Xuan H (2025) A new thermoelastic model for agglomerated and randomly-oriented CNT-reinforced bio-inspired materials. Eng Anal Bound Elem 174:106157.
44. Gardner WA (2023) Transitioning away from stochastic process models. J Sound Vib 565:117871.
45. Lu S, He Q, Wang J (2019) A review of stochastic resonance in rotating machine fault detection. Mech Syst Signal Process 116:230–260.
46. Matsubara M, Takara R, Komatsu T, Furuta S, Khoo PL, Kobayashi M, Mushiaki H, Uesugi K, Kawamura S, Tajiri D (2023) In-situ measurement of dynamic micro X-ray CT and dynamic mechanical analysis for rubber materials. Mech Syst Signal Process 205:110875.
