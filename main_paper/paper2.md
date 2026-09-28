# FaultFormer: Pretraining Transformers for Adaptable Bearing Fault Classification

## Bibliographic Details
- **Journal:** IEEE Access, Volume 12, 2024, pp. 70719–70728 (Research Article)
- **DOI:** 10.1109/ACCESS.2024.3399670
- **Authors:** Anthony Y. Zhou and Amir Barati Farimani
- **Affiliation:** Department of Mechanical Engineering, Carnegie Mellon University, Pittsburgh, PA 15213, USA
- **Corresponding author:** Amir Barati Farimani (barati@cmu.edu)
- **Dates:** Received 5 Apr 2024 / Accepted 7 May 2024 / Published 9 May 2024 / Current version 24 May 2024
- **Associate editor:** Guillermo Valencia-Palomo
- **License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
- **Index terms:** Bearing fault detection, machine health monitoring, signal classification, transformer, pretraining
- **Code library used:** x-transformers (https://github.com/lucidrains/x-transformers)
- **Acknowledgment:** Cooper Lorsung, for discussions on applying transformers to scientific and engineering problems.

---

## 1. Abstract
Global consumption growth motivates deep learning for smart manufacturing and machine health monitoring. Vibration analysis supports predictive maintenance via bearing fault detection. Deep learning models lack generalizability to new tasks or datasets and need expensive labeled data. The paper presents a **self-supervised pretraining and fine-tuning framework based on transformers**, investigating tokenization and data augmentation strategies to reach state-of-the-art accuracies. It demonstrates **self-supervised masked pretraining** for vibration signals and its use in **low-data regimes, task adaptation and dataset adaptation**. Pretraining improves performance on scarce unseen training samples, on fine-tuning to fault classes outside the pretraining distribution, and enables **few-shot generalization to a different dataset**. New paradigm: pretrain on unlabeled data from different bearings, faults and machinery, then quickly deploy to new data-scarce applications.

---

## 2. Introduction

### Motivation
- Emerging economies and modernization drive consumer goods demand; manufacturing must be faster, better, cheaper. Machine health monitoring and predictive maintenance is a key avenue.
- Most plants still use reactive, run-to-fail maintenance; **82% of companies had at least one machine failure in the last three years** [1]. Unplanned downtime costs **Fortune Global 500 companies $1.5 trillion annually** [2].
- Nearly **50% of manufacturers** already integrate IoT sensors [3]. Vibration signals describe structural weakness, looseness in moving components, resonance. Datasets on common failures (bearing faults) exist [4], [5].
- Bearing fault detection is a rich area, with recent work centered on the **CWRU Bearing Dataset [4]**.

### Prior Work Reviewed
- **Classical:** simple time- or frequency-domain approaches [6].
- **CNNs:** accuracies of 92–99%, with better noise resilience and lower training time/data [11–13]. Variants: vanilla CNNs [7], adaptive CNNs [14], multi-scale CNNs [15], attention-guided CNNs [16], time+spectral domain combos [17], extra featurized channels [18], data fusion from multiple training samples [19], long-tailed distribution handling [20].
- **Autoencoders:** denoising autoencoders learn a latent space for a fine-tuned head; accuracy of 99% vs 70% for networks without latent representations [8]. Variants: stacked sparse [21], ensemble [22], wavelet [23].
- **RNNs/LSTMs:** RNNs on CWRU [9]; RNN+CNN combos [24]; GRUs for stable training [25]; alternative feature extraction with RNNs [26].
- **Transformers** [27]: attention avoids vanishing/exploding gradients of long backpropagation through time. Bearing fault applications: [10], [28], [29]; robustness [30] and trustworthiness [31] variants. These do not study self-supervised pretraining, a major transformer advantage (as in BERT [32] and GPT [33]).
- **Pretraining for bearings:** masked pretraining on a data-scarce 4-way classification [34]; contrastive pretraining for long-tailed faults [35] and robust latent features [36]. **Limitation:** they pretrain and fine-tune on the same classes and datasets, i.e., the same distribution.

### This Paper's Positioning and Contributions
Pretrained model adapts to **unseen fault classes** and an **entirely new dataset**, matching real-world use where factory needs differ widely. Contributions:
1. Investigate transformer training strategies for vibration data: tokenization strategies, data augmentations, and masked pretraining with unlabeled data.
2. Validate adapting pretrained transformers to scarce datasets; compare models and training sample sizes.
3. Novel results on fine-tuning pretrained transformers to **new fault classes** and **entirely new data**; fine-tuning shown quicker and more accurate than models fully trained on new tasks/data.

---

## 3. Methods

### 3.1 Proposed Architecture (Fig. 1)
- Transformer encoder that can be pretrained or fine-tuned.
- **(a) End-to-end training / fine-tuning:** Raw signal → optional augmentation → **Tokenizer** → **linear projection (MLP embedding)** → class token prepended + **RoPE (rotary positional embeddings [37])** → Transformer encoder → class embedding → **MLP head** → class (fault type, fault size).
- **(b) Masked pretraining:** Raw signal → tokenizer → linear projection → mask applied after tokenization → Transformer encoder → MLP decoder → **reconstruction loss**.
- Masking: a percentage of tokens are zero-masked, randomly replaced, or left unchanged.

### 3.2 Data Augmentation (Fig. 2)
Inspired by self-supervised contrastive learning (SimCLR [38]); approximates a larger dataset for scarce data and reduces overfitting. Parameters sampled uniformly from ranges so each augmentation yields a unique sample. Augmentation probability is tunable during training; test samples are untouched.
Augmentations: **Gaussian noise, cutout, crop (and resize), shift**, plus combinations.

### 3.3 Signal Tokenization
Needed because raw signals are thousands of points long (too expensive for transformers) and not every point is informative. Three tokenizers:
1. **Constant tokenizer:** reshapes (batch, seq_length, 1) → (batch, seq_length/d, d) — stacks multiple points per token.
2. **CNN tokenizer:** 1D convolutions add channels and shorten the sequence.
3. **Fourier tokenizer:** takes the FFT and extracts the **top 40 modes** per input; real and imaginary amplitudes plus frequency stacked in the channel dimension: (batch, seq_length, 1) → (batch, 40, 3).

### 3.4 Transformer Encoder
Stacked layers, each with multi-head self-attention, a fully connected feed-forward layer, dropout, residual connections and layer norm. Equations:
- Attention: A_n(Q,K,V) = softmax(QKᵗ/√d) V (1)
- Q = XW^q, K = XW^k, V = XW^v (2)
- Head_i = A_n((Q/h)_i, (K/h)_i, (V/h)_i) (3)
- A_h(Q,K,V) = Concat(Head_1, …, Head_n) (4)
- FF(X) = σ(XW_1 + b_1)W_2 + b_2 (5) [nonlinearity σ per GLU variants [39]]
- X_a = LayerNorm(A_h(X) + X) (6)
- X_out = LayerNorm(FF(X_a) + X_a) (7)

### 3.5 Experiments and Datasets
Two datasets: **CWRU** (10 classes, 2,800 samples) and **Paderborn University** (3 classes, 58,000 samples).

**Table 1: Experiments and dataset splits** (* = unlabeled)

| Experiment | Split | Dataset | # Samples | Classes |
|---|---|---|---|---|
| Baseline | Train | CWRU | 2240 | 0–9 |
| Baseline | Test | CWRU | 560 | 0–9 |
| Data Scarcity | Pretraining | CWRU* | 2000 | 0–9 |
| Data Scarcity | Train | CWRU | 400/200/100 | 0–9 |
| Data Scarcity | Test | CWRU | 100 | 0–9 |
| Task Adaptation | Pretraining | CWRU* | 1960 | 0–2, 4–5, 7–8 |
| Task Adaptation | Train | CWRU | 672 | 3, 6, 9 |
| Task Adaptation | Test | CWRU | 168 | 3, 6, 9 |
| Dataset Adaptation | Pretraining | CWRU* | 2800 | 0–9 |
| Dataset Adaptation | Train | Paderborn | 46400 | 0–2 |
| Dataset Adaptation | Test | Paderborn | 11600 | 0–2 |

**Experiment design:**
- **Baseline:** transformers trained end-to-end on standard 80/20 CWRU split; different tokenizers and augmentation probabilities benchmarked against LSTM, CNN, MLP.
- **Data scarcity:** pretrain on unlabeled CWRU, fine-tune on small labeled subsets; models trained from scratch as benchmark.
- **Task adaptation:** pretrain on unlabeled data from a setup, fine-tune on new faults observed on the same setup (hold out certain CWRU classes when pretraining).
- **Dataset adaptation:** pretrain on unlabeled data from one setup, fine-tune on a different setup (different motors, bearings, sensors or defects): CWRU → Paderborn.

---

## 4. Results and Discussion

### 4.1 Baseline Results (Table 2: test accuracy by tokenizer, model, augmentation probability)

| Tokenizer | Model | 0.0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|---|
| Fourier | Transformer | 0.6268 | 0.6321 | 0.65 | **0.9984** |
| Fourier | LSTM | 0.5286 | 0.5571 | 0.7232 | 0.8857 |
| Fourier | CNN | 0.3286 | 0.4482 | 0.5232 | 0.5732 |
| Fourier | MLP | 0.2571 | 0.3482 | 0.2911 | 0.5696 |
| CNN | Transformer | 0.9911 | 0.9893 | 0.9839 | 0.9821 |
| CNN | LSTM | 0.9661 | 0.9946 | 0.9964 | 0.9946 |
| CNN | CNN | 0.9911 | 0.9982 | 0.9911 | 0.9982 |
| CNN | MLP | 0.9054 | 0.9982 | 0.9964 | 0.9964 |
| Constant | Transformer | 0.9107 | 0.9321 | 0.925 | 0.9089 |
| Constant | LSTM | 0.9929 | 0.9946 | 0.9964 | 0.9982 |
| Constant | CNN | 0.9964 | 0.9911 | 0.9929 | 0.9911 |
| Constant | MLP | 0.8946 | 0.9679 | 0.9875 | 0.9946 |

**Findings:**
- High-performing models use frequent augmentation. Transformers perform as well as benchmark models and **outperform others in the Fourier domain** (99.84% at 0.9 augmentation probability).
- LSTM, CNN and MLP overfit the Fourier-basis training data (reaching 100% train, low test accuracy). Fourier coefficients compactly represent global trends but reduce the amount of data; time-domain augmentations substantially change Fourier coefficients, so augmentation creates diverse training samples and makes the task harder and data-rich.
- Even simple **MLPs can reach 99%** with augmentation (not conventional in literature).
- Transformers struggle **without a tokenization strategy**: naive reshaping (constant tokenizer) prevents attention resolving relationships *within* tokens.

### 4.2 Visualizing Results (Fig. 3: attention to Fourier modes)
Attention scores between class token and other tokens per head:
- **Shallow layers (e.g., Layer 0):** heads attend to a single, dominant (largest) Fourier mode; many heads attend to the same token.
- **Deeper layers (e.g., Layer 5):** heads attend to combinations of tokens, each head differentiates to different sets, including smaller Fourier modes.
- Indicates learning of **both global and fine multiscale relationships**, aggregated in the class token.

### 4.3 Masked Pretraining (Fig. 4)
- Language-model-style masking: **50% of tokens masked**; of those, **70% set to zero, 20% replaced by a random value in the signal, 10% left untouched**.
- Aggressive masking helps because signal data is easier to reconstruct than language tokens and forces more expressive latent representations.
- Qualitative result: encoder reconstructs masked signals nearly perfectly (no noticeable error).

### 4.4 Data Scarcity (Table 3: 10-way CWRU classification)

| Model | 100 samples | 200 samples | 400 samples |
|---|---|---|---|
| Transformer-PT (pretrained) | **0.9327** | **0.9615** | **0.9615** |
| Transformer | 0.8077 | 0.8173 | 0.8462 |
| LSTM | 0.8654 | 0.9519 | **0.9615** |
| CNN | 0.9135 | **0.9615** | **0.9615** |
| MLP | 0.7596 | 0.8654 | 0.9038 |

- Pretraining increases performance with small amounts of unseen data; benchmarks converge as samples increase, but pretrained models hit optimum with fewer samples.
- **Non-pretrained transformers perform poorly** in low-data regimes (large data needs, lack of inductive bias) → pretraining is essential for transformers here.

### 4.5 Task Adaptation (Fig. 5: unseen CWRU fault classes)
Pretrain on unlabeled healthy, inner, outer and ball fault data; fine-tune on inner/outer/ball faults with **larger fault sizes** than in pretraining (simulating unseen faults).

| Model | Epoch 1 | 2 | 5 | 20 | 40 |
|---|---|---|---|---|---|
| Transformer-PT | 0.6071 | 0.6429 | **0.9524** | **1.000** | **1.000** |
| Transformer | **0.6190** | **0.7857** | 0.9464 | 0.9821 | 0.9821 |
| LSTM | 0.3631 | 0.3631 | 0.3036 | 0.6190 | 0.9583 |
| CNN | 0.2976 | 0.4762 | 0.9405 | 0.9940 | 0.9821 |
| MLP | 0.3274 | 0.3333 | 0.8690 | 0.8393 | 0.9464 |

- Pretrained models reach better final performance and adapt quickly. An end-to-end transformer initially leads, likely because the pretrained model's head is randomly initialized and needs a few epochs before using the pretrained encoder's context.
- All models converge to high accuracy; the benefit of pretraining is **reaching it faster with less computation**.

### 4.6 Dataset Adaptation (Fig. 6: CWRU → Paderborn)

| Model | Epoch 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| Transformer-PT | **0.8010** | **0.9021** | **0.9691** | **0.9909** | **0.9946** |
| Transformer | 0.6509 | 0.7922 | 0.9257 | 0.9599 | 0.9700 |
| LSTM | 0.4956 | 0.5639 | 0.6562 | 0.8619 | 0.9757 |
| CNN | 0.5509 | 0.7088 | 0.8141 | 0.9111 | 0.9667 |
| MLP | 0.4293 | 0.5323 | 0.6566 | 0.7032 | 0.7964 |

- Pretrained models generalize to a new dataset in a **few-shot manner**: after two epochs, **> 90% accuracy, ~5× faster than the CNN benchmark**, and a higher final accuracy.
- Suggests masked pretraining context generalizes across datasets, enabling pretraining on unlabeled data across machinery, faults and operating conditions.

---

## 5. Discussion
- Deep learning trends emphasize pretraining large models for quick fine-tuning and benefit from shared learning on heterogeneous data; transformer architecture [27] and self-supervised pretraining [33] enabled this in language and vision. This work transfers the paradigm to mechanical/manufacturing domains.
- Self-supervised pretraining exploits a much larger unlabeled dataset when labeled real-world data is scarce; proxy tasks can yield powerful latent representations (theoretical work: [40], [41]).
- Different mechanical faults likely share common features that let models leverage past knowledge on new data.
- Even simple MLPs can classify bearing faults; the challenge is time, compute, dataset collection/labeling and training pipeline. Fine-tuning on top of a pretrained model gives a powerful initialization conditioned on prior time, compute and data.

## 6. Conclusion
- Framework for pretraining/fine-tuning transformers for bearing fault classification. Tokenization and augmentation strategies make transformers match current deep learning approaches. The class token learns diverse dependencies; attention heads focus on different Fourier modes at different layers (coarse and fine details).
- Masked self-supervised pretraining evaluated in three experiments: low-data regimes (outperforms), unseen fault classes (higher accuracy than end-to-end), and unseen datasets (few-shot generalization to new fault, machinery or operating conditions).
- Shift from end-to-end training to pretraining/fine-tuning offers more flexible models in practice.
- **Limitations and future work:**
  - Effective pretraining requires an **extremely large (unlabeled) dataset**.
  - Deep learning predictions in critical maintenance can be hard to understand or quantify in uncertainty.
  - Unlabeled data likely skewed toward **healthy bearings**; methods to weight faulty samples are of interest.
  - Scale to larger transformers and more diverse unlabeled datasets.
  - Explore pretraining beyond masking; include other modalities (video, audio).
  - Vision: **foundation models for mechanical data** that quickly perform diverse tasks on the factory floor.

---

## 7. Appendices

### Appendix A: Data Augmentation Details
Sample a probability uniformly in [0,1]; if less than the augmentation probability, choose among **eight** possibilities with equal probability:
1. **Gaussian noise:** std sampled uniformly from [0, 0.05].
2. **Shift:** shifted by a timestep sampled uniformly from [−l/2, l/2] (l = signal length).
3. **Cutout:** window zeroed out; window length uniform in [100, 500]; start point uniform in [0, l − l_w].
4. **Crop:** window of length l/2 chosen from start uniform on [0, l/2], then **upsampled with linear interpolation** to original length.
5. **Cutout + Shift** (shift then cutout)
6. **Cutout + Gaussian noise** (noise then cutout)
7. **Crop + Shift** (shift then crop)
8. **Crop + Gaussian noise** (noise then crop)

### Appendix B: Tokenizers
- **Constant:** (batch, seq_length, 1) → (batch, seq_length/d, d).
- **CNN tokenizer layers (Table 4):**

| Layer | Output shape | Kernel | Stride |
|---|---|---|---|
| Input | (B, 1, L) | – | – |
| Conv1D | (B, 4, L/2) | 4 | 2 |
| GELU | – | – | – |
| Conv1D | (B, 8, L/4) | 4 | 2 |

- **Fourier:** FFT of input, top 40 modes; real, imaginary amplitudes and frequency stacked → (batch, 40, 3).

### Appendix C: Dataset Details
**A. CWRU dataset**
- Test rig: **2 hp electric motor** driving a shaft with torque transducer, encoder, dynamometer; bearings at drive end and fan end.
- Faults introduced by **electro-discharge machining**, diameters **0.18–0.53 mm**, giving **10 test cases** (Table 5). Motor loads 0–3 hp; vibration measured by accelerometers on the motor housing at 12 kHz and 48 kHz.
- Data used: **48 kHz, 2 hp load, 1750 rpm**. Each sample covers one revolution: ~1670 points per revolution, truncated by 35 points at each end → **1600 points**. The 467,600 data points per class → **280 samples per class**, 2,800 total.

**Table 5: CWRU classes**

| Fault type | Size (mm) | Samples | Label |
|---|---|---|---|
| Normal | N/A | 280 | 0 |
| Ball | .18 | 280 | 1 |
| Ball | .36 | 280 | 2 |
| Ball | .53 | 280 | 3 |
| Inner race | .18 | 280 | 4 |
| Inner race | .36 | 280 | 5 |
| Inner race | .53 | 280 | 6 |
| Outer race | .18 | 280 | 7 |
| Outer race | .36 | 280 | 8 |
| Outer race | .53 | 280 | 9 |

**B. Paderborn University dataset** [5]
- 32 bearings: 6 healthy, 12 artificially damaged, 14 naturally damaged. Split into three classes: **healthy, inner race fault, outer race fault**; 3 bearings with both inner and outer race faults omitted.
- Condition: **1500 rpm, 0.7 Nm torque, 1000 N radial force**. Each bearing used 20 times per load condition → **580 signals**; each 4 s at 64 kHz split into 100 samples → **58,000 samples**, 3 classes.

### Appendix D: Implementation Details
Implemented with **x-transformers** (RoPE, flash attention, GLU activations in feed-forward). **AdamW** with **one-cycle scheduler** (learning-rate warmup).

**Table 6: Fine-tuning hyperparameters / Table 7: Pretraining hyperparameters**

| Hyperparameter | Fine-tuning | Pretraining |
|---|---|---|
| Batch size | 16 | 16 |
| Epochs | 1000 | 1000 |
| Input dimension | variable | variable |
| Model dimension | 256 | 256 |
| Attention heads | 32 | 32 |
| Layers | 4 | 4 |
| Dropout | 0.3 | 0.3 |
| Warmup steps | 100 | 100 |
| Min learning rate | 1e-4 | 1e-4 |
| Max learning rate | 1e-3 | 1e-3 |
| Beta 1 / Beta 2 | 0.9 / 0.98 | 0.9 / 0.98 |
| Mask probability | – | 0.5 |
| Random mask probability | – | 0.2 |
| Replace probability | – | 0.9 |

**Table 8: CNN baseline architecture**

| Layer | Output shape | Kernel | Stride |
|---|---|---|---|
| Input | (B, 8, L) | – | – |
| Conv1D | (B, 32, L/2) | 10 | 2 |
| GELU | | | |
| Conv1D | (B, 256, L/2) | 5 | 1 |
| GELU | | | |
| Conv1D | (B, 512, L/2) | 3 | 1 |
| GELU | | | |
| Conv1D | (B, 256, L/2) | 3 | 1 |
| GELU | | | |
| Conv1D | (B, 256, L/4) | 3 | 2 |
| AdaptiveAvgPool | (B, 256, 8) | – | – |
| Flatten | (B, 2048) | – | – |
| Dropout | – | | |
| Linear | (B, 512) | | |
| GELU | | | |
| Dropout | | | |
| Linear | (B, 526) | | |
| GELU | | | |
| Linear | (B, 10) | | |

**Table 9: MLP baseline architecture:** Input (B, 8, L) → AdaptiveAvgPool (B, 256, 8) → Flatten (B, 2048) → Dropout → Linear (B, 1024) → GELU → Dropout → Linear (B, 1024) → GELU → Linear (B, 512) → GELU → Linear (B, 256) → GELU → Linear (B, 10). Adaptive average pooling handles different input sequence lengths by pooling along length to the model dimension.

*(Note: the paper says pretraining uses "slightly different hyperparameters" because of the mask and lack of a decoder; the pretraining table adds mask/random-mask/replace probabilities. The "Replace probability 0.9" value in Table 7 differs from the 70/20/10 masking split described in the text.)*

---

## 8. Author Biographies
- **Anthony Y. Zhou:** B.S. in mechanical engineering, UC Berkeley (2023). Currently a Ph.D. student in mechanical engineering at Carnegie Mellon University; member of the Mechanical and AI Laboratory (MAIL), advised by Dr. Amir Barati Farimani. Research: intersection of mechanical engineering and machine learning, with applications to surrogate models for partial differential equations.
- **Amir Barati Farimani:** Ph.D. in mechanical science and engineering, University of Illinois at Urbana–Champaign (2015); thesis "Detecting and Sensing Biological Molecules using Nanopores" (atomistic simulations of DNA sensing in biological and solid-state nanopores). Postdoc in Vijay Pande's lab at Stanford, combining ML and molecular dynamics for G-protein coupled receptors (Mu-Opioid receptors: free energy landscape, activation mechanism). Leads MAIL at CMU: ML, data science and molecular dynamics for health and bio-engineering; multidisciplinary group (mechanical, computer science, bio-engineering, physics, materials, chemical engineering); mission: bring state-of-the-art ML to mechanical engineering, developing data-driven models that incorporate physics and noise/stochasticity; use multi-scale simulations (CFD, MD, DFT) to generate data.

---

## 9. References (as listed)
1. *After the Fall: The Costs, Causes and Consequences of Unplanned Downtime*, PTC, 2023.
2. *The True Cost of Downtime 2022*, Siemens AG, 2023.
3. P. Wellener et al., "Exploring the industrial metaverse," Deloitte, 2023.
4. CWRU Bearing Dataset, Case Western Reserve Univ.
5. C. Lessmeier, J. K. Kimotho, D. Zimmer, W. Sextro, "Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: a benchmark data set for data-driven classification," Proc. Eur. Conf. PHM Soc., 2016, pp. 1–10.
6. N. Tandon, "A comparison of some vibration parameters for the condition monitoring of rolling element bearings," Measurement 12(3):285–289, 1994.
7. C. Lu, Z. Wang, B. Zhou, "Intelligent fault diagnosis of rolling bearing using hierarchical convolutional network based health state classification," Adv. Eng. Informat. 32:139–151, 2017.
8. F. Jia, Y. Lei, J. Lin, X. Zhou, N. Lu, "Deep neural networks: a promising tool for fault characteristic mining and intelligent diagnosis of rotating machinery with massive data," Mech. Syst. Signal Process. 72–73:303–315, 2016.
9. H. Jiang, X. Li, H. Shao, K. Zhao, "Intelligent fault diagnosis of rolling bearings using an improved deep recurrent neural network," Meas. Sci. Technol. 29(6), 2018.
10. Y. Ding, M. Jia, Q. Miao, Y. Cao, "A novel time–frequency transformer based on self–attention mechanism and its application in fault diagnosis of rolling bearings," Mech. Syst. Signal Process. 168, 2022.
11. W. A. Smith, R. B. Randall, "Rolling element bearing diagnostics using the CWRU data: a benchmark study," Mech. Syst. Signal Process. 64–65:100–131, 2015.
12. S. Zhang, S. Zhang, B. Wang, T. G. Habetler, "Deep learning algorithms for bearing fault diagnostics—a comprehensive review," IEEE Access 8:29857–29881, 2020.
13. D. Neupane, J. Seok, "Bearing fault detection and diagnosis using CWRU dataset with deep learning approaches: a review," IEEE Access 8:93155–93178, 2020.
14. X. Guo, L. Chen, C. Shen, "Hierarchical adaptive deep convolution neural network and its application to bearing fault diagnosis," Measurement 93:490–502, 2016.
15. Z. Zilong, Q. Wei, "Intelligent fault diagnosis of rolling bearing using one-dimensional multi-scale deep CNN based health state classification," ICNSC 2018.
16. H. Wang, Z. Liu, D. Peng, Z. Cheng, "Attention-guided joint learning CNN with noise robustness for bearing fault diagnosis and vibration signal denoising," ISA Trans. 128:470–484, 2022.
17. S. Li, G. Liu, X. Tang, J. Lu, J. Hu, "An ensemble deep convolutional neural network model with improved D-S evidence fusion for bearing fault diagnosis," Sensors 17(8):1729, 2017.
18. R. Magar, L. Ghule, J. Li, Y. Zhao, A. B. Farimani, "FaultNet: a deep convolutional neural network for bearing fault classification," IEEE Access 9:25189–25199, 2021.
19. M. Xia, T. Li, L. Xu, L. Liu, C. W. de Silva, "Fault diagnosis for rotating machinery using multiple sensors and CNNs," IEEE/ASME Trans. Mechatronics 23(1):101–110, 2018.
20. L. Cui, Z. Dong, H. Xu, D. Zhao, "Triplet attention-enhanced residual tree-inspired decision network," Adv. Eng. Informat. 59, 2024.
21. J. Sun, C. Yan, J. Wen, "Intelligent bearing fault diagnosis method combining compressed data acquisition and deep learning," IEEE Trans. Instrum. Meas. 67(1):185–195, 2018.
22. H. Shao, H. Jiang, Y. Lin, X. Li, "A novel method for intelligent fault diagnosis of rolling bearings using ensemble deep auto-encoders," Mech. Syst. Signal Process. 102:278–297, 2018.
23. W. Du, P. Hu, H. Wang, X. Gong, "Bearing fault diagnosis based on wavelet transform and auto-encoder neural network," SDPC 2019, pp. 469–473.
24. H. Pan, X. He, S. Tang, F. Meng, "An improved bearing fault diagnosis method using one-dimensional CNN and LSTM," J. Mech. Eng./Strojniski Vestnik 64:443–452, 2018.
25. H. Liu, J. Zhou, Y. Zheng, W. Jiang, Y. Zhang, "Fault diagnosis of rolling bearings with recurrent neural network-based autoencoders," ISA Trans. 77:167–178, 2018.
26. L. Guo, N. Li, F. Jia, Y. Lei, J. Lin, "A recurrent neural network based health indicator for remaining useful life prediction of bearings," Neurocomputing 240:98–109, 2017.
27. A. Vaswani et al., "Attention is all you need," 2017, arXiv:1706.03762.
28. Y. Hou, J. Wang, Z. Chen, J. Ma, T. Li, "Diagnosisformer: an efficient rolling bearing fault diagnosis method based on improved transformer," Eng. Appl. Artif. Intell. 124, 2023.
29. H. Fang, J. An, H. Liu, J. Xiang, B. Zhao, F. Dunkin, "A lightweight transformer with strong robustness application in portable bearing fault diagnosis," IEEE Sensors J. 23(9):9649–9657, 2023.
30. Y. Xiao, H. Shao, J. Wang, S. Yan, B. Liu, "Bayesian variational transformer: a generalizable model for rotating machinery fault diagnosis," Mech. Syst. Signal Process. 207, 2024.
31. Y. Xiao, H. Shao, M. Feng, T. Han, J. Wan, B. Liu, "Towards trustworthy rotating machinery fault diagnosis via attention uncertainty in transformer," J. Manuf. Syst. 70:186–201, 2023.
32. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, "BERT," 2018, arXiv:1810.04805.
33. A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, "Improving language understanding by generative pre-training," OpenAI, 2018.
34. J. Cen et al., "A mask self-supervised learning-based transformer for bearing fault diagnosis with limited labeled samples," IEEE Sensors J. 23(10):10359–10369, 2023.
35. R. Hou et al., "Contrastive-weighted self-supervised model for long-tailed data classification with vision transformer augmented," Mech. Syst. Signal Process. 177, 2022.
36. W. Mao, Z. Chen, Y. Zhang, Z. Zhong, "Harmony better than uniformity: a new pre-training anomaly detection method with tensor domain adaptation for early fault evaluation," Eng. Appl. Artif. Intell. 127, 2024.
37. J. Su et al., "RoFormer: enhanced transformer with rotary position embedding," Neurocomputing 568, 2023.
38. T. Chen, S. Kornblith, M. Norouzi, G. E. Hinton, "A simple framework for contrastive learning of visual representations," 2020, arXiv:2002.05709.
39. N. Shazeer, "GLU variants improve transformer," 2020, arXiv:2002.05202.
40. A. Radford et al., "Language models are unsupervised multitask learners," OpenAI Blog 1(8):9, 2019.
41. T. B. Brown et al., "Language models are few-shot learners," 2020, arXiv:2005.14165.

---

## 10. Small Inconsistencies Worth Noting
- The text says 10 classes / 2,800 samples for CWRU, but the Table 1 baseline uses 2,240 train + 560 test (an 80/20 split of 2,800), and the data-scarcity test set is only 100 samples.
- Table 1 data-scarcity pretraining uses 2,000 samples, task-adaptation pretraining 1,960, and dataset-adaptation pretraining 2,800.
- Text describes masking as 50% of tokens with a 70/20/10 zero/random/unchanged split, while Table 7 lists "replace probability 0.9".
- Table 2 lists an odd result: Fourier-tokenizer Transformer at 0.0 augmentation is only 0.6268, but at 0.9 it reaches 0.9984 (best in the table), while the CNN/Constant tokenizers give ~0.99 without augmentation.
- The CNN tokenizer table lists only two conv layers, giving output (B, 8, L/4); the CNN baseline table has a "Linear (B, 526)" layer (likely 512 or 526 as a typo).
