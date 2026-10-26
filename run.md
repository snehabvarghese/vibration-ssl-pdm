Here is how to run the project, including the end-to-end pipeline, the interactive Streamlit dashboard, the Jupyter notebooks, and the test suite:

---

### 1. Prerequisites & Virtual Environment

Open a terminal in the project folder and activate the virtual environment:

```bash
cd /Users/snehabvarghese/vibration-ssl-pdm
source venv/bin/activate
export PYTHONPATH=.
```

---

### 2. Option A: Run the End-to-End Pipeline (One Command)

To execute all 12 pipeline steps sequentially (data download $\to$ preprocessing $\to$ 150-epoch SSL pretraining $\to$ embedding extraction $\to$ anomaly detection $\to$ SSL fine-tuning & evaluation):

```bash
./run_phase1.sh
```

---

### 3. Option B: Run Individual Pipeline Steps

You can also run any single step individually:

```bash
# 1. Raw dataset download & data exploration
python -m preprocessing.download_cwru --all
python -m preprocessing.explore

# 2. Build processed segments dataset
python -m preprocessing.build_dataset

# 3. Train SSL contrastive encoder (150 epochs)
python -m training.train_ssl

# 4. Extract 128-D embeddings & generate t-SNE / UMAP plots
python -m training.extract_embeddings
python -m evaluation.visualization

# 5. Fit & evaluate EDR Anomaly Detector
python -m anomaly_detection.anomaly_detector

# 6. Evaluate Limited-Label Classifiers & SSL Fine-tuning
python -m training.train_classifier
python -m training.train_finetune
python -m training.train_baseline

# 7. Aggregate final comparison table & figures
python -m evaluation.evaluate
```

---

### 4. Option C: Run the Interactive Web Dashboard

To launch the interactive Streamlit dashboard in your web browser:

```bash
streamlit run dashboard/app.py
```

- Access it in your browser at: `http://localhost:8501`
- Select any CWRU recording or upload a custom `.mat` / `.npy` file to inspect live status (**NORMAL / ANOMALOUS**), predicted fault class, EDR anomaly meter, embedding map, and confidence score.

---

### 5. Option D: Run Interactive Jupyter Notebooks

To launch Jupyter and explore the step-by-step notebooks:

```bash
jupyter notebook notebooks/
```

Available notebooks in `notebooks/`:
- [`01_data_exploration.ipynb`](file:///Users/snehabvarghese/vibration-ssl-pdm/notebooks/01_data_exploration.ipynb) — Dataset statistics, signal waveforms, and FFT frequency spectrum analysis.
- [`02_ssl_pretraining.ipynb`](file:///Users/snehabvarghese/vibration-ssl-pdm/notebooks/02_ssl_pretraining.ipynb) — Contrastive pair creation, 1-D CNN architecture, and 150-epoch loss curves.
- [`03_anomaly_detection.ipynb`](file:///Users/snehabvarghese/vibration-ssl-pdm/notebooks/03_anomaly_detection.ipynb) — EDR metric estimation, healthy baseline thresholding, and degradation timelines.
- [`04_evaluation.ipynb`](file:///Users/snehabvarghese/vibration-ssl-pdm/notebooks/04_evaluation.ipynb) — SSL vs Supervised baseline comparisons across 1–20% label budgets and confusion matrices.

---

### 6. Option E: Run Automated Smoke Tests

To run the automated test suite verifying encoder shapes, loss functions, leakage-free splits, and EDR detection:

```bash
python tests/test_pipeline.py
```