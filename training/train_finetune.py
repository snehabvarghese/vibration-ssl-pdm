"""
STEP 9b: Fine-tuning the pretrained SSL encoder on limited labelled subsets.

WHAT IT DOES
------------
Loads the contrastively pretrained encoder from `checkpoints/ssl_encoder.pt`,
attaches a linear classification head, and fine-tunes the entire network
end-to-end on labelled fractions (1 %, 5 %, 10 %, 20 %) of the training split.

Evaluates on the held-out test split to provide the standard SSL fine-tuning benchmark.

OUTPUTS
-------
  results/metrics/finetune_classification.json
  results/figures/10_confusion_ssl_finetune.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CFG, CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR, get_device, set_seed
from evaluation.metrics import (
    classification_metrics,
    plot_confusion,
    sample_labelled_subset,
    summarize_repeats,
)
from models.encoder import CNN1DEncoder
from training.datasets import load_processed
from training.splits import make_split


class FineTuneClassifier(nn.Module):
    """Pretrained 1-D CNN encoder + Linear classification head."""

    def __init__(self, encoder: CNN1DEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return self.head(emb)


def finetune_one_run(
    encoder_ckpt_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    epochs: int = 50,
    lr: float = 2e-4,
    batch_size: int = 64,
    device: str = "cpu",
) -> Dict[str, float]:
    """Fine-tune the encoder on a labelled subset and return test metrics."""
    # Load pretrained encoder
    enc = CNN1DEncoder(CFG.model)
    if encoder_ckpt_path.exists():
        ckpt = torch.load(encoder_ckpt_path, map_location=device, weights_only=False)
        enc.load_state_dict(ckpt["encoder_state_dict"])

    model = FineTuneClassifier(enc, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        tx = torch.from_numpy(x_test).float().to(device)
        logits = model(tx)
        preds = logits.argmax(dim=1).cpu().numpy()

    return classification_metrics(y_test, preds), preds


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end SSL Fine-tuning")
    parser.add_argument("--fractions", type=float, nargs="*", default=CFG.downstream.label_fractions)
    parser.add_argument("--repeats", type=int, default=CFG.downstream.n_repeats)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args(argv)

    set_seed(CFG.seed)
    device = get_device()
    data = load_processed()
    split = make_split(data.record_ids, data.labels, data.loads)

    x_tr, y_tr = data.segments[split.train_idx], data.labels[split.train_idx]
    x_te, y_te = data.segments[split.test_idx], data.labels[split.test_idx]
    classes = data.classes
    num_classes = len(classes)
    ckpt_path = CHECKPOINTS_DIR / CFG.ssl.checkpoint_name

    print(f"Fine-tuning evaluation | Device: {device} | Classes: {num_classes}")

    results: Dict[str, dict] = {}
    last_pred = None

    for frac in args.fractions:
        runs = []
        for rep in range(args.repeats):
            seed = CFG.seed + 1000 * rep
            sub_idx = sample_labelled_subset(y_tr, frac, seed)
            metrics, pred = finetune_one_run(
                ckpt_path,
                x_tr[sub_idx],
                y_tr[sub_idx],
                x_te,
                y_te,
                num_classes,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
            )
            runs.append(metrics)
            last_pred = pred

        summary = summarize_repeats(runs)
        summary["n_labelled"] = len(sub_idx)
        summary["per_run"] = runs
        results[f"{frac}"] = summary

        print(
            f"Fine-tune {frac*100:5.1f}%  n={summary['n_labelled']:4d}  "
            f"acc {summary['accuracy_mean']:.3f}+-{summary['accuracy_std']:.3f}  "
            f"F1 {summary['f1_macro_mean']:.3f}+-{summary['f1_macro_std']:.3f}"
        )

        if frac == max(args.fractions) and last_pred is not None:
            plot_confusion(
                y_te,
                last_pred,
                classes,
                f"SSL Fine-tuned ({frac*100:g}% labels)",
                FIGURES_DIR / "10_confusion_ssl_finetune.png",
            )

    payload = {
        "setup": {
            "model": "SSL Pretrained CNN Fine-tuned",
            "n_train_segments": len(y_tr),
            "n_test_segments": len(y_te),
            "classes": classes,
            "repeats": args.repeats,
        },
        "results": results,
    }
    (METRICS_DIR / "finetune_classification.json").write_text(json.dumps(payload, indent=2))
    print(f"written -> {METRICS_DIR / 'finetune_classification.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
