"""
STEP 10: supervised 1-D CNN baseline, trained on the same limited labels.

FAIRNESS RULES (important for the viva)
---------------------------------------
* Identical backbone to the self-supervised encoder.
* Identical record-level split.
* Identical labelled subsets (same seeds -> `sample_labelled_subset` returns
  exactly the same indices as in training/train_classifier.py).
* Identical test set, evaluated in full.
The only difference is that this model never sees the unlabelled data.

CHECKPOINT SELECTION
--------------------
The baseline needs a validation signal to pick its best epoch. Using the full
validation split would quietly hand it ~1.3 k extra labels while we advertise
the run as "32 labels", so instead we carve the validation set OUT OF the
labelled subset (80/20, stratified). At 1 % that means ~26 train / ~6 val
samples -- realistic for a genuinely label-poor setting, and it keeps the
label budget identical to the self-supervised arm.

OUTPUTS
-------
  checkpoints/baseline_cnn.pt
  results/metrics/baseline_classification.json
  results/figures/11_confusion_baseline.png
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG, CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR, get_device, set_seed
from evaluation.metrics import (
    classification_metrics,
    plot_confusion,
    sample_labelled_subset,
    summarize_repeats,
)
from models.baseline_cnn import BaselineCNN
from training.datasets import SupervisedSegmentDataset, load_processed
from training.extract_embeddings import load_encoder_state
from training.splits import make_split


def _inner_split(y: np.ndarray, seed: int, val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train/val split *within* the labelled subset.

    Every class keeps at least one training sample; classes with a single
    sample contribute nothing to validation.
    """
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = min(len(idx) - 1, int(round(val_frac * len(idx))))
        va.extend(idx[:n_val].tolist())
        tr.extend(idx[n_val:].tolist())
    if not va:                      # too few labels to validate: reuse train
        va = list(tr)
    return np.array(sorted(tr)), np.array(sorted(va))


def train_once(
    x_labelled: np.ndarray,
    y_labelled: np.ndarray,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    init: str = "scratch",
) -> Tuple[BaselineCNN, Dict[str, List[float]]]:
    set_seed(seed)
    # Hold out part of the labelled budget for checkpoint selection.
    tr_idx, va_idx = _inner_split(y_labelled, seed)
    x_train, y_train = x_labelled[tr_idx], y_labelled[tr_idx]
    x_val, y_val = x_labelled[va_idx], y_labelled[va_idx]

    model = BaselineCNN(n_classes)
    if init == "ssl":
        # Same architecture, same labels -- only the starting weights differ.
        model.encoder.load_state_dict(load_encoder_state())
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=CFG.baseline.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        SupervisedSegmentDataset(x_train, y_train),
        batch_size=min(batch_size, max(2, len(y_train))), shuffle=True,
        drop_last=len(y_train) > batch_size,
    )
    val_loader = DataLoader(
        SupervisedSegmentDataset(x_val, y_val), batch_size=256, shuffle=False
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val, best_state = float("inf"), copy.deepcopy(model.state_dict())

    for _ in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if xb.shape[0] < 2:
                continue
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            tot += float(loss.item())
            nb += 1
        train_loss = tot / max(1, nb)

        model.eval()
        vtot, vnb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                vtot += float(criterion(model(xb.to(device)), yb.to(device)).item())
                vnb += 1
        val_loss = vtot / max(1, vnb)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model: BaselineCNN, x: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[s : s + batch_size]).float().unsqueeze(1).to(device)
        out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Supervised 1-D CNN baseline")
    parser.add_argument("--fractions", type=float, nargs="*", default=CFG.downstream.label_fractions)
    parser.add_argument("--repeats", type=int, default=CFG.downstream.n_repeats)
    parser.add_argument("--epochs", type=int, default=CFG.baseline.epochs)
    parser.add_argument("--batch-size", type=int, default=CFG.baseline.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.baseline.lr)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init", type=str, default="scratch", choices=["scratch", "ssl"],
                        help="'scratch' = supervised baseline; "
                             "'ssl' = fine-tune the self-supervised encoder")
    args = parser.parse_args(argv)
    tag = "baseline" if args.init == "scratch" else "ssl-finetune"
    out_name = ("baseline_classification.json" if args.init == "scratch"
                else "finetune_classification.json")
    ckpt_name = (CFG.baseline.checkpoint_name if args.init == "scratch"
                 else "ssl_finetuned_cnn.pt")
    fig_name = ("11_confusion_baseline.png" if args.init == "scratch"
                else "11b_confusion_ssl_finetune.png")

    set_seed(CFG.seed)
    device = args.device or get_device()
    data = load_processed()
    split = make_split(data.record_ids, data.labels, data.loads)
    classes = data.classes

    x_train, y_train = data.segments[split.train_idx], data.labels[split.train_idx]
    x_test, y_test = data.segments[split.test_idx], data.labels[split.test_idx]
    print(f"device {device} | train pool {len(y_train)} | test {len(y_test)}")

    results: Dict[str, dict] = {}
    best_pred, best_frac = None, max(args.fractions)

    for frac in args.fractions:
        runs, last_pred, n_used, last_hist = [], None, 0, None
        for rep in range(args.repeats):
            seed = CFG.seed + 1000 * rep          # same seeds as the SSL classifier
            sub = sample_labelled_subset(y_train, frac, seed)
            n_used = len(sub)
            model, hist = train_once(
                x_train[sub], y_train[sub], len(classes),
                args.epochs, args.batch_size, args.lr, device, seed, args.init,
            )
            pred = predict(model, x_test, device)
            runs.append(classification_metrics(y_test, pred))
            last_pred, last_hist = pred, hist
            if frac == best_frac and rep == args.repeats - 1:
                torch.save(
                    {"state_dict": model.state_dict(), "classes": classes,
                     "label_fraction": frac, "model_config": CFG.model.__dict__},
                    CHECKPOINTS_DIR / ckpt_name,
                )
        res = summarize_repeats(runs)
        res["n_labelled"] = int(n_used)
        res["per_run"] = runs
        res["last_history"] = last_hist
        results[f"{frac}"] = res
        print(f"{tag:13s} {frac*100:5.1f}%  n={n_used:4d}  "
              f"acc {res['accuracy_mean']:.3f}+-{res['accuracy_std']:.3f}  "
              f"F1 {res['f1_macro_mean']:.3f}+-{res['f1_macro_std']:.3f}")
        if frac == best_frac:
            best_pred = last_pred

    if best_pred is not None:
        title = ("Supervised CNN baseline" if args.init == "scratch"
                 else "SSL-pretrained CNN, fine-tuned")
        plot_confusion(y_test, best_pred, classes,
                       f"{title} ({best_frac*100:g}% labels)",
                       FIGURES_DIR / fig_name)

    payload = {
        "setup": {
            "model": ("supervised 1-D CNN trained from scratch" if args.init == "scratch"
                      else "1-D CNN initialised from the self-supervised encoder"),
            "init": args.init,
            "epochs": args.epochs,
            "split_strategy": CFG.split.strategy,
            "note": "trained on the labelled subset only; the epoch is selected on "
                    "a 20% holdout carved out of that same subset, so every arm "
                    "spends exactly the same label budget",
            "classes": classes,
        },
        "results": results,
    }
    (METRICS_DIR / out_name).write_text(json.dumps(payload, indent=2))
    print(f"metrics -> {METRICS_DIR / out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
