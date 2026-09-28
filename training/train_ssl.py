"""
STEP 5: self-supervised contrastive pretraining.

WHAT IT DOES
------------
Trains the Siamese 1-D CNN with NT-Xent on the *training-split segments only*,
WITHOUT their labels. Validation loss is computed on the validation split
(also unlabelled) purely to monitor convergence and pick the best checkpoint.

WHY THE TRAINING SPLIT ONLY
---------------------------
The test records are excluded from pretraining as well. If the encoder had
seen test segments -- even unlabelled -- the downstream evaluation would be
transductive and the generalisation claim would be weaker.

OUTPUTS
-------
  checkpoints/ssl_encoder.pt                encoder + config + loss history
  results/metrics/ssl_training.json         per-epoch train/val loss
  results/figures/04_ssl_loss_curve.png

RUN
---
  python -m training.train_ssl [--epochs N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG, CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR, get_device, set_seed
from models.siamese import NTXentLoss, SiameseContrastiveModel
from training.datasets import ContrastiveSegmentDataset, load_processed
from training.splits import make_split


def run_epoch(
    model: SiameseContrastiveModel,
    loader: DataLoader,
    criterion: NTXentLoss,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """One pass over the loader. Trains when `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)
    total, n_batches = 0.0, 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            if len(batch) == 3:
                view_a, view_b, rec_ids = batch
                rec_ids = rec_ids.to(device)
            else:
                view_a, view_b = batch
                rec_ids = None

            if view_a.shape[0] < 2:
                continue
            view_a, view_b = view_a.to(device), view_b.to(device)
            z_a, z_b = model(view_a, view_b)
            loss = criterion(z_a, z_b, record_ids=rec_ids)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.item())
            n_batches += 1
    return total / max(1, n_batches)


def evaluate_knn_val(model: SiameseContrastiveModel, data, train_idx, val_idx, device: str) -> float:
    """Evaluate 5-NN accuracy on validation embeddings to select best representation checkpoint."""
    from sklearn.neighbors import KNeighborsClassifier

    model.eval()
    with torch.no_grad():
        t_x = torch.from_numpy(data.segments[train_idx]).float().to(device)
        v_x = torch.from_numpy(data.segments[val_idx]).float().to(device)
        t_emb = model.encoder(t_x).cpu().numpy()
        v_emb = model.encoder(v_x).cpu().numpy()

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(t_emb, data.labels[train_idx])
    return float(knn.score(v_emb, data.labels[val_idx]))


def plot_loss(history: Dict[str, List[float]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train", color="#4C72B0")
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], label="validation", color="#C44E52")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent loss")
    ax.set_title("Self-supervised contrastive pretraining")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self-supervised contrastive pretraining")
    parser.add_argument("--epochs", type=int, default=CFG.ssl.epochs)
    parser.add_argument("--batch-size", type=int, default=CFG.ssl.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.ssl.lr)
    parser.add_argument("--temperature", type=float, default=CFG.ssl.temperature)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    set_seed(CFG.seed)
    device = args.device or get_device()

    data = load_processed()
    split = make_split(data.record_ids, data.labels, data.loads)
    
    # Pretraining pool: train + val records (labels are never used; test set stays strictly held out)
    pretrain_idx = np.concatenate([split.train_idx, split.val_idx])
    print(f"Segments: {len(data)}  pretrain pool: {len(pretrain_idx)}  test holdout: {len(split.test_idx)}")
    print("Labels are NOT used in this stage.")

    train_ds = ContrastiveSegmentDataset(data.segments[pretrain_idx], record_ids=data.record_ids[pretrain_idx])
    val_ds = ContrastiveSegmentDataset(data.segments[split.val_idx], record_ids=data.record_ids[split.val_idx])
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=CFG.ssl.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=CFG.ssl.num_workers, drop_last=True,
    )

    model = SiameseContrastiveModel().to(device)
    criterion = NTXentLoss(args.temperature)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=CFG.ssl.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Device: {device}  encoder parameters: {model.encoder.n_parameters():,}")

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_knn_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    ckpt_path = CHECKPOINTS_DIR / CFG.ssl.checkpoint_name
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device) if len(val_ds) else float("nan")
        val_acc = evaluate_knn_val(model, data, split.train_idx, split.val_idx, device)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_knn_acc"].append(val_acc)

        marker = ""
        # Save checkpoint based on best validation kNN accuracy (representation quality)
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            marker = f"  <- best val kNN acc {val_acc:.4f}"
            torch.save(
                {
                    "encoder_state_dict": model.encoder.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "model_config": CFG.model.__dict__,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_knn_acc": val_acc,
                },
                ckpt_path,
            )
        print(f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.4f}  val_loss {val_loss:.4f}  val_knn_acc {val_acc:.4f}{marker}")

    elapsed = time.time() - t0
    metrics = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "temperature": args.temperature,
        "best_val_loss": best_val_loss,
        "best_val_knn_acc": best_val_acc,
        "final_train_loss": history["train_loss"][-1],
        "train_seconds": round(elapsed, 1),
        "n_pretrain_segments": int(pretrain_idx.size),
        "encoder_parameters": model.encoder.n_parameters(),
        "history": history,
    }
    (METRICS_DIR / "ssl_training.json").write_text(json.dumps(metrics, indent=2))
    plot_loss(history, FIGURES_DIR / "04_ssl_loss_curve.png")

    print(f"\nBest validation kNN accuracy: {best_val_acc:.4f} (NT-Xent loss: {best_val_loss:.4f})")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Trained in {elapsed:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
