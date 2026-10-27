"""
Train the Acoustic Anomaly Detection Baseline (Dense Autoencoder).

Key Specifications:
  * Trained on NORMAL clips only (source and target normal clips).
  * Loss: Mean Squared Error (MSE).
  * Optimiser: Adam (lr=1e-3, batch_size=512, epochs=100 with early stopping).
  * Device: CUDA (RTX 3050) when available, CPU fallback.
  * Thresholds: Derived strictly from TRAINING normal clip anomaly scores (95th percentile,
    plus 90th and 99th percentiles). Test data is strictly never used.
  * Checkpoints: Saved to checkpoints/audio/<machine>_ae.pt with scaler stats and JSON metadata.
  * Curves: Saved to results/audio/figures/<machine>_loss.png.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from audio_model.config_audio import (
    AUDIO_CFG,
    CHECKPOINTS_AUDIO_DIR,
    FIGURES_AUDIO_DIR,
    PROCESSED_AUDIO_DIR,
    get_device,
    set_seed,
)
from audio_model.models.autoencoder import DenseAutoencoder


def train_single_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch over frame vectors."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for (batch_x,) in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        optimizer.zero_grad()
        recon = model(batch_x)
        loss = criterion(recon, batch_x)
        loss.backward()
        optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate reconstruction MSE loss on validation frame vectors."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(1, total_samples)


def compute_clip_anomaly_scores(
    model: nn.Module,
    clips_stacked: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Compute clip-level anomaly score (mean frame reconstruction error) for clips.

    Parameters
    ----------
    model : nn.Module
        Trained autoencoder.
    clips_stacked : np.ndarray
        Array of shape (N_clips, n_frames, vector_dim).
    device : torch.device
        Compute device.

    Returns
    -------
    np.ndarray
        1-D array of shape (N_clips,) with reconstruction anomaly scores.
    """
    model.eval()
    n_clips, n_frames, vector_dim = clips_stacked.shape
    clip_scores = np.empty(n_clips, dtype=np.float32)

    with torch.no_grad():
        for i in range(n_clips):
            clip_frames = torch.from_numpy(clips_stacked[i]).to(device)
            recon = model(clip_frames)
            frame_mse = torch.mean((clip_frames - recon) ** 2, dim=-1)
            clip_scores[i] = frame_mse.mean().item()

    return clip_scores


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path,
    machine: str,
) -> None:
    """Plot and save training and validation reconstruction loss curves."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(epochs, train_losses, label="Train MSE", color="#2980b9", lw=2)
    ax.plot(epochs, val_losses, label="Val MSE", color="#e67e22", lw=2, linestyle="--")

    best_epoch = int(np.argmin(val_losses)) + 1
    best_val = val_losses[best_epoch - 1]
    ax.scatter([best_epoch], [best_val], color="#c0392b", s=60, zorder=5, label=f"Best Val: {best_val:.5f} (Ep {best_epoch})")

    ax.set_title(f"Dense Autoencoder Loss Curve: {machine.upper()}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Reconstruction MSE Loss", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def train_machine_autoencoder(
    machine: str,
    processed_dir: Path,
    checkpoints_dir: Path,
    figures_dir: Path,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train Dense Autoencoder on normal training clips for a single machine."""
    set_seed(seed)
    device_str = get_device()
    device = torch.device(device_str)

    machine_dir = processed_dir / machine
    if not machine_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found for machine '{machine}': {machine_dir}")

    print(f"\n{'='*70}\nTraining Autoencoder: {machine.upper()} on device: {device}\n{'='*70}")

    # 1. Load cached processed datasets
    train_stacked = np.load(machine_dir / "train_stacked.npy")  # (N_train_clips, 309, 640)
    val_stacked = np.load(machine_dir / "val_stacked.npy")      # (N_val_clips, 309, 640)
    scaler_data = np.load(machine_dir / "scaler.npz")
    scaler_mean = scaler_data["mean"]
    scaler_std = scaler_data["std"]

    n_train_clips, n_frames, vector_dim = train_stacked.shape
    n_val_clips = val_stacked.shape[0]

    # Flatten clip temporal frames into vector datasets
    train_vectors = train_stacked.reshape(-1, vector_dim).astype(np.float32)
    val_vectors = val_stacked.reshape(-1, vector_dim).astype(np.float32)

    print(f"Train: {n_train_clips} clips ({len(train_vectors):,} frame vectors of dim {vector_dim})")
    print(f"Val  : {n_val_clips} clips ({len(val_vectors):,} frame vectors of dim {vector_dim})")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_vectors)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device_str == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_vectors)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device_str == "cuda"),
    )

    # 2. Build model, optimizer, loss
    model = DenseAutoencoder(input_dim=vector_dim, hidden_dim=128, bottleneck_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    # 3. Training Loop with early stopping
    for epoch in range(1, epochs + 1):
        tr_loss = train_single_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        train_loss_history.append(tr_loss)
        val_loss_history.append(val_loss)

        if val_loss < (best_val_loss - 1e-5):
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or patience_counter == 0:
            marker = "*" if patience_counter == 0 else ""
            print(f"Epoch [{epoch:3d}/{epochs:3d}] | Train MSE: {tr_loss:.6f} | Val MSE: {val_loss:.6f} {marker}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (best epoch {best_epoch} with Val MSE: {best_val_loss:.6f})")
            break

    # Restore best weights
    model.load_state_dict(best_weights)

    # 4. Save Loss Curve
    loss_fig_path = figures_dir / f"{machine}_loss.png"
    plot_loss_curves(train_loss_history, val_loss_history, loss_fig_path, machine)
    print(f"Saved loss curve to: {loss_fig_path}")

    # 5. Compute Training Clip Anomaly Scores & Decision Thresholds
    print("Computing anomaly scores on TRAINING normal clips...")
    train_scores = compute_clip_anomaly_scores(model, train_stacked, device)

    # Decision thresholds: 95th percentile primary, plus 90th and 99th percentiles
    threshold_p90 = float(np.percentile(train_scores, 90))
    threshold_p95 = float(np.percentile(train_scores, 95))
    threshold_p99 = float(np.percentile(train_scores, 99))
    mean_score = float(np.mean(train_scores))
    std_score = float(np.std(train_scores))

    print(f"Training Normal Scores -> Mean: {mean_score:.5f}, Std: {std_score:.5f}")
    print(f"Decision Thresholds    -> P90: {threshold_p90:.5f} | P95: {threshold_p95:.5f} (PRIMARY) | P99: {threshold_p99:.5f}")

    # 6. Save Model Checkpoint (.pt) & Metadata (.json)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"{machine}_ae.pt"
    meta_path = checkpoints_dir / f"{machine}_ae_meta.json"

    torch.save(
        {
            "model_state_dict": best_weights,
            "input_dim": vector_dim,
            "hidden_dim": 128,
            "bottleneck_dim": 8,
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
            "thresholds": {
                "p90": threshold_p90,
                "p95": threshold_p95,
                "p99": threshold_p99,
                "mean": mean_score,
                "std": std_score,
            },
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_train_loss": train_loss_history[best_epoch - 1],
            "machine": machine,
            "seed": seed,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to: {checkpoint_path} ({checkpoint_path.stat().st_size / 1e3:.1f} KB)")

    meta_dict: Dict[str, Any] = {
        "machine": machine,
        "best_epoch": best_epoch,
        "epochs_trained": len(train_loss_history),
        "final_train_mse": float(train_loss_history[best_epoch - 1]),
        "best_val_mse": float(best_val_loss),
        "thresholds": {
            "p90": threshold_p90,
            "p95": threshold_p95,
            "p99": threshold_p99,
            "mean": mean_score,
            "std": std_score,
        },
        "training_clips": n_train_clips,
        "training_frames": len(train_vectors),
        "validation_clips": n_val_clips,
        "validation_frames": len(val_vectors),
        "checkpoint_file": checkpoint_path.name,
        "loss_figure": str(loss_fig_path.name),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    return meta_dict


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train Acoustic Dense Autoencoder on Normal Audio Clips")
    parser.add_argument(
        "--machines",
        nargs="+",
        default=["fan", "bearing"],
        help="Machines to train on (default: fan bearing)",
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_AUDIO_DIR, help="Path to processed data")
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_AUDIO_DIR, help="Path to save checkpoints")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_AUDIO_DIR, help="Path to save figures")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for frame vectors")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=AUDIO_CFG.seed, help="Random seed")
    args = parser.parse_args(argv)

    results: Dict[str, Any] = {}
    for m in args.machines:
        res = train_machine_autoencoder(
            machine=m,
            processed_dir=args.processed_dir,
            checkpoints_dir=args.checkpoints_dir,
            figures_dir=args.figures_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            seed=args.seed,
        )
        results[m] = res

    print("\n" + "=" * 90)
    print("TRAINING SUMMARY REPORT: ACOUSTIC AUTOENCODER")
    print("=" * 90)
    print(f"{'Machine':10s} | {'Best Ep':7s} | {'Train MSE':11s} | {'Val MSE':11s} | {'Threshold P95':14s} | {'P90 / P99':15s}")
    print("-" * 90)
    for m, r in results.items():
        th = r["thresholds"]
        print(
            f"{m:10s} | {r['best_epoch']:3d}/{r['epochs_trained']:3d} | "
            f"{r['final_train_mse']:.6f}    | {r['best_val_mse']:.6f}  | "
            f"{th['p95']:.6f}         | {th['p90']:.5f} / {th['p99']:.5f}"
        )
    print("=" * 90 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
