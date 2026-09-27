"""
STEP 6: embedding extraction.

WHAT IT DOES
------------
Loads the pretrained encoder, discards the projection head, and maps every
segment in the processed dataset to a 128-D embedding:

    segment (3000 samples)  --encoder-->  embedding (128-D)

Embeddings are cached once and then reused by t-SNE/UMAP visualisation, EDR
anomaly detection, and the limited-label classifiers -- so those stages never
have to re-run the network.

WHY THE HEAD IS DROPPED
-----------------------
See models/siamese.py: the projection head specialises in the pretext task.
The encoder output is the transferable representation.

OUTPUT
------
  results/embeddings/embeddings.npz
      embeddings  (N, 128) float32
      labels      (N,)     int64
      classes     (C,)     str
      record_ids  (N,)     str
      loads       (N,)     int64
      split       (N,)     str   -- "train" / "val" / "test"

RUN
---
  python -m training.extract_embeddings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from config import CFG, CHECKPOINTS_DIR, EMBEDDINGS_DIR, ModelConfig, get_device, set_seed
from models.encoder import CNN1DEncoder
from training.datasets import load_processed
from training.splits import Split, make_split

DEFAULT_OUTPUT = EMBEDDINGS_DIR / "embeddings.npz"


def load_encoder(
    checkpoint: Path = CHECKPOINTS_DIR / CFG.ssl.checkpoint_name,
    device: Optional[str] = None,
) -> CNN1DEncoder:
    """Rebuild the encoder from a checkpoint, using the config it was saved with."""
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"{checkpoint} not found -- run `python -m training.train_ssl` first"
        )
    device = device or get_device()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"]) if "model_config" in ckpt else CFG.model
    encoder = CNN1DEncoder(cfg).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    return encoder


def load_encoder_state(
    checkpoint: Path = CHECKPOINTS_DIR / CFG.ssl.checkpoint_name,
) -> Dict[str, torch.Tensor]:
    """Just the pretrained encoder weights, for initialising a fine-tuned model."""
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"{checkpoint} not found -- run `python -m training.train_ssl` first"
        )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return ckpt["encoder_state_dict"]


@torch.no_grad()
def embed_segments(
    encoder: CNN1DEncoder,
    segments: np.ndarray,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> np.ndarray:
    """(N, L) raw segments -> (N, embedding_dim) embeddings."""
    device = device or next(encoder.parameters()).device
    encoder.eval()
    out: List[np.ndarray] = []
    for start in range(0, segments.shape[0], batch_size):
        batch = torch.from_numpy(segments[start : start + batch_size]).float().unsqueeze(1)
        out.append(encoder(batch.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def split_labels(n: int, split: Split) -> np.ndarray:
    tags = np.array(["unused"] * n, dtype=object)
    tags[split.train_idx] = "train"
    tags[split.val_idx] = "val"
    tags[split.test_idx] = "test"
    return tags.astype(str)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract embeddings with the pretrained encoder")
    parser.add_argument("--checkpoint", type=Path,
                        default=CHECKPOINTS_DIR / CFG.ssl.checkpoint_name)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    set_seed(CFG.seed)
    device = args.device or get_device()

    data = load_processed()
    split = make_split(data.record_ids, data.labels, data.loads)
    encoder = load_encoder(args.checkpoint, device)

    emb = embed_segments(encoder, data.segments, args.batch_size, device)

    np.savez_compressed(
        args.output,
        embeddings=emb,
        labels=data.labels,
        classes=np.array(data.classes),
        record_ids=data.record_ids,
        loads=data.loads,
        fault_types=data.fault_types,
        split=split_labels(len(data), split),
    )

    norms = np.linalg.norm(emb, axis=1)
    print(f"Embeddings : {emb.shape}  (dim = {encoder.embedding_dim})")
    print(f"L2 norm    : mean {norms.mean():.3f}  std {norms.std():.3f}")
    print(f"Split      : {split.sizes()}")
    print(f"Saved to   : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
