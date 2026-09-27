"""
STEP 7: embedding visualisation (t-SNE / UMAP) and shared plotting helpers.

WHAT IT DOES
------------
Projects the 128-D embeddings to 2-D and colours the points by their TRUE
dataset label. The labels are used for *display only* -- the encoder never saw
them. If the self-supervised objective learned anything useful, segments of
the same fault class should form clusters without ever having been told which
class they belong to. That is the visual evidence for the SSL stage working.

t-SNE vs UMAP
-------------
t-SNE preserves local neighbourhoods and is the standard in the literature;
UMAP is faster and preserves more global structure. We produce both when
`umap-learn` is installed, and fall back gracefully to t-SNE alone if not.

QUANTITATIVE COMPANION
----------------------
A picture can mislead, so we also compute the silhouette score of the true
classes in the raw 128-D embedding space (not the 2-D projection). Higher =
better separated clusters; it is reported alongside the figures.

OUTPUTS
-------
  results/figures/05_tsne_embeddings.png
  results/figures/06_umap_embeddings.png   (if umap-learn is available)
  results/metrics/embedding_quality.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from config import CFG, EMBEDDINGS_DIR, FIGURES_DIR, METRICS_DIR, set_seed

DEFAULT_EMBEDDINGS = EMBEDDINGS_DIR / "embeddings.npz"


def load_embeddings(path: Path = DEFAULT_EMBEDDINGS) -> Dict[str, np.ndarray]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m training.extract_embeddings` first"
        )
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def scatter_2d(
    xy: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab20")
    for c, name in enumerate(class_names):
        m = labels == c
        if not m.any():
            continue
        ax.scatter(
            xy[m, 0], xy[m, 1],
            s=6, alpha=0.65, color=cmap(c % 20),
            label=f"{name} (n={int(m.sum())})",
        )
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(markerscale=2, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_tsne(emb: np.ndarray, seed: int = CFG.seed, perplexity: float = 30.0) -> np.ndarray:
    perplexity = min(perplexity, max(5.0, (emb.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        learning_rate="auto", random_state=seed,
    ).fit_transform(emb)


def run_umap(emb: np.ndarray, seed: int = CFG.seed) -> Optional[np.ndarray]:
    """Return UMAP coordinates, or None if umap-learn is unavailable."""
    try:
        import umap
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"UMAP unavailable ({exc}); skipping the UMAP figure.")
        return None
    return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                     random_state=seed).fit_transform(emb)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Visualise learned embeddings")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "train", "val", "test"])
    parser.add_argument("--max-points", type=int, default=4000,
                        help="subsample for t-SNE runtime")
    args = parser.parse_args(argv)

    set_seed(CFG.seed)
    d = load_embeddings(args.embeddings)
    emb, labels = d["embeddings"], d["labels"]
    classes = [str(c) for c in d["classes"]]

    if args.split != "all":
        m = d["split"].astype(str) == args.split
        emb, labels = emb[m], labels[m]

    rng = np.random.default_rng(CFG.seed)
    if emb.shape[0] > args.max_points:
        sel = rng.choice(emb.shape[0], args.max_points, replace=False)
        emb, labels = emb[sel], labels[sel]

    # Cluster quality in the ORIGINAL 128-D space, not the 2-D projection.
    sil = float(silhouette_score(emb, labels)) if len(np.unique(labels)) > 1 else float("nan")

    print(f"Projecting {emb.shape[0]} embeddings of dim {emb.shape[1]} ...")
    tsne_xy = run_tsne(emb)
    scatter_2d(tsne_xy, labels, classes,
               "t-SNE of self-supervised embeddings (colour = true class, unseen by the encoder)",
               FIGURES_DIR / "05_tsne_embeddings.png")

    umap_xy = run_umap(emb)
    if umap_xy is not None:
        scatter_2d(umap_xy, labels, classes,
                   "UMAP of self-supervised embeddings (colour = true class)",
                   FIGURES_DIR / "06_umap_embeddings.png")

    metrics = {
        "n_points": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]),
        "split": args.split,
        "silhouette_score_128d": sil,
        "umap_available": umap_xy is not None,
    }
    (METRICS_DIR / "embedding_quality.json").write_text(json.dumps(metrics, indent=2))
    print(f"Silhouette score (true classes, 128-D): {sil:.3f}")
    print(f"Figures written to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
