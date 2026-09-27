"""
Vibration-specific data augmentation for contrastive learning.

WHY
---
Contrastive SSL needs two *different views of the same underlying sample* that
still represent the same machine condition:

    original segment
          |
      +---+---+
      |       |
   aug A   aug B      -> positive pair (must end up close in embedding space)

Views of *different* segments are negatives. The augmentations must therefore
be strong enough that the network cannot solve the task trivially, but must
not destroy the fault signature (otherwise a "positive pair" would no longer
share a condition, and the encoder would learn nothing useful).

CHOSEN AUGMENTATIONS (all amplitude- or time-domain, all cheap)
  jitter        additive Gaussian noise      -> sensor noise robustness
  scaling       multiply by a random gain    -> sensor gain / load variation
  time shift    circular shift               -> impulse phase is arbitrary
  time mask     zero a short span            -> dropout / occlusion robustness
  permutation   shuffle a few sub-segments   -> forces reliance on local
                                                impulse morphology rather than
                                                on global window layout

NOT USED: frequency-domain masking and time warping, which can move or remove
the very fault harmonics we want the encoder to key on.

INPUTS  : (window,) or (batch, window) float arrays, already z-scored
OUTPUTS : arrays of the same shape
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from config import CFG, AugmentConfig


# --------------------------------------------------------------------------
# Individual augmentations
# --------------------------------------------------------------------------
def jitter(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise (sigma is relative to unit-variance input)."""
    return x + rng.normal(0.0, sigma, size=x.shape)


def scaling(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Multiply the whole window by one random gain drawn from N(1, sigma)."""
    factor = rng.normal(1.0, sigma)
    return x * factor


def time_shift(x: np.ndarray, max_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Circularly shift the window; fault impulses have arbitrary phase."""
    n = x.shape[-1]
    shift = int(rng.integers(-int(max_frac * n), int(max_frac * n) + 1))
    return np.roll(x, shift, axis=-1)


def time_mask(x: np.ndarray, max_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Zero out one contiguous span of up to `max_frac` of the window."""
    n = x.shape[-1]
    span = int(rng.integers(1, max(2, int(max_frac * n))))
    start = int(rng.integers(0, n - span + 1))
    out = x.copy()
    out[..., start : start + span] = 0.0
    return out


def permutation(x: np.ndarray, n_segments: int, rng: np.random.Generator) -> np.ndarray:
    """Split the window into `n_segments` chunks and shuffle their order."""
    n = x.shape[-1]
    if n_segments < 2 or n_segments > n:
        return x
    bounds = np.array_split(np.arange(n), n_segments)
    order = rng.permutation(len(bounds))
    return np.concatenate([x[..., bounds[i]] for i in order], axis=-1)


# --------------------------------------------------------------------------
# Composite view generator
# --------------------------------------------------------------------------
def augment_signal(
    x: np.ndarray,
    cfg: Optional[AugmentConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply the configured stochastic augmentation chain to one window.

    Each augmentation fires independently with its own probability, so two
    calls on the same input almost always give two different views.
    """
    cfg = cfg or CFG.augment
    rng = rng or np.random.default_rng()
    out = np.asarray(x, dtype=np.float64).copy()

    if rng.random() < cfg.jitter_prob:
        out = jitter(out, cfg.jitter_sigma, rng)
    if rng.random() < cfg.scaling_prob:
        out = scaling(out, cfg.scaling_sigma, rng)
    if rng.random() < cfg.time_shift_prob:
        out = time_shift(out, cfg.time_shift_max_frac, rng)
    if rng.random() < cfg.time_mask_prob:
        out = time_mask(out, cfg.time_mask_max_frac, rng)
    if rng.random() < cfg.permutation_prob:
        out = permutation(out, cfg.permutation_segments, rng)
    return out


def make_two_views(
    x: np.ndarray,
    cfg: Optional[AugmentConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the positive pair (view A, view B) for one segment."""
    rng = rng or np.random.default_rng()
    return augment_signal(x, cfg, rng), augment_signal(x, cfg, rng)
