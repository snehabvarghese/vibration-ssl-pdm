"""
Multimodal Decision and Score Fusion Engine for Vibration, Audio, and Motor PdM.

Implements calibrated late fusion over standardized ModelResult inputs:
- Methods: "max", "weighted", "vote"
- Three-tier status verdict: NORMAL, WARNING, ALARM
- Resilient to missing modalities (missing modalities are never assumed normal)
- Full traceability with score ratios, agreement counting, and readable reasons
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np

from core.models_api.interface import ModelResult


@dataclass
class FusionConfig:
    """Configurable thresholds, weights, and voting parameters for sensor fusion.

    Attributes
    ----------
    warning_ratio_threshold : float
        Threshold on fused ratio to trigger a WARNING (default: 0.8).
    alarm_ratio_threshold : float
        Threshold on fused ratio to trigger an ALARM when multiple models agree (default: 1.0).
    single_model_alarm_ratio : float
        Extreme ratio on any single modality that immediately triggers an ALARM (default: 2.0).
    min_agreement_for_alarm : int
        Minimum number of models agreeing anomalous for ratio-based ALARM (default: 2).
    weights : Dict[str, float]
        Relative weights for "weighted" fusion (default: equal 1/3 per modality).
    expected_modalities : List[str]
        Complete expected list of modalities (default: ["vibration", "audio", "motor"]).
    """

    warning_ratio_threshold: float = 0.8
    alarm_ratio_threshold: float = 1.0
    single_model_alarm_ratio: float = 2.0
    min_agreement_for_alarm: int = 2
    weights: Dict[str, float] = field(
        default_factory=lambda: {"vibration": 1.0 / 3.0, "audio": 1.0 / 3.0, "motor": 1.0 / 3.0}
    )
    expected_modalities: List[str] = field(
        default_factory=lambda: ["vibration", "audio", "motor"]
    )


@dataclass
class FusionResult:
    """Standardized output schema for multimodal fusion verdict.

    Attributes
    ----------
    fused_score : float
        Combined score ratio.
    fused_status : {"NORMAL", "WARNING", "ALARM", "UNAVAILABLE"}
        Overall machine operational verdict.
    method : {"max", "weighted", "vote"}
        Fusion aggregation strategy used.
    per_model_ratios : Dict[str, float]
        Individual score ratios (anomaly_score / threshold) per active model.
    missing_models : List[str]
        List of expected modalities that were missing or unavailable.
    reasons : List[str]
        Human-readable diagnostic explanations for the verdict.
    confidence : float
        Confidence score in [0.0, 1.0] reflecting modality availability and agreement.
    is_replay : bool
        Flag indicating if any input came from a recorded replay dataset.
    timestamp : str
        ISO 8601 acquisition timestamp.
    details : Dict[str, Any]
        Diagnostic metadata, weights, and raw inputs.
    """

    fused_score: float
    fused_status: Literal["NORMAL", "WARNING", "ALARM", "UNAVAILABLE"]
    method: Literal["max", "weighted", "vote"]
    per_model_ratios: Dict[str, float]
    missing_models: List[str]
    reasons: List[str]
    confidence: float
    is_replay: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to clean JSON-serializable dictionary."""
        d = asdict(self)
        return _make_serializable(d)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def fuse(
    results: Sequence[ModelResult],
    method: Literal["max", "weighted", "vote"] = "max",
    config: Optional[FusionConfig] = None,
) -> FusionResult:
    """Fuse diagnostic outputs from any subset of ModelResult instances.

    Parameters
    ----------
    results : Sequence[ModelResult]
        Collection of diagnostic results from vibration, audio, or motor adapters.
    method : {"max", "weighted", "vote"}, default "max"
        - "max": Fused score is the maximum ratio across available models.
        - "weighted": Fused score is the weighted mean ratio across available models.
        - "vote": k-of-n majority consensus (k=2 for 3 models, k=1 for 1 model).
    config : Optional[FusionConfig]
        Configurable parameters for weights, thresholds, and alarm criteria.

    Returns
    -------
    FusionResult
        Unified decision verdict and rationale.
    """
    cfg = config or FusionConfig()
    meth = method.strip().lower()
    if meth not in ("max", "weighted", "vote"):
        raise ValueError(f"Unknown fusion method '{method}'. Supported: 'max', 'weighted', 'vote'")

    # 1. Filter available models (missing modalities are NEVER assumed normal)
    available_results = [r for r in results if r.status in ("NORMAL", "ANOMALOUS")]
    active_modalities = {r.modality for r in available_results}
    missing_models = [m for m in cfg.expected_modalities if m not in active_modalities]

    is_replay = any(bool(r.is_replay) for r in results)

    if not available_results:
        return FusionResult(
            fused_score=0.0,
            fused_status="UNAVAILABLE",
            method=meth,  # type: ignore[arg-type]
            per_model_ratios={},
            missing_models=missing_models,
            reasons=["All diagnostic models are unavailable or missing."],
            confidence=0.0,
            is_replay=is_replay,
            details={"error": "No available models to fuse"},
        )

    # 2. Extract score ratios
    # Map modality -> score_ratio (if multiple models for same modality, take highest)
    per_model_ratios: Dict[str, float] = {}
    for r in available_results:
        ratio = float(r.score_ratio)
        if r.modality not in per_model_ratios or ratio > per_model_ratios[r.modality]:
            per_model_ratios[r.modality] = ratio

    n_available = len(per_model_ratios)
    anomalous_models = list({
        r.modality for r in available_results
        if r.status == "ANOMALOUS" or r.score_ratio >= 1.0
    })
    num_anomalous = len(anomalous_models)

    # 3. Calculate fused score according to method
    reasons: List[str] = []

    if meth == "max":
        fused_score = float(max(per_model_ratios.values()))
        top_mod = [m for m, r in per_model_ratios.items() if math.isclose(r, fused_score, rel_tol=1e-5)][0]
        reasons.append(f"Max ratio {fused_score:.2f} driven by {top_mod}.")

    elif meth == "weighted":
        # Renormalize weights across available modalities
        raw_weights = {m: float(cfg.weights.get(m, 1.0)) for m in per_model_ratios}
        total_w = sum(raw_weights.values())
        if total_w <= 0:
            total_w = 1.0
            norm_weights = {m: 1.0 / n_available for m in per_model_ratios}
        else:
            norm_weights = {m: w / total_w for m, w in raw_weights.items()}

        fused_score = float(sum(per_model_ratios[m] * norm_weights[m] for m in per_model_ratios))
        w_strs = [f"{m}: {norm_weights[m]:.2f} (r={per_model_ratios[m]:.2f})" for m in per_model_ratios]
        reasons.append(f"Weighted ratio {fused_score:.2f} across available models [{', '.join(w_strs)}].")

    elif meth == "vote":
        # k-of-n consensus: k=2 with 3 models, k=2 with 2 models, k=1 with 1 model
        k = 2 if n_available >= 2 else 1
        if num_anomalous >= k:
            fused_score = float(max(per_model_ratios[m] for m in anomalous_models if m in per_model_ratios))
            reasons.append(
                f"Voting consensus reached: {num_anomalous}/{n_available} models flagged anomaly (required {k})."
            )
        else:
            normal_ratios = [r for r in per_model_ratios.values() if r < 1.0]
            fused_score = float(np.mean(normal_ratios)) if normal_ratios else float(np.mean(list(per_model_ratios.values())))
            reasons.append(
                f"Voting consensus normal: only {num_anomalous}/{n_available} models flagged anomaly (required {k})."
            )

    # 4. Status determination via config rules
    # Rule 1: ALARM
    # ALARM when fused ratio >= 1.0 with at least two models agreeing, or any single ratio >= 2.0
    single_alarm_models = [m for m, r in per_model_ratios.items() if r >= cfg.single_model_alarm_ratio]
    agreeing_alarm_models = [
        m for m, r in per_model_ratios.items()
        if (r >= cfg.alarm_ratio_threshold) or (m in anomalous_models)
    ]
    is_multi_alarm = (fused_score >= cfg.alarm_ratio_threshold) and (
        len(agreeing_alarm_models) >= min(cfg.min_agreement_for_alarm, n_available)
    )

    if single_alarm_models:
        fused_status: Literal["NORMAL", "WARNING", "ALARM", "UNAVAILABLE"] = "ALARM"
        reasons.append(
            f"ALARM triggered: critical single-modality ratio >= {cfg.single_model_alarm_ratio} on {', '.join(single_alarm_models)}."
        )
    elif is_multi_alarm:
        fused_status = "ALARM"
        reasons.append(
            f"ALARM triggered: fused ratio {fused_score:.2f} >= {cfg.alarm_ratio_threshold} with {num_anomalous} models agreeing."
        )
    # Rule 2: WARNING
    # WARNING when fused ratio >= 0.8 or one model anomalous
    elif (fused_score >= cfg.warning_ratio_threshold) or (num_anomalous >= 1):
        fused_status = "WARNING"
        if num_anomalous == 1:
            reasons.append(f"WARNING triggered: solitary model '{anomalous_models[0]}' flagged anomalous.")
        else:
            reasons.append(f"WARNING triggered: fused ratio {fused_score:.2f} >= {cfg.warning_ratio_threshold}.")
    # Rule 3: NORMAL
    else:
        fused_status = "NORMAL"
        reasons.append(f"NORMAL: All active sensor ratios below warning threshold ({cfg.warning_ratio_threshold}).")

    # 5. Missing modalities caveat
    if missing_models:
        reasons.append(f"Note: modalities {missing_models} were unavailable and omitted from fusion.")

    # 6. Confidence based on modality availability & agreement
    # Baseline availability ratio [0.33, 1.0]
    avail_factor = n_available / max(1, len(cfg.expected_modalities))
    if fused_status == "ALARM":
        agreement_factor = 1.0 if (num_anomalous >= 2 or len(single_alarm_models) > 0) else 0.8
    elif fused_status == "WARNING":
        agreement_factor = 0.75
    else:
        agreement_factor = 1.0 if num_anomalous == 0 else 0.7

    confidence = round(float(avail_factor * agreement_factor), 2)
    confidence = max(0.1, min(1.0, confidence))

    return FusionResult(
        fused_score=round(fused_score, 4),
        fused_status=fused_status,
        method=meth,  # type: ignore[arg-type]
        per_model_ratios={m: round(r, 4) for m, r in per_model_ratios.items()},
        missing_models=missing_models,
        reasons=reasons,
        confidence=confidence,
        is_replay=is_replay,
        details={
            "num_available": n_available,
            "num_anomalous": num_anomalous,
            "warning_threshold": cfg.warning_ratio_threshold,
            "alarm_threshold": cfg.alarm_ratio_threshold,
            "single_alarm_threshold": cfg.single_model_alarm_ratio,
        },
    )
