"""
Unit tests for the Multimodal Decision and Score Fusion Engine (core/fusion).

Tests cover:
- All three fusion methods: "max", "weighted", "vote"
- Missing models resilience (missing modalities never assumed normal)
- Boundary condition: ratio exactly 1.0
- Config-driven WARNING and ALARM transitions
- Extreme ratio single-sensor alarm (ratio >= 2.0)
- JSON serialization via to_dict()
"""

from __future__ import annotations

import json
import pytest

from core.models_api.interface import ModelResult
from core.fusion.fusion import FusionConfig, FusionResult, fuse


def _make_result(modality: str, score_ratio: float, status: str = "NORMAL") -> ModelResult:
    """Helper to construct dummy ModelResult with specified score_ratio."""
    threshold = 1.0
    anomaly_score = score_ratio * threshold
    st = "ANOMALOUS" if score_ratio >= 1.0 else status
    return ModelResult(
        model_name=f"{modality}_model",
        modality=modality,  # type: ignore[arg-type]
        status=st,          # type: ignore[arg-type]
        anomaly_score=anomaly_score,
        threshold=threshold,
        score_ratio=score_ratio,
        prediction_label="Normal" if st == "NORMAL" else "Fault",
        confidence=0.9,
        is_replay=True,
    )


# ---------------------------------------------------------------------------
# 1. Fusion Methods on Hand-Made Inputs
# ---------------------------------------------------------------------------

def test_fuse_max_method():
    r1 = _make_result("vibration", 0.5)
    r2 = _make_result("audio", 1.4)
    r3 = _make_result("motor", 0.7)

    res = fuse([r1, r2, r3], method="max")
    assert res.method == "max"
    assert res.fused_score == 1.4
    assert res.per_model_ratios == {"vibration": 0.5, "audio": 1.4, "motor": 0.7}
    assert res.missing_models == []
    assert res.confidence > 0.0


def test_fuse_weighted_method_equal_weights():
    r1 = _make_result("vibration", 0.6)
    r2 = _make_result("audio", 0.9)
    r3 = _make_result("motor", 0.3)

    res = fuse([r1, r2, r3], method="weighted")
    assert res.method == "weighted"
    # (0.6 + 0.9 + 0.3) / 3 = 0.6
    assert pytest.approx(res.fused_score, rel=1e-3) == 0.6


def test_fuse_weighted_custom_weights():
    cfg = FusionConfig(weights={"vibration": 0.8, "audio": 0.1, "motor": 0.1})
    r1 = _make_result("vibration", 1.0)
    r2 = _make_result("audio", 0.0)
    r3 = _make_result("motor", 0.0)

    res = fuse([r1, r2, r3], method="weighted", config=cfg)
    assert pytest.approx(res.fused_score, rel=1e-3) == 0.8


def test_fuse_vote_method_three_models_consensus():
    # 2 of 3 anomalous -> consensus anomalous
    r1 = _make_result("vibration", 1.3)
    r2 = _make_result("audio", 1.1)
    r3 = _make_result("motor", 0.4)

    res = fuse([r1, r2, r3], method="vote")
    assert res.method == "vote"
    assert res.fused_score >= 1.0
    assert res.fused_status == "ALARM"  # 2 models agreeing >= 1.0


def test_fuse_vote_method_single_anomaly_no_consensus():
    # 1 of 3 anomalous -> consensus normal (k=2 required)
    r1 = _make_result("vibration", 1.2)
    r2 = _make_result("audio", 0.4)
    r3 = _make_result("motor", 0.5)

    res = fuse([r1, r2, r3], method="vote")
    assert res.fused_score < 1.0
    # Still WARNING because 1 model is anomalous
    assert res.fused_status == "WARNING"


def test_fuse_vote_method_single_available_model():
    # With 1 model, k=1
    r1 = _make_result("vibration", 1.2)
    res = fuse([r1], method="vote")
    assert res.fused_score == 1.2


# ---------------------------------------------------------------------------
# 2. Missing Models & Availability
# ---------------------------------------------------------------------------

def test_missing_modality_is_never_assumed_normal():
    # Only vibration provided; audio and motor missing
    r1 = _make_result("vibration", 0.5)
    res = fuse([r1], method="max")

    assert "audio" in res.missing_models
    assert "motor" in res.missing_models
    # Confidence is degraded because only 1 of 3 expected models is present
    assert res.confidence <= 0.4
    assert res.fused_status == "NORMAL"


def test_all_models_unavailable():
    res = fuse([], method="max")
    assert res.fused_status == "UNAVAILABLE"
    assert res.confidence == 0.0
    assert len(res.missing_models) == 3


# ---------------------------------------------------------------------------
# 3. Ratio Boundary: Exactly 1.0
# ---------------------------------------------------------------------------

def test_ratio_exactly_one_point_zero():
    # Exactly 1.0 means at/above threshold
    r1 = _make_result("vibration", 1.0)
    r2 = _make_result("audio", 1.0)
    r3 = _make_result("motor", 0.5)

    res = fuse([r1, r2, r3], method="max")
    assert res.fused_score == 1.0
    # 2 models agree at 1.0 -> triggers ALARM
    assert res.fused_status == "ALARM"


def test_single_model_at_one_point_zero():
    r1 = _make_result("vibration", 1.0)
    r2 = _make_result("audio", 0.5)
    r3 = _make_result("motor", 0.5)

    res = fuse([r1, r2, r3], method="max")
    # 1 model at 1.0 (no multi-agreement, not >= 2.0) -> WARNING
    assert res.fused_status == "WARNING"


# ---------------------------------------------------------------------------
# 4. Config-Driven WARNING and ALARM Transitions
# ---------------------------------------------------------------------------

def test_warning_triggered_by_fused_ratio_above_0_8():
    # All individually < 1.0, but fused ratio >= 0.8
    r1 = _make_result("vibration", 0.85)
    r2 = _make_result("audio", 0.7)
    r3 = _make_result("motor", 0.6)

    res = fuse([r1, r2, r3], method="max")
    assert res.fused_score == 0.85
    assert res.fused_status == "WARNING"


def test_single_model_extreme_ratio_triggers_alarm():
    # Single ratio >= 2.0 triggers ALARM even with no agreement
    r1 = _make_result("vibration", 2.1)
    r2 = _make_result("audio", 0.2)
    r3 = _make_result("motor", 0.1)

    res = fuse([r1, r2, r3], method="weighted")
    assert res.fused_status == "ALARM"
    assert any("critical" in reason.lower() for reason in res.reasons)


def test_custom_config_thresholds():
    # Lower alarm threshold to 0.75
    custom_cfg = FusionConfig(warning_ratio_threshold=0.5, alarm_ratio_threshold=0.75)
    r1 = _make_result("vibration", 0.78)
    r2 = _make_result("audio", 0.80)

    # 2 models agreeing above 0.75 with custom cfg
    r1.status = "ANOMALOUS"
    r2.status = "ANOMALOUS"
    res = fuse([r1, r2], method="max", config=custom_cfg)
    assert res.fused_status == "ALARM"


# ---------------------------------------------------------------------------
# 5. Serialization via to_dict()
# ---------------------------------------------------------------------------

def test_fusion_result_to_dict():
    r1 = _make_result("vibration", 1.1)
    res = fuse([r1], method="max")
    d = res.to_dict()

    assert isinstance(d, dict)
    assert d["fused_score"] == 1.1
    assert d["is_replay"] is True
    # Verify JSON serializability
    json_str = json.dumps(d)
    assert isinstance(json_str, str)
