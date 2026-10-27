"""
End-to-End Evaluation & Latency Benchmarking Script.

Executes all 5 replay scenarios, collects step-by-step multimodal fusion verdicts,
measures API endpoint latencies (median and p95), saves detailed JSON results to
results/integration/e2e_results.json, and generates diagnostic trajectory plots.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app
from core.replay import get_scenario, list_replay_scenarios, REPLAY_SOURCE_LABEL

client = TestClient(app)

RESULTS_DIR = PROJECT_ROOT / "results" / "integration"
FIGURES_DIR = RESULTS_DIR / "figures"


def benchmark_endpoint_latency(
    name: str,
    method: str,
    url: str,
    payload: Any = None,
    n_iterations: int = 25,
) -> Dict[str, float]:
    """Measure request latency in milliseconds across multiple iterations."""
    latencies_ms: List[float] = []

    # Warmup
    for _ in range(2):
        if method == "GET":
            client.get(url)
        elif method == "POST":
            client.post(url, json=payload)

    # Measurement iterations
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        if method == "GET":
            resp = client.get(url)
        elif method == "POST":
            resp = client.post(url, json=payload)
        t1 = time.perf_counter()
        if resp.status_code == 200:
            latencies_ms.append((t1 - t0) * 1000.0)

    if not latencies_ms:
        return {"median_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    arr = np.array(latencies_ms)
    return {
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "n_samples": len(arr),
    }


def evaluate_all_scenarios() -> Dict[str, Any]:
    """Collect step-by-step fusion status and trajectory data for all scenarios."""
    scenarios_meta = list_replay_scenarios()
    scenarios_results: Dict[str, Any] = {}

    for meta in scenarios_meta:
        sc_id = meta["id"]
        steps = get_scenario(sc_id)

        step_records: List[Dict[str, Any]] = []
        status_counts = {"NORMAL": 0, "WARNING": 0, "ALARM": 0, "UNAVAILABLE": 0}
        first_warning_step: int | None = None
        first_alarm_step: int | None = None

        for s in steps:
            fusion = s["fusion"]
            status = fusion["fused_status"]
            status_counts[status] = status_counts.get(status, 0) + 1

            if status == "WARNING" and first_warning_step is None:
                first_warning_step = s["step"]
            if status == "ALARM" and first_alarm_step is None:
                first_alarm_step = s["step"]

            models = s["models"]
            step_records.append({
                "step": s["step"],
                "timestamp": s["timestamp"],
                "fused_score": fusion["fused_score"],
                "fused_status": status,
                "reasons": fusion.get("reasons", []),
                "vibration_ratio": models.get("vibration", {}).get("score_ratio", 0.0),
                "audio_ratio": models.get("audio", {}).get("score_ratio", 0.0),
                "motor_ratio": models.get("motor", {}).get("score_ratio", 0.0),
                "is_replay": s["is_replay"],
            })

        scenarios_results[sc_id] = {
            "id": sc_id,
            "name": meta["name"],
            "description": meta["description"],
            "total_steps": len(steps),
            "status_counts": status_counts,
            "first_warning_step": first_warning_step,
            "first_alarm_step": first_alarm_step,
            "steps": step_records,
            "is_replay": True,
        }

    return scenarios_results


def run_latency_benchmarks() -> Dict[str, Any]:
    """Benchmark key FastAPI endpoints."""
    # Find sample IDs
    samples_vib = client.get("/samples/vibration").json()
    vib_sid = samples_vib[0]["sample_id"] if samples_vib else "105"

    samples_audio = client.get("/samples/audio").json()
    audio_sid = samples_audio[0]["sample_id"] if samples_audio else "fan_001"

    fusion_payload = {
        "results": [
            {
                "model_name": "vibration_cwru",
                "modality": "vibration",
                "status": "NORMAL",
                "anomaly_score": 0.35,
                "threshold": 1.0,
                "score_ratio": 0.35,
            },
            {
                "model_name": "audio_ae",
                "modality": "audio",
                "status": "ANOMALOUS",
                "anomaly_score": 1.25,
                "threshold": 1.0,
                "score_ratio": 1.25,
            },
            {
                "model_name": "motor_mcsa",
                "modality": "motor",
                "status": "NORMAL",
                "anomaly_score": 0.22,
                "threshold": 0.5,
                "score_ratio": 0.44,
            },
        ],
        "method": "max",
    }

    benchmarks = {
        "GET /models": benchmark_endpoint_latency("models", "GET", "/models"),
        "GET /replay/scenarios": benchmark_endpoint_latency("replay_scenarios", "GET", "/replay/scenarios"),
        f"POST /predict/vibration/sample/{vib_sid}": benchmark_endpoint_latency(
            "predict_vibration", "POST", f"/predict/vibration/sample/{vib_sid}"
        ),
        f"POST /predict/audio/sample/{audio_sid}": benchmark_endpoint_latency(
            "predict_audio", "POST", f"/predict/audio/sample/{audio_sid}"
        ),
        "POST /predict/motor/sample/paderborn_demo_normal": benchmark_endpoint_latency(
            "predict_motor", "POST", "/predict/motor/sample/paderborn_demo_normal"
        ),
        "POST /fusion": benchmark_endpoint_latency(
            "fusion", "POST", "/fusion", payload=fusion_payload
        ),
    }

    return benchmarks


def plot_scenario_trajectories(scenarios_data: Dict[str, Any], output_path: Path) -> None:
    """Plot multi-sensor score ratio trajectories and fused scores across all scenarios."""
    fig, axes = plt.subplots(len(scenarios_data), 1, figsize=(12, 16), sharex=True)
    if len(scenarios_data) == 1:
        axes = [axes]

    color_vib = "#0284c7"  # Sky blue
    color_aud = "#9333ea"  # Purple
    color_mot = "#d97706"  # Amber
    color_fus = "#dc2626"  # Red

    for ax, (sc_id, sc_info) in zip(axes, scenarios_data.items()):
        steps = sc_info["steps"]
        x = [s["step"] for s in steps]
        v_r = [s["vibration_ratio"] for s in steps]
        a_r = [s["audio_ratio"] for s in steps]
        m_r = [s["motor_ratio"] for s in steps]
        f_s = [s["fused_score"] for s in steps]

        ax.plot(x, v_r, label="Vibration Ratio", color=color_vib, linewidth=1.5, alpha=0.8)
        ax.plot(x, a_r, label="Audio Ratio", color=color_aud, linewidth=1.5, alpha=0.8)
        ax.plot(x, m_r, label="Motor Ratio", color=color_mot, linewidth=1.5, alpha=0.8)
        ax.plot(x, f_s, label="Fused Score", color=color_fus, linewidth=2.5, linestyle="-")

        # Threshold line at 1.0
        ax.axhline(1.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Threshold (1.0)")
        ax.axhline(0.8, color="#f59e0b", linestyle=":", linewidth=1.0, label="Warning (0.8)")

        # Onset markers if present
        if sc_info["first_warning_step"]:
            ax.axvline(sc_info["first_warning_step"], color="#f59e0b", linestyle=":", alpha=0.6)
        if sc_info["first_alarm_step"]:
            ax.axvline(sc_info["first_alarm_step"], color="#dc2626", linestyle="--", alpha=0.6)

        ax.set_title(f"Scenario: {sc_info['name']} — {sc_info['description']}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score Ratio", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(0, max(2.5, max(f_s) * 1.15))

    axes[-1].set_xlabel("Replay Step (seconds)", fontsize=11)
    axes[0].legend(loc="upper left", ncol=5, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_latency_benchmarks(benchmarks: Dict[str, Any], output_path: Path) -> None:
    """Generate bar chart of median and p95 API endpoint latencies."""
    fig, ax = plt.subplots(figsize=(10, 5))

    endpoints = [k.replace("POST /predict/", "predict/").replace("/sample/", "/") for k in benchmarks.keys()]
    medians = [benchmarks[k]["median_ms"] for k in benchmarks.keys()]
    p95s = [benchmarks[k]["p95_ms"] for k in benchmarks.keys()]

    x = np.arange(len(endpoints))
    width = 0.35

    rects1 = ax.bar(x - width / 2, medians, width, label="Median Latency (ms)", color="#3b82f6")
    rects2 = ax.bar(x + width / 2, p95s, width, label="95th-Percentile Latency (ms)", color="#8b5cf6")

    ax.set_ylabel("Latency (milliseconds)", fontsize=11)
    ax.set_title("FastAPI End-to-End Endpoint Latencies (Benchmark)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(endpoints, rotation=20, ha="right", fontsize=9)
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" Running End-to-End Scenarios & Latency Benchmarks")
    print("=" * 80)

    # 1. Scenarios Evaluation
    print("\n[1/3] Simulating all 5 replay scenarios (60 steps each)...")
    scenarios_data = evaluate_all_scenarios()
    for sc_id, sc in scenarios_data.items():
        print(f"  [OK] {sc['name']:<35} -> NORMAL: {sc['status_counts']['NORMAL']:>2}, "
              f"WARN: {sc['status_counts']['WARNING']:>2}, ALARM: {sc['status_counts']['ALARM']:>2} "
              f"(Onset: step {sc['first_warning_step'] or '-'})")

    # 2. Latency Benchmarks
    print("\n[2/3] Measuring API endpoint latencies (median & p95)...")
    benchmarks = run_latency_benchmarks()
    for ep, stats in benchmarks.items():
        print(f"  * {ep:<48} : median={stats['median_ms']:>6.2f} ms | p95={stats['p95_ms']:>6.2f} ms")

    # 3. Save JSON Results
    print("\n[3/3] Saving results and generating figures...")
    e2e_results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "All end-to-end data is replayed from held-out recorded benchmark datasets (CWRU, MIMII DG, Paderborn). No live sensors were used in Phase 1.",
        "scenarios": {
            sc_id: {
                "id": sc["id"],
                "name": sc["name"],
                "description": sc["description"],
                "total_steps": sc["total_steps"],
                "status_counts": sc["status_counts"],
                "first_warning_step": sc["first_warning_step"],
                "first_alarm_step": sc["first_alarm_step"],
                "is_replay": True,
            }
            for sc_id, sc in scenarios_data.items()
        },
        "latency_benchmarks": benchmarks,
        "scenarios_trajectory_sample": {
            sc_id: sc["steps"][:10] for sc_id, sc in scenarios_data.items()
        },
    }

    json_path = RESULTS_DIR / "e2e_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(e2e_results, f, indent=2)
    print(f"  [OK] Saved JSON results to: {json_path}")

    # Generate Figures
    fig_trajectories = FIGURES_DIR / "e2e_scenario_trajectories.png"
    plot_scenario_trajectories(scenarios_data, fig_trajectories)
    print(f"  [OK] Saved trajectory plot to: {fig_trajectories}")

    fig_latency = FIGURES_DIR / "e2e_api_latency.png"
    plot_latency_benchmarks(benchmarks, fig_latency)
    print(f"  [OK] Saved latency plot to: {fig_latency}")

    print("\nAll integration artifacts successfully generated!\n")



if __name__ == "__main__":
    main()
