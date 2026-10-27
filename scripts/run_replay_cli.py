"""
CLI Replay Player for Multimodal PdM Telemetry Scenarios.

Streams or prints 60-second telemetry replay sequences as a formatted console table.
All telemetry outputs are tagged:
REPLAY: recorded CWRU / MIMII DG / Paderborn samples
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.replay import get_scenario, list_replay_scenarios, REPLAY_SOURCE_LABEL


def format_status(status: str) -> str:
    """Add colored or formatted indicators for status."""
    if status == "ALARM":
        return "ALARM   "
    elif status == "WARNING":
        return "WARNING "
    elif status == "NORMAL":
        return "NORMAL  "
    return "UNAVAIL "


def run_replay_cli(
    scenario_id: str = "multi_fault_escalation",
    speed: float = 0.0,
    max_steps: int = 60,
) -> int:
    """Print scenario replay telemetry steps as a formatted table."""
    try:
        steps = get_scenario(scenario_id)[:max_steps]
    except KeyError:
        print(f"Error: Unknown scenario '{scenario_id}'.")
        print("Available scenarios:")
        for sc in list_replay_scenarios():
            print(f"  - {sc['id']}: {sc['name']}")
        return 1

    print("\n" + "=" * 115)
    print(f" MULTIMODAL PDM TELEMETRY REPLAY: {scenario_id.upper()}")
    print(f" Source: {REPLAY_SOURCE_LABEL}")
    print(f" Disclaimer: Scenarios pair held-out test windows from CWRU, MIMII DG, & Paderborn.")
    print("=" * 115)

    header = (
        f"{'Step':>4} | {'Timestamp':<24} | {'Vibration':<18} | {'Audio':<18} | "
        f"{'Motor':<18} | {'Fused Ratio':>11} | {'Verdict':<8}"
    )
    print(header)
    print("-" * 115)

    delay = 1.0 / speed if speed > 0 else 0.0

    for step_data in steps:
        step_idx = step_data["step"]
        ts = step_data["timestamp"]
        models = step_data["models"]
        fusion = step_data["fusion"]

        v_res = models.get("vibration", {})
        a_res = models.get("audio", {})
        m_res = models.get("motor", {})

        v_str = f"r={v_res.get('score_ratio', 0.0):.2f} ({v_res.get('status', 'N/A')[:4]})"
        a_str = f"r={a_res.get('score_ratio', 0.0):.2f} ({a_res.get('status', 'N/A')[:4]})"
        m_str = f"r={m_res.get('score_ratio', 0.0):.2f} ({m_res.get('status', 'N/A')[:4]})"

        fused_score = fusion.get("fused_score", 0.0)
        verdict = format_status(fusion.get("fused_status", "NORMAL"))

        row = (
            f"{step_idx:>4} | {ts:<24} | {v_str:<18} | {a_str:<18} | "
            f"{m_str:<18} | {fused_score:>11.2f} | {verdict:<8}"
        )
        print(row)

        if delay > 0:
            time.sleep(delay)

    print("-" * 115)
    print(f"Replay complete ({len(steps)} steps). All samples flagged is_replay=True.\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal PdM Scenario Replay CLI")
    parser.add_argument(
        "--scenario",
        type=str,
        default="multi_fault_escalation",
        help="Scenario ID (default: multi_fault_escalation)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="Playback speed (steps/second, 0 = print instantly)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="Number of steps to play (default: 60)",
    )
    args = parser.parse_args()
    return run_replay_cli(scenario_id=args.scenario, speed=args.speed, max_steps=args.steps)


if __name__ == "__main__":
    sys.exit(main())
