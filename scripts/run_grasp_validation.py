r"""Run the ordered, hardware-free Panda + Allegro grasp validation gates.

Examples:

    & .\.venv-robosuite\Scripts\python.exe .\scripts\run_grasp_validation.py --gate 1
    & .\.venv-robosuite\Scripts\python.exe .\scripts\run_grasp_validation.py --gate 2
    & .\.venv-robosuite\Scripts\python.exe .\scripts\run_grasp_validation.py --gate 3
    & .\.venv-robosuite\Scripts\python.exe .\scripts\run_grasp_validation.py --gate 2 --visualize

Requests for Gate 2 and Gate 3 re-run their lower prerequisite gate(s) under
the exact same snapshot.  No SpaceMouse, camera, motion planner, or legacy
PyBullet component is constructed by this entry point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.grasp_validation import (  # noqa: E402
    DEFAULT_VISUAL_OUTPUT_ROOT,
    GraspValidationRunner,
    run_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="optional parent directory for timestamped validation artifacts",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="disable offscreen frame capture (only for headless CI diagnosis)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="replay the current locked Gate 2 candidate in the real MuJoCo viewer",
    )
    parser.add_argument(
        "--locked-config",
        type=Path,
        default=None,
        help="optional qualified locked_gate2_configuration.json; defaults to the newest lock",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="optional MP4 path; falls back to PNG frames if a video codec is unavailable",
    )
    parser.add_argument(
        "--auto-exit-after-final-s",
        type=float,
        default=None,
        help="testing only: close the visual viewer this many wall-clock seconds after the final result",
    )
    args = parser.parse_args()
    if args.visualize:
        if args.gate != "2":
            parser.error("--visualize requires --gate 2 and never runs Gate 3 or later")
        output_root = DEFAULT_VISUAL_OUTPUT_ROOT if args.output_root is None else args.output_root
        runner = GraspValidationRunner(
            output_root=output_root,
            record_frames=(not args.no_frames) or args.record is not None,
        )
        code, output, summary = runner.run_gate2_visualization(
            locked_config_path=args.locked_config,
            record_path=args.record,
            auto_exit_after_final_s=args.auto_exit_after_final_s,
        )
        compact = {
            "status": summary.get("status"),
            "output": str(output),
            "exit_reason": summary.get("exit_reason"),
            "physics_parameters_changed": summary.get("physics_parameters_changed"),
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2))
        return int(code)
    if args.locked_config is not None or args.record is not None or args.auto_exit_after_final_s is not None:
        parser.error("--locked-config, --record, and --auto-exit-after-final-s require --visualize")
    gate: int | str = "all" if args.gate == "all" else int(args.gate)
    code, output, summary = run_validation(
        gate,
        output_root=args.output_root,
        record_frames=not args.no_frames,
    )
    compact = {
        "status": summary.get("status"),
        "output": str(output),
        "gate_results": {
            name: bool(result.get("passed"))
            for name, result in dict(summary.get("gates", {})).items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
