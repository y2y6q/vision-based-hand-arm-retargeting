"""SpaceMouse teleoperation for robosuite Panda + RobotiqThreeFingerGripper.

The six SpaceMouse axes drive only Panda world-frame OSC_POSE.  Hold the
configured left/right buttons to continuously close/open the official
three-finger gripper. Press B once to enlarge the Lift cube, R to reset, and
Q or Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.gripper_teleop import run_gripper_teleop  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "spacemouse_wrist.json")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--record", type=Path, default=None, help="front-view MP4 output; PNG fallback when needed")
    parser.add_argument("--cube-enlarge-factor", type=float, default=1.5)
    parser.add_argument(
        "--gripper-rate",
        type=float,
        default=None,
        help="optional Robotiq policy-command ramp in units/s; default is the faster 12.0 units/s",
    )
    parser.add_argument("--viewer-smoke", action="store_true")
    parser.add_argument("--headless-smoke", action="store_true")
    parser.add_argument("--no-three-view", action="store_true")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.cube_enlarge_factor <= 1.0:
        parser.error("--cube-enlarge-factor must be greater than one")
    if args.gripper_rate is not None and args.gripper_rate <= 0.0:
        parser.error("--gripper-rate must be positive")
    if args.viewer_smoke and args.headless:
        parser.error("--viewer-smoke requires a visible MuJoCo viewer")
    if args.headless_smoke and not args.headless:
        parser.error("--headless-smoke requires --headless")
    if args.viewer_smoke and args.headless_smoke:
        parser.error("choose either viewer smoke mode")
    return args


def main() -> int:
    args = parse_args()
    return run_gripper_teleop(
        robot="Panda", gripper_type="RobotiqThreeFingerGripper", config=WristConfig.load(args.config),
        headless=args.headless, duration_s=args.duration, viewer_smoke=args.viewer_smoke,
        headless_smoke=args.headless_smoke, enable_three_view=not args.no_three_view,
        cube_enlarge_factor=args.cube_enlarge_factor, record_path=args.record,
        three_finger_rate_per_s=args.gripper_rate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
