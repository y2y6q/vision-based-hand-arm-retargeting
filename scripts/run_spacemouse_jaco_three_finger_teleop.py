"""SpaceMouse teleoperation for robosuite Jaco + JacoThreeFingerGripper.

Six SpaceMouse axes are interpreted relative to the configured primary camera,
then drive the Jaco end effector through world-frame ``OSC_POSE``. Hold the
configured left button to continuously close the
official one-DoF three-finger gripper; hold the right button to continuously
open it. Releasing either button sends exact zero gripper velocity. ``R``
resets; ``Q`` and Ctrl+C safely send a zero action and release resources.
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
    parser.add_argument("--headless", action="store_true", help="run without a MuJoCo viewer")
    parser.add_argument("--duration", type=float, default=None, help="optional bounded run duration in seconds")
    parser.add_argument(
        "--viewer-smoke",
        action="store_true",
        help="run bounded visible synthetic zero-motion + close/open smoke validation",
    )
    parser.add_argument(
        "--headless-smoke",
        action="store_true",
        help="run bounded headless synthetic zero-motion + close/open smoke validation",
    )
    parser.add_argument(
        "--no-three-view",
        action="store_true",
        help="do not enable the optional project three-view visualization hook",
    )
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.viewer_smoke and args.headless:
        parser.error("--viewer-smoke requires a visible MuJoCo viewer; use --headless-smoke with --headless")
    if args.headless_smoke and not args.headless:
        parser.error("--headless-smoke requires --headless")
    if args.viewer_smoke and args.headless_smoke:
        parser.error("choose either --viewer-smoke or --headless-smoke")
    return args


def main() -> int:
    args = parse_args()
    return run_gripper_teleop(
        robot="Jaco",
        config=WristConfig.load(args.config),
        headless=args.headless,
        duration_s=args.duration,
        viewer_smoke=args.viewer_smoke,
        headless_smoke=args.headless_smoke,
        enable_three_view=not args.no_three_view,
    )


if __name__ == "__main__":
    raise SystemExit(main())
