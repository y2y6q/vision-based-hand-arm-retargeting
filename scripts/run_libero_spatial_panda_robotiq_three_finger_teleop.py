"""Teleoperate one official LIBERO Spatial state with Panda + Robotiq three-finger gripper.

Hold SpaceMouse left button to close and right button to open. Run only with
``.venv-libero`` because official LIBERO uses robosuite 1.4 in that venv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.libero_spatial_teleop import (  # noqa: E402
    LIBERO_ARM_ROTATION_SPEED_SCALE,
    LIBERO_ARM_TRANSLATION_SPEED_SCALE,
    PANDA_ROBOTIQ_THREE_FINGER_SPEC,
    run_libero_spatial_teleop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "spacemouse_wrist.json")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-order", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--headless-smoke", action="store_true")
    parser.add_argument("--viewer-smoke", action="store_true",
                        help="visible windows with synthetic zero SpaceMouse input (no HID needed)")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--gripper-rate", type=float, default=None)
    parser.add_argument("--no-three-view", action="store_true")
    parser.add_argument("--arm-speed", type=float, default=LIBERO_ARM_TRANSLATION_SPEED_SCALE,
                        help="SpaceMouse translation speed multiplier (0.1-5)")
    parser.add_argument("--rot-speed", type=float, default=LIBERO_ARM_ROTATION_SPEED_SCALE,
                        help="SpaceMouse rotation speed multiplier (0.1-5)")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.gripper_rate is not None and args.gripper_rate <= 0:
        parser.error("--gripper-rate must be positive")
    if args.headless_smoke and not args.headless:
        parser.error("--headless-smoke requires --headless")
    if args.viewer_smoke and (args.headless or args.headless_smoke):
        parser.error("--viewer-smoke shows windows; do not combine it with --headless")
    return args


def main() -> int:
    args = parse_args()
    return run_libero_spatial_teleop(
        PANDA_ROBOTIQ_THREE_FINGER_SPEC,
        WristConfig.load(args.config),
        task_id=args.task_id,
        task_order=args.task_order,
        init_state_id=args.init_state_id,
        seed=args.seed,
        headless=args.headless,
        duration_s=args.duration if args.duration is not None else (2.0 if args.headless_smoke else (10.0 if args.viewer_smoke else None)),
        camera_index=0,
        camera_display=False,
        camera_mirror=True,
        hand_control="off",
        synthetic_input=args.headless_smoke or args.viewer_smoke,
        three_view=not args.no_three_view,
        arm_speed_scale=args.arm_speed,
        rotation_speed_scale=args.rot_speed,
        three_finger_rate_per_s=args.gripper_rate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
