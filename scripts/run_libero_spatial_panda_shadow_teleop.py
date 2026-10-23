"""Teleoperate one official LIBERO Spatial state with Panda + Shadow Hand.

SpaceMouse moves only the Panda end effector. The camera drives Shadow through
official dex-retargeting DexPilot by default (``--hand-control dex``).

Run only with ``.venv-libero``. The released benchmark uses robosuite 1.4;
the project's normal robosuite mainline uses 1.5.2 in a different environment.
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
    PANDA_SHADOW_SPEC,
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
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-no-mirror", action="store_true")
    parser.add_argument("--no-camera-display", action="store_true")
    parser.add_argument("--hand-control", choices=("dex", "legacy-curl", "off"), default="dex")
    parser.add_argument("--no-three-view", action="store_true")
    parser.add_argument("--arm-speed", type=float, default=LIBERO_ARM_TRANSLATION_SPEED_SCALE,
                        help="SpaceMouse translation speed multiplier (0.1-5)")
    parser.add_argument("--rot-speed", type=float, default=LIBERO_ARM_ROTATION_SPEED_SCALE,
                        help="SpaceMouse rotation speed multiplier (0.1-5)")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.headless_smoke and not args.headless:
        parser.error("--headless-smoke requires --headless")
    if args.viewer_smoke and (args.headless or args.headless_smoke):
        parser.error("--viewer-smoke shows windows; do not combine it with --headless")
    return args


def main() -> int:
    args = parse_args()
    return run_libero_spatial_teleop(
        PANDA_SHADOW_SPEC,
        WristConfig.load(args.config),
        task_id=args.task_id,
        task_order=args.task_order,
        init_state_id=args.init_state_id,
        seed=args.seed,
        headless=args.headless,
        duration_s=args.duration if args.duration is not None else (2.0 if args.headless_smoke else (10.0 if args.viewer_smoke else None)),
        camera_index=args.camera_index,
        camera_display=not args.no_camera_display,
        camera_mirror=not args.camera_no_mirror,
        hand_control=args.hand_control,
        synthetic_input=args.headless_smoke or args.viewer_smoke,
        three_view=not args.no_three_view,
        arm_speed_scale=args.arm_speed,
        rotation_speed_scale=args.rot_speed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
