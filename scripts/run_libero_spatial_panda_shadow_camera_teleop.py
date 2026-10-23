"""Camera-only teleoperation of an official LIBERO Spatial scene: Panda + Shadow Hand.

No SpaceMouse. Your RIGHT hand in front of the webcam drives everything:

* move it left / right, up / down: the end effector follows like a mouse
  pointer (slow motion = fine, fast motion = farther);
* hold it nearer the camera than where it started: the end effector moves
  away at a speed set by how much nearer; farther: it comes back; in between
  (the deadzone shown on the depth gauge) it stays;
* turn or tilt it: the end effector follows the palm's rotation (only
  while the hand is not translating);
* the fingers are retargeted to the Shadow Hand (DexPilot by default).

Show your LEFT hand next to the right one to clutch: the arm holds while you
reposition the right hand. Run only with ``.venv-libero``.

Keys: C pause / resume, - / = arm gain, R reset, Q quit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.camera_arm_control import CameraArmSettings  # noqa: E402
from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.libero_camera_teleop import run_libero_camera_teleop  # noqa: E402


def parse_args() -> argparse.Namespace:
    defaults = CameraArmSettings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "spacemouse_wrist.json")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-order", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-no-mirror", action="store_true")
    parser.add_argument("--no-camera-display", action="store_true")
    parser.add_argument("--hand-control", choices=("dex", "legacy-curl", "off"), default="dex",
                        help="finger mapping; the arm always follows the wrist")
    parser.add_argument("--gain", type=float, default=defaults.translation_gain,
                        help="multiplier on the speed-dependent left/right and up/down gain")
    parser.add_argument("--depth-gain", type=float, default=defaults.depth_gain,
                        help="robot m/s per metre of hand offset from the neutral depth (beyond the deadzone)")
    parser.add_argument("--rotation-gain", type=float, default=defaults.rotation_gain)
    parser.add_argument("--camera-fov", type=float, default=65.0,
                        help="webcam horizontal field of view in degrees (sets the depth scale)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--headless-smoke", action="store_true",
                        help="no windows; scripted synthetic hand instead of the webcam")
    parser.add_argument("--viewer-smoke", action="store_true",
                        help="windows on; scripted synthetic hand instead of the webcam")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--no-three-view", action="store_true")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.headless_smoke and not args.headless:
        parser.error("--headless-smoke requires --headless")
    if args.viewer_smoke and args.headless:
        parser.error("--viewer-smoke shows windows; do not combine it with --headless")
    if not 20.0 <= args.camera_fov <= 120.0:
        parser.error("--camera-fov must be between 20 and 120 degrees")
    for name in ("gain", "depth_gain"):
        if not 0.2 <= getattr(args, name) <= 4.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0.2, 4]")
    return args


def main() -> int:
    import math

    args = parse_args()
    smoke = args.headless_smoke or args.viewer_smoke
    settings = CameraArmSettings(
        focal_width_normalized=0.5 / math.tan(math.radians(args.camera_fov) / 2.0),
        translation_gain=args.gain, depth_gain=args.depth_gain, rotation_gain=args.rotation_gain,
    )
    return run_libero_camera_teleop(
        WristConfig.load(args.config), settings,
        task_id=args.task_id, task_order=args.task_order, init_state_id=args.init_state_id, seed=args.seed,
        headless=args.headless,
        duration_s=args.duration if args.duration is not None else (11.0 if smoke else None),
        camera_index=args.camera_index, camera_display=not args.no_camera_display,
        camera_mirror=not args.camera_no_mirror,
        hand_control="off" if smoke and args.hand_control == "dex" else args.hand_control,
        synthetic_hand=smoke, three_view=not args.no_three_view,
    )


if __name__ == "__main__":
    raise SystemExit(main())
