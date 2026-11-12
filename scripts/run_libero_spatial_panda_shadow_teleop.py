"""Teleoperate one official LIBERO state (Spatial by default, ``--suite`` for others) with Panda + Shadow Hand.

SpaceMouse moves only the Panda end effector. The camera drives Shadow through
official dex-retargeting DexPilot by default (``--hand-control dex``).
``--hand-mount down`` turns the hand on the flange so the fingers point down.
``--object-mass-scale`` makes the task objects heavier (LIBERO's weigh 5-30 g).
``--object-mass NAME=GRAMS ...`` sets single objects (class like ``AkitaBlackBowl`` or
instance like ``butter_2``); the others follow ``--object-mass-scale``.
``--thumb-lift-gain`` scales how far the thumb lifts off the palm (camera depth is compressed).
``--hand-delay-ms`` sets the fixed delay at which each tick reads the camera hand (default 65 ms).

Simulator window: main view, a close-up and a top view that follow the hand
(``V`` turns the close-up around the hand), webcam preview in the corner.

Run only with ``.venv-libero``. The released benchmark uses robosuite 1.4;
the project's normal robosuite mainline uses 1.5.2 in a different environment.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.libero_spatial_teleop import (  # noqa: E402
    LIBERO_ARM_ROTATION_SPEED_SCALE,
    LIBERO_ARM_TRANSLATION_SPEED_SCALE,
    LIBERO_SUITES,
    OBJECT_MASS_RANGE_G,
    OBJECT_MASS_SCALE_RANGE,
    parse_object_mass_overrides,
    run_libero_spatial_teleop,
    shadow_spec,
)
from wrist_teleop.libero_views import parse_view_size  # noqa: E402
from wrist_teleop.shadow_dex import SHADOW_THUMB_LIFT_GAIN_RANGE  # noqa: E402
from wrist_teleop.shadow_gripper import SHADOW_MOUNTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "spacemouse_wrist.json")
    parser.add_argument("--suite", choices=LIBERO_SUITES, default="libero_spatial",
                        help="official task suite; --task-id picks a task inside it")
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
    parser.add_argument("--dex-official", action="store_true",
                        help="unmodified official DexPilot (without the project mapping terms)")
    parser.add_argument("--thumb-lift-gain", type=float, default=None,
                        help="multiply the thumb's lift off the palm as the camera sees it "
                             "({:g}-{:g}; default 1.5 from shadow_grasp_terms.json, 1 = as measured)".format(
                                 *SHADOW_THUMB_LIFT_GAIN_RANGE))
    parser.add_argument("--hand-mount", choices=SHADOW_MOUNTS, default="forward",
                        help="forward: fingers forward, palm down (default); down: fingers point down")
    parser.add_argument("--object-mass-scale", type=float, default=1.0,
                        help="multiply every task object's mass ({:g}-{:g}; 1 = official)".format(*OBJECT_MASS_SCALE_RANGE))
    parser.add_argument("--object-mass", nargs="+", action="extend", default=[], metavar="NAME=GRAMS",
                        help="set one object's mass in grams ({:g}-{:g}), e.g. AkitaBlackBowl=300 butter_2=110; "
                             "NAME is a class from docs/libero_tasks_object_masses.md or an instance; "
                             "objects not named follow --object-mass-scale".format(*OBJECT_MASS_RANGE_G))
    parser.add_argument("--view-size", type=parse_view_size, default=None, metavar="WxH",
                        help="simulator window size in pixels (default: fill the primary monitor)")
    parser.add_argument("--hand-delay-ms", type=float, default=None,
                        help="fixed camera-hand delay behind each tick, 0-300 ms (default: config hand_target_delay_s)")
    args = parser.parse_args()
    if args.hand_delay_ms is not None and not 0.0 <= args.hand_delay_ms <= 300.0:
        parser.error("--hand-delay-ms must be within 0-300")
    if args.thumb_lift_gain is not None:
        if args.dex_official:
            parser.error("--thumb-lift-gain belongs to the project mapping; drop --dex-official")
        if not SHADOW_THUMB_LIFT_GAIN_RANGE[0] <= args.thumb_lift_gain <= SHADOW_THUMB_LIFT_GAIN_RANGE[1]:
            parser.error("--thumb-lift-gain must be within {:g}-{:g}".format(*SHADOW_THUMB_LIFT_GAIN_RANGE))
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.headless_smoke and not args.headless:
        parser.error("--headless-smoke requires --headless")
    if args.viewer_smoke and (args.headless or args.headless_smoke):
        parser.error("--viewer-smoke shows windows; do not combine it with --headless")
    try:
        args.object_masses_kg = parse_object_mass_overrides(args.object_mass)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    args = parse_args()
    config = WristConfig.load(args.config)
    if args.hand_delay_ms is not None:
        config = replace(config, hand_target_delay_s=args.hand_delay_ms / 1e3)
    return run_libero_spatial_teleop(
        shadow_spec(args.hand_mount),
        config,
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
        suite=args.suite,
        object_mass_scale=args.object_mass_scale,
        object_masses_kg=args.object_masses_kg,
        view_size=args.view_size,
        dex_grasp_terms=not args.dex_official,
        thumb_lift_gain=args.thumb_lift_gain,
    )


if __name__ == "__main__":
    raise SystemExit(main())
