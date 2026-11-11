"""LIBERO Panda + Shadow with an approach planner (copy of the Shadow SpaceMouse entry; that one is unchanged).

Number keys 1-9 pick a task object / fixture (small number badges in the
views, names printed at start). A motion planner moves the arm until the
hand's grasp space is above it; moving the SpaceMouse stops the planner and
takes over at once, and after arrival the SpaceMouse does the fine approach.
The camera drives the Shadow fingers throughout (``--hand-control dex`` by
default). ``0`` cancels a plan, ``L`` toggles the five fingertip rays, ``M``
switches the hand mount (fingers down <-> palm parallel to the table) and
restarts the task.

``--planner ompl`` (default, ``ompl_planner`` Conda env) checks straight lift /
traverse / descend routes with the hand orientation fixed and MuJoCo
clearance to the whole robot, and searches with OMPL RRT-Connect only when
those are blocked; ``--planner straight`` drives the first such route
without any check, for comparison. ``--planner-speed`` sets its peak speed.
``--sim-speed`` runs the simulation slower than real time (1, 0.8, 0.6, 0.4).

Covers libero_spatial and libero_object. Run only with ``.venv-libero``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.libero_motion_planning import (  # noqa: E402
    PLANNER_BACKENDS,
    PLANNER_MAX_SPEED_M_S,
    PLANNER_SPEED_RANGE_M_S,
    PREGRASP_CLEARANCE_M,
)
from wrist_teleop.libero_planner_teleop import (  # noqa: E402
    PLANNER_SUITES,
    SIM_SPEED_CHOICES,
    run_libero_planner_teleop,
)
from wrist_teleop.libero_spatial_teleop import (  # noqa: E402
    LIBERO_ARM_ROTATION_SPEED_SCALE,
    LIBERO_ARM_TRANSLATION_SPEED_SCALE,
    OBJECT_MASS_RANGE_G,
    OBJECT_MASS_SCALE_RANGE,
    parse_object_mass_overrides,
    shadow_spec,
)
from wrist_teleop.libero_views import parse_view_size  # noqa: E402
from wrist_teleop.shadow_dex import SHADOW_THUMB_LIFT_GAIN_RANGE  # noqa: E402
from wrist_teleop.shadow_gripper import SHADOW_MOUNTS  # noqa: E402


def parse_smoke_keys(text: str) -> list[tuple[float, str]]:
    """``"0.5:3,6:m"`` -> ``[(0.5, "3"), (6.0, "m")]``."""
    keys = []
    for item in filter(None, (part.strip() for part in text.split(","))):
        at, _, key = item.partition(":")
        try:
            seconds = float(at)
        except ValueError:
            raise ValueError("--smoke-keys entries are SECONDS:KEY, got {!r}".format(item)) from None
        if len(key) != 1 or seconds < 0.0:
            raise ValueError("--smoke-keys entries are SECONDS:KEY with one key, got {!r}".format(item))
        keys.append((seconds, key))
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "spacemouse_wrist.json")
    parser.add_argument("--suite", choices=PLANNER_SUITES, default="libero_spatial",
                        help="official task suite; --task-id picks a task inside it")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-order", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--headless-smoke", action="store_true",
                        help="with --headless: synthetic zero SpaceMouse, no camera, short run")
    parser.add_argument("--smoke-target", type=int, default=None,
                        help="request this target number once after 0.5 s (testing)")
    parser.add_argument("--smoke-keys", default="", metavar="T:KEY,...",
                        help="press keys at run times in seconds, e.g. 0.5:3,6:m (testing)")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-no-mirror", action="store_true")
    parser.add_argument("--no-camera-display", action="store_true")
    parser.add_argument("--hand-control", choices=("dex", "legacy-curl", "off"), default="dex")
    parser.add_argument("--arm-speed", type=float, default=LIBERO_ARM_TRANSLATION_SPEED_SCALE,
                        help="SpaceMouse translation speed multiplier (0.1-5)")
    parser.add_argument("--rot-speed", type=float, default=LIBERO_ARM_ROTATION_SPEED_SCALE,
                        help="SpaceMouse rotation speed multiplier (0.1-5)")
    parser.add_argument("--dex-official", action="store_true",
                        help="unmodified official DexPilot (without the project mapping terms)")
    parser.add_argument("--thumb-lift-gain", type=float, default=None,
                        help="thumb lift gain ({:g}-{:g}; default from shadow_grasp_terms.json)".format(
                            *SHADOW_THUMB_LIFT_GAIN_RANGE))
    parser.add_argument("--hand-mount", choices=SHADOW_MOUNTS, default="down",
                        help="start mount; down (default here): fingers point down; forward: palm parallel to "
                             "the table, fingers forward. M switches while running")
    parser.add_argument("--object-mass-scale", type=float, default=1.0,
                        help="multiply every task object's mass ({:g}-{:g}; 1 = official)".format(*OBJECT_MASS_SCALE_RANGE))
    parser.add_argument("--object-mass", nargs="+", action="extend", default=[], metavar="NAME=GRAMS",
                        help="set one object's mass in grams ({:g}-{:g})".format(*OBJECT_MASS_RANGE_G))
    parser.add_argument("--view-size", type=parse_view_size, default=None, metavar="WxH")
    parser.add_argument("--hand-delay-ms", type=float, default=None,
                        help="fixed camera-hand delay behind each tick, 0-300 ms (default: config)")
    parser.add_argument("--sim-speed", type=float, choices=SIM_SPEED_CHOICES, default=1.0,
                        help="simulated seconds per real second (default 1 = real time)")
    parser.add_argument("--planner", choices=PLANNER_BACKENDS, default="ompl",
                        help="ompl: checked straight routes, then OMPL RRT-Connect; straight: lift / traverse / "
                             "descend, no collision check")
    parser.add_argument("--planner-speed", type=float, default=PLANNER_MAX_SPEED_M_S,
                        help="planner peak hand speed in m/s, simulated time ({:g}-{:g}, default {:g})".format(
                            *PLANNER_SPEED_RANGE_M_S, PLANNER_MAX_SPEED_M_S))
    parser.add_argument("--planner-python", default=None,
                        help="python.exe of the ompl_planner Conda env (default: found next to panda_retarget)")
    parser.add_argument("--clearance-cm", type=float, default=100.0 * PREGRASP_CLEARANCE_M,
                        help="lowest fingertip above the target's top on arrival, cm (default 2)")
    parser.add_argument("--no-rays", action="store_true", help="start with the fingertip rays off (L toggles)")
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
    if not 0.0 <= args.clearance_cm <= 20.0:
        parser.error("--clearance-cm must be within 0-20")
    if args.smoke_target is not None and not 1 <= args.smoke_target <= 9:
        parser.error("--smoke-target must be 1-9")
    if not PLANNER_SPEED_RANGE_M_S[0] <= args.planner_speed <= PLANNER_SPEED_RANGE_M_S[1]:
        parser.error("--planner-speed must be within {:g}-{:g}".format(*PLANNER_SPEED_RANGE_M_S))
    try:
        args.smoke_key_list = parse_smoke_keys(args.smoke_keys)
    except ValueError as exc:
        parser.error(str(exc))
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
    smoke = args.headless_smoke
    return run_libero_planner_teleop(
        shadow_spec(args.hand_mount),
        config,
        task_id=args.task_id,
        task_order=args.task_order,
        init_state_id=args.init_state_id,
        seed=args.seed,
        headless=args.headless,
        duration_s=args.duration if args.duration is not None else (8.0 if smoke else None),
        camera_index=args.camera_index,
        camera_display=not args.no_camera_display,
        camera_mirror=not args.camera_no_mirror,
        hand_control="off" if smoke else args.hand_control,
        synthetic_input=smoke,
        suite=args.suite,
        arm_speed_scale=args.arm_speed,
        rotation_speed_scale=args.rot_speed,
        object_mass_scale=args.object_mass_scale,
        object_masses_kg=args.object_masses_kg,
        view_size=args.view_size,
        dex_grasp_terms=not args.dex_official,
        thumb_lift_gain=args.thumb_lift_gain,
        sim_speed=args.sim_speed,
        planner_backend=args.planner,
        planner_python=args.planner_python,
        clearance_m=args.clearance_cm / 100.0,
        rays_enabled=not args.no_rays,
        smoke_target=args.smoke_target,
        planner_speed_m_s=args.planner_speed,
        smoke_keys=args.smoke_key_list,
    )


if __name__ == "__main__":
    raise SystemExit(main())
