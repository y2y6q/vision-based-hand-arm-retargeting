"""LIBERO data collection on the planner entry: Panda + Shadow, a CSV row per attempt, a video after you press S.

Same controls as the planner entry (main command 2): SpaceMouse moves the
arm, the camera drives the Shadow fingers, number keys 1-9 send the hand
above a task object / fixture with the approach planner (moving the
SpaceMouse stops it and takes over; 0 cancels), five fingertip rays (L),
number badges in the views. The close-up is that entry's close-up turned
twice with V, fixed there and mirrored left-right.

Each task's own BDDL goal is checked live (every condition shown in the
window); when the official ``_check_success()`` has held for 10 steps
(0.5 s) the attempt is a success, the next random layout loads, and after
``--demos-per-task`` successes the next task. Running the same command again
continues where it stopped.

Output in outputs/libero_collect/<suite>/, nothing else:
  attempts.csv  one row per attempt you started: result (success / failed /
                quit), time, layout, hand mount, what ended it, video name
                (empty without a video);
  videos/       a video of each attempt where you pressed S, from that
                moment (from 1 s before the first move if you pressed before
                it) to the end: the window exactly as you see it (webcam
                inset included), the timer on top and SUCCESS / FAILED /
                QUIT at the end. S again cancels it (deleted). Without S
                nothing is recorded.
``--save-demos`` also writes the raw simulator demos (for training later)
to outputs/libero_collect_demos/<suite>/; ``--share-dir`` copies the CSV and
the videos to a synced cloud folder after the run.

Keys in the simulator window: 1-9 approach a target | 0 cancel | L rays |
M switch the hand mount (same layout) | S record this attempt from now |
R new layout | Q quit (R / M count as a failure once the arm moved).

Run only with ``.venv-libero``. See docs/libero_data_collection_zh.md.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import time


def reopen_stderr_after_sounddevice() -> bool:
    """Import sounddevice now; if that moved fd 2 under a console ``sys.stderr``, open ``sys.stderr`` again.

    mediapipe (the camera thread) imports sounddevice, which points fd 2 at
    NUL while PortAudio starts and then puts a copy of the old handle back.
    The console handle that Python 3.8's ``sys.stderr`` holds is closed by
    that, so every later ``print(..., file=sys.stderr)`` raised ``OSError:
    [WinError 6]`` while stdout kept working. On 2026-11-10 this ended two
    runs right after a success: attempts.csv was open in WPS, the row waited
    in memory as designed, and the warning about it raised. Done first, before
    robosuite and the rest bind loggers to ``sys.stderr``; the camera thread's
    import then finds sounddevice loaded. Returns whether it opened
    ``sys.stderr`` again (only on a Windows console, never under a pipe).
    """

    if sys.platform != "win32":
        return False
    import msvcrt

    try:
        before = msvcrt.get_osfhandle(2)
    except OSError:  # no fd 2 at all
        return False
    try:
        import sounddevice  # noqa: F401  (PortAudio starts here)
    except Exception:  # not installed / no PortAudio: mediapipe copes with that, and so does this
        pass
    try:
        moved = msvcrt.get_osfhandle(2) != before
    except OSError:
        return False
    raw = getattr(getattr(sys.stderr, "buffer", None), "raw", None)
    if not moved or type(raw).__name__ != "_WindowsConsoleIO":
        return False  # a pipe or file writes through fd 2 itself and keeps working
    old = sys.stderr
    sys.stderr = open(2, "w", encoding=old.encoding, errors=old.errors, buffering=1, closefd=False)
    return True


STDERR_REOPENED = reopen_stderr_after_sounddevice()  # before the imports below

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.libero_collection import (  # noqa: E402
    ATTEMPTS_CSV,
    COLLECT_ROOT,
    DEMOS_PER_TASK,
    AttemptTable,
    describe_predicate,
    parse_task_list,
    share_collection,
)
from wrist_teleop.libero_motion_planning import (  # noqa: E402
    PLANNER_BACKENDS,
    PLANNER_MAX_SPEED_M_S,
    PLANNER_SPEED_RANGE_M_S,
    PREGRASP_CLEARANCE_M,
)
from wrist_teleop.libero_planner_teleop import PLANNER_SUITES  # noqa: E402
from wrist_teleop.libero_spatial_teleop import (  # noqa: E402
    LIBERO_ARM_ROTATION_SPEED_SCALE,
    LIBERO_ARM_TRANSLATION_SPEED_SCALE,
    OBJECT_MASS_RANGE_G,
    OBJECT_MASS_SCALE_RANGE,
    parse_object_mass_overrides,
)
from wrist_teleop.libero_views import parse_view_size  # noqa: E402
from wrist_teleop.shadow_dex import SHADOW_THUMB_LIFT_GAIN_RANGE  # noqa: E402
from wrist_teleop.shadow_gripper import SHADOW_MOUNTS  # noqa: E402


def parse_smoke_keys(text: str) -> list[tuple[float, str]]:
    """``"1:1,4:0"`` -> ``[(1.0, "1"), (4.0, "0")]`` (as the planner entry's --smoke-keys)."""
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
    return sorted(keys)


def smoke_key_hook(keys: list[tuple[float, str]]):
    """A step hook that presses ``keys`` at their run times (seconds from the first step)."""
    pending = list(keys)
    started: list[float] = []

    def hook(_env, _info):
        now = time.perf_counter()
        if not started:
            started.append(now)
        if pending and now - started[0] >= pending[0][0]:
            return pending.pop(0)[1]
        return None

    return hook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "spacemouse_wrist.json")
    parser.add_argument("--suite", choices=PLANNER_SUITES, default="libero_object")
    parser.add_argument("--tasks", default="all", help="all (default), 3, 0-4 or 1,3,7-9")
    parser.add_argument("--demos-per-task", type=int, default=DEMOS_PER_TASK,
                        help="successes per task before the next task (default {})".format(DEMOS_PER_TASK))
    parser.add_argument("--list-tasks", action="store_true", help="print the tasks, their success rules and counts")
    parser.add_argument("--hand-mount", choices=SHADOW_MOUNTS, default="down",
                        help="start mount; down (default, as the planner entry): fingers point down; forward: palm "
                             "parallel to the table, fingers forward. M switches while running")
    parser.add_argument("--seed", type=int, default=7, help="base seed of the random layouts")
    parser.add_argument("--operator", default="", help="name written in every CSV row (optional)")
    parser.add_argument("--output-root", type=Path, default=COLLECT_ROOT)
    parser.add_argument("--save-demos", action="store_true",
                        help="also save the raw simulator demo of every success (states / actions, for training "
                             "later) to outputs/libero_collect_demos/<suite>/")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--headless-smoke", action="store_true",
                        help="with --headless: synthetic zero SpaceMouse, no camera, 8 s run; nothing is written "
                             "unless --smoke-keys moves the arm")
    parser.add_argument("--smoke-keys", default="", metavar="T:KEY,...",
                        help="press keys at run times in seconds, e.g. 1:1,4:0 (testing)")
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
    parser.add_argument("--object-mass-scale", type=float, default=1.0,
                        help="multiply every task object's mass ({:g}-{:g}; 1 = official)".format(*OBJECT_MASS_SCALE_RANGE))
    parser.add_argument("--object-mass", nargs="+", action="extend", default=[], metavar="NAME=GRAMS",
                        help="set one object's mass in grams ({:g}-{:g})".format(*OBJECT_MASS_RANGE_G))
    parser.add_argument("--view-size", type=parse_view_size, default=None, metavar="WxH")
    parser.add_argument("--hand-delay-ms", type=float, default=None,
                        help="fixed camera-hand delay behind each tick, 0-300 ms (default: config)")
    planner = parser.add_argument_group("approach planner (as the planner entry)")
    planner.add_argument("--planner", choices=PLANNER_BACKENDS, default="ompl",
                         help="ompl: checked straight routes, then OMPL RRT-Connect; straight: lift / traverse / "
                              "descend, no collision check")
    planner.add_argument("--planner-speed", type=float, default=PLANNER_MAX_SPEED_M_S,
                         help="planner peak hand speed in m/s, simulated time ({:g}-{:g}, default {:g})".format(
                             *PLANNER_SPEED_RANGE_M_S, PLANNER_MAX_SPEED_M_S))
    planner.add_argument("--planner-python", default=None,
                         help="python.exe of the ompl_planner Conda env (default: found next to panda_retarget)")
    planner.add_argument("--clearance-cm", type=float, default=100.0 * PREGRASP_CLEARANCE_M,
                         help="lowest fingertip above the target's top on arrival, cm (default 2)")
    planner.add_argument("--no-rays", action="store_true", help="start with the fingertip rays off (L toggles)")
    share = parser.add_argument_group("sharing")
    share.add_argument("--share-dir", type=Path, default=os.environ.get("LIBERO_SHARE_DIR") or None,
                       help="synced cloud folder: attempts.csv and the videos are copied to <share-dir>/<suite>/ "
                            "after the run (default: env LIBERO_SHARE_DIR)")
    share.add_argument("--share-url", default=os.environ.get("LIBERO_SHARE_URL") or None,
                       help="that folder's share link, printed after the run (default: env LIBERO_SHARE_URL)")
    args = parser.parse_args()
    if args.demos_per_task < 1:
        parser.error("--demos-per-task must be at least 1")
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
    if not PLANNER_SPEED_RANGE_M_S[0] <= args.planner_speed <= PLANNER_SPEED_RANGE_M_S[1]:
        parser.error("--planner-speed must be within {:g}-{:g}".format(*PLANNER_SPEED_RANGE_M_S))
    try:
        args.smoke_key_list = parse_smoke_keys(args.smoke_keys)
        args.object_masses_kg = parse_object_mass_overrides(args.object_mass)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def list_tasks(args: argparse.Namespace, table: list[dict]) -> None:
    attempts = AttemptTable(Path(args.output_root) / args.suite / ATTEMPTS_CSV)
    print("{}: {} tasks, {} successes each (table: {})".format(args.suite, len(table), args.demos_per_task,
                                                              attempts.path))
    for row in table:
        rule = " AND ".join(describe_predicate(state) for state in row["goal"])
        print("{:>3}  {:>2}/{:<2}  rate {:<14} {}\n       success: {}".format(
            row["task_id"], attempts.success_count(row["task_id"]), args.demos_per_task,
            attempts.stats(row["task_id"]).describe(), row["language"], rule))


def main() -> int:
    args = parse_args()
    from wrist_teleop.libero_collect_teleop import run_libero_collect, suite_task_table

    table = suite_task_table(args.suite)
    try:
        tasks = parse_task_list(args.tasks, len(table))
    except ValueError as exc:
        print("--tasks: {}".format(exc), file=sys.stderr)
        return 2
    if args.list_tasks:
        list_tasks(args, table)
        return 0
    config = WristConfig.load(args.config)
    if args.hand_delay_ms is not None:
        config = replace(config, hand_target_delay_s=args.hand_delay_ms / 1e3)
    smoke = args.headless_smoke
    result = run_libero_collect(
        config,
        suite=args.suite,
        tasks=tasks,
        demos_per_task=args.demos_per_task,
        hand_mount=args.hand_mount,
        seed=args.seed,
        headless=args.headless,
        duration_s=args.duration if args.duration is not None else (8.0 if smoke else None),
        camera_index=args.camera_index,
        camera_display=not args.no_camera_display,
        camera_mirror=not args.camera_no_mirror,
        hand_control="off" if smoke else args.hand_control,
        synthetic_input=smoke,
        arm_speed_scale=args.arm_speed,
        rotation_speed_scale=args.rot_speed,
        object_mass_scale=args.object_mass_scale,
        object_masses_kg=args.object_masses_kg,
        view_size=args.view_size,
        dex_grasp_terms=not args.dex_official,
        thumb_lift_gain=args.thumb_lift_gain,
        planner_backend=args.planner,
        planner_python=args.planner_python,
        clearance_m=args.clearance_cm / 100.0,
        planner_speed_m_s=args.planner_speed,
        rays_enabled=not args.no_rays,
        output_root=args.output_root,
        operator=args.operator,
        save_demos=args.save_demos,
        step_hook=smoke_key_hook(args.smoke_key_list) if args.smoke_key_list else None,
    )
    if args.share_dir and not smoke and result.table_path.is_file():
        try:
            copied = share_collection(result.suite_root, args.share_dir, args.suite)
        except OSError as exc:
            print("could not copy to the share folder: {}".format(exc), file=sys.stderr)
        else:
            print("copied {} file{} to {} (the sync client uploads them)".format(
                len(copied), "" if len(copied) == 1 else "s", Path(args.share_dir) / args.suite))
            if args.share_url:
                print("share link: {}".format(args.share_url))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
