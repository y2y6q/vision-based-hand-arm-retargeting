"""Panda + official Shadow Hand teleoperation.

This fourth entry keeps the Panda + Allegro interaction structure: SpaceMouse
motion owns only six OSC wrist values; camera curls and button gestures own
only the Shadow hand's 20 official position actuators.  The hand is the local
MuJoCo Menagerie Shadow Hand E3M5 model (24 physical joints, 20 actuators),
not an Allegro substitute. Shadow dex-retargeting is deliberately deferred.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.device import MotionSample, SpaceMouseHid  # noqa: E402
from wrist_teleop.grasp_telemetry import GraspTelemetry  # noqa: E402
from wrist_teleop.hand_input import CameraHandWorker  # noqa: E402
from wrist_teleop.interactive_cube import InteractiveCubeScaler  # noqa: E402
from wrist_teleop.mapper import SixDofMapper  # noqa: E402
from wrist_teleop.runtime_controls import KeyboardCommandRouter, install_viewer_key_callback  # noqa: E402
from wrist_teleop.shadow_gripper import make_shadow_panda_env  # noqa: E402
from wrist_teleop.shadow_hand import audit_shadow_hand_support  # noqa: E402
from wrist_teleop.shadow_hand_input import ShadowHandGestureController, ShadowLegacyHandInput  # noqa: E402
from wrist_teleop.shadow_integrated import PandaShadowActionComposer  # noqa: E402
from wrist_teleop.visualization import LaserPointer, TriViewCompositor  # noqa: E402


_CUBE_FACTOR = 1.5


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


class _RunLog:
    def __init__(self, config: WristConfig) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = ROOT / config.output_dir / f"{stamp}_shadow_integrated"
        self.path.mkdir(parents=True, exist_ok=False)
        self._stream = (self.path / "telemetry.jsonl").open("w", encoding="utf-8")

    def metadata(self, contents: dict[str, Any]) -> None:
        (self.path / "metadata.json").write_text(json.dumps(contents, indent=2, default=_jsonable), encoding="utf-8")

    def event(self, contents: dict[str, Any]) -> None:
        self._stream.write(json.dumps(contents, default=_jsonable, separators=(",", ":")) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class _NullCamera:
    error: str | None = None
    running = False

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None


class _NeutralInput:
    """Synthetic exact-zero SpaceMouse used only for explicit smoke modes."""

    def open(self) -> dict[str, Any]:
        return {"product_string": "synthetic-neutral-smoke", "vendor_id": None, "product_id": None}

    def reset_input_state(self) -> None:
        return None

    def poll(self) -> MotionSample:
        zeros = np.zeros(6, dtype=np.float64)
        return MotionSample(zeros, (False,) * 8, True, True, time.monotonic(), buttons_valid=True,
                            calibrated_axes=zeros.copy(), zero_bias_axes=zeros.copy(), neutral_locked=True,
                            calibration_state="synthetic_neutral")

    def close(self) -> None:
        return None


def _laser_config(config: WristConfig):
    return replace(config.laser, start_site="palm_laser_site", direction_frame="world", world_direction=(0.0, 0.0, -1.0))


def _sleep(cycle_started: float, control_hz: float) -> None:
    remaining = 1.0 / float(control_hz) - (time.monotonic() - cycle_started)
    if remaining > 0.0:
        time.sleep(remaining)


def _reset(
    env: Any, *, mapper: SixDofMapper, device: Any, hand_input: ShadowLegacyHandInput,
    gestures: ShadowHandGestureController | None, buttons: Sequence[bool], keypress_callback: Any | None,
) -> tuple[PandaShadowActionComposer, ShadowHandGestureController, Any, np.ndarray]:
    """Complete one R/reset without reopening HID or the camera."""
    env.reset()
    if keypress_callback is not None:
        install_viewer_key_callback(env, keypress_callback)
    composer = PandaShadowActionComposer(env)
    composer.initialize_goal()
    mapper.reset()
    device.reset_input_state()
    hand_input.reset()
    if gestures is None:
        gestures = ShadowHandGestureController(
            config=mapper.config.hand_gestures,
            lower_limits=composer.hand_lower_limits,
            upper_limits=composer.hand_upper_limits,
        )
    required = max(gestures.config.left_button_index, gestures.config.right_button_index) + 1
    seeded_buttons = tuple(bool(value) for value in buttons)
    if len(seeded_buttons) < required:
        seeded_buttons = (False,) * required
    hand = gestures.reset(buttons=seeded_buttons)
    action = composer.compose(np.zeros(6), hand.joint_targets)
    env.step(action)
    return composer, gestures, hand, action


def _close(env: Any | None, composer: PandaShadowActionComposer | None, mapper: SixDofMapper, device: Any, camera: Any) -> np.ndarray | None:
    action = None
    try:
        if env is not None and composer is not None:
            action = composer.compose(np.zeros(6), None)
            env.step(action)
    finally:
        mapper.reset()
        device.close()
        camera.close()
        if env is not None:
            env.close()
    return action


def _write_audit(config: WristConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / config.output_dir / f"{stamp}_shadow_hand_audit"
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "shadow_hand_availability.json"
    path.write_text(json.dumps(audit_shadow_hand_support().to_dict(), indent=2), encoding="utf-8")
    return path


def run_shadow(
    config: WristConfig, *, headless: bool, duration_s: float | None, camera_index: int,
    camera_display: bool, hand_control: str, synthetic_input: bool, three_view: bool,
) -> int:
    """Run real Panda + Shadow physics; no Shadow dex path is selected here."""
    if hand_control not in {"legacy-curl", "off"}:
        raise ValueError("Shadow accepts only legacy-curl or off; Shadow dex is not connected")
    log = _RunLog(config)
    mapper = SixDofMapper(config)
    device: Any = _NeutralInput() if synthetic_input else SpaceMouseHid(config)
    keyboard = KeyboardCommandRouter()
    use_three_view = bool(three_view and config.tri_view.enabled)
    camera: Any = _NullCamera()
    env = composer = gestures = laser = tri_view = grasp = cube = None
    hand_input: ShadowLegacyHandInput | None = None
    last_buttons = (False,) * 8
    control_frame = mapper.describe_control_frame()
    resize_lock = threading.Lock()
    resize_source: str | None = None
    ui_lock = threading.Lock()
    ui_state: dict[str, Any] = {"hand_mode": "shadow-legacy-curl" if hand_control == "legacy-curl" else "shadow-safe-open", "hand_source": "safe_open/hold", "active_override": None}
    reason, exit_code = "completed", 0

    def route(key: int, source: str) -> None:
        nonlocal resize_source
        keyboard.feed(key, source=source)
        if isinstance(key, int) and 0 <= key <= 255 and chr(key).lower() == "b":
            with resize_lock:
                if resize_source is None:
                    resize_source = source

    def take_resize() -> str | None:
        nonlocal resize_source
        with resize_lock:
            source, resize_source = resize_source, None
            return source

    def camera_status() -> dict[str, Any]:
        with ui_lock:
            return dict(ui_state)

    def set_hand_status(command: Any) -> None:
        with ui_lock:
            ui_state["hand_source"] = command.source
            ui_state["active_override"] = command.active_override

    def create_tri_view() -> tuple[LaserPointer | None, TriViewCompositor | None, dict[str, Any]]:
        if not use_three_view:
            return None, None, {"enabled": False, "reason": "three_view_disabled"}
        assert env is not None
        next_laser = LaserPointer(env, _laser_config(config))
        next_tri = TriViewCompositor(
            config.tri_view, show_window=True, key_handler=lambda key: route(key, "tri_view"),
            on_close=lambda: keyboard.feed("q", source="tri_view_window"), laser=next_laser,
        )
        next_tri.configure_cameras(env)
        nonlocal_control = mapper.set_camera_from_sim(env, config.tri_view.front_camera_name)
        return next_laser, next_tri, {"laser": next_laser.update(), "control_frame": nonlocal_control}

    try:
        env = make_shadow_panda_env(config, has_renderer=not headless and not use_three_view,
                                    has_offscreen_renderer=bool(use_three_view and not headless))
        device_info = device.open()
        # Read limits from the actual loaded 20-actuator model, then let R reset
        # rebuild it once into the normal no-startup-jump OSC state.
        env.reset()
        preview = PandaShadowActionComposer(env)
        hand_input = ShadowLegacyHandInput(preview.hand_lower_limits, preview.hand_upper_limits)
        if hand_control == "legacy-curl":
            camera = CameraHandWorker(hand_input, camera_index=camera_index, display=camera_display,
                                      keypress_callback=lambda key: route(key, "camera"), status_provider=camera_status)
        composer, gestures, reset_hand, reset_action = _reset(
            env, mapper=mapper, device=device, hand_input=hand_input, gestures=None, buttons=last_buttons,
            keypress_callback=None if headless or use_three_view else lambda key: route(key, "mujoco"),
        )
        cube = InteractiveCubeScaler(env, factor=_CUBE_FACTOR)
        laser, tri_view, visual_initial = create_tri_view()
        if visual_initial.get("control_frame") is not None:
            control_frame = visual_initial["control_frame"]
        grasp = GraspTelemetry(env, config.grasp_telemetry)
        if hand_control == "legacy-curl":
            camera.start()
        set_hand_status(reset_hand)
        log.metadata({
            "mode": "panda_shadow_integrated", "headless": headless, "synthetic_input": synthetic_input,
            "three_view": use_three_view, "camera_index": camera_index, "camera_display": camera_display,
            "config": config.to_dict(), "device": device_info, "shadow_hand": composer.describe(),
            "hand_mode": hand_control, "shadow_dex": {"integrated": False, "reason": "deferred by design"},
            "camera_hand_input": {"algorithm": "MediaPipe landmarks + Shadow legacy curl" if hand_control == "legacy-curl" else "safe-open/off", "actuator_count": 20},
            "button_gestures": {"config": config.hand_gestures.to_dict(), "targets": gestures.gesture_targets},
            "visualization": {"laser_config": _laser_config(config).__dict__, "initial": visual_initial},
            "control_frame": control_frame, "interactive_cube": cube.status(),
        })
        print("Panda + Shadow Hand ready: 6 OSC wrist dimensions + 20 official Shadow actuator dimensions.")
        print("hand mode=legacy-curl" if hand_control == "legacy-curl" else "hand mode=off (safe-open)")
        print("Shadow dex-retargeting is intentionally not enabled. B=larger cube; R=reset; Q/Ctrl+C=exit.")
        log.event({"event": "environment_reset", "reason": "startup", "timestamp_monotonic_s": time.monotonic(),
                   "action": reset_action, "hand_source": reset_hand.source, "cube": cube.status()})
        started = previous = time.monotonic()
        screenshot: Path | None = None
        while duration_s is None or time.monotonic() - started < duration_s:
            if keyboard.stop_event.is_set():
                reason = keyboard.quit_reason or "Q"
                break
            if keyboard.take_reset():
                assert hand_input is not None
                prior_cube = cube.restore_default() if cube is not None else {"available": False}
                with resize_lock:
                    resize_source = None
                if tri_view is not None:
                    tri_view.close()
                if laser is not None:
                    laser.close()
                composer, gestures, reset_hand, reset_action = _reset(
                    env, mapper=mapper, device=device, hand_input=hand_input, gestures=gestures, buttons=last_buttons,
                    keypress_callback=None if headless or use_three_view else lambda key: route(key, "mujoco"),
                )
                cube = InteractiveCubeScaler(env, factor=_CUBE_FACTOR)
                laser, tri_view, visual_reset = create_tri_view()
                if visual_reset.get("control_frame") is not None:
                    control_frame = visual_reset["control_frame"]
                grasp = GraspTelemetry(env, config.grasp_telemetry)
                set_hand_status(reset_hand)
                log.event({"event": "environment_reset", "reason": "R", "timestamp_monotonic_s": time.monotonic(),
                           "action": reset_action, "hand_source": reset_hand.source, "cube_before_reset": prior_cube,
                           "cube": cube.status(), "visualization": visual_reset})
                print("environment_reset")
                previous = time.monotonic()
                continue
            source = take_resize()
            if source is not None and cube is not None:
                resized = cube.enlarge()
                if resized.get("changed") or resized.get("reason") != "already_enlarged":
                    log.event({"event": "cube_size_changed", "timestamp_monotonic_s": time.monotonic(), "key": "B", "source": source, "cube": resized})
            cycle_started = time.monotonic()
            sample = device.poll()
            last_buttons = sample.buttons if sample.buttons_valid else (False,) * len(sample.buttons)
            wrist = mapper.map(sample.control_axes, valid=sample.valid and sample.armed)
            assert hand_input is not None and gestures is not None and composer is not None
            camera_hand = hand_input.latest(cycle_started)
            hand = gestures.update(last_buttons, camera_hand.joint_targets, camera_valid=camera_hand.valid, timestamp_s=cycle_started)
            set_hand_status(hand)
            action = composer.compose(wrist, hand.joint_targets)
            _obs, reward, done, _info = env.step(action)
            grasp_record = grasp.update() if grasp is not None else {"available": False}
            laser_record = laser.update() if laser is not None else {"enabled": False}
            tri_record = tri_view.update(
                env, status_text=(f"Panda + Shadow | hand={hand.source} | wrist={np.linalg.norm(wrist):.4f} | "
                                  f"cube={'1.5x' if cube is not None and cube.enlarged else 'default'} | B=enlarge"),
            ) if tri_view is not None else {"enabled": False}
            if tri_view is not None and screenshot is None and tri_record.get("rendered"):
                screenshot = log.path / "tri_view_shadow_initial.png"
                tri_view.save_last_frame(screenshot)
                tri_record["screenshot"] = str(screenshot)
            if tri_record.get("closed"):
                keyboard.feed("q", source="tri_view")
            elapsed = max(cycle_started - previous, 1e-9)
            previous = cycle_started
            log.event({
                "event": "step", "timestamp_monotonic_s": cycle_started, "raw_axes": sample.raw_axes,
                "calibrated_axes": sample.control_axes, "mapped_world_delta": wrist, "mapper": mapper.diagnostics(),
                "buttons": sample.buttons, "buttons_valid": sample.buttons_valid, "input_valid": sample.valid,
                "input_armed": sample.armed, "input_error": sample.error, "update_hz": 1.0 / elapsed,
                "action": action, "arm_action": action[composer.arm_slice], "hand_action": action[composer.hand_slice],
                "hand_targets": composer.held_hand_targets, "hand_qpos": composer.hand_qpos(), "ee_pose": composer.ee_pose(),
                "hand_source": hand.source, "hand_override": hand.active_override,
                "hand_transition": {"active": hand.transition_active, "progress": hand.transition_progress},
                "button_rising_edges": hand.button_rising_edges, "button_event": hand.event,
                "camera": {"valid": camera_hand.valid, "tracking_state": camera_hand.tracking_state,
                           "curls": camera_hand.curls, "error": camera_hand.error or camera.error,
                           "worker_running": camera.running, "worker_status": camera.status() if hasattr(camera, "status") else None},
                "visualization": {"laser": laser_record, "tri_view": tri_record}, "grasp": grasp_record,
                "reward": reward, "done": done,
            })
            _sleep(cycle_started, config.control_hz)
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    except Exception as exc:
        reason, exit_code = f"error: {type(exc).__name__}: {exc}", 1
        log.event({"event": "exception", "reason": reason, "timestamp_monotonic_s": time.monotonic()})
        print(reason, file=sys.stderr)
    finally:
        final_action = None
        try:
            if grasp is not None:
                log.event({"event": "grasp_summary", "timestamp_monotonic_s": time.monotonic(), "grasp": grasp.summary()})
            if tri_view is not None:
                tri_view.close()
            if laser is not None:
                laser.close()
            final_action = _close(env, composer, mapper, device, camera)
        except Exception as exc:
            cleanup = f"cleanup error: {type(exc).__name__}: {exc}"
            log.event({"event": "cleanup_exception", "reason": cleanup, "timestamp_monotonic_s": time.monotonic()})
            print(cleanup, file=sys.stderr)
            if exit_code == 0:
                reason, exit_code = cleanup, 1
        log.event({"event": "exit", "reason": reason, "timestamp_monotonic_s": time.monotonic(), "last_zero_wrist_action": final_action})
        log.close()
        print(f"Shadow teleoperation stopped: {reason}. Log: {log.path}")
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "spacemouse_wrist.json")
    parser.add_argument("--audit", action="store_true", help="write the local model/action audit and exit")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--no-camera-display", action="store_true")
    parser.add_argument("--hand-control", choices=("legacy-curl", "off"), default="legacy-curl")
    parser.add_argument("--no-three-view", action="store_true")
    parser.add_argument("--viewer-smoke", action="store_true")
    parser.add_argument("--headless-smoke", action="store_true")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.audit and (args.headless or args.duration is not None or args.viewer_smoke or args.headless_smoke):
        parser.error("--audit does not start simulation; do not combine it with run options")
    if args.viewer_smoke and args.headless:
        parser.error("--viewer-smoke requires a visible tri-view window")
    if args.headless_smoke and not args.headless:
        parser.error("--headless-smoke requires --headless")
    if args.viewer_smoke and args.headless_smoke:
        parser.error("choose either --viewer-smoke or --headless-smoke")
    if args.viewer_smoke or args.headless_smoke:
        args.hand_control = "off"
        if args.duration is None:
            args.duration = 2.0
    return args


def main() -> int:
    args = parse_args()
    config = WristConfig.load(args.config)
    if args.audit:
        path = _write_audit(config)
        audit = audit_shadow_hand_support()
        print(f"Shadow Hand availability audit: {path}")
        print("Shadow Hand audit completed. No simulation, HID, or camera was started.")
        return 0 if audit.ready else 2
    return run_shadow(
        config, headless=args.headless, duration_s=args.duration, camera_index=args.camera_index,
        camera_display=not args.headless and not args.no_camera_display, hand_control=args.hand_control,
        synthetic_input=args.viewer_smoke or args.headless_smoke,
        three_view=not args.headless and not args.no_three_view,
    )


if __name__ == "__main__":
    raise SystemExit(main())
