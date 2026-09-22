"""The single robosuite SpaceMouse wrist teleoperation entry point.

``diagnostic`` uses only HID acquisition; ``teleop`` runs the verified Panda
OSC_POSE wrist baseline; ``integrated`` combines the wrist source with one of
three explicit hand modes: official dex-retargeting, the retained legacy curl
mapper, or safe-open/off. SpaceMouse motion data only writes the Panda's six
OSC entries, and camera / button gesture targets only write the Allegro's
sixteen position entries.
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
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop import (  # noqa: E402
    CameraHandWorker,
    DexHandInputAdapter,
    MotionSample,
    OscPoseComposer,
    PandaAllegroActionComposer,
    SixDofMapper,
    SpaceMouseHid,
    HandInputAdapter,
    WristConfig,
    create_panda_osc_pose_env,
)
from wrist_teleop.allegro_model import make_allegro_panda_env  # noqa: E402
from wrist_teleop.allegro_model import ALLEGRO_JOINT_NAMES, source_joint_specs  # noqa: E402
from wrist_teleop.dex_retargeting import DexRetargetingAdapter, DexRetargetingUnavailable  # noqa: E402
from wrist_teleop.hand_gestures import AllegroHandGestureController  # noqa: E402
from wrist_teleop.integrated_runtime import (  # noqa: E402
    close_integrated_resources,
    reset_integrated_environment,
)
from wrist_teleop.runtime_controls import KeyboardCommandRouter  # noqa: E402
from wrist_teleop.visualization import LaserPointer, TriViewCompositor  # noqa: E402
from wrist_teleop.grasp_telemetry import GraspTelemetry  # noqa: E402
from wrist_teleop.interactive_cube import InteractiveCubeScaler  # noqa: E402


_INTERACTIVE_CUBE_ENLARGE_FACTOR = 1.5
_PANDA_ALLEGRO_PALM_LASER_SITE = "palm_laser_site"


def _panda_allegro_palm_laser_config(config: WristConfig):
    """Return this entry's palm-origin, world-vertical laser settings.

    Other teleoperation entries retain their own site-local laser configuration.
    """
    return replace(
        config.laser,
        start_site=_PANDA_ALLEGRO_PALM_LASER_SITE,
        direction_frame="world",
        world_direction=(0.0, 0.0, -1.0),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


class RunLog:
    def __init__(self, config: WristConfig, mode: str):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = ROOT / config.output_dir / f"{stamp}_{mode}"
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
    """Cleanup-compatible camera stand-in used by hand-control=off smoke runs."""

    error: str | None = None
    running = False

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None


class _NeutralInputDevice:
    """A declared smoke-test device; normal interactive runs always use HID."""

    def __init__(self) -> None:
        self._opened = False

    def open(self) -> dict[str, Any]:
        self._opened = True
        return {"product_string": "synthetic-neutral-smoke", "vendor_id": None, "product_id": None}

    def reset_input_state(self) -> None:
        return None

    def poll(self) -> MotionSample:
        return MotionSample(
            raw_axes=np.zeros(6, dtype=np.float64),
            buttons=(False,) * 8,
            valid=True,
            armed=True,
            timestamp_s=time.monotonic(),
            buttons_valid=True,
        )

    def close(self) -> None:
        self._opened = False


def _make_hand_input(config: WristConfig, hand_control: str):
    """Create a hand source without allowing dex to silently fall back."""

    if hand_control == "legacy-curl" or hand_control == "off":
        return HandInputAdapter(), None
    if hand_control != "dex":
        raise ValueError(f"unsupported hand control mode: {hand_control}")
    specs = source_joint_specs()
    retargeter = DexRetargetingAdapter(
        allegro_joint_names=ALLEGRO_JOINT_NAMES,
        lower_limits=[spec.lower for spec in specs],
        upper_limits=[spec.upper for spec in specs],
        runtime=config.dex_hand.runtime,
        sidecar_python=config.dex_hand.sidecar_python,
    )
    return DexHandInputAdapter(
        retargeter,
        lower_limits=[spec.lower for spec in specs],
        upper_limits=[spec.upper for spec in specs],
        joint_names=ALLEGRO_JOINT_NAMES,
        stale_timeout_s=config.dex_hand.stale_timeout_s,
        expected_handedness=config.dex_hand.expected_handedness,
        input_is_mirrored=config.dex_hand.input_is_mirrored,
    ), retargeter


def _status(sample, command: np.ndarray, rate_hz: float, mapper: SixDofMapper) -> dict[str, Any]:
    return {
        "raw_axes": sample.raw_axes,
        "calibrated_axes": sample.control_axes,
        "zero_bias_axes": sample.zero_bias_axes,
        "neutral_locked": sample.neutral_locked,
        "input_calibration_state": sample.calibration_state,
        "mapped_world_delta": command,
        "mapper": mapper.diagnostics(),
        "buttons": sample.buttons,
        "buttons_valid": sample.buttons_valid,
        "valid": sample.valid,
        "armed": sample.armed,
        "update_hz": rate_hz,
        "error": sample.error,
    }


def run_diagnostic(config: WristConfig, duration_s: float | None) -> int:
    """Show HID motion / mapping / buttons only; never starts robosuite."""
    log = RunLog(config, "diagnostic")
    mapper = SixDofMapper(config)
    device = SpaceMouseHid(config)
    reason = "completed"
    try:
        info = device.open()
        log.metadata({"mode": "diagnostic", "config": config.to_dict(), "device": info})
        print(f"SpaceMouse connected: {info.get('product_string')} ({info.get('vendor_id'):04x}:{info.get('product_id'):04x})")
        print("Diagnostic is wrist-only. Ctrl+C stops it and records raw button-bit states for integrated setup.")
        start = previous = time.monotonic()
        shown_at = start
        cycles = 0
        while duration_s is None or time.monotonic() - start < duration_s:
            cycle_start = time.monotonic()
            sample = device.poll()
            command = mapper.map(sample.control_axes, valid=sample.valid and sample.armed)
            cycles += 1
            elapsed = max(cycle_start - previous, 1e-9)
            previous = cycle_start
            status = _status(sample, command, 1.0 / elapsed, mapper)
            status.update(timestamp_monotonic_s=cycle_start, event="sample")
            log.event(status)
            if cycle_start - shown_at >= 0.2:
                print(json.dumps(status, default=_jsonable))
                shown_at = cycle_start
            _sleep_to_rate(cycle_start, config.control_hz)
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    except Exception as exc:
        reason = f"error: {type(exc).__name__}: {exc}"
        log.event({"event": "exception", "reason": reason, "timestamp_monotonic_s": time.monotonic()})
        print(reason, file=sys.stderr)
        return 1
    finally:
        mapper.reset()
        device.close()
        log.event({"event": "exit", "reason": reason, "timestamp_monotonic_s": time.monotonic()})
        log.close()
        print(f"Diagnostic stopped: {reason}. Log: {log.path}")
    return 0


def _sleep_to_rate(cycle_start: float, control_hz: float) -> None:
    remaining = (1.0 / control_hz) - (time.monotonic() - cycle_start)
    if remaining > 0:
        time.sleep(remaining)


def run_teleop(config: WristConfig, headless: bool, duration_s: float | None) -> int:
    """Run SpaceMouse -> camera-relative mapper -> world-frame OSC_POSE -> Panda EE."""
    log = RunLog(config, "teleop")
    mapper = SixDofMapper(config)
    device = SpaceMouseHid(config)
    env = None
    reason = "completed"
    try:
        env = create_panda_osc_pose_env(config, has_renderer=not headless)
        env.reset()
        composer = OscPoseComposer(env)
        composer.initialize_goal()
        info = device.open()
        log.metadata({
            "mode": "teleop", "headless": headless, "config": config.to_dict(),
            "device": info, "osc_pose": composer.describe(),
        })
        print("Panda OSC_POSE ready: camera-relative six-axis input, world-frame wrist deltas only.")
        print("SpaceMouse buttons are logged only in wrist-only mode. Ctrl+C exits.")
        start = previous = time.monotonic()
        while duration_s is None or time.monotonic() - start < duration_s:
            cycle_start = time.monotonic()
            sample = device.poll()
            command = mapper.map(sample.control_axes, valid=sample.valid and sample.armed)
            action = composer.compose(command)
            observation, reward, done, info = env.step(action)
            elapsed = max(cycle_start - previous, 1e-9)
            previous = cycle_start
            event = _status(sample, command, 1.0 / elapsed, mapper)
            event.update(
                event="step", timestamp_monotonic_s=cycle_start, action=action,
                ee_pose=composer.ee_pose(), reward=reward, done=done,
                loop_seconds=time.monotonic() - cycle_start,
            )
            log.event(event)
            _sleep_to_rate(cycle_start, config.control_hz)
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    except Exception as exc:
        reason = f"error: {type(exc).__name__}: {exc}"
        log.event({"event": "exception", "reason": reason, "timestamp_monotonic_s": time.monotonic()})
        print(reason, file=sys.stderr)
        return 1
    finally:
        mapper.reset()
        device.close()
        if env is not None:
            env.close()
        log.event({"event": "exit", "reason": reason, "timestamp_monotonic_s": time.monotonic()})
        log.close()
        print(f"Teleoperation stopped: {reason}. Log: {log.path}")
    return 0


def run_integrated(
    config: WristConfig,
    headless: bool,
    duration_s: float | None,
    camera_index: int,
    camera_display: bool,
    *,
    hand_control: str = "dex",
    synthetic_input: bool = False,
    three_view: bool = False,
) -> int:
    """Run independent SpaceMouse wrist and camera-Allegro hand sources.

    The MediaPipe worker publishes only a latest 16-joint target and may fail
    safely (for example, when a headless smoke host has no camera). The main
    simulator loop never waits for a camera frame. SpaceMouse motion remains
    wrist-only; its two configured button rising edges select bounded Allegro
    gesture overrides without altering the six-dimensional OSC command.
    """
    log = RunLog(config, "integrated")
    mapper = SixDofMapper(config)
    device: Any = _NeutralInputDevice() if synthetic_input else SpaceMouseHid(config)
    hand_input: Any = None
    retargeter: DexRetargetingAdapter | None = None
    keyboard = KeyboardCommandRouter()
    use_three_view = bool(three_view and config.tri_view.enabled)
    cube_resize_lock = threading.Lock()
    cube_resize_source: str | None = None
    # The camera worker only reads this snapshot to annotate the already-open
    # capture.  It never owns or writes simulator / hand actions.
    camera_ui_lock = threading.Lock()
    camera_ui_state: dict[str, Any] = {
        "hand_mode": hand_control,
        "hand_source": "safe_open/hold",
        "active_override": None,
    }

    def camera_ui_status() -> dict[str, Any]:
        with camera_ui_lock:
            return dict(camera_ui_state)

    def update_camera_ui_status(*, source: str, active_override: str | None) -> None:
        with camera_ui_lock:
            camera_ui_state["hand_source"] = str(source)
            camera_ui_state["active_override"] = active_override

    def route_key(key: int, *, source: str) -> None:
        """Keep R/Q semantics and queue B without blocking a GUI callback."""
        nonlocal cube_resize_source
        keyboard.feed(key, source=source)
        normalized = chr(key).lower() if isinstance(key, int) and 0 <= key <= 255 else ""
        if normalized == "b":
            with cube_resize_lock:
                # HighGUI can repeat a held key. One pending request is enough;
                # the model-side operation is idempotent after enlargement.
                if cube_resize_source is None:
                    cube_resize_source = source

    def take_cube_resize_source() -> str | None:
        nonlocal cube_resize_source
        with cube_resize_lock:
            source, cube_resize_source = cube_resize_source, None
            return source

    def clear_cube_resize_request() -> None:
        nonlocal cube_resize_source
        with cube_resize_lock:
            cube_resize_source = None

    def route_camera_key(key: int) -> None:
        route_key(key, source="camera")

    def route_mujoco_key(key: int) -> None:
        route_key(key, source="mujoco")

    def route_tri_view_key(key: int) -> None:
        route_key(key, source="tri_view")

    camera: Any = _NullCamera()
    env = None
    composer: PandaAllegroActionComposer | None = None
    gestures: AllegroHandGestureController | None = None
    laser: LaserPointer | None = None
    tri_view: TriViewCompositor | None = None
    tri_view_snapshot: Path | None = None
    grasp: GraspTelemetry | None = None
    cube_scaler: InteractiveCubeScaler | None = None
    laser_config = _panda_allegro_palm_laser_config(config)
    reason = "completed"
    exit_code = 0
    last_buttons = (False,) * 8
    control_frame: dict[str, Any] = mapper.describe_control_frame()
    try:
        log.event({
            "event": "hand_mode_requested",
            "mode": hand_control,
            "timestamp_monotonic_s": time.monotonic(),
            "legacy_curl_fallback_allowed": False,
        })
        hand_input, retargeter = _make_hand_input(config, hand_control)
        if retargeter is not None:
            # Record the native solver identity before HID / camera startup so
            # a later hardware error cannot hide what dex runtime was used.
            dex_metadata = retargeter.metadata()
            log.event({"event": "dex_runtime_ready", "timestamp_monotonic_s": time.monotonic(), "dex": dex_metadata})
            print("hand mode=dex")
            print(
                "dex-retargeting=" + dex_metadata["version"]
                + " | runtime=" + str(dex_metadata["runtime"])
                + " | config=" + dex_metadata["config_path"]
                + " | sha256=" + dex_metadata["config_sha256"]
            )
            print(
                "dex human_indices=" + json.dumps(dex_metadata["human_indices"])
                + " | output joints=" + json.dumps(dex_metadata["retargeting_joint_names"])
            )
        if hand_control != "off":
            camera = CameraHandWorker(
                hand_input,
                camera_index=camera_index,
                display=camera_display,
                keypress_callback=route_camera_key,
                status_provider=camera_ui_status,
            )
        env = make_allegro_panda_env(
            config,
            # The compositor owns the only visible window in three-view mode.
            # Keeping robosuite's native viewer disabled prevents two GUI event
            # loops from competing for R/Q and for renderer ownership.
            has_renderer=not headless and not use_three_view,
            has_offscreen_renderer=bool(use_three_view and not headless),
        )
        info = device.open()
        composer, gestures, reset_hand, reset_action = reset_integrated_environment(
            env,
            mapper=mapper,
            device=device,
            hand_input=hand_input,
            gestures=None,
            gesture_config=config.hand_gestures,
            buttons=last_buttons,
            keypress_callback=None if headless or use_three_view else route_mujoco_key,
        )
        update_camera_ui_status(
            source=reset_hand.source,
            active_override=reset_hand.active_override,
        )
        cube_scaler = InteractiveCubeScaler(env, factor=_INTERACTIVE_CUBE_ENLARGE_FACTOR)
        if use_three_view:
            laser = LaserPointer(env, laser_config)
            tri_view = TriViewCompositor(
                config.tri_view,
                show_window=True,
                key_handler=route_tri_view_key,
                on_close=lambda: keyboard.feed("q", source="tri_view_window"),
                laser=laser,
            )
            tri_view.configure_cameras(env)
            control_frame = mapper.set_camera_from_sim(env, config.tri_view.front_camera_name)
            initial_laser = laser.update()
        else:
            initial_laser = {"enabled": False, "reason": "three_view_disabled"}
        grasp = GraspTelemetry(env, config.grasp_telemetry)
        if hand_control != "off":
            camera.start()
        if retargeter is None:
            if hand_control == "legacy-curl":
                print("hand mode=legacy-curl (selected explicitly; dex is not active)")
            else:
                print("hand mode=off (safe-open; camera retargeting is disabled)")
        log.metadata({
            "mode": "integrated",
            "headless": headless,
            "synthetic_input": synthetic_input,
            "three_view": use_three_view,
            "camera_index": camera_index,
            "camera_display": camera_display,
            "config": config.to_dict(),
            "device": info,
            "osc_pose_and_allegro": composer.describe(),
            "control_frame": control_frame,
            "camera_hand_input": {
                "algorithm": (
                    "official dex-retargeting vector mapping" if hand_control == "dex"
                    else "MediaPipe landmarks + legacy rule-based finger curl mapping" if hand_control == "legacy-curl"
                    else "safe-open/off"
                ),
                "mode": hand_control,
                "smoothing_alpha": hand_input.smoothing_alpha,
                "stale_timeout_s": hand_input.stale_timeout_s,
                "calibration_state": hand_input.calibration_state,
                "dex": None if retargeter is None else retargeter.metadata(),
            },
            "button_gestures": {
                "config": config.hand_gestures.to_dict(),
                "targets_rad": gestures.gesture_targets,
                "button_mapping_status": "configured from HID bit indices; physical left/right requires manual confirmation",
            },
            "keyboard": {
                "reset": "R/r queues one complete environment reset",
                "quit": "Q/q queues safe shutdown with a final zero wrist delta",
                "cube_enlarge": "B/b enlarges the interactive Lift cube once to 1.5x; R restores its default size",
                "sources": (
                    (["synchronized tri-view OpenCV window"] if use_three_view else ["MuJoCo viewer"])
                    + (["Panda + Allegro camera OpenCV window"] if camera_display and hand_control != "off" else [])
                ),
            },
            "visualization": {
                "three_view": use_three_view,
                "tri_view_config": config.tri_view.__dict__,
                "laser_config": laser_config.__dict__,
                "initial_laser": initial_laser,
            },
            "interactive_cube": {
                "key": "B/b",
                "restore_key": "R/r",
                "state": cube_scaler.status(),
            },
            "grasp_telemetry": {
                "config": config.grasp_telemetry.to_dict(),
                "initial": grasp.update(),
            },
        })
        print(
            "Panda + Allegro ready: SpaceMouse motion uses the primary-camera frame and writes only wrist deltas; "
            f"hand mode is {hand_control}."
        )
        print(
            "B enlarges the interactive Lift cube once to 1.5x; R restores its default size and resets the environment; "
            "Q and Ctrl+C safely exit. Configured SpaceMouse buttons toggle Allegro gestures."
        )
        log.event({
            "event": "environment_reset",
            "reason": "startup",
            "timestamp_monotonic_s": time.monotonic(),
            "action": reset_action,
            "hand_source": reset_hand.source,
            "active_override": reset_hand.active_override,
            "cube": cube_scaler.status(),
        })
        start = previous = time.monotonic()
        while duration_s is None or time.monotonic() - start < duration_s:
            if keyboard.stop_event.is_set():
                reason = keyboard.quit_reason or "Q"
                break
            if keyboard.take_reset():
                cube_before_reset = (
                    cube_scaler.restore_default()
                    if cube_scaler is not None
                    else {"available": False, "changed": False, "reason": "not_initialized"}
                )
                clear_cube_resize_request()
                # ``env.reset()`` reconstructs MuJoCo objects. Close first so
                # the previous scene's visual site is restored, then bind the
                # laser / compositor to the fresh scene below.
                if tri_view is not None:
                    tri_view.close()
                    tri_view = None
                if laser is not None:
                    laser.close()
                    laser = None
                composer, gestures, reset_hand, reset_action = reset_integrated_environment(
                    env,
                    mapper=mapper,
                    device=device,
                    hand_input=hand_input,
                    gestures=gestures,
                    buttons=last_buttons,
                    keypress_callback=None if headless or use_three_view else route_mujoco_key,
                )
                if use_three_view:
                    laser = LaserPointer(env, laser_config)
                    tri_view = TriViewCompositor(
                        config.tri_view,
                        show_window=True,
                        key_handler=route_tri_view_key,
                        on_close=lambda: keyboard.feed("q", source="tri_view_window"),
                        laser=laser,
                    )
                    tri_view.configure_cameras(env)
                    control_frame = mapper.set_camera_from_sim(env, config.tri_view.front_camera_name)
                    reset_laser = laser.update()
                else:
                    reset_laser = {"enabled": False, "reason": "three_view_disabled"}
                grasp = GraspTelemetry(env, config.grasp_telemetry)
                cube_scaler = InteractiveCubeScaler(env, factor=_INTERACTIVE_CUBE_ENLARGE_FACTOR)
                log.event({
                    "event": "environment_reset",
                    "reason": "R",
                    "timestamp_monotonic_s": time.monotonic(),
                    "action": reset_action,
                    "hand_source": reset_hand.source,
                    "active_override": reset_hand.active_override,
                    "laser": reset_laser,
                    "control_frame": control_frame,
                    "grasp": grasp.update(),
                    "cube_before_reset": cube_before_reset,
                    "cube": cube_scaler.status(),
                })
                update_camera_ui_status(
                    source=reset_hand.source,
                    active_override=reset_hand.active_override,
                )
                print("environment_reset")
                previous = time.monotonic()
                continue
            cube_resize_request = take_cube_resize_source()
            if cube_resize_request is not None:
                cube_event = (
                    cube_scaler.enlarge()
                    if cube_scaler is not None
                    else {"available": False, "changed": False, "reason": "not_initialized"}
                )
                if cube_event.get("changed", False) or cube_event.get("reason") != "already_enlarged":
                    log.event({
                        "event": "cube_size_changed",
                        "timestamp_monotonic_s": time.monotonic(),
                        "key": "B",
                        "source": cube_resize_request,
                        "cube": cube_event,
                    })
                    print(
                        "cube_size_changed: "
                        + (
                            f"factor={cube_event.get('current_factor')}"
                            if cube_event.get("changed", False)
                            else str(cube_event.get("reason"))
                        )
                    )
            cycle_start = time.monotonic()
            sample = device.poll()
            last_buttons = sample.buttons if sample.buttons_valid else (False,) * len(sample.buttons)
            wrist_command = mapper.map(sample.control_axes, valid=sample.valid and sample.armed)
            hand_command = hand_input.latest(cycle_start)
            gesture_command = gestures.update(
                last_buttons,
                hand_command.joint_targets,
                camera_valid=hand_command.valid,
                timestamp_s=cycle_start,
            )
            update_camera_ui_status(
                source=gesture_command.source,
                active_override=gesture_command.active_override,
            )
            action = composer.compose(wrist_command, gesture_command.joint_targets)
            observation, reward, done, info = env.step(action)
            grasp_event = grasp.update() if grasp is not None else {"available": False, "reason": "not_initialized"}
            laser_event = laser.update() if laser is not None else {"enabled": False}
            tri_view_event = (
                tri_view.update(
                    env,
                    status_text=(
                        f"Panda + Allegro | hand={gesture_command.source} | "
                        f"wrist={np.linalg.norm(wrist_command):.4f} | "
                        f"cube={'1.5x' if cube_scaler is not None and cube_scaler.enlarged else 'default'} | B=enlarge"
                    ),
                )
                if tri_view is not None
                else {"enabled": False}
            )
            if (
                tri_view is not None
                and tri_view_snapshot is None
                and bool(tri_view_event.get("rendered", False))
            ):
                candidate = log.path / "tri_view_laser_initial.png"
                if tri_view.save_last_frame(candidate):
                    tri_view_snapshot = candidate
                    tri_view_event["screenshot"] = str(candidate)
                else:
                    tri_view_event["screenshot_error"] = "could_not_write_tri_view_frame"
            if bool(tri_view_event.get("closed", False)):
                keyboard.feed("q", source="tri_view")
            elapsed = max(cycle_start - previous, 1e-9)
            previous = cycle_start
            event = _status(sample, wrist_command, 1.0 / elapsed, mapper)
            event.update(
                event="step",
                timestamp_monotonic_s=cycle_start,
                action=action,
                arm_action=action[composer.arm_slice],
                hand_action=action[composer.hand_slice],
                camera={
                    "valid": hand_command.valid,
                    "tracking_state": hand_command.tracking_state,
                    "calibration_state": hand_command.calibration_state,
                    "timestamp_s": hand_command.timestamp_s,
                    "curls": hand_command.curls,
                    "error": hand_command.error or camera.error,
                    "worker_running": camera.running,
                    "worker_status": camera.status() if hasattr(camera, "status") else None,
                    "details": hand_command.details,
                },
                hand_source=gesture_command.source,
                hand_override=gesture_command.active_override,
                hand_transition={
                    "active": gesture_command.transition_active,
                    "progress": gesture_command.transition_progress,
                },
                button_rising_edges=gesture_command.button_rising_edges,
                button_event=gesture_command.event,
                hand_targets=composer.held_hand_targets,
                hand_qpos=composer.hand_qpos(),
                ee_pose=composer.ee_pose(),
                reward=reward,
                done=done,
                visualization={"laser": laser_event, "tri_view": tri_view_event},
                cube_scale_factor=(cube_scaler.status().get("current_factor") if cube_scaler is not None else None),
                grasp=grasp_event,
                loop_seconds=time.monotonic() - cycle_start,
            )
            log.event(event)
            _sleep_to_rate(cycle_start, config.control_hz)
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    except Exception as exc:
        reason = f"error: {type(exc).__name__}: {exc}"
        exit_code = 1
        log.event({"event": "exception", "reason": reason, "timestamp_monotonic_s": time.monotonic()})
        print(reason, file=sys.stderr)
    finally:
        last_action = None
        try:
            try:
                if grasp is not None:
                    log.event({
                        "event": "grasp_summary",
                        "timestamp_monotonic_s": time.monotonic(),
                        "grasp": grasp.summary(),
                    })
                if tri_view is not None:
                    tri_view.close()
                if laser is not None:
                    laser.close()
                last_action = close_integrated_resources(
                    env=env,
                    composer=composer,
                    mapper=mapper,
                    device=device,
                    camera=camera,
                )
            finally:
                # The camera worker is stopped by close_integrated_resources
                # before its solver pipe is released. This also runs if an
                # unrelated cleanup operation reports an error.
                if retargeter is not None:
                    retargeter.close()
        except Exception as exc:
            cleanup_reason = f"cleanup error: {type(exc).__name__}: {exc}"
            log.event({"event": "cleanup_exception", "reason": cleanup_reason,
                       "timestamp_monotonic_s": time.monotonic()})
            print(cleanup_reason, file=sys.stderr)
            if exit_code == 0:
                reason = cleanup_reason
                exit_code = 1
        log.event({
            "event": "exit",
            "reason": reason,
            "timestamp_monotonic_s": time.monotonic(),
            "last_zero_wrist_action": last_action,
        })
        log.close()
        print(f"Integrated teleoperation stopped: {reason}. Log: {log.path}")
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="integrated", choices=("teleop", "diagnostic", "integrated"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "spacemouse_wrist.json")
    parser.add_argument("--headless", action="store_true", help="teleop / integrated: no MuJoCo or camera viewer")
    parser.add_argument("--duration", type=float, default=None, help="optional bounded run for smoke tests")
    parser.add_argument("--camera-index", type=int, default=0, help="integrated only: OpenCV camera index")
    parser.add_argument("--no-camera-display", action="store_true", help="integrated only: hide the camera status window")
    parser.add_argument(
        "--hand-control",
        choices=("dex", "legacy-curl", "off"),
        default="dex",
        help="integrated only: official dex (default), explicit legacy curl, or safe-open/off",
    )
    parser.add_argument(
        "--no-three-view",
        action="store_true",
        help="integrated only: use the legacy single MuJoCo viewer instead of the synchronized three-view window",
    )
    parser.add_argument(
        "--viewer-smoke",
        action="store_true",
        help="integrated only: synthetic neutral input, safe-open hand, three-view render, and bounded automatic exit",
    )
    parser.add_argument(
        "--headless-smoke",
        action="store_true",
        help="integrated only: synthetic neutral input and bounded headless automatic exit",
    )
    parser.add_argument(
        "--dex-smoke",
        action="store_true",
        help="integrated only: keep official dex active, use synthetic wrist input, and run a bounded headless smoke test",
    )
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.mode == "diagnostic" and (args.headless or args.no_camera_display or args.camera_index != 0):
        parser.error("camera and --headless options do not apply to diagnostic")
    if args.mode == "teleop" and (args.no_camera_display or args.camera_index != 0):
        parser.error("camera options apply only to integrated")
    if args.mode != "integrated" and (
        args.hand_control != "dex" or args.no_three_view or args.viewer_smoke or args.headless_smoke or args.dex_smoke
    ):
        parser.error("--hand-control, --no-three-view, --viewer-smoke, --headless-smoke, and --dex-smoke apply only to integrated")
    if args.viewer_smoke and args.headless:
        parser.error("--viewer-smoke requires a visible tri-view window; use --headless-smoke with --headless")
    if args.headless_smoke and not args.headless:
        parser.error("--headless-smoke requires --headless")
    if sum(bool(value) for value in (args.viewer_smoke, args.headless_smoke, args.dex_smoke)) > 1:
        parser.error("choose only one of --viewer-smoke, --headless-smoke, or --dex-smoke")
    if args.viewer_smoke:
        args.hand_control = "off"
        if args.duration is None:
            args.duration = 2.0
    if args.headless_smoke:
        args.hand_control = "off"
        if args.duration is None:
            args.duration = 2.0
    if args.dex_smoke:
        if not args.headless:
            parser.error("--dex-smoke requires --headless")
        if args.hand_control != "dex":
            parser.error("--dex-smoke requires --hand-control dex")
        if args.duration is None:
            args.duration = 2.0
    return args


def main() -> int:
    args = parse_args()
    config = WristConfig.load(args.config)
    if args.mode == "diagnostic":
        return run_diagnostic(config, args.duration)
    if args.mode == "integrated":
        use_three_view = not args.headless and not args.no_three_view
        return run_integrated(
            config, args.headless, args.duration, args.camera_index,
            camera_display=not args.headless and not args.no_camera_display,
            hand_control=args.hand_control,
            synthetic_input=args.viewer_smoke or args.headless_smoke or args.dex_smoke,
            three_view=use_three_view,
        )
    return run_teleop(config, args.headless, args.duration)


if __name__ == "__main__":
    raise SystemExit(main())
