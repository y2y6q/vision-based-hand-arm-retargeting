"""Camera-only LIBERO Spatial teleoperation: Panda + Shadow Hand, no SpaceMouse.

One webcam drives everything: the right wrist moves the Panda end effector
(left / right and up / down like a mouse with pointer acceleration, toward /
away from the camera as a rate control, the palm rotation as the end
effector's orientation); the same
detection retargets the Shadow fingers (DexPilot by default). See
``camera_arm_control`` for the estimation, filtering, mapping, workspace and
singularity protection. Nothing assists the grasp: every motion is the
operator's, the protections only scale down or stop it.

Display: the main view plus two orthographic operator views (top = plan,
side = elevation, see ``operator_views``), a vertical palm laser with its
clearance, and the webcam preview with a safe zone and depth gauge.

Clutch: show your LEFT hand next to the right one; the arm holds while you
reposition the right hand (like lifting a mouse), and continues from there
when the left hand leaves.

Keys (simulator window): ``C`` pause / resume, ``-`` / ``=`` arm gain,
``R`` reset, ``Q`` quit.
"""

from __future__ import annotations

from dataclasses import replace
import math
import sys
import time
from typing import Any, Optional

import numpy as np

from .camera_arm_control import (
    ArmSafetyGuard,
    CameraArmHandInput,
    CameraArmMapper,
    CameraArmSettings,
    TargetFollower,
    WorkspaceLimits,
    matrix_to_rotvec,
    rotvec_to_matrix,
)
from .config import WristConfig
from .hand_input import CameraHandWorker
from .libero_spatial_teleop import (
    LIBERO_MAIN_VIEW_WIDTH_PX,
    LIBERO_SIDE_VIEW_WIDTH_PX,
    PANDA_SHADOW_SPEC,
    LiberoActionComposer,
    OfficialLiberoSpatialSession,
    _RunLog,
    _WindowsTimerResolution,
    _shadow_laser_config,
    _shadow_tri_view_config,
    _sleep,
)
from .mapper import SixDofMapper
from .operator_views import (
    BLUE,
    GREY,
    RED,
    WHITE,
    YELLOW,
    OperatorOverlay,
    apply_ortho_view,
    plan_and_elevation_views,
)
from .runtime_controls import KeyboardCommandRouter
from .shadow_dex import ShadowDexHandInput, ShadowDexRetargeter
from .shadow_hand_input import ShadowHandGestureController, ShadowLegacyHandInput
from .synthetic_hand import SyntheticHandWorker
from .visualization import LaserPointer, TriViewCompositor

CAMERA_SPEC_LOG_NAME = "panda_shadow_camera"
MAX_SPEED_M_S = 0.20
MAX_ROT_SPEED_RAD_S = 1.2
# The accumulated target may lead the commanded goal by at most this much
# travel time: a fast hand cannot queue motion that continues after it stops.
LEASH_S = 0.25
STUCK_ERROR_M = 0.03
STUCK_TIME_S = 1.0
TOGGLE_DEBOUNCE_S = 0.35  # a held key repeats; one press toggles once
# Workspace: the scene's objects / fixtures plus this margin, inside the
# reach box around the initial hand pose; orientation within these limits.
OBJECT_MARGIN_M = 0.15
MAX_YAW_RAD = math.radians(90.0)
MAX_TILT_RAD = math.radians(30.0)
TOP_CAMERA = "birdview"
SIDE_CAMERA = "sideview"
# Side view from the operator's left: from the right, the wooden cabinet of
# LIBERO Spatial stands between the camera and the bowls.
SIDE_VIEW_SIGN = -1.0


def _floor_height(sim: Any, start: np.ndarray, robot_root_body: int) -> Optional[float]:
    """Height of the first non-robot surface straight below ``start``."""

    import mujoco

    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    point = np.asarray(start, dtype=np.float64).copy()
    direction = np.array((0.0, 0.0, -1.0))
    geom = np.zeros(1, dtype=np.int32)
    travelled = 0.0
    for _ in range(12):
        distance = mujoco.mj_ray(model, data, point, direction, None, 1, -1, geom)
        if distance < 0.0 or geom[0] < 0:
            return None
        body = int(model.geom_bodyid[geom[0]])
        travelled += distance
        if int(model.body_rootid[body]) != robot_root_body:
            return float(start[2] - travelled)
        point = point + direction * (distance + 1e-4)
        travelled += 1e-4
    return None


def scene_points_xy(env: Any) -> np.ndarray:
    """Horizontal positions of the task's objects and fixtures (may be empty)."""

    sim = env.sim
    points = []
    for registry in (getattr(env, "objects_dict", None) or {}, getattr(env, "fixtures_dict", None) or {}):
        for item in registry.values():
            try:
                body = sim.model.body_name2id(item.root_body)
            except Exception:
                continue
            points.append(np.array(sim.data.body_xpos[body][:2], dtype=np.float64))
    return np.array(points).reshape(-1, 2)


def build_workspace(env: Any, composer: LiberoActionComposer) -> WorkspaceLimits:
    """Safe target region: the task area of this scene, within reach of the initial pose."""

    sim = env.sim
    position, orientation = composer.ee_pose()
    base = np.array(sim.data.body_xpos[sim.model.body_name2id("robot0_base")], dtype=np.float64)
    site_body = int(sim.model.site_bodyid[sim.model.site_name2id(composer.controller.eef_name)])
    model = getattr(sim.model, "_model", sim.model)
    floor = _floor_height(sim, position, int(model.body_rootid[site_body]))
    z_min = (floor + 0.02) if floor is not None else position[2] - 0.30
    box_min = np.array((position[0] - 0.35, position[1] - 0.40, z_min))
    box_max = np.array((position[0] + 0.35, position[1] + 0.40, position[2] + 0.30))
    points = np.vstack((scene_points_xy(env), position[:2]))
    box_min[:2] = np.maximum(box_min[:2], points.min(axis=0) - OBJECT_MARGIN_M)
    box_max[:2] = np.minimum(box_max[:2], points.max(axis=0) + OBJECT_MARGIN_M)
    return WorkspaceLimits(
        box_min=box_min, box_max=box_max,
        shoulder=base + np.array((0.0, 0.0, 0.333)),
        base_axis_xy=base[:2].copy(),
        nominal_ori=orientation.copy(),
        max_yaw_rad=MAX_YAW_RAD, max_tilt_rad=MAX_TILT_RAD,
    )


def leash(target_pos: np.ndarray, target_ori: np.ndarray, goal_pos: np.ndarray, goal_ori: np.ndarray,
          max_m: float, max_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep the target within ``max_m`` / ``max_rad`` of the commanded goal."""

    offset = np.asarray(target_pos, dtype=np.float64) - goal_pos
    distance = float(np.linalg.norm(offset))
    if distance > max_m:
        target_pos = goal_pos + offset * (max_m / distance)
    turn = matrix_to_rotvec(np.asarray(target_ori) @ np.asarray(goal_ori).T)
    angle = float(np.linalg.norm(turn))
    if angle > max_rad:
        target_ori = rotvec_to_matrix(turn * (max_rad / angle)) @ goal_ori
    return np.asarray(target_pos, dtype=np.float64), np.asarray(target_ori, dtype=np.float64)


def arm_step(
    arm: CameraArmMapper,
    target_pos: np.ndarray,
    target_ori: np.ndarray,
    *,
    limits: WorkspaceLimits,
    follower: TargetFollower,
    guard: ArmSafetyGuard,
    goal: tuple[np.ndarray, np.ndarray],
    measured: tuple[np.ndarray, np.ndarray],
    dt: float,
) -> tuple[np.ndarray, list[str], list[str], dict[str, Any]]:
    """One tracking tick: mapper target -> safe world delta for the composer.

    The target is clamped to the workspace and leashed to the commanded goal,
    and whatever a limit or the guard refuses is removed from the mapper's
    accumulated target, so a command beyond a limit is dropped, not wound up.
    """

    goal_pos, goal_ori = goal
    measured_pos, measured_ori = measured
    target_pos, target_ori, limit_reasons = limits.clamp(target_pos, target_ori)
    target_pos, target_ori = leash(target_pos, target_ori, goal_pos, goal_ori,
                                   follower.max_speed_m_s * LEASH_S, follower.max_rot_speed_rad_s * LEASH_S)
    arm.sync_target(target_pos, target_ori)
    delta = follower.delta(target_pos, target_ori, goal_pos, goal_ori, dt)
    lead = np.concatenate((goal_pos - measured_pos, matrix_to_rotvec(goal_ori @ measured_ori.T)))
    delta, guard_reasons, guard_info = guard.check(delta, lead)
    if guard_reasons:
        arm.sync_target(goal_pos + delta[:3], rotvec_to_matrix(delta[3:]) @ goal_ori)
    return delta, limit_reasons, guard_reasons, guard_info


def run_libero_camera_teleop(
    config: WristConfig,
    settings: CameraArmSettings,
    *,
    task_id: int,
    task_order: int,
    init_state_id: int,
    seed: int,
    headless: bool,
    duration_s: Optional[float],
    camera_index: int,
    camera_display: bool,
    camera_mirror: bool,
    hand_control: str,
    synthetic_hand: bool,
    three_view: bool,
) -> int:
    if hand_control not in ("dex", "legacy-curl", "off"):
        raise ValueError("hand_control must be dex, legacy-curl or off")
    spec = PANDA_SHADOW_SPEC
    log = _RunLog(replace(spec, name=CAMERA_SPEC_LOG_NAME))
    session = OfficialLiberoSpatialSession(
        spec, config, task_id=task_id, task_order=task_order, init_state_id=init_state_id,
        seed=seed, has_offscreen_renderer=bool(three_view),
    )
    keyboard = KeyboardCommandRouter()
    arm = CameraArmMapper(settings)
    basis_mapper = SixDofMapper(config)  # only for the primary-camera basis
    follower = TargetFollower(MAX_SPEED_M_S, MAX_ROT_SPEED_RAD_S)
    tri_view = laser = camera = retargeter = overlay = None
    composer = guard = limits = None
    views: dict[str, Any] = {}
    exit_code, reason = 0, "completed"
    screenshot_written = False
    loop_periods: list[float] = []
    notices: list[str] = []
    last_toggle: dict[str, float] = {}

    def route(key: int, source: str) -> None:
        try:
            char = chr(int(key) & 0xFF).lower()
        except (TypeError, ValueError):
            char = ""
        if char == "c":
            now = time.monotonic()
            if now - last_toggle.get(char, -math.inf) < TOGGLE_DEBOUNCE_S:
                last_toggle[char] = now
                return
            last_toggle[char] = now
        if char == "c":
            paused = arm.toggle_pause()
            notices.append("arm paused (C to resume)" if paused else "arm resumed from the current hand pose")
        elif char in ("-", "_"):
            notices.append("arm gain x{:.2f}".format(arm.scale_gains(1.0 / 1.25)))
        elif char in ("=", "+"):
            notices.append("arm gain x{:.2f}".format(arm.scale_gains(1.25)))
        else:
            keyboard.feed(key, source=source)

    def rebuild_protection(env: Any) -> None:
        nonlocal guard, limits
        limits = build_workspace(env, composer)
        guard = ArmSafetyGuard(env.sim, env.robots[0], composer.controller.eef_name)

    def setup_views(env: Any) -> None:
        """Main-camera basis, then the plan / elevation cameras aligned with it."""

        nonlocal basis
        tri_view.configure_cameras(env)
        basis_mapper.set_camera_from_sim(env, tri_config.front_camera_name)
        basis = basis_mapper.control_rotation
        sizes = tri_config.panel_sizes
        points = np.vstack((scene_points_xy(env), composer.ee_pose()[0][:2]))
        top, side = plan_and_elevation_views(
            points, (float(limits.box_min[2]) - 0.05, float(limits.box_max[2])), basis,
            top_camera=TOP_CAMERA, top_size=sizes["wrist"], side_camera=SIDE_CAMERA, side_size=sizes["side"],
            side_sign=SIDE_VIEW_SIGN,
        )
        apply_ortho_view(env.sim, top)
        apply_ortho_view(env.sim, side)
        overlay.top, overlay.side = top, side
        views.update({name: {"camera": view.camera_name, "position_m": view.position, "extent_m": view.extent_m,
                             "image_right_world": view.x_axis, "image_up_world": view.y_axis}
                      for name, view in (("top", top), ("side", side))})

    timer_resolution = _WindowsTimerResolution().__enter__()
    try:
        session.open()
        env = session.env
        composer = LiberoActionComposer(env, spec)
        composer.initialize_goal()
        rebuild_protection(env)
        basis = basis_mapper.control_rotation
        tri_config = replace(
            _shadow_tri_view_config(config),
            wrist_camera_name=TOP_CAMERA, side_camera_name=SIDE_CAMERA,
            image_width=LIBERO_MAIN_VIEW_WIDTH_PX,
            thumbnail_scale=LIBERO_SIDE_VIEW_WIDTH_PX / LIBERO_MAIN_VIEW_WIDTH_PX,
            display_width=None, display_height=None,
            window_name="LIBERO camera teleop: Panda + Shadow",
        )
        # A vertical palm laser: where the hand is above the table / bowl, in
        # the main and side views (seen end-on from the top).
        laser_config = replace(
            _shadow_laser_config(config), direction_frame="world", world_direction=(0.0, 0.0, -1.0),
            visible_views=("front", "wrist", "side"), hide_robot_in_wrist_view=False,
        )
        if three_view and tri_config.enabled:
            laser = LaserPointer(env, laser_config)
            overlay = OperatorOverlay(None, None, depth_deadzone_m=settings.depth_deadzone_m,
                                      edge_warning=settings.edge_warning)
            tri_view = TriViewCompositor(
                tri_config, show_window=not headless,
                key_handler=lambda key: route(key, "tri_view"),
                on_close=lambda: keyboard.feed("q", source="tri_view_window"), laser=laser, overlay=overlay,
            )
            setup_views(env)

        lower, upper = composer.gripper.lower_limits, composer.gripper.upper_limits
        if hand_control == "dex":
            print("Starting official dex-retargeting sidecar for Shadow (first launch can take ~60 s)...")
            retargeter = ShadowDexRetargeter()
            fingers = ShadowDexHandInput(
                retargeter, lower, upper, stale_timeout_s=config.dex_hand.stale_timeout_s,
                expected_handedness=config.dex_hand.expected_handedness, unmirror_world_x=camera_mirror,
            )
        else:
            fingers = ShadowLegacyHandInput(lower, upper)
        hand_input = CameraArmHandInput(fingers, arm, camera_mirror=camera_mirror,
                                        expected_handedness=config.dex_hand.expected_handedness)
        gestures = ShadowHandGestureController(config=config.hand_gestures, lower_limits=lower, upper_limits=upper)
        no_buttons = (False,) * 8
        gestures.reset(buttons=no_buttons)
        status_snapshot: dict[str, Any] = {"hand_mode": hand_control, "hand_source": "arm: waiting"}
        # The webcam preview is embedded in the simulator view when there is
        # one; otherwise it keeps its own window.
        embed_preview = tri_view is not None
        if synthetic_hand:
            camera = SyntheticHandWorker(hand_input, mirrored=camera_mirror,
                                         focal_width_normalized=settings.focal_width_normalized)
        else:
            camera = CameraHandWorker(
                hand_input, camera_index=camera_index, display=(camera_display or embed_preview) and not headless,
                mirror_input=camera_mirror, keypress_callback=lambda key: route(key, "camera"),
                status_provider=lambda: dict(status_snapshot),
                window_name="LIBERO camera teleop: right hand", display_text=not embed_preview,
            )
        camera.start()

        log.metadata({
            "mode": "libero_spatial_camera_only_teleoperation", "session": session.describe(),
            "hand_control": hand_control, "camera_mirror": camera_mirror, "synthetic_hand": synthetic_hand,
            "camera_arm_settings": dict(vars(settings)), "world_from_control": basis,
            "max_speed_m_s": MAX_SPEED_M_S, "max_rot_speed_rad_s": MAX_ROT_SPEED_RAD_S,
            "leash_s": LEASH_S,
            "workspace": {"box_min": limits.box_min, "box_max": limits.box_max, "shoulder": limits.shoulder,
                          "max_reach_m": limits.max_reach_m, "min_radius_m": limits.min_radius_m,
                          "max_yaw_rad": limits.max_yaw_rad, "max_tilt_rad": limits.max_tilt_rad},
            "safety": {"joint_margin_rad": guard.joint_margin, "reference_manipulability": guard.reference_manipulability,
                       "min_manipulability": guard.min_manipulability, "wall_recheck_s": guard.wall_recheck_s},
            "operator_views": views,
            "dex": retargeter.metadata() if retargeter is not None else None,
        })
        print("LIBERO camera teleop ready: {}".format(session.task.language))
        print("Show your open RIGHT hand to the camera; the arm starts after a few stable frames.")
        print("Left/right, up/down: move the hand (faster = farther). Away/back: hold the hand nearer/farther.")
        print("Keys: LEFT hand = clutch | C pause | -/= gain | R reset | Q quit")

        started = previous = time.monotonic()
        stuck_since: Optional[float] = None
        last_state = None
        while duration_s is None or time.monotonic() - started < duration_s:
            if keyboard.stop_event.is_set():
                reason = keyboard.quit_reason or "Q"
                break
            if keyboard.take_reset():
                session.reset()
                composer = LiberoActionComposer(env, spec)
                composer.initialize_goal()
                rebuild_protection(env)
                arm.request_anchor()
                fingers.reset()
                gestures.reset(buttons=no_buttons)
                if tri_view is not None:
                    laser = LaserPointer(env, laser_config)  # hard reset freed the old sim
                    tri_view.laser = laser
                    setup_views(env)
                log.event({"event": "environment_reset", "timestamp_monotonic_s": time.monotonic()})
                previous = time.monotonic()
                continue
            tick = time.monotonic()
            dt = float(np.clip(tick - previous, 1.0 / config.control_hz, 2.0 / config.control_hz))
            measured_pos, measured_ori = composer.ee_pose()
            goal_pos = composer.goal_pos if composer.goal_pos is not None else measured_pos
            goal_ori = composer.goal_ori if composer.goal_ori is not None else measured_ori
            status = arm.target(goal_pos, goal_ori, basis, tick)
            delta = np.zeros(6)
            limit_reasons: list[str] = []
            guard_reasons: list[str] = []
            guard_info: dict[str, Any] = {}
            if status.state == "tracking":
                delta, limit_reasons, guard_reasons, guard_info = arm_step(
                    arm, status.target_pos, status.target_ori, limits=limits, follower=follower, guard=guard,
                    goal=(goal_pos, goal_ori), measured=(measured_pos, measured_ori), dt=dt,
                )
            # Anti-lock: a goal that the arm cannot reach for a second (contact,
            # limit) is dropped to the measured pose and the hand re-anchored,
            # instead of straining against it.
            goal_error = 0.0 if composer.goal_pos is None else float(np.linalg.norm(composer.goal_pos - measured_pos))
            if goal_error > STUCK_ERROR_M:
                stuck_since = tick if stuck_since is None else stuck_since
                if tick - stuck_since > STUCK_TIME_S:
                    composer.initialize_goal()
                    arm.request_anchor()
                    delta = np.zeros(6)
                    stuck_since = None
                    notices.append("arm could not reach target: holding here")
                    log.event({"event": "stuck_recovery", "goal_error_m": goal_error, "timestamp_monotonic_s": tick})
            else:
                stuck_since = None

            camera_command = fingers.latest(tick)
            hand_command = gestures.update(
                no_buttons, camera_command.joint_targets,
                camera_valid=bool(camera_command.valid and hand_control != "off"), timestamp_s=tick,
            )
            action = composer.compose(delta, hand_command.joint_targets)
            _obs, reward, _done, _info = env.step(action)

            if status.state != last_state:
                print("arm:", status.state)
                last_state = status.state
            while notices:
                message = notices.pop(0)
                print(message)
                log.event({"event": "notice", "message": message, "timestamp_monotonic_s": tick})
            protections = limit_reasons + guard_reasons
            status_snapshot["hand_source"] = "arm: {}{}".format(
                status.state, " | " + ",".join(protections) if protections else "")
            laser_record = laser.update() if laser is not None else {"enabled": False}
            clearance = hit = palm = None
            if laser_record.get("active"):
                palm = np.asarray(laser_record["origin_world_m"])
                if laser_record.get("hit_geom_id") is not None:
                    clearance = float(laser_record["length_m"])
                    hit = palm + np.asarray(laser_record["direction_world"]) * clearance
            if overlay is not None:
                details = status.details
                lines = [
                    ("ARM {}{}".format(
                        status.state.upper(), "   rotation held while moving" if details.get("rotation_frozen") else ""),
                     WHITE),
                    ("gain x{:.2f}  now {:.2f}   depth {:+.0f} cm/s".format(
                        settings.translation_gain, details.get("effective_gain", 0.0),
                        100.0 * details.get("depth_rate_m_s", 0.0)), WHITE),
                ]
                if clearance is not None:
                    lines.append(("palm {:.1f} cm above {}".format(100.0 * clearance, laser_record.get("hit_geom")), WHITE))
                if protections:
                    lines.append(("limit: " + ", ".join(sorted(set(protections))), RED))
                lines.append(("LEFT hand = clutch | C pause | -/= gain | R reset | Q quit", GREY))
                if status.state == "clutched":
                    badge = ("CLUTCH", BLUE)
                elif status.state in ("lost", "waiting"):
                    badge = ("SHOW RIGHT HAND", RED)
                elif status.state == "anchoring":
                    badge = ("HOLD STILL...", YELLOW)
                elif status.state == "paused":
                    badge = ("PAUSED (C)", YELLOW)
                else:
                    badge = None
                frame_getter = getattr(camera, "latest_frame", None)
                overlay.state = {
                    "palm_world": palm, "hit_world": hit, "clearance_m": clearance, "lines": lines, "badge": badge,
                    "camera_frame": frame_getter()[0] if embed_preview and callable(frame_getter) else None,
                    "edge_distance": details.get("edge_distance"), "pose_error": hand_input.last_pose_error,
                    "depth_offset_m": details.get("depth_offset_m"),
                }
            visual = tri_view.update(env, status_text=session.task.language) if tri_view is not None else {"enabled": False}
            if tri_view is not None and visual.get("rendered") and not screenshot_written:
                screenshot = log.path / "tri_view_initial.png"
                screenshot_written = bool(tri_view.save_last_frame(screenshot))
            if tri_view is not None and visual.get("closed"):
                keyboard.feed("q", source="tri_view")
            if camera_display and not headless and not embed_preview and hasattr(camera, "show_latest"):
                camera.show_latest()
            elapsed = max(tick - previous, 1e-9)
            loop_periods.append(elapsed)
            previous = tick
            log.event({
                "event": "step", "timestamp_monotonic_s": tick, "arm_state": status.state,
                "hand_offset_control_m": status.hand_offset_m, "hand_rotvec_control": status.hand_rotation_rad,
                "target_pos": status.target_pos, "measured_pos": measured_pos,
                "measured_rotvec": matrix_to_rotvec(measured_ori), "goal_error_m": goal_error,
                "arm_action": action[composer.arm_slice], "delta": delta,
                "limits": limit_reasons, "guard": guard_reasons, "guard_info": guard_info,
                "pose_error": hand_input.last_pose_error, "arm_details": status.details,
                "clearance_m": clearance,
                "hand_source": hand_command.source, "hand_action": action[composer.hand_slice],
                "reward": reward, "task_success": bool(env._check_success()),
                "visualization": {"laser": laser_record, "tri_view": visual},
            })
            _sleep(tick, config.control_hz)
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    except Exception as exc:
        exit_code, reason = 1, "error: {}: {}".format(type(exc).__name__, exc)
        log.event({"event": "exception", "reason": reason, "timestamp_monotonic_s": time.monotonic()})
        print(reason, file=sys.stderr)
    finally:
        try:
            if composer is not None and session.env is not None:
                final = composer.compose(np.zeros(6), None)
                session.env.step(final)
                log.event({"event": "safe_zero", "action": final, "timestamp_monotonic_s": time.monotonic()})
            if tri_view is not None:
                tri_view.save_last_frame(log.path / "tri_view_final.png")
                tri_view.close()
            if laser is not None:
                laser.close()
            if camera is not None:
                camera.close()
            if retargeter is not None:
                retargeter.close()
            session.close()
        finally:
            timer_resolution.__exit__()
            loop_hz = 1.0 / float(np.median(loop_periods[1:])) if len(loop_periods) > 1 else None
            log.event({"event": "exit", "reason": reason, "median_loop_hz": loop_hz,
                       "timestamp_monotonic_s": time.monotonic()})
            log.close()
        if loop_hz is not None:
            print("median control/render loop: {:.1f} Hz (target {:.0f} Hz)".format(loop_hz, config.control_hz))
        print("LIBERO camera teleoperation stopped: {}. Log: {}".format(reason, log.path))
    return exit_code
