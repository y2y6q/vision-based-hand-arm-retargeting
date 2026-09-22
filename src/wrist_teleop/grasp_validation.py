"""Reproducible, hardware-free Gate 1--3 Panda + Allegro grasp validation.

The live teleoperation path remains deliberately untouched.  This module owns
only scripted simulator experiments and never constructs a SpaceMouse or a
camera worker.  A single runner executes the gates in order:

``Gate 1`` verifies the compiled hand, actuators, mounting transform and TCP;
``Gate 2`` evaluates a temporarily supported static hand grasp after the
support is removed; and ``Gate 3`` performs the same grasp through a scripted
world-frame OSC trajectory.

Every invocation writes self-contained evidence below ``outputs/grasp_validation``.
The individual physics tests are intentionally conservative: a failed gate is
recorded with one stable failure category and stops the higher gate.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import mujoco
import numpy as np

from .allegro_model import ALLEGRO_JOINT_NAMES, SOURCE_URDF, make_allegro_panda_env
from .config import WristConfig
from .hand_input import curls_to_allegro_targets
from .integrated import PandaAllegroActionComposer
from .runtime_controls import KeyboardCommandRouter, install_viewer_key_callback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "grasp_validation"
DEFAULT_VISUAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "grasp_validation_visual"

GATE2_FAILURES = frozenset(
    {
        "no_contact",
        "insufficient_opposition",
        "object_ejected",
        "object_slip",
        "penetration_artifact",
        "model_collision",
        "actuator_saturation",
        "joint_limit",
        "joint_tracking_error",
        "unstable_contact",
        "timeout",
        "unknown",
    }
)
GATE3_FAILURES = frozenset(
    {
        "pregrasp_pose_error",
        "approach_collision",
        "osc_tracking_error",
        "missed_object",
        "grasp_not_closed",
        "contact_lost_during_lift",
        "object_slip",
        "insufficient_lift",
        "hold_failure",
        "timeout",
        "unknown",
    }
)

FINGER_GROUPS: Mapping[str, tuple[int, ...]] = {
    "index": (0, 1, 2, 3),
    "middle": (4, 5, 6, 7),
    "ring": (8, 9, 10, 11),
    "thumb": (12, 13, 14, 15),
}


@dataclass(frozen=True)
class GraspValidationConfig:
    """All fixed parameters used by the Gate experiments.

    The cube location is expressed in the actual OSC wrist/TCP frame rather
    than assuming that robosuite's wrist reference is the hand palm.  This is
    important for the custom Allegro installation: the current ``grip_site``
    is a wrist reference, while the generated Allegro root is the palm frame.
    """

    config_version: str = "grasp-gates-v9-substep-fixture-normal-search"
    change_note: str = (
        "v9 replaces the 20 Hz zero-order-held temporary support with a critically "
        "damped force fixture updated on every MuJoCo substep, removes the release "
        "velocity write, records canonical cube-to-hand normals, and runs the "
        "fixed Gate 2 staged geometry search before qualification."
    )
    seed: int = 20260909
    control_hz: float = 20.0
    reuse_compiled_model_across_trials: bool = True
    panda_initial_joint_positions_rad: tuple[float, float, float, float, float, float, float] = (
        0.0, 0.19634954, 0.0, -2.61799388, 0.0, 2.20, 0.78539816,
    )
    # Gate 1
    joint_probe_fraction: float = 0.15
    open_settle_s: float = 0.75
    probe_settle_s: float = 0.75
    joint_target_error_rad: float = 0.070
    joint_motion_fraction_required: float = 0.30
    actuator_saturation_fraction: float = 0.98
    sustained_saturation_s: float = 0.50
    joint_limit_margin_rad: float = 0.010
    significant_penetration_m: float = 0.004
    gate1_close_curls: tuple[float, float, float, float] = (0.60, 0.60, 0.60, 0.80)
    # Gate 2: a physically held, temporary placement fixture is released
    # before the three-second gravity-on evaluation.  It is never a weld and
    # it never follows the wrist after release.
    static_cube_center_tcp_m: tuple[float, float, float] = (0.085, 0.030, 0.115)
    static_close_curls: tuple[float, float, float, float] = (0.50, 0.50, 0.50, 0.85)
    close_duration_s: float = 1.20
    temporary_support_preclose_s: float = 0.30
    temporary_support_settle_s: float = 3.80
    temporary_support_release_ramp_s: float = 0.30
    # The support gains are derived from the compiled cube mass:
    # Kp=m(2*pi*f_n)^2 and Kd=2*zeta*m(2*pi*f_n).  The correction component
    # is acceleration-limited before gravity compensation is added.
    temporary_support_natural_frequency_hz: float = 4.0
    temporary_support_damping_ratio: float = 1.0
    temporary_support_max_correction_accel_mps2: float = 25.0
    fixture_peak_speed_limit_mps: float = 0.20
    # A recorded, gravity-on interval after the fixture is removed lets the
    # freely simulated cube settle into already-closing fingers. It is not
    # excluded from penetration, velocity, collision, or ejection metrics;
    # the following 3.1 s is the qualification hold itself.
    gravity_release_settle_s: float = 0.25
    normal_opposition_dot_max: float = -0.50
    normal_contact_min_force_n: float = 0.05
    static_hold_s: float = 3.10
    static_slip_limit_m: float = 0.025
    gate2_y_candidates_m: tuple[float, float, float] = (0.010, 0.015, 0.020)
    gate2_x_candidates_m: tuple[float, float, float] = (0.080, 0.085, 0.090)
    gate2_z_candidates_m: tuple[float, float, float] = (0.110, 0.115, 0.120)
    gate2_search_top_geometry_candidates: int = 2
    # Gate 3
    table_cube_xy_m: tuple[float, float] = (-0.050, 0.000)
    pregrasp_clearance_m: float = 0.085
    motion_position_tolerance_m: float = 0.012
    motion_orientation_tolerance_rad: float = 0.12
    motion_timeout_s: float = 5.0
    osc_tracking_error_m: float = 0.035
    lift_command_m: float = 0.050
    lift_success_m: float = 0.030
    lift_hold_s: float = 3.10
    # Qualification policy shared by Gate 2 and Gate 3.
    qualification_trials: int = 3
    formal_trials: int = 10
    required_formal_successes: int = 8
    frame_every_s: float = 0.50

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.reuse_compiled_model_across_trials, bool):
            raise ValueError("reuse_compiled_model_across_trials must be boolean")
        if not 0.10 <= float(self.joint_probe_fraction) <= 0.20:
            raise ValueError("joint_probe_fraction must be in [0.10, 0.20]")
        for name in (
            "control_hz", "open_settle_s", "probe_settle_s", "joint_target_error_rad",
            "joint_motion_fraction_required", "actuator_saturation_fraction",
            "sustained_saturation_s", "joint_limit_margin_rad", "significant_penetration_m",
            "close_duration_s", "temporary_support_preclose_s", "temporary_support_settle_s", "temporary_support_release_ramp_s",
            "temporary_support_natural_frequency_hz", "temporary_support_damping_ratio",
            "temporary_support_max_correction_accel_mps2", "fixture_peak_speed_limit_mps",
            "gravity_release_settle_s", "normal_contact_min_force_n",
            "static_hold_s", "static_slip_limit_m", "pregrasp_clearance_m",
            "motion_position_tolerance_m", "motion_orientation_tolerance_rad", "motion_timeout_s",
            "osc_tracking_error_m", "lift_command_m", "lift_success_m", "lift_hold_s",
            "frame_every_s",
        ):
            if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < float(self.joint_motion_fraction_required) <= 1.0:
            raise ValueError("joint_motion_fraction_required must be in (0, 1]")
        if not 0.0 < float(self.actuator_saturation_fraction) <= 1.0:
            raise ValueError("actuator_saturation_fraction must be in (0, 1]")
        if not -1.0 <= float(self.normal_opposition_dot_max) <= 1.0:
            raise ValueError("normal_opposition_dot_max must be in [-1, 1]")
        for name in ("static_cube_center_tcp_m",):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, values)
        panda_qpos = tuple(float(value) for value in self.panda_initial_joint_positions_rad)
        if len(panda_qpos) != 7 or not np.all(np.isfinite(panda_qpos)):
            raise ValueError("panda_initial_joint_positions_rad must contain seven finite values")
        object.__setattr__(self, "panda_initial_joint_positions_rad", panda_qpos)
        xy = tuple(float(value) for value in self.table_cube_xy_m)
        if len(xy) != 2 or not np.all(np.isfinite(xy)):
            raise ValueError("table_cube_xy_m must contain two finite values")
        object.__setattr__(self, "table_cube_xy_m", xy)
        for name in ("gate1_close_curls", "static_close_curls"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 4 or not np.all(np.isfinite(values)) or any(value < 0 or value > 1 for value in values):
                raise ValueError(f"{name} must contain four curl fractions in [0, 1]")
            object.__setattr__(self, name, values)
        for name in ("gate2_y_candidates_m", "gate2_x_candidates_m", "gate2_z_candidates_m"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, values)
        if int(self.gate2_search_top_geometry_candidates) != 2:
            raise ValueError("gate2_search_top_geometry_candidates is fixed at two")
        if self.qualification_trials != 3 or self.formal_trials != 10 or self.required_formal_successes != 8:
            raise ValueError("acceptance policy is fixed at 3 qualifying and 8/10 formal trials")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


DEFAULT_CONFIG = GraspValidationConfig()

# Parameter history is deliberately stored with every new snapshot.  It is an
# evidence ledger, not a second configuration source: DEFAULT_CONFIG remains
# the only executable configuration for a run.
TUNING_HISTORY: tuple[Mapping[str, Any], ...] = (
    {
        "version": "v1",
        "change": "initial static grasp fixture trial",
        "reason": "establish baseline Gate 1 and Gate 2 evidence",
    },
    {
        "version": "v2",
        "change": "correct fixture wrench slot after a torque-slot instability",
        "reason": "MuJoCo xfrc_applied stores force in entries 0:3 and torque in 3:6",
    },
    {
        "version": "v3",
        "change": "set fixture gains to Kp=100 N/m, Kd=10 N/(m/s), max=8 N; use cube offset [0.085, 0.030, 0.115] and curls [0.50, 0.50, 0.50, 0.85]",
        "reason": "remove high-gain fixture instability while retaining normal gravity after release",
    },
    {
        "version": "v4",
        "change": "set Panda joint_6 from 2.94159265 to 1.20 rad",
        "reason": "test gravity orientation for thumb opposition; Gate 1 found hand--Panda collision",
    },
    {
        "version": "v5",
        "change": "set Panda joint_6 from 1.20 to 2.20 rad",
        "reason": "collision scan found this tilt starts without hand--Panda collision",
    },
    {
        "version": "v6",
        "change": "refresh OSC reference and null-space initial joints after the scripted Panda pose write; preserve all v5 physical values",
        "reason": "ensure recorded Panda pose remains the configured 2.20-rad setting throughout Gate 2",
    },
    {
        "version": "v7",
        "change": "set validation-only robosuite hard_reset from true to false (reuse one compiled model across resets)",
        "reason": "lock cube size, mass, friction, collision geometry, and model SHA during single / 3 / 10 trial protocols",
    },
    {
        "version": "v8",
        "change": "pass seed=20260909 directly to robosuite Lift during environment construction",
        "reason": "make the compiled cube parameters repeat across fresh validation invocations, not only resets within one invocation",
    },
    {
        "version": "v9",
        "change": "replace the 20 Hz held virtual fixture with mass-derived, critically damped per-physics-substep support; remove release qvel reset; add canonical-normal Gate 2 screening",
        "reason": "v8 showed a 2.521 m/s fixture transient and 9.843 mm index-tip impact during close/settle, while initial placement and gravity hold were stable",
    },
)


@dataclass(frozen=True)
class Gate2Candidate:
    """A fully recorded static grasp candidate used only inside Gate 2.

    The first two search stages intentionally retain the v8 hand target and
    vary only the requested cube TCP coordinates.  The final stage may use a
    short, explicit finger schedule after geometry has been ranked.
    """

    candidate_id: str
    stage: str
    cube_center_tcp_m: tuple[float, float, float]
    final_curls: tuple[float, float, float, float]
    close_schedule: tuple[tuple[float, tuple[float, float, float, float]], ...]
    rationale: str

    def __post_init__(self) -> None:
        cube = tuple(float(value) for value in self.cube_center_tcp_m)
        if len(cube) != 3 or not np.all(np.isfinite(cube)):
            raise ValueError("Gate 2 candidate cube_center_tcp_m must contain three finite values")
        curls = tuple(float(value) for value in self.final_curls)
        if len(curls) != 4 or not np.all(np.isfinite(curls)) or any(value < 0.0 or value > 1.0 for value in curls):
            raise ValueError("Gate 2 candidate final_curls must contain four fractions in [0, 1]")
        schedule: list[tuple[float, tuple[float, float, float, float]]] = []
        for duration_s, target in self.close_schedule:
            duration = float(duration_s)
            values = tuple(float(value) for value in target)
            if duration <= 0.0 or len(values) != 4 or any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError("Gate 2 candidate close_schedule must contain positive duration and four curl fractions")
            schedule.append((duration, values))
        if not schedule:
            raise ValueError("Gate 2 candidate close_schedule cannot be empty")
        object.__setattr__(self, "cube_center_tcp_m", cube)
        object.__setattr__(self, "final_curls", curls)
        object.__setattr__(self, "close_schedule", tuple(schedule))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "cube_center_tcp_m": list(self.cube_center_tcp_m),
            "final_curls": list(self.final_curls),
            "close_schedule": [
                {"duration_s": duration, "curls": list(curls)}
                for duration, curls in self.close_schedule
            ],
            "rationale": self.rationale,
        }


def _json_value(value: Any) -> Any:
    """Convert NumPy / pathlib values into strict JSON values without NaNs."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, Path):
        value = str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("validation artifacts cannot contain NaN or infinity")
        return value
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_locked_gate2_candidate(path: Path, *, config: GraspValidationConfig = DEFAULT_CONFIG) -> Gate2Candidate:
    """Load the already-qualified Gate 2 candidate without re-running its search.

    The visual replay is deliberately tied to the formal Gate 2 lock.  It
    rejects a stale or malformed lock rather than silently substituting the
    base configuration.
    """

    lock_path = Path(path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if int(payload.get("gate", -1)) != 2:
        raise ValueError(f"{lock_path} is not a Gate 2 lock")
    if str(payload.get("base_config_sha256")) != config.sha256:
        raise ValueError(
            f"{lock_path} was created for config {payload.get('base_config_sha256')}, "
            f"but the active Gate 2 config is {config.sha256}"
        )
    candidate_data = dict(payload.get("candidate", {}))
    try:
        schedule = tuple(
            (float(item["duration_s"]), tuple(float(value) for value in item["curls"]))
            for item in candidate_data["close_schedule"]
        )
        candidate = Gate2Candidate(
            candidate_id=str(candidate_data["candidate_id"]),
            stage=str(candidate_data["stage"]),
            cube_center_tcp_m=tuple(float(value) for value in candidate_data["cube_center_tcp_m"]),
            final_curls=tuple(float(value) for value in candidate_data["final_curls"]),
            close_schedule=schedule,
            rationale=str(candidate_data["rationale"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{lock_path} has an invalid Gate 2 candidate") from exc
    expected = str(payload.get("candidate_sha256", ""))
    actual = canonical_sha256({"gate2_base_config": config.to_dict(), "candidate": candidate.to_dict()})
    if expected != actual:
        raise ValueError(f"{lock_path} candidate SHA-256 does not match its contents")
    return candidate


def newest_locked_gate2_config(root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Return the most recently written qualified Gate 2 lock."""

    locks = sorted(Path(root).glob("*_gate2*/locked_gate2_configuration.json"), key=lambda item: item.stat().st_mtime)
    if not locks:
        raise FileNotFoundError(f"no Gate 2 lock found below {root}")
    return locks[-1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class ArtifactWriter:
    """Small, flushed writer for reproducible evidence and crash diagnostics."""

    def __init__(self, root: Path, *, config_sha256: str, frame_every_s: float, record_frames: bool = True):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.frames = self.root / "frames"
        self.frames.mkdir()
        self.config_sha256 = config_sha256
        self._trials = (self.root / "trials.jsonl").open("w", encoding="utf-8", newline="\n")
        self._contacts = (self.root / "contacts.jsonl").open("w", encoding="utf-8", newline="\n")
        self._events = (self.root / "events.jsonl").open("w", encoding="utf-8", newline="\n")
        self._run_log = (self.root / "run.log").open("w", encoding="utf-8", newline="\n")
        self.frame_paths: list[str] = []
        self._frame_index = 0
        self._next_frame_time = -np.inf
        self._frame_every_s = float(frame_every_s)
        self._record_frames = bool(record_frames)
        self.frame_error: str | None = None

    @staticmethod
    def _write(stream: Any, value: Mapping[str, Any]) -> None:
        stream.write(json.dumps(_json_value(value), sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()

    def write_json(self, name: str, value: Mapping[str, Any]) -> Path:
        path = self.root / name
        path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def log(self, message: str) -> None:
        self._run_log.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
        self._run_log.flush()

    def event(self, value: Mapping[str, Any]) -> None:
        record = dict(value)
        record.setdefault("config_sha256", self.config_sha256)
        self._write(self._events, record)

    def trial(self, value: Mapping[str, Any]) -> None:
        record = dict(value)
        record.setdefault("config_sha256", self.config_sha256)
        failure = record.get("failure_category")
        if bool(record.get("success")):
            record["failure_category"] = None
        elif not isinstance(failure, str):
            record["failure_category"] = "unknown"
        self._write(self._trials, record)

    def contacts(self, values: Iterable[Mapping[str, Any]]) -> None:
        for value in values:
            record = dict(value)
            record.setdefault("config_sha256", self.config_sha256)
            self._write(self._contacts, record)

    def capture(self, env: Any, label: str, *, sim_time: float, force: bool = False) -> str | None:
        """Capture a normal simulator frame sequence without affecting physics.

        Failure to obtain an offscreen OpenGL context is logged rather than
        hiding a physics result.  On the supported local environment this uses
        robosuite's existing offscreen renderer and writes PNG frames.
        """

        if not self._record_frames:
            return None
        if not force and sim_time + 1e-9 < self._next_frame_time:
            return None
        self._next_frame_time = sim_time + self._frame_every_s
        try:
            image = env.sim.render(width=640, height=480, camera_name="agentview")
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[2] < 3:
                raise RuntimeError(f"unexpected renderer output shape {image.shape}")
            try:
                import cv2
            except Exception as exc:  # pragma: no cover - OpenCV is present in the validated environment
                raise RuntimeError(f"OpenCV PNG writer unavailable: {exc}") from exc
            filename = f"{self._frame_index:04d}_{label}.png"
            path = self.frames / filename
            # MuJoCo's pixel output is RGB and vertically flipped for image files.
            bgr = np.ascontiguousarray(image[::-1, :, :3][:, :, ::-1])
            if not cv2.imwrite(str(path), bgr):
                raise RuntimeError("cv2.imwrite returned false")
            self._frame_index += 1
            relative = path.relative_to(self.root).as_posix()
            self.frame_paths.append(relative)
            self.event({"event": "frame", "label": label, "sim_time_s": sim_time, "path": relative})
            return relative
        except Exception as exc:  # pragma: no cover - host GL availability is external
            self.frame_error = f"{type(exc).__name__}: {exc}"
            self.log(f"frame_capture_error {self.frame_error}")
            self.event({"event": "frame_capture_error", "sim_time_s": sim_time, "error": self.frame_error})
            return None

    def capture_validation_views(
        self,
        env: Any,
        label: str,
        *,
        sim_time: float,
        overwrite: bool = False,
    ) -> list[str]:
        """Save the three Gate 2 close views without changing physics.

        ``frontview``, ``sideview``, and ``birdview`` are configured by
        ``configure_validation_cameras`` to aim closely at the palm/cube.  A
        fixed overwrite name lets the final deepest-penetration frame replace
        earlier candidates without leaving a per-step image sequence.
        """

        if not self._record_frames:
            return []
        paths: list[str] = []
        try:
            try:
                import cv2
            except Exception as exc:  # pragma: no cover - OpenCV is validated with robosuite
                raise RuntimeError(f"OpenCV PNG writer unavailable: {exc}") from exc
            for view, camera_name in (
                ("palm_front", "frontview"),
                ("palm_side", "sideview"),
                ("palm_top", "birdview"),
            ):
                image = np.asarray(env.sim.render(width=800, height=600, camera_name=camera_name))
                if image.ndim != 3 or image.shape[2] < 3:
                    raise RuntimeError(f"unexpected renderer output shape {image.shape}")
                filename = (
                    f"{label}_{view}.png"
                    if overwrite
                    else f"{self._frame_index:04d}_{label}_{view}.png"
                )
                path = self.frames / filename
                bgr = np.ascontiguousarray(image[::-1, :, :3][:, :, ::-1])
                if not cv2.imwrite(str(path), bgr):
                    raise RuntimeError("cv2.imwrite returned false")
                relative = path.relative_to(self.root).as_posix()
                if relative not in self.frame_paths:
                    self.frame_paths.append(relative)
                paths.append(relative)
                self.event({
                    "event": "validation_frame",
                    "label": label,
                    "view": view,
                    "camera": camera_name,
                    "sim_time_s": sim_time,
                    "path": relative,
                    "overwrite": bool(overwrite),
                })
            if not overwrite:
                self._frame_index += 1
            return paths
        except Exception as exc:  # pragma: no cover - host GL availability is external
            self.frame_error = f"{type(exc).__name__}: {exc}"
            self.log(f"validation_frame_capture_error {self.frame_error}")
            self.event({"event": "validation_frame_capture_error", "sim_time_s": sim_time, "error": self.frame_error})
            return []

    def close(self) -> None:
        for stream in (self._trials, self._contacts, self._events, self._run_log):
            stream.close()


class VisualizationInterrupted(RuntimeError):
    """Stop a visual replay at a safe control-step boundary."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Gate2VisualizationRecorder:
    """Optional front-view recording that never participates in simulation."""

    def __init__(self, path: Path | None, *, fps: float):
        self.requested_path = None if path is None else Path(path)
        self.fps = float(fps)
        self.mode = "disabled" if path is None else "pending"
        self.output_path: Path | None = None
        self.frame_count = 0
        self.error: str | None = None
        self._writer: Any = None
        self._cv2: Any = None

    def capture(self, env: Any) -> None:
        if self.requested_path is None or self.mode == "unavailable":
            return
        try:
            import cv2

            image = np.asarray(env.sim.render(width=800, height=600, camera_name="frontview"))
            if image.shape[:2] != (600, 800) or image.ndim != 3 or image.shape[2] < 3:
                raise RuntimeError(f"unexpected recording frame shape {image.shape}")
            bgr = np.ascontiguousarray(image[::-1, :, :3][:, :, ::-1])
            if self.mode == "pending":
                self._cv2 = cv2
                self.requested_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(self.requested_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (800, 600),
                )
                if writer.isOpened():
                    self._writer = writer
                    self.mode = "mp4"
                    self.output_path = self.requested_path
                else:
                    writer.release()
                    self.output_path = self.requested_path.with_suffix("").with_name(
                        f"{self.requested_path.stem}_frames"
                    )
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    self.mode = "frames"
            if self.mode == "mp4":
                self._writer.write(bgr)
            elif self.mode == "frames":
                frame = self.output_path / f"{self.frame_count:05d}.png"
                if not cv2.imwrite(str(frame), bgr):
                    raise RuntimeError("cv2.imwrite returned false")
            self.frame_count += 1
        except Exception as exc:  # pragma: no cover - host codecs and OpenGL are external
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            self.error = f"{type(exc).__name__}: {exc}"
            self.mode = "unavailable"

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def summary(self) -> dict[str, Any]:
        return {
            "requested_path": None if self.requested_path is None else str(self.requested_path),
            "mode": self.mode,
            "output_path": None if self.output_path is None else str(self.output_path),
            "frame_count": self.frame_count,
            "error": self.error,
            "fps": self.fps,
        }


def _write_visual_status(message: str) -> None:
    """Write UTF-8 status without failing on a legacy Windows console code page."""

    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is not None:
            buffer.write((message + "\n").encode("utf-8"))
            buffer.flush()
        else:  # pragma: no cover - retained for redirected text-only streams
            print(message.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _pose_from_body(sim: Any, body_name: str) -> dict[str, list[float]]:
    body_id = sim.model.body_name2id(body_name)
    return {
        "body": body_name,
        "position_m": np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).tolist(),
        "rotation_matrix": np.asarray(sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3).tolist(),
    }


def _pose_from_site(sim: Any, site_name: str) -> dict[str, list[float]]:
    site_id = sim.model.site_name2id(site_name)
    return {
        "site": site_name,
        "position_m": np.asarray(sim.data.site_xpos[site_id], dtype=np.float64).tolist(),
        "rotation_matrix": np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).tolist(),
    }


def _relative_pose(parent: Mapping[str, Sequence[float]], child: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    parent_position = np.asarray(parent["position_m"], dtype=np.float64)
    parent_rotation = np.asarray(parent["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    child_position = np.asarray(child["position_m"], dtype=np.float64)
    child_rotation = np.asarray(child["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    return {
        "translation_m": (parent_rotation.T @ (child_position - parent_position)).tolist(),
        "rotation_matrix": (parent_rotation.T @ child_rotation).tolist(),
    }


def _look_at_quaternion(camera_position_m: np.ndarray, focus_position_m: np.ndarray) -> np.ndarray:
    """Return a MuJoCo camera quaternion whose local -Z looks at ``focus``."""

    forward = np.asarray(focus_position_m, dtype=np.float64) - np.asarray(camera_position_m, dtype=np.float64)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-9:
        raise ValueError("validation camera position must differ from its focus")
    forward /= forward_norm
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_hint))) > 0.95:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward))
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    return quaternion


def configure_validation_cameras(
    env: Any,
    *,
    palm_position_m: Sequence[float],
    palm_rotation: Sequence[float],
    focus_position_m: Sequence[float],
) -> dict[str, Any]:
    """Aim the three world-fixed validation cameras close to the grasp.

    This changes only MuJoCo camera metadata.  It does not alter a body pose,
    visual/collision geometry, mass, friction, or any control state.
    """

    model = env.sim.model
    palm_position = np.asarray(palm_position_m, dtype=np.float64)
    palm_rotation_matrix = np.asarray(palm_rotation, dtype=np.float64).reshape(3, 3)
    focus = np.asarray(focus_position_m, dtype=np.float64)
    # The offsets are defined in the compiled palm frame to retain front,
    # side, and top/oblique evidence even when the Panda pose changes.
    local_offsets = {
        "frontview": np.array([0.00, -0.34, 0.08], dtype=np.float64),
        "sideview": np.array([0.34, 0.00, 0.07], dtype=np.float64),
        "birdview": np.array([0.06, -0.03, 0.36], dtype=np.float64),
    }
    configured: dict[str, Any] = {}
    for camera_name, offset in local_offsets.items():
        camera_id = int(model.camera_name2id(camera_name))
        if int(model.cam_bodyid[camera_id]) != 0:
            raise RuntimeError(f"validation camera {camera_name} is not world-fixed")
        position = palm_position + palm_rotation_matrix @ offset
        quaternion = _look_at_quaternion(position, focus)
        model.cam_pos[camera_id] = position
        model.cam_quat[camera_id] = quaternion
        model.cam_fovy[camera_id] = 42.0
        configured[camera_name] = {
            "position_m": position.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "focus_position_m": focus.tolist(),
            "fovy_deg": 42.0,
        }
    env.sim.forward()
    return configured


def _named_geom(model: Any, geom_id: int) -> str:
    return model.geom_id2name(int(geom_id)) or f"geom_{int(geom_id)}"


def _named_body(model: Any, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[int(geom_id)])
    return model.body_id2name(body_id) or f"body_{body_id}"


def _is_cube(name: str) -> bool:
    return name.startswith("cube_")


def _is_table(name: str) -> bool:
    return "table" in name.lower()


def _is_hand(name: str) -> bool:
    return name.startswith("gripper0_right_")


def _is_panda(name: str) -> bool:
    return name.startswith("robot0_")


def _finger_for_geom(name: str) -> str | None:
    match = re.search(r"link_(\d+)\.0", name)
    if match is None:
        return None
    index = int(match.group(1))
    for finger, members in FINGER_GROUPS.items():
        if index in members:
            return finger
    return None


def _cube_to_hand_normal(raw_normal_world: Sequence[float], *, cube_is_geom1: bool) -> np.ndarray:
    """Normalize MuJoCo's geom1->geom2 normal into cube->hand direction."""

    normal = np.asarray(raw_normal_world, dtype=np.float64)
    if normal.shape != (3,):
        raise ValueError("contact normal must have three coordinates")
    return normal.copy() if cube_is_geom1 else -normal


@dataclass
class ContactSummary:
    cube_fingers: set[str]
    cube_contact_count: int = 0
    cube_hand_max_penetration_m: float = 0.0
    total_contacts: int = 0
    # ``table_contact`` remains the historical cube--table fact used by Gate
    # 2. Gate 3 additionally needs to distinguish normal resting support from
    # an unsafe arm / hand collision with the table.
    table_contact: bool = False
    hand_table_contact: bool = False
    panda_table_contact: bool = False
    hand_self_contact: bool = False
    hand_panda_contact: bool = False
    max_penetration_m: float = 0.0
    cube_hand_contacts: list[dict[str, Any]] | None = None
    normal_opposition: dict[str, Any] | None = None
    records: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.records is None:
            self.records = []
        if self.cube_hand_contacts is None:
            self.cube_hand_contacts = []


def collect_contacts(
    env: Any,
    *,
    gate: int,
    trial_id: str,
    phase: str,
    writer: ArtifactWriter | None = None,
) -> ContactSummary:
    """Read full MuJoCo contact evidence and derive conservative categories."""

    sim = env.sim
    model, data = sim.model, sim.data
    summary = ContactSummary(cube_fingers=set())
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        name1, name2 = _named_geom(model, geom1), _named_geom(model, geom2)
        dist = float(contact.dist)
        force = np.zeros(6, dtype=np.float64)
        try:
            mujoco.mj_contactForce(model._model, data._data, contact_index, force)
        except Exception:  # pragma: no cover - wrappers without raw objects are unsupported here
            pass
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        # MuJoCo documents contact.frame[0] as pointing from geom1 to geom2.
        # Keep this raw value, then normalize every cube--hand pair below so
        # normal comparisons cannot depend on MuJoCo's geom ordering.
        raw_normal = frame[0].copy()
        record = {
            "event": "contact",
            "gate": gate,
            "trial_id": trial_id,
            "phase": phase,
            "sim_time_s": float(data.time),
            "contact_index": contact_index,
            "geom1": name1,
            "geom2": name2,
            "body1": _named_body(model, geom1),
            "body2": _named_body(model, geom2),
            "position_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
            "normal_world": raw_normal.tolist(),
            "distance_m": dist,
            "force_contact_frame": force.tolist(),
            "normal_force": float(force[0]),
        }
        summary.records.append(record)
        summary.total_contacts += 1
        summary.max_penetration_m = min(summary.max_penetration_m, dist)
        pair = (name1, name2)
        is_cube_pair = (_is_cube(name1) and _is_hand(name2)) or (_is_cube(name2) and _is_hand(name1))
        if is_cube_pair:
            summary.cube_contact_count += 1
            hand_geom = name2 if _is_hand(name2) else name1
            finger = _finger_for_geom(hand_geom)
            cube_is_geom1 = _is_cube(name1)
            cube_to_hand = _cube_to_hand_normal(raw_normal, cube_is_geom1=cube_is_geom1)
            record.update({
                "cube_geom": name1 if cube_is_geom1 else name2,
                "hand_geom": hand_geom,
                "finger": finger,
                "cube_to_hand_normal_world": cube_to_hand.tolist(),
                "hand_to_cube_normal_world": (-cube_to_hand).tolist(),
                "normal_force_abs_n": float(abs(force[0])),
            })
            summary.cube_hand_max_penetration_m = min(summary.cube_hand_max_penetration_m, dist)
            summary.cube_hand_contacts.append({
                "finger": finger,
                "cube_geom": name1 if cube_is_geom1 else name2,
                "hand_geom": hand_geom,
                "cube_to_hand_normal_world": cube_to_hand.tolist(),
                "normal_force_abs_n": float(abs(force[0])),
                "distance_m": dist,
            })
            if finger is not None:
                summary.cube_fingers.add(finger)
        if (_is_cube(name1) and _is_table(name2)) or (_is_cube(name2) and _is_table(name1)):
            summary.table_contact = True
        if (_is_hand(name1) and _is_table(name2)) or (_is_hand(name2) and _is_table(name1)):
            summary.hand_table_contact = True
        if (_is_panda(name1) and _is_table(name2)) or (_is_panda(name2) and _is_table(name1)):
            summary.panda_table_contact = True
        if _is_hand(name1) and _is_hand(name2):
            summary.hand_self_contact = True
        if (_is_hand(name1) and _is_panda(name2)) or (_is_hand(name2) and _is_panda(name1)):
            summary.hand_panda_contact = True
    if writer is not None:
        writer.contacts(summary.records)
    return summary


def evaluate_cube_hand_opposition(
    summary: ContactSummary,
    *,
    dot_max: float,
    minimum_normal_force_n: float,
) -> dict[str, Any]:
    """Evaluate one control frame using cube-to-hand canonical normals.

    A physical pinch needs one thumb contact and two distinct non-thumb
    contacts whose normals point back toward the thumb-side contact.  This
    deliberately does not infer opposition merely from a set of finger names.
    """

    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in FINGER_GROUPS}
    for record in summary.cube_hand_contacts or []:
        finger = record.get("finger")
        if (
            finger in grouped
            and float(record.get("normal_force_abs_n", 0.0)) >= minimum_normal_force_n
        ):
            grouped[str(finger)].append(record)
    thumb_contacts = grouped["thumb"]
    pairs: list[dict[str, Any]] = []
    opposed_fingers: list[str] = []
    per_finger_min_dot: dict[str, float] = {}
    for finger in ("index", "middle", "ring"):
        best_pair: dict[str, Any] | None = None
        for thumb in thumb_contacts:
            thumb_normal = np.asarray(thumb["cube_to_hand_normal_world"], dtype=np.float64)
            for other in grouped[finger]:
                other_normal = np.asarray(other["cube_to_hand_normal_world"], dtype=np.float64)
                dot = float(np.clip(np.dot(thumb_normal, other_normal), -1.0, 1.0))
                pair = {
                    "thumb_geom": thumb["hand_geom"],
                    "finger": finger,
                    "finger_geom": other["hand_geom"],
                    "thumb_normal_world": thumb_normal.tolist(),
                    "finger_normal_world": other_normal.tolist(),
                    "dot": dot,
                    "thumb_normal_force_abs_n": float(thumb["normal_force_abs_n"]),
                    "finger_normal_force_abs_n": float(other["normal_force_abs_n"]),
                }
                if best_pair is None or dot < float(best_pair["dot"]):
                    best_pair = pair
        if best_pair is not None:
            per_finger_min_dot[finger] = float(best_pair["dot"])
            pairs.append(best_pair)
            if float(best_pair["dot"]) <= dot_max:
                opposed_fingers.append(finger)
    digits_present = sorted(
        finger for finger in ("index", "middle", "ring") if grouped[finger]
    )
    return {
        "dot_max": float(dot_max),
        "minimum_normal_force_n": float(minimum_normal_force_n),
        "thumb_contact_count": len(thumb_contacts),
        "nonthumb_fingers_with_force": digits_present,
        "opposed_nonthumb_fingers": opposed_fingers,
        "pair_minimum_dots": per_finger_min_dot,
        "pairs": pairs,
        "has_thumb_and_two_digits": bool(thumb_contacts and len(digits_present) >= 2),
        "valid": bool(thumb_contacts and len(opposed_fingers) >= 2),
    }


def _cube_handles(env: Any) -> dict[str, int]:
    model = env.sim.model
    joint_id = int(model.joint_name2id("cube_joint0"))
    return {
        "joint_id": joint_id,
        "qpos_start": int(model.jnt_qposadr[joint_id]),
        "qvel_start": int(model.jnt_dofadr[joint_id]),
        "body_id": int(model.body_name2id("cube_main")),
        "geom_id": int(model.geom_name2id("cube_g0")),
    }


def _cube_state(env: Any, handles: Mapping[str, int]) -> dict[str, Any]:
    sim = env.sim
    model, data = sim.model, sim.data
    body_id = int(handles["body_id"])
    qvel_start = int(handles["qvel_start"])
    return {
        "position_m": np.asarray(data.body_xpos[body_id], dtype=np.float64).tolist(),
        "rotation_matrix": np.asarray(data.body_xmat[body_id], dtype=np.float64).reshape(3, 3).tolist(),
        "linear_velocity_mps": np.asarray(data.qvel[qvel_start:qvel_start + 3], dtype=np.float64).tolist(),
        "angular_velocity_radps": np.asarray(data.qvel[qvel_start + 3:qvel_start + 6], dtype=np.float64).tolist(),
        "mass_kg": float(model.body_mass[body_id]),
        "geom_half_size_m": np.asarray(model.geom_size[int(handles["geom_id"])], dtype=np.float64).tolist(),
        "friction": np.asarray(model.geom_friction[int(handles["geom_id"])], dtype=np.float64).tolist(),
    }


def set_cube_pose(env: Any, handles: Mapping[str, int], position_m: Sequence[float], quaternion_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0)) -> None:
    """Perform one recorded initial placement before a trial, never during evaluation."""

    position = np.asarray(position_m, dtype=np.float64)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("cube placement requires finite xyz and wxyz values")
    sim = env.sim
    qpos_start, qvel_start = int(handles["qpos_start"]), int(handles["qvel_start"])
    sim.data.qpos[qpos_start:qpos_start + 3] = position
    sim.data.qpos[qpos_start + 3:qpos_start + 7] = quaternion / np.linalg.norm(quaternion)
    sim.data.qvel[qvel_start:qvel_start + 6] = 0.0
    sim.forward()


class TemporaryCubeSupport:
    """Temporary external virtual fixture used only while initially closing.

    This is a force-based placement support, not a weld and not an object
    teleport. Its wrench is re-evaluated at every MuJoCo physics substep and
    set to zero before the gravity-on evaluation begins. The configured target
    is fixed in the world frame and never follows Panda motion.
    """

    def __init__(self, env: Any, handles: Mapping[str, int], target_position_m: Sequence[float], config: GraspValidationConfig):
        self.env = env
        self.handles = dict(handles)
        self.target = np.asarray(target_position_m, dtype=np.float64)
        self.config = config
        self.active = True
        self.force_scale = 1.0
        sim = self.env.sim
        model = sim.model
        body_id = int(self.handles["body_id"])
        self.mass_kg = float(model.body_mass[body_id])
        self.physics_timestep_s = float(model.opt.timestep)
        self.control_timestep_s = float(getattr(env, "control_timestep", 1.0 / config.control_hz))
        self.natural_frequency_radps = 2.0 * np.pi * float(config.temporary_support_natural_frequency_hz)
        self.kp_n_per_m = self.mass_kg * self.natural_frequency_radps**2
        self.kd_n_per_mps = 2.0 * float(config.temporary_support_damping_ratio) * self.mass_kg * self.natural_frequency_radps
        self.max_correction_force_n = self.mass_kg * float(config.temporary_support_max_correction_accel_mps2)
        self.expected_substeps_per_control = max(1, int(round(self.control_timestep_s / self.physics_timestep_s)))
        self._writer: ArtifactWriter | None = None
        self._context: dict[str, Any] = {}
        self._control_samples: list[dict[str, Any]] = []
        self._total_samples = 0
        self._clamped_samples = 0
        self._max_raw_correction_force_n = 0.0
        self._max_applied_force_n = 0.0
        self._max_cube_speed_mps = 0.0
        self._max_displacement_m = 0.0

    def begin_control_step(self, writer: ArtifactWriter, *, gate: int, trial_id: str, phase: str) -> None:
        self._writer = writer
        self._context = {"gate": gate, "trial_id": trial_id, "phase": phase}
        self._control_samples = []

    def apply_substep(self) -> None:
        """Apply a fresh translation-only fixture force for one physics step."""

        if not self.active:
            return
        sim = self.env.sim
        model, data = sim.model, sim.data
        body_id = int(self.handles["body_id"])
        position = np.asarray(data.body_xpos[body_id], dtype=np.float64)
        qvel_start = int(self.handles["qvel_start"])
        velocity = np.asarray(data.qvel[qvel_start:qvel_start + 3], dtype=np.float64)
        displacement = self.target - position
        raw_correction = self.kp_n_per_m * displacement - self.kd_n_per_mps * velocity
        raw_correction_norm = float(np.linalg.norm(raw_correction))
        correction = raw_correction.copy()
        clamped = raw_correction_norm > self.max_correction_force_n
        if clamped:
            correction *= self.max_correction_force_n / raw_correction_norm
        # Gravity remains enabled. This external term cancels only the cube's
        # weight while the fixture is active; it does not lock position or
        # orientation and it is removed outright at release.
        gravity_compensation = -self.mass_kg * np.asarray(model.opt.gravity, dtype=np.float64)
        unscaled_force = correction + gravity_compensation
        applied_force = self.force_scale * unscaled_force
        data.xfrc_applied[body_id, :] = 0.0
        data.xfrc_applied[body_id, :3] = applied_force
        sample = {
            "event": "fixture_substep",
            **self._context,
            "sim_time_s": float(data.time),
            "substep_index_in_control": len(self._control_samples) + 1,
            "physics_timestep_s": self.physics_timestep_s,
            "control_timestep_s": self.control_timestep_s,
            "expected_substeps_per_control": self.expected_substeps_per_control,
            "cube_mass_kg": self.mass_kg,
            "natural_frequency_radps": self.natural_frequency_radps,
            "damping_ratio": float(self.config.temporary_support_damping_ratio),
            "kp_n_per_m": self.kp_n_per_m,
            "kd_n_per_mps": self.kd_n_per_mps,
            "max_correction_force_n": self.max_correction_force_n,
            "support_force_scale": self.force_scale,
            "displacement_to_target_m": displacement.tolist(),
            "displacement_norm_m": float(np.linalg.norm(displacement)),
            "cube_linear_velocity_mps": velocity.tolist(),
            "cube_speed_mps": float(np.linalg.norm(velocity)),
            "raw_correction_force_n": raw_correction.tolist(),
            "raw_correction_force_norm_n": raw_correction_norm,
            "gravity_compensation_force_n": gravity_compensation.tolist(),
            "unscaled_support_force_n": unscaled_force.tolist(),
            "applied_support_force_n": applied_force.tolist(),
            "applied_support_force_norm_n": float(np.linalg.norm(applied_force)),
            "correction_force_clamped": bool(clamped),
        }
        self._control_samples.append(sample)
        self._total_samples += 1
        self._clamped_samples += int(clamped)
        self._max_raw_correction_force_n = max(self._max_raw_correction_force_n, raw_correction_norm)
        self._max_applied_force_n = max(self._max_applied_force_n, float(np.linalg.norm(applied_force)))
        self._max_cube_speed_mps = max(self._max_cube_speed_mps, float(np.linalg.norm(velocity)))
        self._max_displacement_m = max(self._max_displacement_m, float(np.linalg.norm(displacement)))
        if self._writer is not None:
            self._writer.event(sample)

    @contextmanager
    def substep_context(self, writer: ArtifactWriter, *, gate: int, trial_id: str, phase: str) -> Iterable[None]:
        """Wrap only this environment's pre-action hook for one ``env.step``.

        robosuite calls ``_pre_action`` once per physics substep. Calling the
        original hook first preserves its controller behavior; the fixture is
        then updated before MuJoCo integrates that same 2 ms substep.
        """

        self.begin_control_step(writer, gate=gate, trial_id=trial_id, phase=phase)
        original_pre_action = self.env._pre_action

        def support_pre_action(action: Any, policy_step: bool = False) -> None:
            original_pre_action(action, policy_step)
            self.apply_substep()

        self.env._pre_action = support_pre_action
        try:
            yield
        finally:
            self.env._pre_action = original_pre_action
            self.clear_applied_force()

    def control_step_diagnostics(self) -> dict[str, Any]:
        samples = self._control_samples
        return {
            "substep_count": len(samples),
            "expected_substeps": self.expected_substeps_per_control,
            "substep_count_matches_expected": len(samples) == self.expected_substeps_per_control,
            "max_raw_correction_force_n": max((float(item["raw_correction_force_norm_n"]) for item in samples), default=0.0),
            "max_applied_support_force_n": max((float(item["applied_support_force_norm_n"]) for item in samples), default=0.0),
            "max_cube_speed_mps": max((float(item["cube_speed_mps"]) for item in samples), default=0.0),
            "max_displacement_m": max((float(item["displacement_norm_m"]) for item in samples), default=0.0),
            "correction_force_clamped": any(bool(item["correction_force_clamped"]) for item in samples),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "type": "fixed-world external-force virtual fixture",
            "target_position_m": self.target.tolist(),
            "mass_kg": self.mass_kg,
            "physics_timestep_s": self.physics_timestep_s,
            "control_timestep_s": self.control_timestep_s,
            "expected_substeps_per_control": self.expected_substeps_per_control,
            "natural_frequency_hz": float(self.config.temporary_support_natural_frequency_hz),
            "natural_frequency_radps": self.natural_frequency_radps,
            "damping_ratio": float(self.config.temporary_support_damping_ratio),
            "kp_n_per_m": self.kp_n_per_m,
            "kd_n_per_mps": self.kd_n_per_mps,
            "max_correction_accel_mps2": float(self.config.temporary_support_max_correction_accel_mps2),
            "max_correction_force_n": self.max_correction_force_n,
            "release_ramp_duration_s": float(self.config.temporary_support_release_ramp_s),
            "total_physics_substeps": self._total_samples,
            "clamped_substeps": self._clamped_samples,
            "max_raw_correction_force_n": self._max_raw_correction_force_n,
            "max_applied_support_force_n": self._max_applied_force_n,
            "max_cube_speed_mps": self._max_cube_speed_mps,
            "max_displacement_m": self._max_displacement_m,
        }

    def clear_applied_force(self) -> None:
        self.env.sim.data.xfrc_applied[int(self.handles["body_id"]), :] = 0.0

    def set_force_scale(self, value: float) -> None:
        if not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("temporary support force scale must be finite in [0, 1]")
        self.force_scale = float(value)

    def release(self) -> None:
        self.active = False
        self.force_scale = 0.0
        self.clear_applied_force()


@dataclass
class Gate2KeyframeTracker:
    """Keep only the Gate 2 visual evidence that explains a trial result."""

    writer: ArtifactWriter
    trial_id: str
    first_contact_captured: bool = False
    deepest_cube_hand_penetration_m: float = 0.0

    def observe(self, env: Any, contacts: ContactSummary, *, phase: str, sim_time: float) -> None:
        prefix = f"g2_{self.trial_id}"
        if contacts.cube_contact_count and not self.first_contact_captured:
            self.writer.capture_validation_views(env, f"{prefix}_first_contact", sim_time=sim_time)
            self.first_contact_captured = True
        penetration = float(contacts.cube_hand_max_penetration_m)
        if penetration < self.deepest_cube_hand_penetration_m - 1e-9:
            self.deepest_cube_hand_penetration_m = penetration
            # A stable file name means only the actual deepest frame survives.
            self.writer.capture_validation_views(
                env,
                f"{prefix}_max_penetration",
                sim_time=sim_time,
                overwrite=True,
            )


def _axis_angle_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Stable world-frame axis-angle vector for a proper 3x3 rotation."""

    trace = float(np.trace(rotation))
    cosine = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
        dtype=np.float64,
    )
    denominator = 2.0 * np.sin(angle)
    if abs(denominator) < 1e-7:
        # The Gate trajectory does not command a 180-degree reorientation.
        # Treat this unusual condition as a bounded error vector instead of
        # producing NaNs in a safety experiment.
        return np.zeros(3, dtype=np.float64)
    return axis / denominator * angle


def _quaternion_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper world rotation matrix to MuJoCo's wxyz quaternion."""

    values = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, values.reshape(-1))
    return quaternion


def _ee_pose(composer: PandaAllegroActionComposer) -> tuple[np.ndarray, np.ndarray]:
    pose = composer.ee_pose()
    return np.asarray(pose["position_m"], dtype=np.float64), np.asarray(pose["rotation_matrix"], dtype=np.float64).reshape(3, 3)


def _actuator_saturation(env: Any, composer: PandaAllegroActionComposer, fraction: float) -> bool:
    model, data = env.sim.model, env.sim.data
    actuator_ids = np.asarray(composer.robot._ref_joint_gripper_actuator_indexes["right"], dtype=np.intp)
    ranges = np.asarray(model.actuator_forcerange[actuator_ids], dtype=np.float64)
    force = np.asarray(data.actuator_force[actuator_ids], dtype=np.float64)
    limits = np.max(np.abs(ranges), axis=1)
    valid = limits > 1e-8
    return bool(np.any(np.abs(force[valid]) >= fraction * limits[valid]))


def _unexpected_hand_collision(summary: ContactSummary) -> bool:
    return bool(summary.hand_self_contact or summary.hand_panda_contact)


def _hand_at_limit(composer: PandaAllegroActionComposer, margin_rad: float) -> bool:
    qpos = composer.hand_qpos()
    return bool(np.any(qpos <= composer.hand_lower_limits + margin_rad) or np.any(qpos >= composer.hand_upper_limits - margin_rad))


def _hand_targets(composer: PandaAllegroActionComposer, curls: Sequence[float]) -> np.ndarray:
    return np.asarray(
        curls_to_allegro_targets(curls, composer.hand_lower_limits, composer.hand_upper_limits),
        dtype=np.float64,
    )


def _step(
    env: Any,
    composer: PandaAllegroActionComposer,
    *,
    hand_targets: Sequence[float],
    wrist_delta: Sequence[float],
    gate: int,
    trial_id: str,
    phase: str,
    writer: ArtifactWriter,
    config: GraspValidationConfig,
    support: TemporaryCubeSupport | None = None,
    target_pose: tuple[np.ndarray, np.ndarray] | None = None,
    keyframes: Gate2KeyframeTracker | None = None,
    step_callback: Callable[[Any, ContactSummary, Mapping[str, Any]], None] | None = None,
) -> tuple[ContactSummary, dict[str, Any]]:
    action = composer.compose(wrist_delta, hand_targets)
    if support is None:
        env.step(action)
    else:
        with support.substep_context(writer, gate=gate, trial_id=trial_id, phase=phase):
            env.step(action)
    fixture_diagnostics = support.control_step_diagnostics() if support is not None else None
    contacts = collect_contacts(env, gate=gate, trial_id=trial_id, phase=phase, writer=writer)
    contacts.normal_opposition = evaluate_cube_hand_opposition(
        contacts,
        dot_max=config.normal_opposition_dot_max,
        minimum_normal_force_n=config.normal_contact_min_force_n,
    )
    ee_position, ee_rotation = _ee_pose(composer)
    event: dict[str, Any] = {
        "event": "step",
        "gate": gate,
        "trial_id": trial_id,
        "phase": phase,
        "sim_time_s": float(env.sim.data.time),
        "osc_action": np.asarray(action[composer.arm_slice], dtype=np.float64).tolist(),
        "hand_action": np.asarray(action[composer.hand_slice], dtype=np.float64).tolist(),
        "ee_position_m": ee_position.tolist(),
        "ee_rotation_matrix": ee_rotation.tolist(),
        "hand_target_rad": np.asarray(hand_targets, dtype=np.float64).tolist(),
        "hand_actual_rad": composer.hand_qpos().tolist(),
        "actuator_force": np.asarray(
            env.sim.data.actuator_force[np.asarray(composer.robot._ref_joint_gripper_actuator_indexes["right"], dtype=np.intp)],
            dtype=np.float64,
        ).tolist(),
        "panda_joint_positions_rad": np.asarray(composer.robot._joint_positions, dtype=np.float64).tolist(),
        "cube_fingers": sorted(contacts.cube_fingers),
        "cube_contact_count": contacts.cube_contact_count,
        "cube_hand_max_penetration_m": contacts.cube_hand_max_penetration_m,
        "table_contact": contacts.table_contact,
        "max_penetration_m": contacts.max_penetration_m,
        "support_active": bool(support is not None and support.active),
        "normal_opposition": contacts.normal_opposition,
    }
    if fixture_diagnostics is not None:
        event["fixture"] = fixture_diagnostics
    try:
        cube = _cube_state(env, _cube_handles(env))
        event.update(
            cube_position_m=cube["position_m"],
            cube_linear_velocity_mps=cube["linear_velocity_mps"],
            cube_angular_velocity_radps=cube["angular_velocity_radps"],
        )
    except Exception:
        # Gate construction is explicitly based on Lift's standard cube. The
        # narrow fallback keeps lower-level unit tests usable with tiny fakes.
        pass
    if target_pose is not None:
        target_position, target_rotation = target_pose
        rotation_error = _axis_angle_from_rotation(target_rotation @ ee_rotation.T)
        event.update(
            ee_target_position_m=np.asarray(target_position, dtype=np.float64).tolist(),
            ee_target_rotation_matrix=np.asarray(target_rotation, dtype=np.float64).tolist(),
            ee_position_error_m=float(np.linalg.norm(np.asarray(target_position) - ee_position)),
            ee_orientation_error_rad=float(np.linalg.norm(rotation_error)),
        )
    if keyframes is not None:
        keyframes.observe(env, contacts, phase=phase, sim_time=float(env.sim.data.time))
    writer.event(event)
    if step_callback is not None:
        step_callback(env, contacts, event)
    return contacts, event


def _run_for(
    env: Any,
    composer: PandaAllegroActionComposer,
    *,
    duration_s: float,
    hand_targets: np.ndarray,
    gate: int,
    trial_id: str,
    phase: str,
    writer: ArtifactWriter,
    config: GraspValidationConfig,
    support: TemporaryCubeSupport | None = None,
    target_pose: tuple[np.ndarray, np.ndarray] | None = None,
    keyframes: Gate2KeyframeTracker | None = None,
    step_callback: Callable[[Any, ContactSummary, Mapping[str, Any]], None] | None = None,
) -> list[tuple[ContactSummary, dict[str, Any]]]:
    steps = max(1, int(round(duration_s * config.control_hz)))
    result: list[tuple[ContactSummary, dict[str, Any]]] = []
    for _ in range(steps):
        result.append(_step(
            env, composer, hand_targets=hand_targets, wrist_delta=np.zeros(6), gate=gate,
            trial_id=trial_id, phase=phase, writer=writer, config=config, support=support,
            target_pose=target_pose, keyframes=keyframes, step_callback=step_callback,
        ))
    return result


def _merge_contact_metrics(records: Iterable[tuple[ContactSummary, Mapping[str, Any]]]) -> dict[str, Any]:
    fingers: set[str] = set()
    max_penetration = 0.0
    cube_hand_max_penetration = 0.0
    table_contact = False
    hand_table_contact = False
    panda_table_contact = False
    unexpected_collision = False
    cube_contacts = 0
    for summary, _ in records:
        fingers.update(summary.cube_fingers)
        max_penetration = min(max_penetration, summary.max_penetration_m)
        cube_hand_max_penetration = min(cube_hand_max_penetration, summary.cube_hand_max_penetration_m)
        table_contact = table_contact or summary.table_contact
        hand_table_contact = hand_table_contact or summary.hand_table_contact
        panda_table_contact = panda_table_contact or summary.panda_table_contact
        unexpected_collision = unexpected_collision or _unexpected_hand_collision(summary)
        cube_contacts += summary.cube_contact_count
    return {
        "cube_fingers": sorted(fingers),
        "cube_contacts": cube_contacts,
        "max_penetration_m": max_penetration,
        "cube_hand_max_penetration_m": cube_hand_max_penetration,
        "table_contact": table_contact,
        "hand_table_contact": hand_table_contact,
        "panda_table_contact": panda_table_contact,
        "unexpected_hand_collision": unexpected_collision,
    }


def build_environment_snapshot(env: Any, composer: PandaAllegroActionComposer, config: GraspValidationConfig) -> dict[str, Any]:
    """Capture the exact compiled API, transforms, model and cube facts."""

    sim = env.sim
    model, data = sim.model, sim.data
    handles = _cube_handles(env)
    hand_qpos_ids = np.asarray(composer.robot._ref_gripper_joint_pos_indexes["right"], dtype=np.intp)
    hand_actuator_ids = np.asarray(composer.robot._ref_joint_gripper_actuator_indexes["right"], dtype=np.intp)
    flange = _pose_from_body(sim, "robot0_right_hand")
    palm = _pose_from_body(sim, "gripper0_right_allegro_right_hand")
    tcp = _pose_from_site(sim, composer.hand.important_sites["grip_site"])
    joint_rows: list[dict[str, Any]] = []
    for index, (name, qpos_id, actuator_id) in enumerate(zip(composer.hand_joint_names, hand_qpos_ids, hand_actuator_ids)):
        joint_id = int(model.actuator_trnid[int(actuator_id), 0])
        joint_rows.append({
            "source_index": index,
            "source_name": name,
            "compiled_joint": model.joint_id2name(joint_id),
            "qpos_index": int(qpos_id),
            "actuator_index": int(actuator_id),
            "actuator_name": model.actuator_id2name(int(actuator_id)),
            "lower_rad": float(composer.hand_lower_limits[index]),
            "upper_rad": float(composer.hand_upper_limits[index]),
            "actual_rad": float(data.qpos[int(qpos_id)]),
            "ctrlrange_rad": np.asarray(model.actuator_ctrlrange[int(actuator_id)], dtype=np.float64).tolist(),
            "forcerange_nm": np.asarray(model.actuator_forcerange[int(actuator_id)], dtype=np.float64).tolist(),
            "gainprm": np.asarray(model.actuator_gainprm[int(actuator_id)], dtype=np.float64).tolist(),
            "biasprm": np.asarray(model.actuator_biasprm[int(actuator_id)], dtype=np.float64).tolist(),
        })
    try:
        import robosuite
        robosuite_version = getattr(robosuite, "__version__", "unknown")
    except Exception:  # pragma: no cover - environment construction already establishes this import
        robosuite_version = "unknown"
    cube = _cube_state(env, handles)
    table_site_id = int(model.site_name2id("table_top"))
    return {
        "config": config.to_dict(),
        "config_sha256": config.sha256,
        "tuning_history": list(TUNING_HISTORY),
        "environment": "robosuite Lift + Panda + custom AllegroRightHand",
        "versions": {
            "python": sys.version.split()[0],
            "robosuite": robosuite_version,
            "mujoco": getattr(mujoco, "__version__", "unknown"),
            "allegro_source_urdf": str(SOURCE_URDF),
            "allegro_source_urdf_sha256": sha256_file(SOURCE_URDF),
        },
        "random_seed": config.seed,
        "control": {
            "control_hz": config.control_hz,
            "mujoco_timestep_s": float(model.opt.timestep),
            "gravity_mps2": np.asarray(model.opt.gravity, dtype=np.float64).tolist(),
            "action_dim": int(env.action_dim),
            "action_spec": {"low": np.asarray(env.action_spec[0], dtype=np.float64).tolist(), "high": np.asarray(env.action_spec[1], dtype=np.float64).tolist()},
            "arm_slice": [composer.arm_slice.start, composer.arm_slice.stop],
            "hand_slice": [composer.hand_slice.start, composer.hand_slice.stop],
            "controller": composer.describe(),
        },
        "panda_initial": {
            "joint_positions_rad": np.asarray(composer.robot._joint_positions, dtype=np.float64).tolist(),
            "ee": {"position_m": _ee_pose(composer)[0].tolist(), "rotation_matrix": _ee_pose(composer)[1].tolist()},
        },
        "allegro": {
            "joint_order": list(ALLEGRO_JOINT_NAMES),
            "joint_rows": joint_rows,
            "safe_open_targets_rad": _hand_targets(composer, (0.0, 0.0, 0.0, 0.0)).tolist(),
            "gate2_close_targets_rad": _hand_targets(composer, config.static_close_curls).tolist(),
            "flange": flange,
            "palm": palm,
            "osc_wrist_tcp": tcp,
            "palm_from_flange": _relative_pose(flange, palm),
            "osc_wrist_tcp_from_flange": _relative_pose(flange, tcp),
            "osc_wrist_tcp_from_palm": _relative_pose(palm, tcp),
            "grasp_center_in_osc_tcp_m": list(config.static_cube_center_tcp_m),
            "tcp_note": "gripper0_right_grip_site is the existing OSC wrist reference; static grasp placement is recorded separately in that frame.",
        },
        "cube": {
            "free_joint": "cube_joint0",
            "body": "cube_main",
            "collision_geom": "cube_g0",
            "initial": cube,
            "table_top_position_m": np.asarray(data.site_xpos[table_site_id], dtype=np.float64).tolist(),
        },
    }


def classify_gate2(metrics: Mapping[str, Any], config: GraspValidationConfig = DEFAULT_CONFIG) -> tuple[bool, str | None]:
    """Return Gate 2 success and exactly one permitted failure category."""

    if bool(metrics.get("model_collision", False)):
        return False, "model_collision"
    if (
        float(metrics.get("initial_cube_hand_penetration_m", 0.0)) < -config.significant_penetration_m
        or int(metrics.get("initial_cube_hand_contact_count", 0)) > 0
        or float(metrics.get("cube_hand_max_penetration_m", metrics.get("max_penetration_m", 0.0))) < -config.significant_penetration_m
    ):
        return False, "penetration_artifact"
    if float(metrics.get("actuator_saturation_s", 0.0)) >= config.sustained_saturation_s:
        return False, "actuator_saturation"
    if float(metrics.get("joint_limit_s", 0.0)) >= config.sustained_saturation_s:
        return False, "joint_limit"
    if float(metrics.get("hand_target_error_rad", 0.0)) > config.joint_target_error_rad:
        return False, "joint_tracking_error"
    if bool(metrics.get("table_contact", False)) or bool(metrics.get("object_ejected", False)):
        return False, "object_ejected"
    if int(metrics.get("cube_contacts", 0)) <= 0:
        return False, "no_contact"
    fingers = set(metrics.get("cube_fingers", ()))
    if (
        "thumb" not in fingers
        or len(fingers & {"index", "middle", "ring"}) < 2
        or not bool(metrics.get("opposition_seen", False))
        or float(metrics.get("opposition_hold_s", 0.0)) + 1e-9 < config.static_hold_s
    ):
        return False, "insufficient_opposition"
    if float(metrics.get("slip_distance_m", 0.0)) > config.static_slip_limit_m:
        return False, "object_slip"
    if bool(metrics.get("contact_lost_after_opposition", False)):
        return False, "unstable_contact"
    flags = metrics.get("diagnostic_flags", {})
    if isinstance(flags, Mapping) and bool(flags.get("fixture_unstable", False)):
        return False, "unstable_contact"
    if not bool(metrics.get("support_released", False)) or not bool(metrics.get("gravity_enabled", False)):
        return False, "unknown"
    if float(metrics.get("hold_duration_s", 0.0)) + 1e-9 < config.static_hold_s:
        return False, "timeout"
    return True, None


def classify_gate3(metrics: Mapping[str, Any], config: GraspValidationConfig = DEFAULT_CONFIG) -> tuple[bool, str | None]:
    """Return Gate 3 success and exactly one permitted failure category."""

    if bool(metrics.get("pregrasp_pose_error", False)):
        return False, "pregrasp_pose_error"
    if bool(metrics.get("approach_collision", False)):
        return False, "approach_collision"
    if bool(metrics.get("osc_tracking_error", False)):
        return False, "osc_tracking_error"
    if bool(metrics.get("timed_out", False)):
        return False, "timeout"
    if int(metrics.get("cube_contacts", 0)) <= 0:
        return False, "missed_object"
    if not bool(metrics.get("hand_closed", False)):
        return False, "grasp_not_closed"
    if bool(metrics.get("contact_lost_during_lift", False)):
        return False, "contact_lost_during_lift"
    if float(metrics.get("slip_distance_m", 0.0)) > config.static_slip_limit_m:
        return False, "object_slip"
    if float(metrics.get("lift_m", 0.0)) < config.lift_success_m:
        return False, "insufficient_lift"
    if bool(metrics.get("table_contact_after_lift", False)) or float(metrics.get("hold_duration_s", 0.0)) < config.lift_hold_s:
        return False, "hold_failure"
    return True, None


def aggregate_protocol(trials: Sequence[Mapping[str, Any]], *, config_sha256: str, config: GraspValidationConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Evaluate single-success -> 3/3 qualification -> locked 10-trial policy."""

    if any(str(trial.get("config_sha256")) != config_sha256 for trial in trials):
        return {"passed": False, "failure_category": "unknown", "reason": "configuration hash changed during protocol"}
    single = [trial for trial in trials if str(trial.get("protocol_phase")) == "single"]
    qualifying = [trial for trial in trials if str(trial.get("protocol_phase")) == "qualification"]
    formal = [trial for trial in trials if str(trial.get("protocol_phase")) == "formal"]
    single_ok = len(single) == 1 and bool(single[0].get("success"))
    qualification_ok = len(qualifying) == config.qualification_trials and all(bool(trial.get("success")) for trial in qualifying)
    formal_successes = sum(bool(trial.get("success")) for trial in formal)
    formal_ok = len(formal) == config.formal_trials and formal_successes >= config.required_formal_successes
    return {
        "passed": bool(single_ok and qualification_ok and formal_ok),
        "single_success": single_ok,
        "qualification_successes": sum(bool(trial.get("success")) for trial in qualifying),
        "qualification_required": config.qualification_trials,
        "formal_successes": formal_successes,
        "formal_trials": len(formal),
        "formal_required_successes": config.required_formal_successes,
        "failure_category": None if single_ok and qualification_ok and formal_ok else "protocol_not_qualified",
    }


class GraspValidationRunner:
    """One hardware-free runner for the ordered Gate 1--3 validation."""

    def __init__(self, config: GraspValidationConfig = DEFAULT_CONFIG, *, output_root: Path | None = None, record_frames: bool = True):
        self.config = config
        self.output_root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
        self.record_frames = bool(record_frames)

    def _new_environment(self, *, has_renderer: bool = False) -> Any:
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        wrist_config = WristConfig(control_hz=self.config.control_hz, smoothing_alpha=None)
        return make_allegro_panda_env(
            wrist_config,
            has_renderer=has_renderer,
            has_offscreen_renderer=self.record_frames,
            initialization_noise=None,
            hard_reset=not self.config.reuse_compiled_model_across_trials,
            seed=self.config.seed,
        )

    def _reset(self, env: Any) -> PandaAllegroActionComposer:
        env.reset()
        robot = env.robots[0]
        qvel_ids = np.asarray(robot._ref_joint_vel_indexes, dtype=np.intp)
        initial_joints = np.asarray(self.config.panda_initial_joint_positions_rad, dtype=np.float64)
        robot.set_robot_joint_positions(initial_joints)
        env.sim.data.qvel[qvel_ids] = 0.0
        env.sim.forward()
        composer = PandaAllegroActionComposer(env)
        # Refresh OSC's cached reference and null-space target after writing
        # the scripted fixed pose, then latch a zero-delta goal there.
        composer.controller.update_initial_joints(initial_joints)
        composer.controller.update(force=True)
        composer.initialize_goal()
        return composer

    def _output_directory(self, suffix: str) -> Path:
        base = self.output_root / f"{_utc_stamp()}_{suffix}"
        candidate, index = base, 1
        while candidate.exists():
            candidate = self.output_root / f"{base.name}_{index}"
            index += 1
        return candidate

    def run(self, gate: int | str = "all") -> tuple[int, Path, dict[str, Any]]:
        if gate == "all":
            max_gate = 3
        else:
            max_gate = int(gate)
            if max_gate not in (1, 2, 3):
                raise ValueError("gate must be 1, 2, 3, or 'all'")
        output = self._output_directory(f"gate{max_gate}")
        writer = ArtifactWriter(
            output,
            config_sha256=self.config.sha256,
            frame_every_s=self.config.frame_every_s,
            record_frames=self.record_frames,
        )
        env = None
        summary: dict[str, Any] = {"requested_gate": gate, "config_sha256": self.config.sha256, "gates": {}}
        exit_code = 1
        try:
            writer.log(f"starting Gate validation requested_gate={gate}")
            env = self._new_environment()
            composer = self._reset(env)
            snapshot = build_environment_snapshot(env, composer, self.config)
            writer.write_json("config_snapshot.json", snapshot)
            writer.write_json("environment_versions.json", snapshot["versions"])
            gate1 = self.run_gate1(env, writer)
            summary["gates"]["gate1"] = gate1
            if not gate1["passed"]:
                summary["status"] = "failed_gate1"
                exit_code = 2
                return exit_code, output, summary
            if max_gate == 1:
                summary["status"] = "passed_gate1"
                exit_code = 0
                return exit_code, output, summary
            gate2 = self.run_gate2_protocol(env, writer)
            summary["gates"]["gate2"] = gate2
            if not gate2["passed"]:
                summary["status"] = "failed_gate2"
                exit_code = 2
                return exit_code, output, summary
            if max_gate == 2:
                summary["status"] = "passed_gate2"
                exit_code = 0
                return exit_code, output, summary
            gate3 = self.run_gate3_protocol(env, writer)
            summary["gates"]["gate3"] = gate3
            if not gate3["passed"]:
                summary["status"] = "failed_gate3"
                exit_code = 2
                return exit_code, output, summary
            summary["status"] = "passed_gate3"
            exit_code = 0
            return exit_code, output, summary
        except Exception as exc:
            summary["status"] = "runner_exception"
            summary["exception"] = f"{type(exc).__name__}: {exc}"
            writer.log(f"exception {summary['exception']}")
            writer.event({"event": "exception", "error": summary["exception"]})
            exit_code = 1
            return exit_code, output, summary
        finally:
            summary["frame_paths"] = list(writer.frame_paths)
            summary["frame_capture_error"] = writer.frame_error
            summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            writer.write_json("summary.json", summary)
            writer.close()
            if env is not None:
                env.close()

    def run_gate1(self, env: Any, writer: ArtifactWriter) -> dict[str, Any]:
        """Probe every compiled hand joint from an interior pose in both directions."""

        composer = self._reset(env)
        trial_id = "gate1_model_audit"
        open_target = _hand_targets(composer, (0.0, 0.0, 0.0, 0.0))
        initial_records = _run_for(
            env, composer, duration_s=self.config.open_settle_s, hand_targets=open_target, gate=1,
            trial_id=trial_id, phase="safe_open", writer=writer, config=self.config,
        )
        initial = _merge_contact_metrics(initial_records)
        writer.capture(env, "g1_safe_open", sim_time=float(env.sim.data.time), force=True)
        lower, upper = composer.hand_lower_limits, composer.hand_upper_limits
        neutral = (lower + upper) * 0.5
        neutral_records = _run_for(
            env, composer, duration_s=self.config.probe_settle_s, hand_targets=neutral, gate=1,
            trial_id=trial_id, phase="probe_neutral", writer=writer, config=self.config,
        )
        joint_rows: list[dict[str, Any]] = []
        global_unexpected = bool(initial["unexpected_hand_collision"] or _merge_contact_metrics(neutral_records)["unexpected_hand_collision"])
        all_saturation_samples = 0
        all_limit_samples = 0
        total_probe_samples = 0
        for index, name in enumerate(composer.hand_joint_names):
            range_rad = float(upper[index] - lower[index])
            amplitude = range_rad * self.config.joint_probe_fraction
            directions: list[dict[str, Any]] = []
            for sign, label in ((1.0, "positive"), (-1.0, "negative")):
                _run_for(
                    env, composer, duration_s=self.config.probe_settle_s * 0.5, hand_targets=neutral, gate=1,
                    trial_id=trial_id, phase=f"{name}_{label}_return_neutral", writer=writer, config=self.config,
                )
                before = composer.hand_qpos()
                target = neutral.copy()
                target[index] += sign * amplitude
                target = np.clip(target, lower, upper)
                records = _run_for(
                    env, composer, duration_s=self.config.probe_settle_s, hand_targets=target, gate=1,
                    trial_id=trial_id, phase=f"{name}_{label}", writer=writer, config=self.config,
                )
                after = composer.hand_qpos()
                actuator_ids = np.asarray(composer.robot._ref_joint_gripper_actuator_indexes["right"], dtype=np.intp)
                ctrl = np.asarray(env.sim.data.ctrl[actuator_ids], dtype=np.float64)
                steady_error = float(abs(after[index] - target[index]))
                actual_delta = float(after[index] - before[index])
                expected_delta = float(target[index] - neutral[index])
                direction_correct = bool(
                    np.sign(actual_delta) == np.sign(expected_delta)
                    and abs(actual_delta) >= self.config.joint_motion_fraction_required * abs(expected_delta)
                )
                actuator_target_correct = bool(abs(ctrl[index] - target[index]) <= 1e-8)
                contacts = _merge_contact_metrics(records)
                saturation_count = sum(_actuator_saturation(env, composer, self.config.actuator_saturation_fraction) for _summary, _event in records)
                limit_count = sum(_hand_at_limit(composer, self.config.joint_limit_margin_rad) for _summary, _event in records)
                all_saturation_samples += saturation_count
                all_limit_samples += limit_count
                total_probe_samples += len(records)
                global_unexpected = global_unexpected or bool(contacts["unexpected_hand_collision"])
                directions.append({
                    "direction": label,
                    "target_rad": float(target[index]),
                    "actual_rad": float(after[index]),
                    "actual_delta_rad": actual_delta,
                    "expected_delta_rad": expected_delta,
                    "steady_error_rad": steady_error,
                    "direction_correct": direction_correct,
                    "actuator_target_correct": actuator_target_correct,
                    "max_penetration_m": contacts["max_penetration_m"],
                    "saturation_samples": saturation_count,
                    "limit_samples": limit_count,
                })
            actuator_id = int(composer.robot._ref_joint_gripper_actuator_indexes["right"][index])
            compiled_joint_id = int(env.sim.model.actuator_trnid[actuator_id, 0])
            joint_rows.append({
                "index": index,
                "source_joint": name,
                "compiled_joint": env.sim.model.joint_id2name(compiled_joint_id),
                "action_index": composer.hand_slice.start + index,
                "actuator": env.sim.model.actuator_id2name(actuator_id),
                "range_rad": [float(lower[index]), float(upper[index])],
                "probe_amplitude_rad": amplitude,
                "directions": directions,
                "passed": all(
                    row["direction_correct"] and row["actuator_target_correct"]
                    and row["steady_error_rad"] <= self.config.joint_target_error_rad
                    and row["max_penetration_m"] >= -self.config.significant_penetration_m
                    for row in directions
                ),
            })
        close_target = _hand_targets(composer, self.config.gate1_close_curls)
        close_records = _run_for(
            env, composer, duration_s=self.config.probe_settle_s, hand_targets=close_target, gate=1,
            trial_id=trial_id, phase="close_preset", writer=writer, config=self.config,
        )
        close_qpos = composer.hand_qpos()
        close_error = float(np.max(np.abs(close_qpos - close_target)))
        close_contacts = _merge_contact_metrics(close_records)
        writer.capture(env, "g1_close_preset", sim_time=float(env.sim.data.time), force=True)
        _run_for(
            env, composer, duration_s=self.config.open_settle_s, hand_targets=open_target, gate=1,
            trial_id=trial_id, phase="reopen", writer=writer, config=self.config,
        )
        # The mount / TCP facts are taken from the compiled scene after all
        # probes.  They are stable geometry facts, not inferred from drawings.
        flange = _pose_from_body(env.sim, "robot0_right_hand")
        palm = _pose_from_body(env.sim, "gripper0_right_allegro_right_hand")
        tcp = _pose_from_site(env.sim, composer.hand.important_sites["grip_site"])
        mount = _relative_pose(flange, palm)
        tcp_from_palm = _relative_pose(palm, tcp)
        transform_valid = bool(
            0.050 <= np.linalg.norm(np.asarray(mount["translation_m"], dtype=np.float64)) <= 0.150
            and np.all(np.isfinite(np.asarray(tcp_from_palm["translation_m"], dtype=np.float64)))
        )
        sustained_saturation = total_probe_samples > 0 and all_saturation_samples / total_probe_samples >= self.config.sustained_saturation_s * self.config.control_hz / max(total_probe_samples, 1)
        sustained_limits = total_probe_samples > 0 and all_limit_samples / total_probe_samples >= self.config.sustained_saturation_s * self.config.control_hz / max(total_probe_samples, 1)
        reasons: list[str] = []
        if not all(row["passed"] for row in joint_rows):
            reasons.append("joint_mapping_or_direction")
        if initial["max_penetration_m"] < -self.config.significant_penetration_m or initial["unexpected_hand_collision"]:
            reasons.append("initial_collision")
        if global_unexpected or close_contacts["unexpected_hand_collision"]:
            reasons.append("model_collision")
        if close_contacts["max_penetration_m"] < -self.config.significant_penetration_m:
            reasons.append("penetration_artifact")
        if close_error > self.config.joint_target_error_rad:
            reasons.append("close_tracking_error")
        if sustained_saturation:
            reasons.append("actuator_saturation")
        if sustained_limits:
            reasons.append("joint_limit")
        if not transform_valid:
            reasons.append("mount_or_tcp_transform")
        passed = not reasons
        result = {
            "gate": 1,
            "trial_id": trial_id,
            "protocol_phase": "audit",
            "success": passed,
            "passed": passed,
            "failure_category": None if passed else reasons[0],
            "failure_reasons": reasons,
            "joint_rows": joint_rows,
            "initial_contacts": initial,
            "close_preset": {
                "curls": list(self.config.gate1_close_curls),
                "targets_rad": close_target.tolist(),
                "actual_rad": close_qpos.tolist(),
                "max_target_error_rad": close_error,
                "contacts": close_contacts,
            },
            "actuator_saturation_samples": all_saturation_samples,
            "joint_limit_samples": all_limit_samples,
            "probe_samples": total_probe_samples,
            "panda_allegro_mount": {"palm_from_flange": mount, "flange": flange, "palm": palm},
            "tcp": {"osc_wrist_tcp": tcp, "tcp_from_palm": tcp_from_palm, "role": "wrist reference; not assumed to be grasp center"},
        }
        writer.trial(result)
        writer.log(f"gate1 {'passed' if passed else 'failed'} reasons={reasons}")
        return result

    def _base_gate2_candidate(self) -> Gate2Candidate:
        return Gate2Candidate(
            candidate_id="fixture_baseline",
            stage="fixture_smoke",
            cube_center_tcp_m=self.config.static_cube_center_tcp_m,
            final_curls=self.config.static_close_curls,
            close_schedule=((self.config.close_duration_s, self.config.static_close_curls),),
            rationale="v8 static pose and curl target used only to validate the repaired fixture before geometry screening",
        )

    def _candidate_sha256(self, candidate: Gate2Candidate) -> str:
        return canonical_sha256({"gate2_base_config": self.config.to_dict(), "candidate": candidate.to_dict()})

    @staticmethod
    def _candidate_name_coordinate(value_m: float) -> str:
        return f"{int(round(value_m * 1000.0)):03d}mm"

    def _prepare_static_trial(
        self,
        env: Any,
        writer: ArtifactWriter,
        trial_id: str,
        candidate: Gate2Candidate,
        step_callback: Callable[[Any, ContactSummary, Mapping[str, Any]], None] | None = None,
    ) -> tuple[PandaAllegroActionComposer, dict[str, int], np.ndarray, np.ndarray, np.ndarray, ContactSummary, dict[str, Any]]:
        composer = self._reset(env)
        handles = _cube_handles(env)
        open_target = _hand_targets(composer, (0.0, 0.0, 0.0, 0.0))
        close_target = _hand_targets(composer, candidate.final_curls)
        # Configure the close camera before the opening phase as well. This
        # changes camera metadata only, so the visual replay has one stable
        # close view from its first displayed control step.
        preview_tcp_position, preview_tcp_rotation = _ee_pose(composer)
        preview_placement = preview_tcp_position + preview_tcp_rotation @ np.asarray(candidate.cube_center_tcp_m, dtype=np.float64)
        preview_palm = _pose_from_body(env.sim, "gripper0_right_allegro_right_hand")
        configure_validation_cameras(
            env,
            palm_position_m=preview_palm["position_m"],
            palm_rotation=preview_palm["rotation_matrix"],
            focus_position_m=preview_placement,
        )
        _run_for(
            env, composer, duration_s=self.config.open_settle_s, hand_targets=open_target, gate=2,
            trial_id=trial_id, phase="safe_open", writer=writer, config=self.config,
            step_callback=step_callback,
        )
        tcp_position, tcp_rotation = _ee_pose(composer)
        placement = tcp_position + tcp_rotation @ np.asarray(candidate.cube_center_tcp_m, dtype=np.float64)
        set_cube_pose(env, handles, placement, _quaternion_wxyz_from_rotation(tcp_rotation))
        placement_contacts = collect_contacts(
            env,
            gate=2,
            trial_id=trial_id,
            phase="initial_cube_placement",
            writer=writer,
        )
        palm = _pose_from_body(env.sim, "gripper0_right_allegro_right_hand")
        camera_configuration = configure_validation_cameras(
            env,
            palm_position_m=palm["position_m"],
            palm_rotation=palm["rotation_matrix"],
            focus_position_m=placement,
        )
        writer.event({
            "event": "initial_cube_placement",
            "gate": 2,
            "trial_id": trial_id,
            "phase": "temporary_support_setup",
            "sim_time_s": float(env.sim.data.time),
            "cube_position_m": placement.tolist(),
            "placement_frame": "osc_wrist_tcp",
            "cube_center_tcp_m": list(candidate.cube_center_tcp_m),
            "candidate": candidate.to_dict(),
            "candidate_sha256": self._candidate_sha256(candidate),
            "initial_cube_hand_contact_count": placement_contacts.cube_contact_count,
            "initial_cube_hand_penetration_m": placement_contacts.cube_hand_max_penetration_m,
            "gravity_mps2": np.asarray(env.sim.model.opt.gravity, dtype=np.float64).tolist(),
            "validation_cameras": camera_configuration,
        })
        writer.capture_validation_views(env, f"g2_{trial_id}_placed", sim_time=float(env.sim.data.time))
        return composer, handles, open_target, close_target, placement, placement_contacts, camera_configuration

    def _run_candidate_close(
        self,
        env: Any,
        composer: PandaAllegroActionComposer,
        *,
        candidate: Gate2Candidate,
        open_target: np.ndarray,
        support: TemporaryCubeSupport,
        trial_id: str,
        writer: ArtifactWriter,
        keyframes: Gate2KeyframeTracker,
        step_callback: Callable[[Any, ContactSummary, Mapping[str, Any]], None] | None = None,
    ) -> list[tuple[ContactSummary, dict[str, Any]]]:
        """Execute a candidate's explicit, continuous hand-close schedule."""

        records: list[tuple[ContactSummary, dict[str, Any]]] = []
        previous = open_target.copy()
        for segment_index, (duration_s, curls) in enumerate(candidate.close_schedule, start=1):
            endpoint = _hand_targets(composer, curls)
            steps = max(1, int(round(duration_s * self.config.control_hz)))
            writer.event({
                "event": "gate2_close_schedule_segment",
                "gate": 2,
                "trial_id": trial_id,
                "phase": "close_with_temporary_support",
                "segment_index": segment_index,
                "duration_s": duration_s,
                "steps": steps,
                "start_target_rad": previous.tolist(),
                "end_target_rad": endpoint.tolist(),
                "curls": list(curls),
            })
            for step_index in range(steps):
                fraction = float(step_index + 1) / float(steps)
                target = previous + fraction * (endpoint - previous)
                records.append(_step(
                    env, composer, hand_targets=target, wrist_delta=np.zeros(6), gate=2,
                    trial_id=trial_id, phase="close_with_temporary_support", writer=writer,
                    config=self.config, support=support, keyframes=keyframes, step_callback=step_callback,
                ))
            previous = endpoint
        return records

    def _hold_normal_metrics(self, records: Sequence[tuple[ContactSummary, Mapping[str, Any]]]) -> dict[str, Any]:
        control_dt = 1.0 / self.config.control_hz
        valid_frames = 0
        two_digit_frames = 0
        opposition_seen = False
        contact_lost_after_opposition = False
        current_s = 0.0
        longest_s = 0.0
        pair_rows: list[dict[str, Any]] = []
        for summary, _event in records:
            normal = summary.normal_opposition or evaluate_cube_hand_opposition(
                summary,
                dot_max=self.config.normal_opposition_dot_max,
                minimum_normal_force_n=self.config.normal_contact_min_force_n,
            )
            two_digit_frames += int(bool(normal["has_thumb_and_two_digits"]))
            pair_rows.extend(dict(row) for row in normal["pairs"])
            if bool(normal["valid"]):
                valid_frames += 1
                opposition_seen = True
                current_s += control_dt
                longest_s = max(longest_s, current_s)
            else:
                if opposition_seen:
                    contact_lost_after_opposition = True
                current_s = 0.0
        by_finger: dict[str, dict[str, Any]] = {}
        for finger in ("index", "middle", "ring"):
            rows = [row for row in pair_rows if row.get("finger") == finger]
            if rows:
                best = min(rows, key=lambda row: float(row["dot"]))
                by_finger[finger] = {
                    "minimum_dot": float(best["dot"]),
                    "thumb_geom": best["thumb_geom"],
                    "finger_geom": best["finger_geom"],
                    "thumb_normal_force_abs_n": float(best["thumb_normal_force_abs_n"]),
                    "finger_normal_force_abs_n": float(best["finger_normal_force_abs_n"]),
                }
        return {
            "hold_frames": len(records),
            "valid_opposition_frames": valid_frames,
            "thumb_two_digit_frames": two_digit_frames,
            "opposition_seen": opposition_seen,
            "opposition_hold_s": longest_s,
            "contact_lost_after_opposition": contact_lost_after_opposition,
            "per_finger_best_pairs": by_finger,
            "normal_opposition_dot_max": self.config.normal_opposition_dot_max,
            "normal_contact_min_force_n": self.config.normal_contact_min_force_n,
        }

    @staticmethod
    def _record_speed(record: Mapping[str, Any]) -> float:
        velocity = np.asarray(record.get("cube_linear_velocity_mps", (0.0, 0.0, 0.0)), dtype=np.float64)
        return float(np.linalg.norm(velocity)) if velocity.shape == (3,) else 0.0

    def run_gate2_trial(
        self,
        env: Any,
        writer: ArtifactWriter,
        *,
        trial_id: str,
        protocol_phase: str,
        candidate: Gate2Candidate | None = None,
        write_trial: bool = True,
        step_callback: Callable[[Any, ContactSummary, Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        candidate = self._base_gate2_candidate() if candidate is None else candidate
        candidate_sha256 = self._candidate_sha256(candidate)
        (
            composer,
            handles,
            open_target,
            close_target,
            placement,
            placement_contacts,
            camera_configuration,
        ) = self._prepare_static_trial(env, writer, trial_id, candidate, step_callback=step_callback)
        keyframes = Gate2KeyframeTracker(writer=writer, trial_id=trial_id)
        support = TemporaryCubeSupport(env, handles, placement, self.config)
        preclose_records = _run_for(
            env, composer, duration_s=self.config.temporary_support_preclose_s, hand_targets=open_target,
            gate=2, trial_id=trial_id, phase="post_placement_temporary_support", writer=writer,
            config=self.config, support=support, keyframes=keyframes, step_callback=step_callback,
        )
        close_records = self._run_candidate_close(
            env, composer, candidate=candidate, open_target=open_target, support=support,
            trial_id=trial_id, writer=writer, keyframes=keyframes, step_callback=step_callback,
        )
        support_records = _run_for(
            env, composer, duration_s=self.config.temporary_support_settle_s, hand_targets=close_target, gate=2,
            trial_id=trial_id, phase="close_settle_with_temporary_support", writer=writer, config=self.config,
            support=support, keyframes=keyframes, step_callback=step_callback,
        )
        release_ramp_records: list[tuple[ContactSummary, dict[str, Any]]] = []
        release_ramp_steps = max(1, int(round(self.config.temporary_support_release_ramp_s * self.config.control_hz)))
        for step_index in range(release_ramp_steps):
            force_scale = 1.0 - float(step_index + 1) / float(release_ramp_steps)
            support.set_force_scale(force_scale)
            writer.event({
                "event": "temporary_support_release_ramp",
                "gate": 2,
                "trial_id": trial_id,
                "phase": "temporary_support_release_ramp",
                "step_index": step_index + 1,
                "step_count": release_ramp_steps,
                "support_force_scale": force_scale,
                "sim_time_s": float(env.sim.data.time),
            })
            release_ramp_records.append(_step(
                env, composer, hand_targets=close_target, wrist_delta=np.zeros(6), gate=2,
                trial_id=trial_id, phase="temporary_support_release_ramp", writer=writer,
                config=self.config, support=support, keyframes=keyframes, step_callback=step_callback,
            ))
        fixture_records = preclose_records + close_records + support_records + release_ramp_records
        # Release only the external force. No cube qpos or qvel is written at
        # release: the ensuing hold is a genuine gravity-on free simulation.
        release_cube = _cube_state(env, handles)
        support.release()
        release_time = float(env.sim.data.time)
        writer.event({
            "event": "temporary_support_released",
            "gate": 2,
            "trial_id": trial_id,
            "phase": "gravity_hold",
            "sim_time_s": release_time,
            "cube_position_m": release_cube["position_m"],
            "cube_linear_velocity_mps": release_cube["linear_velocity_mps"],
            "xfrc_applied": np.asarray(env.sim.data.xfrc_applied[handles["body_id"]], dtype=np.float64).tolist(),
            "cube_free_joint_velocity_reset": False,
            "cube_qpos_rewritten_at_release": False,
            "gravity_mps2": np.asarray(env.sim.model.opt.gravity, dtype=np.float64).tolist(),
        })
        writer.capture_validation_views(env, f"g2_{trial_id}_support_released", sim_time=release_time)
        release_settle_records = _run_for(
            env, composer, duration_s=self.config.gravity_release_settle_s, hand_targets=close_target, gate=2,
            trial_id=trial_id, phase="gravity_release_settle", writer=writer, config=self.config,
            keyframes=keyframes, step_callback=step_callback,
        )
        hold_records = _run_for(
            env, composer, duration_s=self.config.static_hold_s, hand_targets=close_target, gate=2,
            trial_id=trial_id, phase="gravity_hold", writer=writer, config=self.config,
            keyframes=keyframes, step_callback=step_callback,
        )
        unassisted_records = release_settle_records + hold_records
        all_records = fixture_records + unassisted_records
        fixture_metrics = _merge_contact_metrics(fixture_records)
        unassisted_metrics = _merge_contact_metrics(unassisted_records)
        hold_metrics = _merge_contact_metrics(hold_records)
        normal_metrics = self._hold_normal_metrics(hold_records)
        saturation_samples = sum(
            _actuator_saturation(env, composer, self.config.actuator_saturation_fraction)
            for _summary, _event in all_records
        )
        limit_samples = sum(
            _hand_at_limit(composer, self.config.joint_limit_margin_rad)
            for _summary, _event in all_records
        )
        first_contact_index = next(
            (index for index, (summary, _event) in enumerate(fixture_records) if summary.cube_contact_count > 0),
            None,
        )
        fixture_speeds = [self._record_speed(event) for _summary, event in fixture_records]
        release_settle_speeds = [self._record_speed(event) for _summary, event in release_settle_records]
        hold_speeds = [self._record_speed(event) for _summary, event in hold_records]
        peak_fixture_speed = max(fixture_speeds, default=0.0)
        peak_speed_after_first_contact = (
            max(fixture_speeds[first_contact_index:], default=0.0)
            if first_contact_index is not None else 0.0
        )
        substep_count_mismatch = any(
            not bool(event.get("fixture", {}).get("substep_count_matches_expected", False))
            for _summary, event in fixture_records
        )
        release_position = np.asarray(release_cube["position_m"], dtype=np.float64)
        final_cube = _cube_state(env, handles)
        final_position = np.asarray(final_cube["position_m"], dtype=np.float64)
        max_slip = float(np.linalg.norm(final_position - release_position))
        cube_hand_penetration = min(
            float(fixture_metrics["cube_hand_max_penetration_m"]),
            float(unassisted_metrics["cube_hand_max_penetration_m"]),
        )
        all_penetration = min(float(fixture_metrics["max_penetration_m"]), float(unassisted_metrics["max_penetration_m"]))
        control_dt = 1.0 / self.config.control_hz
        expected_hold_frames = max(1, int(round(self.config.static_hold_s * self.config.control_hz)))
        diagnostic_flags = {
            "penetration": bool(
                placement_contacts.cube_hand_max_penetration_m < -self.config.significant_penetration_m
                or cube_hand_penetration < -self.config.significant_penetration_m
            ),
            "insufficient_digits": bool(normal_metrics["thumb_two_digit_frames"] < expected_hold_frames),
            "normal_not_opposed": bool(normal_metrics["valid_opposition_frames"] < expected_hold_frames),
            "fixture_unstable": bool(
                first_contact_index is None
                or peak_speed_after_first_contact > self.config.fixture_peak_speed_limit_mps
                or substep_count_mismatch
            ),
        }
        metrics = {
            "cube_contacts": hold_metrics["cube_contacts"],
            "cube_fingers": hold_metrics["cube_fingers"],
            "max_penetration_m": cube_hand_penetration,
            "all_contact_max_penetration_m": all_penetration,
            "cube_hand_max_penetration_m": cube_hand_penetration,
            "initial_cube_hand_contact_count": placement_contacts.cube_contact_count,
            "initial_cube_hand_penetration_m": placement_contacts.cube_hand_max_penetration_m,
            "fixture_cube_hand_max_penetration_m": fixture_metrics["cube_hand_max_penetration_m"],
            "gravity_release_settle_cube_hand_max_penetration_m": _merge_contact_metrics(release_settle_records)["cube_hand_max_penetration_m"],
            "gravity_hold_cube_hand_max_penetration_m": hold_metrics["cube_hand_max_penetration_m"],
            "unassisted_cube_hand_max_penetration_m": unassisted_metrics["cube_hand_max_penetration_m"],
            "table_contact": bool(unassisted_metrics["table_contact"]),
            "model_collision": bool(fixture_metrics["unexpected_hand_collision"] or unassisted_metrics["unexpected_hand_collision"]),
            "object_ejected": bool(unassisted_metrics["table_contact"] or final_position[2] < release_position[2] - 0.12),
            "slip_distance_m": max_slip,
            "contact_lost_after_opposition": normal_metrics["contact_lost_after_opposition"],
            "support_released": True,
            "gravity_enabled": bool(np.linalg.norm(np.asarray(env.sim.model.opt.gravity, dtype=np.float64)) > 1e-9),
            "hold_duration_s": float(len(hold_records) * control_dt),
            "qualified_hold_duration_s": float(len(hold_records) * control_dt),
            "gravity_release_settle_duration_s": float(len(release_settle_records) * control_dt),
            "actuator_saturation_s": saturation_samples * control_dt,
            "joint_limit_s": limit_samples * control_dt,
            "opposition_seen": normal_metrics["opposition_seen"],
            "opposition_hold_s": normal_metrics["opposition_hold_s"],
            "normal_opposition": normal_metrics,
            "hand_target_error_rad": float(np.max(np.abs(composer.hand_qpos() - close_target))),
            "close_cube_contacts": fixture_metrics["cube_contacts"],
            "close_cube_fingers": fixture_metrics["cube_fingers"],
            "fixture_peak_cube_speed_mps": peak_fixture_speed,
            "fixture_peak_speed_after_first_contact_mps": peak_speed_after_first_contact,
            "gravity_release_settle_peak_cube_speed_mps": max(release_settle_speeds, default=0.0),
            "gravity_hold_peak_cube_speed_mps": max(hold_speeds, default=0.0),
            "unassisted_peak_cube_speed_mps": max(release_settle_speeds + hold_speeds, default=0.0),
            "fixture_first_contact_control_step": first_contact_index,
            "fixture_first_contact_sim_time_s": (
                float(fixture_records[first_contact_index][1]["sim_time_s"])
                if first_contact_index is not None else None
            ),
            "fixture_substep_count_mismatch": substep_count_mismatch,
            "diagnostic_flags": diagnostic_flags,
            "total_control_steps": len(all_records),
        }
        success, failure = classify_gate2(metrics, self.config)
        support_summary = support.summary()
        support_summary.update({
            "released_at_sim_time_s": release_time,
            "free_joint_velocity_reset_at_release": False,
            "cube_qpos_rewritten_at_release": False,
            "no_weld": True,
            "normal_gravity_during_hold": True,
        })
        result = {
            "gate": 2,
            "trial_id": trial_id,
            "protocol_phase": protocol_phase,
            "config_sha256": candidate_sha256,
            "base_config_sha256": self.config.sha256,
            "candidate": candidate.to_dict(),
            "candidate_sha256": candidate_sha256,
            "success": success,
            "passed": success,
            "failure_category": failure,
            "metrics": metrics,
            "support": support_summary,
            "validation_cameras": camera_configuration,
            "hand": {
                "targets_rad": close_target.tolist(),
                "actual_rad": composer.hand_qpos().tolist(),
            },
            "cube_release": release_cube,
            "cube_final": final_cube,
        }
        if write_trial:
            writer.trial(result)
        writer.log(f"gate2 {trial_id} {'passed' if success else 'failed'} failure={failure} candidate={candidate.candidate_id}")
        writer.capture_validation_views(env, f"g2_{trial_id}_final_hold", sim_time=float(env.sim.data.time))
        return result

    def _run_protocol(
        self,
        trial_runner: Callable[..., dict[str, Any]],
        env: Any,
        writer: ArtifactWriter,
        *,
        gate: int,
    ) -> dict[str, Any]:
        trials: list[dict[str, Any]] = []
        single = trial_runner(env, writer, trial_id=f"gate{gate}_single", protocol_phase="single")
        single.setdefault("config_sha256", self.config.sha256)
        trials.append(single)
        if not single["success"]:
            counts = Counter(str(trial.get("failure_category")) for trial in trials if not trial.get("success"))
            return {"gate": gate, "passed": False, "trials": trials, "protocol": aggregate_protocol(trials, config_sha256=self.config.sha256, config=self.config), "failure_counts": dict(counts)}
        for index in range(self.config.qualification_trials):
            trial = trial_runner(env, writer, trial_id=f"gate{gate}_qualification_{index + 1}", protocol_phase="qualification")
            trial.setdefault("config_sha256", self.config.sha256)
            trials.append(trial)
            if not trial["success"]:
                counts = Counter(str(item.get("failure_category")) for item in trials if not item.get("success"))
                return {"gate": gate, "passed": False, "trials": trials, "protocol": aggregate_protocol(trials, config_sha256=self.config.sha256, config=self.config), "failure_counts": dict(counts)}
        for index in range(self.config.formal_trials):
            trial = trial_runner(env, writer, trial_id=f"gate{gate}_formal_{index + 1}", protocol_phase="formal")
            trial.setdefault("config_sha256", self.config.sha256)
            trials.append(trial)
        protocol = aggregate_protocol(trials, config_sha256=self.config.sha256, config=self.config)
        counts = Counter(str(trial.get("failure_category")) for trial in trials if not trial.get("success"))
        return {"gate": gate, "passed": bool(protocol["passed"]), "trials": trials, "protocol": protocol, "failure_counts": dict(counts)}

    def _gate2_fixture_smoke(self, env: Any, writer: ArtifactWriter) -> dict[str, Any]:
        """Run the repaired fixture before any grasp-geometry screening."""

        candidate = self._base_gate2_candidate()
        trial = self.run_gate2_trial(
            env,
            writer,
            trial_id="gate2_fixture_smoke",
            protocol_phase="fixture_smoke",
            candidate=candidate,
        )
        metrics = trial["metrics"]
        support = trial["support"]
        passed_checks = {
            "no_initial_cube_hand_contact": int(metrics["initial_cube_hand_contact_count"]) == 0,
            "no_initial_cube_hand_penetration": float(metrics["initial_cube_hand_penetration_m"]) >= -1e-9,
            "first_contact_observed": metrics["fixture_first_contact_sim_time_s"] is not None,
            "post_contact_peak_speed_within_limit": float(metrics["fixture_peak_speed_after_first_contact_mps"]) <= self.config.fixture_peak_speed_limit_mps,
            "full_fixture_penetration_within_limit": float(metrics["fixture_cube_hand_max_penetration_m"]) >= -self.config.significant_penetration_m,
            "gravity_hold_penetration_within_limit": float(metrics["gravity_hold_cube_hand_max_penetration_m"]) >= -self.config.significant_penetration_m,
            "support_updated_each_physics_substep": not bool(metrics["fixture_substep_count_mismatch"]),
            "release_did_not_rewrite_cube_state": (
                not bool(support["free_joint_velocity_reset_at_release"])
                and not bool(support["cube_qpos_rewritten_at_release"])
            ),
            "gravity_enabled": bool(metrics["gravity_enabled"]),
        }
        smoke = {
            "gate": 2,
            "stage": "fixture_smoke",
            "candidate": candidate.to_dict(),
            "candidate_sha256": self._candidate_sha256(candidate),
            "passed": all(passed_checks.values()),
            "checks": passed_checks,
            "metrics": {
                "initial_cube_hand_contact_count": metrics["initial_cube_hand_contact_count"],
                "initial_cube_hand_penetration_m": metrics["initial_cube_hand_penetration_m"],
                "fixture_peak_speed_after_first_contact_mps": metrics["fixture_peak_speed_after_first_contact_mps"],
                "fixture_cube_hand_max_penetration_m": metrics["fixture_cube_hand_max_penetration_m"],
                "gravity_hold_cube_hand_max_penetration_m": metrics["gravity_hold_cube_hand_max_penetration_m"],
                "gravity_hold_peak_cube_speed_mps": metrics["gravity_hold_peak_cube_speed_mps"],
                "support": support,
            },
            "trial_failure_category": trial["failure_category"],
        }
        writer.write_json("fixture_smoke.json", smoke)
        writer.event({"event": "fixture_smoke_result", **smoke})
        writer.log(f"gate2 fixture_smoke {'passed' if smoke['passed'] else 'failed'} checks={passed_checks}")
        return smoke

    def _screen_gate2_candidate(
        self,
        env: Any,
        writer: ArtifactWriter,
        *,
        candidate: Gate2Candidate,
    ) -> dict[str, Any]:
        result = self.run_gate2_trial(
            env,
            writer,
            trial_id=f"gate2_screen_{candidate.candidate_id}",
            protocol_phase=f"search_{candidate.stage}",
            candidate=candidate,
            write_trial=False,
        )
        metrics = result["metrics"]
        flags = dict(metrics.get("diagnostic_flags", {}))
        hard_rejections: list[str] = []
        if int(metrics["initial_cube_hand_contact_count"]) > 0 or float(metrics["initial_cube_hand_penetration_m"]) < -1e-9:
            hard_rejections.append("initial_cube_hand_contact_or_penetration")
        if bool(flags.get("fixture_unstable", False)):
            hard_rejections.append("fixture_unstable")
        if bool(flags.get("penetration", False)):
            hard_rejections.append("penetration_over_4mm")
        if bool(metrics["model_collision"]):
            hard_rejections.append("unexpected_hand_collision")
        if float(metrics["actuator_saturation_s"]) >= self.config.sustained_saturation_s:
            hard_rejections.append("actuator_saturation")
        if float(metrics["joint_limit_s"]) >= self.config.sustained_saturation_s:
            hard_rejections.append("joint_limit")
        if bool(metrics["table_contact"]):
            hard_rejections.append("table_support")
        normal = dict(metrics["normal_opposition"])
        per_finger = dict(normal["per_finger_best_pairs"])
        pair_worst_dot = max(
            (float(row["minimum_dot"]) for row in per_finger.values()),
            default=1.0,
        )
        sort_key = (
            int(bool(hard_rejections)),
            -int(normal["valid_opposition_frames"]),
            -float(normal["opposition_hold_s"]),
            -int(normal["thumb_two_digit_frames"]),
            pair_worst_dot,
            abs(float(metrics["cube_hand_max_penetration_m"])),
            float(metrics["fixture_peak_speed_after_first_contact_mps"]),
            float(metrics["gravity_hold_peak_cube_speed_mps"]),
            float(metrics["slip_distance_m"]),
            float(metrics["hand_target_error_rad"]),
            candidate.candidate_id,
        )
        result["screening"] = {
            "hard_rejections": hard_rejections,
            "diagnostic_flags": flags,
            "rank_sort_key": list(sort_key),
            "rank_components": {
                "valid_opposition_frames": normal["valid_opposition_frames"],
                "longest_opposition_hold_s": normal["opposition_hold_s"],
                "thumb_two_digit_frames": normal["thumb_two_digit_frames"],
                "worst_required_pair_dot": pair_worst_dot,
                "cube_hand_max_penetration_m": metrics["cube_hand_max_penetration_m"],
                "fixture_peak_speed_after_first_contact_mps": metrics["fixture_peak_speed_after_first_contact_mps"],
                "gravity_hold_peak_cube_speed_mps": metrics["gravity_hold_peak_cube_speed_mps"],
                "slip_distance_m": metrics["slip_distance_m"],
                "hand_target_error_rad": metrics["hand_target_error_rad"],
            },
        }
        writer.trial(result)
        return result

    @staticmethod
    def _rank_screenings(rows: Sequence[tuple[Gate2Candidate, Mapping[str, Any]]]) -> list[tuple[Gate2Candidate, Mapping[str, Any]]]:
        return sorted(rows, key=lambda item: tuple(item[1]["screening"]["rank_sort_key"]))

    def _write_candidate_table(
        self,
        writer: ArtifactWriter,
        *,
        stage: str,
        rows: Sequence[tuple[Gate2Candidate, Mapping[str, Any]]],
        selected_candidate_id: str | None,
    ) -> None:
        ranked = self._rank_screenings(rows)
        table_rows: list[dict[str, Any]] = []
        for rank, (candidate, result) in enumerate(ranked, start=1):
            table_rows.append({
                "rank": rank,
                "selected_for_next_stage": candidate.candidate_id == selected_candidate_id,
                "candidate": candidate.to_dict(),
                "candidate_sha256": self._candidate_sha256(candidate),
                "success": result["success"],
                "primary_failure_category": result["failure_category"],
                "screening": result["screening"],
                "metrics": result["metrics"],
            })
        writer.write_json(
            f"candidate_table_{stage}.json",
            {
                "gate": 2,
                "stage": stage,
                "base_config_sha256": self.config.sha256,
                "candidate_count": len(table_rows),
                "ranking_basis": [
                    "hard_rejections",
                    "valid_opposition_frames_desc",
                    "longest_opposition_hold_s_desc",
                    "thumb_two_digit_frames_desc",
                    "worst_required_pair_dot_asc",
                    "cube_hand_penetration_abs_asc",
                    "fixture_peak_speed_asc",
                    "hold_peak_speed_asc",
                    "slip_asc",
                    "target_error_asc",
                    "candidate_id",
                ],
                "rows": table_rows,
            },
        )

    def _geometry_y_candidates(self) -> list[Gate2Candidate]:
        x, _unused_y, z = self.config.static_cube_center_tcp_m
        return [
            Gate2Candidate(
                candidate_id=f"y_{self._candidate_name_coordinate(y)}",
                stage="geometry_y",
                cube_center_tcp_m=(x, y, z),
                final_curls=self.config.static_close_curls,
                close_schedule=((self.config.close_duration_s, self.config.static_close_curls),),
                rationale="stage 1: vary only cube TCP local Y; preserve v8 X, Z, orientation, and final curls",
            )
            for y in self.config.gate2_y_candidates_m
        ]

    def _geometry_xz_candidates(self, y: float) -> list[Gate2Candidate]:
        return [
            Gate2Candidate(
                candidate_id=(
                    f"xz_x{self._candidate_name_coordinate(x)}_"
                    f"y{self._candidate_name_coordinate(y)}_z{self._candidate_name_coordinate(z)}"
                ),
                stage="geometry_xz",
                cube_center_tcp_m=(x, y, z),
                final_curls=self.config.static_close_curls,
                close_schedule=((self.config.close_duration_s, self.config.static_close_curls),),
                rationale="stage 2: vary only cube TCP local X/Z after selecting the stage-1 Y; preserve v8 curls",
            )
            for x in self.config.gate2_x_candidates_m
            for z in self.config.gate2_z_candidates_m
        ]

    def _finger_schedule_candidates(self, geometry_candidates: Sequence[Gate2Candidate]) -> list[Gate2Candidate]:
        variants = (
            ("c1", (0.50, 0.50, 0.15, 0.85)),
            ("c2", (0.50, 0.50, 0.15, 0.80)),
        )
        candidates: list[Gate2Candidate] = []
        for geometry in geometry_candidates:
            for label, final_curls in variants:
                candidates.append(Gate2Candidate(
                    candidate_id=f"fingers_{geometry.candidate_id}_{label}",
                    stage="finger_schedule",
                    cube_center_tcp_m=geometry.cube_center_tcp_m,
                    final_curls=final_curls,
                    close_schedule=(
                        (0.35, (0.35, 0.35, 0.10, 0.30)),
                        (0.35, (0.50, 0.50, 0.15, 0.55)),
                        (0.50, final_curls),
                    ),
                    rationale="stage 3: index/middle approach first, thumb closes second, ring remains backed away",
                ))
        return candidates

    def _run_gate2_locked_protocol(
        self,
        env: Any,
        writer: ArtifactWriter,
        *,
        candidate: Gate2Candidate,
    ) -> dict[str, Any]:
        candidate_sha256 = self._candidate_sha256(candidate)
        trials: list[dict[str, Any]] = []
        single = self.run_gate2_trial(
            env, writer, trial_id="gate2_single", protocol_phase="single", candidate=candidate,
        )
        trials.append(single)
        if not single["success"]:
            protocol = aggregate_protocol(trials, config_sha256=candidate_sha256, config=self.config)
            return {"trials": trials, "protocol": protocol, "locked": False}
        for index in range(self.config.qualification_trials):
            trial = self.run_gate2_trial(
                env,
                writer,
                trial_id=f"gate2_qualification_{index + 1}",
                protocol_phase="qualification",
                candidate=candidate,
            )
            trials.append(trial)
            if not trial["success"]:
                protocol = aggregate_protocol(trials, config_sha256=candidate_sha256, config=self.config)
                return {"trials": trials, "protocol": protocol, "locked": False}
        locked = {
            "gate": 2,
            "candidate": candidate.to_dict(),
            "candidate_sha256": candidate_sha256,
            "base_config_sha256": self.config.sha256,
            "qualification_trials": self.config.qualification_trials,
            "qualification_successes": self.config.qualification_trials,
        }
        writer.write_json("locked_gate2_configuration.json", locked)
        writer.event({"event": "gate2_configuration_locked", **locked})
        for index in range(self.config.formal_trials):
            trials.append(self.run_gate2_trial(
                env,
                writer,
                trial_id=f"gate2_formal_{index + 1}",
                protocol_phase="formal",
                candidate=candidate,
            ))
        protocol = aggregate_protocol(trials, config_sha256=candidate_sha256, config=self.config)
        return {"trials": trials, "protocol": protocol, "locked": True}

    def run_gate2_protocol(self, env: Any, writer: ArtifactWriter) -> dict[str, Any]:
        """Execute fixture smoke -> staged screening -> one/3/10 Gate 2 protocol."""

        smoke = self._gate2_fixture_smoke(env, writer)
        if not smoke["passed"]:
            return {
                "gate": 2,
                "passed": False,
                "fixture_smoke": smoke,
                "search": {"skipped": True, "reason": "fixture_smoke_failed"},
                "trials": [],
                "protocol": {"passed": False, "failure_category": "fixture_smoke_failed"},
                "failure_counts": {"fixture_smoke_failed": 1},
            }
        y_rows = [
            (candidate, self._screen_gate2_candidate(env, writer, candidate=candidate))
            for candidate in self._geometry_y_candidates()
        ]
        ranked_y = self._rank_screenings(y_rows)
        selected_y = ranked_y[0][0]
        self._write_candidate_table(
            writer,
            stage="geometry_y",
            rows=y_rows,
            selected_candidate_id=selected_y.candidate_id,
        )
        xz_rows = [
            (candidate, self._screen_gate2_candidate(env, writer, candidate=candidate))
            for candidate in self._geometry_xz_candidates(selected_y.cube_center_tcp_m[1])
        ]
        ranked_xz = self._rank_screenings(xz_rows)
        top_geometry = [candidate for candidate, _result in ranked_xz[:self.config.gate2_search_top_geometry_candidates]]
        selected_geometry = ranked_xz[0][0]
        self._write_candidate_table(
            writer,
            stage="geometry_xz",
            rows=xz_rows,
            selected_candidate_id=selected_geometry.candidate_id,
        )
        finger_rows = [
            (candidate, self._screen_gate2_candidate(env, writer, candidate=candidate))
            for candidate in self._finger_schedule_candidates(top_geometry)
        ]
        ranked_fingers = self._rank_screenings(finger_rows)
        selected_finger = ranked_fingers[0][0]
        self._write_candidate_table(
            writer,
            stage="finger_schedule",
            rows=finger_rows,
            selected_candidate_id=selected_finger.candidate_id,
        )
        all_rows = y_rows + xz_rows + finger_rows
        selected_candidate = self._rank_screenings(all_rows)[0][0]
        writer.write_json("gate2_search_selection.json", {
            "gate": 2,
            "fixture_smoke_passed": True,
            "screened_candidate_count": len(all_rows),
            "selected_candidate": selected_candidate.to_dict(),
            "selected_candidate_sha256": self._candidate_sha256(selected_candidate),
            "stage_winners": {
                "geometry_y": selected_y.to_dict(),
                "geometry_xz": selected_geometry.to_dict(),
                "finger_schedule": selected_finger.to_dict(),
            },
        })
        locked_run = self._run_gate2_locked_protocol(env, writer, candidate=selected_candidate)
        trials = locked_run["trials"]
        protocol = locked_run["protocol"]
        counts = Counter(str(trial.get("failure_category")) for trial in trials if not trial.get("success"))
        return {
            "gate": 2,
            "passed": bool(protocol.get("passed", False)),
            "fixture_smoke": smoke,
            "search": {
                "skipped": False,
                "screened_candidate_count": len(all_rows),
                "selected_candidate": selected_candidate.to_dict(),
                "selected_candidate_sha256": self._candidate_sha256(selected_candidate),
            },
            "trials": trials,
            "protocol": protocol,
            "configuration_locked_after_qualification": bool(locked_run["locked"]),
            "failure_counts": dict(counts),
        }

    def run_gate2_visualization(
        self,
        *,
        locked_config_path: Path | None = None,
        record_path: Path | None = None,
        auto_exit_after_final_s: float | None = None,
    ) -> tuple[int, Path, dict[str, Any]]:
        """Replay one locked Gate 2 grasp in a real MuJoCo viewer.

        This is a display mode, not another search or qualification run.  It
        keeps the locked candidate, compiled-model seed, 20 Hz control, and
        every physics parameter unchanged.  Only wall-clock pacing is slower
        for the short opening, placement, close, and release phases.
        """

        if auto_exit_after_final_s is not None and float(auto_exit_after_final_s) < 0.0:
            raise ValueError("auto_exit_after_final_s must be non-negative")
        lock_path = newest_locked_gate2_config() if locked_config_path is None else Path(locked_config_path)
        candidate = load_locked_gate2_candidate(lock_path, config=self.config)
        output = self._output_directory("gate2_visual")
        writer = ArtifactWriter(
            output,
            config_sha256=self.config.sha256,
            frame_every_s=self.config.frame_every_s,
            record_frames=self.record_frames,
        )
        recorder = Gate2VisualizationRecorder(record_path, fps=self.config.control_hz)
        router = KeyboardCommandRouter()
        env = None
        replay_count = 0
        last_result: dict[str, Any] | None = None
        exit_reason = "completed"
        summary: dict[str, Any] = {
            "requested_gate": 2,
            "mode": "visualize",
            "base_config_sha256": self.config.sha256,
            "locked_config_path": str(lock_path),
            "locked_candidate": candidate.to_dict(),
            "physics_parameters_changed": False,
            "replays": [],
        }
        # These are wall-clock display periods only. The MuJoCo control and
        # substep schedule remains the locked 20 Hz / 2 ms experiment.
        display_step_s = {
            "safe_open": 1.0 / max(1, int(round(self.config.open_settle_s * self.config.control_hz))),
            "post_placement_temporary_support": 1.0 / max(1, int(round(self.config.temporary_support_preclose_s * self.config.control_hz))),
            "close_with_temporary_support": 2.0 / max(1, int(round(self.config.close_duration_s * self.config.control_hz))),
            "close_settle_with_temporary_support": 1.0 / self.config.control_hz,
            "temporary_support_release_ramp": 0.60 / max(1, int(round(self.config.temporary_support_release_ramp_s * self.config.control_hz))),
            "gravity_release_settle": 1.0 / self.config.control_hz,
            "gravity_hold": 1.0 / self.config.control_hz,
        }
        try:
            env = self._new_environment(has_renderer=True)
            # Create robosuite's viewer wrapper before installing the project
            # callback. The reset immediately before each replay discards this
            # priming state, so it cannot affect the measured grasp.
            primer = self._reset(env)
            if getattr(env, "viewer", None) is None:
                env.step(primer.compose(np.zeros(6), primer.hand_qpos()))
            if not install_viewer_key_callback(env, lambda key: router.feed(key, source="gate2_viewer")):
                raise RuntimeError("MuJoCo viewer keyboard callback is unavailable")
            writer.write_json("visualization_config.json", {
                "locked_config_path": str(lock_path),
                "candidate": candidate.to_dict(),
                "candidate_sha256": self._candidate_sha256(candidate),
                "base_config_sha256": self.config.sha256,
                "control_hz": self.config.control_hz,
                "physics_parameters_changed": False,
                "display_step_s": display_step_s,
                "camera": "frontview",
                "keyboard": {"reset": "R/r", "quit": "Q/q", "interrupt": "Ctrl+C"},
            })
            writer.log(f"visualization_start lock={lock_path} candidate={candidate.candidate_id}")
            writer.event({
                "event": "visualization_start",
                "gate": 2,
                "locked_config_path": str(lock_path),
                "candidate": candidate.to_dict(),
                "physics_parameters_changed": False,
            })

            while not router.stop_event.is_set():
                replay_count += 1
                trial_id = f"gate2_visual_replay_{replay_count:03d}"
                state: dict[str, Any] = {
                    "phase": None,
                    "max_penetration_m": 0.0,
                    "fingers": [],
                    "release_position_m": None,
                    "unassisted_start_s": None,
                    "hold_start_s": None,
                    "last_sim_time_s": 0.0,
                    "last_console_s": -np.inf,
                    "camera_set": False,
                    "panda_initial": None,
                    "panda_max_deviation_rad": 0.0,
                }

                def observe_step(step_env: Any, _contacts: ContactSummary, event: Mapping[str, Any]) -> None:
                    phase = str(event["phase"])
                    sim_time = float(event["sim_time_s"])
                    if phase != state["phase"]:
                        state["phase"] = phase
                        writer.event({
                            "event": "visualization_phase_start",
                            "gate": 2,
                            "trial_id": trial_id,
                            "phase": phase,
                            "sim_time_s": sim_time,
                        })
                    if not state["camera_set"]:
                        camera_id = int(step_env.sim.model.camera_name2id("frontview"))
                        step_env.viewer.set_camera(camera_id)
                        state["camera_set"] = True
                        writer.event({
                            "event": "visualization_camera_set",
                            "gate": 2,
                            "trial_id": trial_id,
                            "camera": "frontview",
                            "sim_time_s": sim_time,
                        })
                    state["max_penetration_m"] = min(
                        float(state["max_penetration_m"]),
                        float(event.get("cube_hand_max_penetration_m", 0.0)),
                    )
                    state["fingers"] = list(event.get("cube_fingers", ()))
                    state["last_sim_time_s"] = sim_time
                    panda = np.asarray(event.get("panda_joint_positions_rad", ()), dtype=np.float64)
                    if panda.shape == (7,):
                        if state["panda_initial"] is None:
                            state["panda_initial"] = panda.copy()
                        state["panda_max_deviation_rad"] = max(
                            float(state["panda_max_deviation_rad"]),
                            float(np.max(np.abs(panda - state["panda_initial"]))),
                        )
                    if phase == "gravity_release_settle" and state["unassisted_start_s"] is None:
                        state["unassisted_start_s"] = sim_time
                        state["release_position_m"] = list(event.get("cube_position_m", ()))
                        writer.event({
                            "event": "visualization_unassisted_start",
                            "gate": 2,
                            "trial_id": trial_id,
                            "sim_time_s": sim_time,
                            "support_released": True,
                        })
                    if phase == "gravity_hold" and state["hold_start_s"] is None:
                        state["hold_start_s"] = sim_time
                        writer.event({
                            "event": "visualization_gravity_hold_start",
                            "gate": 2,
                            "trial_id": trial_id,
                            "sim_time_s": sim_time,
                        })
                    release = state["release_position_m"]
                    cube_position = np.asarray(event.get("cube_position_m", ()), dtype=np.float64)
                    slip_m = 0.0
                    if release is not None and cube_position.shape == (3,):
                        slip_m = float(np.linalg.norm(cube_position - np.asarray(release, dtype=np.float64)))
                    now = time.monotonic()
                    if now - float(state["last_console_s"]) >= 0.25:
                        hold_s = 0.0 if state["hold_start_s"] is None else max(0.0, sim_time - float(state["hold_start_s"]))
                        fingers = ",".join(state["fingers"]) or "无"
                        _write_visual_status(
                            f"Gate 2 可视化 | 阶段={phase} | 无支撑保持={hold_s:.2f}/{self.config.static_hold_s:.2f}s "
                            f"| 滑移={slip_m * 1000.0:.2f}mm | 最大穿透={-float(state['max_penetration_m']) * 1000.0:.2f}mm "
                            f"| 承力手指={fingers}",
                        )
                        state["last_console_s"] = now
                    recorder.capture(step_env)
                    if router.stop_event.is_set():
                        raise VisualizationInterrupted("quit")
                    if router.take_reset():
                        raise VisualizationInterrupted("reset")
                    time.sleep(float(display_step_s.get(phase, 1.0 / self.config.control_hz)))

                writer.event({"event": "visualization_replay_start", "gate": 2, "trial_id": trial_id, "replay": replay_count})
                try:
                    result = self.run_gate2_trial(
                        env,
                        writer,
                        trial_id=trial_id,
                        protocol_phase="visual_replay",
                        candidate=candidate,
                        step_callback=observe_step,
                    )
                except VisualizationInterrupted as interrupted:
                    exit_reason = interrupted.reason
                    writer.event({
                        "event": "visualization_interrupted",
                        "gate": 2,
                        "trial_id": trial_id,
                        "reason": interrupted.reason,
                        "sim_time_s": float(state["last_sim_time_s"]),
                    })
                    if interrupted.reason == "reset":
                        continue
                    break

                last_result = result
                metrics = dict(result["metrics"])
                writer.event({
                    "event": "visualization_gravity_hold_end",
                    "gate": 2,
                    "trial_id": trial_id,
                    "sim_time_s": float(state["last_sim_time_s"]),
                    "hold_duration_s": float(metrics["hold_duration_s"]),
                    "slip_distance_m": float(metrics["slip_distance_m"]),
                    "max_penetration_m": float(metrics["max_penetration_m"]),
                    "cube_fingers": list(metrics["cube_fingers"]),
                })
                writer.event({
                    "event": "visualization_result",
                    "gate": 2,
                    "trial_id": trial_id,
                    "success": bool(result["success"]),
                    "failure_category": result["failure_category"],
                    "panda_max_deviation_rad": float(state["panda_max_deviation_rad"]),
                    "support_released": bool(metrics["support_released"]),
                    "gravity_enabled": bool(metrics["gravity_enabled"]),
                    "cube_qpos_rewritten_at_release": False,
                })
                verdict = "PASS" if result["success"] else f"FAIL: {result['failure_category']}"
                _write_visual_status(f"Gate 2 可视化完成 | {verdict}。按 R 重播，按 Q 退出。")

                final_deadline = (
                    None
                    if auto_exit_after_final_s is None
                    else time.monotonic() + float(auto_exit_after_final_s)
                )
                while not router.stop_event.is_set():
                    native_viewer = getattr(getattr(env, "viewer", None), "viewer", None)
                    is_running = getattr(native_viewer, "is_running", None)
                    if callable(is_running) and not bool(is_running()):
                        exit_reason = "viewer_closed"
                        break
                    env.viewer.update()
                    if router.take_reset():
                        exit_reason = "reset"
                        break
                    if final_deadline is not None and time.monotonic() >= final_deadline:
                        exit_reason = "auto_exit"
                        break
                    time.sleep(0.02)
                if exit_reason == "reset":
                    continue
                break
        except KeyboardInterrupt:
            exit_reason = "ctrl_c"
            writer.event({"event": "visualization_interrupted", "gate": 2, "reason": "ctrl_c"})
        except Exception as exc:
            exit_reason = "exception"
            summary["exception"] = f"{type(exc).__name__}: {exc}"
            writer.log(f"visualization_exception {summary['exception']}")
            writer.event({"event": "visualization_exception", "gate": 2, "error": summary["exception"]})
        finally:
            recorder.close()
            summary["replay_count"] = replay_count
            summary["exit_reason"] = exit_reason
            summary["recording"] = recorder.summary()
            summary["last_result"] = last_result
            summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            if "exception" in summary:
                summary["status"] = "visualization_exception"
            elif last_result is None:
                summary["status"] = "visualization_stopped"
            else:
                summary["status"] = "passed_gate2_visualization" if last_result["success"] else "failed_gate2_visualization"
            writer.write_json("summary.json", summary)
            writer.close()
            if env is not None:
                env.close()
        return (1 if "exception" in summary else 0), output, summary

    def _move_to_pose(
        self,
        env: Any,
        composer: PandaAllegroActionComposer,
        *,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        hand_targets: np.ndarray,
        gate: int,
        trial_id: str,
        phase: str,
        writer: ArtifactWriter,
    ) -> dict[str, Any]:
        max_steps = max(1, int(round(self.config.motion_timeout_s * self.config.control_hz)))
        max_position_error = 0.0
        max_orientation_error = 0.0
        initial_position_error: float | None = None
        initial_orientation_error: float | None = None
        records: list[tuple[ContactSummary, dict[str, Any]]] = []
        for _ in range(max_steps):
            current_position, current_rotation = _ee_pose(composer)
            position_error = target_position - current_position
            rotation_error = _axis_angle_from_rotation(target_rotation @ current_rotation.T)
            position_norm = float(np.linalg.norm(position_error))
            orientation_norm = float(np.linalg.norm(rotation_error))
            if initial_position_error is None:
                initial_position_error = position_norm
                initial_orientation_error = orientation_norm
            max_position_error = max(max_position_error, position_norm)
            max_orientation_error = max(max_orientation_error, orientation_norm)
            if position_norm <= self.config.motion_position_tolerance_m and orientation_norm <= self.config.motion_orientation_tolerance_rad:
                return {
                    "reached": True,
                    "timed_out": False,
                    "records": records,
                    "max_position_error_m": max_position_error,
                    "max_orientation_error_rad": max_orientation_error,
                    "initial_command_distance_m": initial_position_error,
                    "initial_command_orientation_error_rad": initial_orientation_error,
                    "final_position_error_m": position_norm,
                    "final_orientation_error_rad": orientation_norm,
                }
            wrist_delta = np.concatenate((
                np.clip(position_error, -0.004, 0.004),
                np.clip(rotation_error, -0.04, 0.04),
            ))
            records.append(_step(
                env, composer, hand_targets=hand_targets, wrist_delta=wrist_delta, gate=gate,
                trial_id=trial_id, phase=phase, writer=writer, config=self.config,
                target_pose=(target_position, target_rotation),
            ))
        current_position, current_rotation = _ee_pose(composer)
        position_norm = float(np.linalg.norm(target_position - current_position))
        orientation_norm = float(np.linalg.norm(_axis_angle_from_rotation(target_rotation @ current_rotation.T)))
        return {
            "reached": False,
            "timed_out": True,
            "records": records,
            "max_position_error_m": max(max_position_error, position_norm),
            "max_orientation_error_rad": max(max_orientation_error, orientation_norm),
            "initial_command_distance_m": initial_position_error,
            "initial_command_orientation_error_rad": initial_orientation_error,
            "final_position_error_m": position_norm,
            "final_orientation_error_rad": orientation_norm,
        }

    def _prepare_table_cube(self, env: Any, handles: Mapping[str, int]) -> np.ndarray:
        model, data = env.sim.model, env.sim.data
        table_top = np.asarray(data.site_xpos[int(model.site_name2id("table_top"))], dtype=np.float64)
        half_size = np.asarray(model.geom_size[int(handles["geom_id"])], dtype=np.float64)
        position = np.array([self.config.table_cube_xy_m[0], self.config.table_cube_xy_m[1], table_top[2] + half_size[2] + 0.001], dtype=np.float64)
        set_cube_pose(env, handles, position)
        return position

    def run_gate3_trial(self, env: Any, writer: ArtifactWriter, *, trial_id: str, protocol_phase: str) -> dict[str, Any]:
        def phase_event(event: str, phase: str, **values: Any) -> None:
            writer.event({
                "event": event,
                "gate": 3,
                "trial_id": trial_id,
                "phase": phase,
                "sim_time_s": float(env.sim.data.time),
                **values,
            })

        composer = self._reset(env)
        handles = _cube_handles(env)
        open_target = _hand_targets(composer, (0.0, 0.0, 0.0, 0.0))
        close_target = _hand_targets(composer, self.config.static_close_curls)
        _run_for(env, composer, duration_s=self.config.open_settle_s, hand_targets=open_target, gate=3, trial_id=trial_id, phase="safe_open", writer=writer, config=self.config)
        cube_start_position = self._prepare_table_cube(env, handles)
        wrist_position, wrist_rotation = _ee_pose(composer)
        grasp_center = np.asarray(self.config.static_cube_center_tcp_m, dtype=np.float64)
        pregrasp_center = grasp_center + np.array([0.0, 0.0, self.config.pregrasp_clearance_m], dtype=np.float64)
        pregrasp_target = cube_start_position - wrist_rotation @ pregrasp_center
        approach_target = cube_start_position - wrist_rotation @ grasp_center
        writer.event({
            "event": "gate3_targets",
            "gate": 3,
            "trial_id": trial_id,
            "cube_initial_position_m": cube_start_position.tolist(),
            "pregrasp_target_position_m": pregrasp_target.tolist(),
            "approach_target_position_m": approach_target.tolist(),
            "target_rotation_matrix": wrist_rotation.tolist(),
        })
        writer.capture(env, f"g3_{trial_id}_table_start", sim_time=float(env.sim.data.time), force=True)
        phase_event("phase_start", "move_to_pregrasp", target_position_m=pregrasp_target.tolist())
        pregrasp = self._move_to_pose(
            env, composer, target_position=pregrasp_target, target_rotation=wrist_rotation, hand_targets=open_target,
            gate=3, trial_id=trial_id, phase="move_to_pregrasp", writer=writer,
        )
        phase_event("phase_end", "move_to_pregrasp", reached=bool(pregrasp["reached"]), timed_out=bool(pregrasp["timed_out"]))
        not_started_move = {
            "reached": False,
            "timed_out": False,
            "records": [],
            "max_position_error_m": 0.0,
            "max_orientation_error_rad": 0.0,
            "final_position_error_m": None,
            "final_orientation_error_rad": None,
            "initial_command_distance_m": None,
            "initial_command_orientation_error_rad": None,
        }
        approach = dict(not_started_move)
        close_records: list[tuple[ContactSummary, dict[str, Any]]] = []
        lift = dict(not_started_move)
        hold_records: list[tuple[ContactSummary, dict[str, Any]]] = []
        if pregrasp["reached"]:
            phase_event("phase_start", "approach", target_position_m=approach_target.tolist())
            approach = self._move_to_pose(
                env, composer, target_position=approach_target, target_rotation=wrist_rotation, hand_targets=open_target,
                gate=3, trial_id=trial_id, phase="approach", writer=writer,
            )
            phase_event("phase_end", "approach", reached=bool(approach["reached"]), timed_out=bool(approach["timed_out"]))
        if pregrasp["reached"] and approach["reached"]:
            phase_event("phase_start", "close_allegro")
            close_steps = max(1, int(round(self.config.close_duration_s * self.config.control_hz)))
            for index in range(close_steps):
                fraction = float(index + 1) / float(close_steps)
                target = open_target + fraction * (close_target - open_target)
                close_records.append(_step(
                    env, composer, hand_targets=target, wrist_delta=np.zeros(6), gate=3, trial_id=trial_id,
                    phase="close_allegro", writer=writer, config=self.config, target_pose=(approach_target, wrist_rotation),
                ))
            phase_event("phase_end", "close_allegro")
            cube_before_lift = _cube_state(env, handles)
            ee_before_lift, _ = _ee_pose(composer)
            lift_target = approach_target + np.array([0.0, 0.0, self.config.lift_command_m], dtype=np.float64)
            phase_event("phase_start", "lift", target_position_m=lift_target.tolist())
            lift = self._move_to_pose(
                env, composer, target_position=lift_target, target_rotation=wrist_rotation, hand_targets=close_target,
                gate=3, trial_id=trial_id, phase="lift", writer=writer,
            )
            phase_event("phase_end", "lift", reached=bool(lift["reached"]), timed_out=bool(lift["timed_out"]))
            if lift["reached"]:
                phase_event("phase_start", "hold")
                hold_records = _run_for(
                    env, composer, duration_s=self.config.lift_hold_s, hand_targets=close_target, gate=3,
                    trial_id=trial_id, phase="hold", writer=writer, config=self.config, target_pose=(lift_target, wrist_rotation),
                )
                phase_event("phase_end", "hold")
        else:
            cube_before_lift = _cube_state(env, handles)
            ee_before_lift, _ = _ee_pose(composer)
        all_records = list(pregrasp["records"]) + list(approach["records"]) + close_records + list(lift["records"]) + hold_records
        approach_contacts = _merge_contact_metrics(list(pregrasp["records"]) + list(approach["records"]))
        close_contacts = _merge_contact_metrics(close_records)
        lift_contacts = _merge_contact_metrics(list(lift["records"]))
        hold_contacts = _merge_contact_metrics(hold_records)
        lift_hold_contacts = _merge_contact_metrics(list(lift["records"]) + hold_records)
        cube_final = _cube_state(env, handles)
        cube_final_position = np.asarray(cube_final["position_m"], dtype=np.float64)
        cube_before_lift_position = np.asarray(cube_before_lift["position_m"], dtype=np.float64)
        ee_final_position, _ = _ee_pose(composer)
        lift_and_hold_events = list(lift["records"]) + hold_records
        cube_max_height = max(
            [cube_start_position[2], cube_before_lift_position[2]] + [
                np.asarray(event.get("cube_position_m", cube_final_position), dtype=np.float64)[2]
                for _summary, event in lift_and_hold_events
                if "cube_position_m" in event
            ] + [cube_final_position[2]]
        )
        hold_heights = [
            float(np.asarray(event["cube_position_m"], dtype=np.float64)[2])
            for _summary, event in hold_records if "cube_position_m" in event
        ]
        if hold_heights:
            minimum_hold_height = min(hold_heights + [float(cube_final_position[2])])
        else:
            minimum_hold_height = float(cube_final_position[2])
        max_lift_m = float(cube_max_height - cube_start_position[2])
        minimum_hold_lift_m = float(minimum_hold_height - cube_start_position[2])
        cube_fingers = set(close_contacts["cube_fingers"]) | set(lift_hold_contacts["cube_fingers"])
        contact_seen = bool(close_contacts["cube_contacts"] > 0)
        def opposed(summary: ContactSummary) -> bool:
            fingers = summary.cube_fingers
            return "thumb" in fingers and len(fingers & {"index", "middle", "ring"}) >= 2

        close_opposition_seen = any(opposed(summary) for summary, _event in close_records)
        lift_opposition_seen = any(opposed(summary) for summary, _event in lift["records"])
        hold_opposition_current_s = 0.0
        hold_opposition_longest_s = 0.0
        for summary, _event in hold_records:
            if opposed(summary):
                hold_opposition_current_s += 1.0 / self.config.control_hz
                hold_opposition_longest_s = max(hold_opposition_longest_s, hold_opposition_current_s)
            else:
                hold_opposition_current_s = 0.0
        # A union of contacts across frames cannot prove a stable grasp. Require
        # simultaneous thumb + two-finger opposition during close, lift, and
        # the full measured hold interval.
        contact_lost = bool(
            not close_opposition_seen
            or not lift_opposition_seen
            or hold_opposition_longest_s + 1e-9 < self.config.lift_hold_s
        )
        hand_error = float(np.max(np.abs(composer.hand_qpos() - close_target)))
        table_contact_after_lift = bool(hold_contacts["table_contact"])
        # Relative motion with respect to the wrist after close is the slip
        # metric. No arm or cube pose is written after initial placement.
        slip_distance = float(np.linalg.norm(
            (cube_final_position - cube_before_lift_position)
            - (ee_final_position - ee_before_lift)
        ))
        unsafe_collision = bool(
            approach_contacts["unexpected_hand_collision"]
            or approach_contacts["hand_table_contact"]
            or approach_contacts["panda_table_contact"]
            or close_contacts["unexpected_hand_collision"]
            or close_contacts["hand_table_contact"]
            or close_contacts["panda_table_contact"]
            or lift_hold_contacts["unexpected_hand_collision"]
            or lift_hold_contacts["hand_table_contact"]
            or lift_hold_contacts["panda_table_contact"]
        )
        completed_move_errors = [
            float(phase["final_position_error_m"])
            for phase in (pregrasp, approach, lift)
            if phase.get("reached") and phase.get("final_position_error_m") is not None
        ]
        metrics = {
            "pregrasp_pose_error": not bool(pregrasp["reached"]),
            "approach_collision": unsafe_collision,
            "osc_tracking_error": bool(any(error > self.config.osc_tracking_error_m for error in completed_move_errors)),
            "timed_out": bool(pregrasp["timed_out"] or approach["timed_out"] or lift["timed_out"]),
            "cube_contacts": int(close_contacts["cube_contacts"] + lift_hold_contacts["cube_contacts"]),
            "cube_fingers": sorted(cube_fingers),
            "hand_closed": bool(hand_error <= self.config.joint_target_error_rad),
            "contact_lost_during_lift": contact_lost,
            "slip_distance_m": slip_distance,
            "lift_m": minimum_hold_lift_m,
            "max_lift_m": max_lift_m,
            "minimum_hold_lift_m": minimum_hold_lift_m,
            "hold_duration_s": float(len(hold_records) / self.config.control_hz),
            "table_contact_after_lift": table_contact_after_lift,
            "cube_initial_height_m": float(cube_start_position[2]),
            "cube_height_before_lift_m": float(cube_before_lift_position[2]),
            "cube_final_height_m": float(cube_final_position[2]),
            "cube_max_height_m": float(cube_max_height),
            "cube_minimum_hold_height_m": minimum_hold_height,
            "ee_final_position_m": ee_final_position.tolist(),
            "hand_target_error_rad": hand_error,
            "close_opposition_seen": close_opposition_seen,
            "lift_opposition_seen": lift_opposition_seen,
            "hold_opposition_s": hold_opposition_longest_s,
            "unsafe_collision": unsafe_collision,
        }
        success, failure = classify_gate3(metrics, self.config)
        result = {
            "gate": 3,
            "trial_id": trial_id,
            "protocol_phase": protocol_phase,
            "config_sha256": self.config.sha256,
            "success": success,
            "passed": success,
            "failure_category": failure,
            "metrics": metrics,
            "phases": {
                "move_to_pregrasp": {key: value for key, value in pregrasp.items() if key != "records"},
                "approach": {key: value for key, value in approach.items() if key != "records"},
                "lift": {key: value for key, value in lift.items() if key != "records"},
                "close_duration_s": self.config.close_duration_s,
                "hold_duration_s": self.config.lift_hold_s,
            },
            "cube_final": cube_final,
            "hand": {"targets_rad": close_target.tolist(), "actual_rad": composer.hand_qpos().tolist()},
        }
        writer.trial(result)
        writer.log(f"gate3 {trial_id} {'passed' if success else 'failed'} failure={failure}")
        writer.capture(env, f"g3_{trial_id}_final", sim_time=float(env.sim.data.time), force=True)
        return result

    def run_gate3_protocol(self, env: Any, writer: ArtifactWriter) -> dict[str, Any]:
        return self._run_protocol(self.run_gate3_trial, env, writer, gate=3)


def run_validation(gate: int | str = "all", *, output_root: Path | None = None, record_frames: bool = True, config: GraspValidationConfig = DEFAULT_CONFIG) -> tuple[int, Path, dict[str, Any]]:
    """Convenience API used by the one CLI entry and focused tests."""

    return GraspValidationRunner(config=config, output_root=output_root, record_frames=record_frames).run(gate)
