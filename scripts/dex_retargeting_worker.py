"""Dedicated Python worker for official ``dex-retargeting==0.4.6``.

This file deliberately has no project-package imports.  It is launched by the
robosuite process with a separate Python interpreter which can load the native
Pinocchio stack.  Its standard output is a newline-delimited JSON protocol;
all third-party chatter is redirected to stderr so the parent never needs to
parse log output as a reply.

Request / response protocol
===========================

Every request is one JSON object per input line.  All replies have the request
``id`` (or ``null`` for malformed JSON) and an ``ok`` boolean.

``init``
    ``{"id": 1, "op": "init", "wheel_path": "...", "config_path": "...",
    "urdf_root": "..."}``

    The worker reads the supplied wheel metadata, requires the official
    ``dex_retargeting`` version 0.4.6, adds that exact wheel before importing
    dex, resolves the supplied URDF root, and builds ``RetargetingConfig``.

``retarget``
    ``{"id": 2, "op": "retarget", "landmarks": [[x, y, z], ...]}``

    ``landmarks`` must be a finite 21 by 3 MANO-space array.  The worker uses
    the official vector configuration's human-index matrix to form
    ``ref_value`` and returns the source-order solver output.

``reset`` and ``close``
    Reset clears official solver / filter history.  Close replies once and
    then terminates the process cleanly.

Optional ``grasp_terms`` in ``init`` (project extension, DexPilot only)
    A JSON object of settings (see ``GRASP_TERM_DEFAULTS``): posture terms,
    a thumb lift gain, per-finger length scaling and a continuous pinch
    mapping in place of the official snap (optionally measured between finger
    pad points instead of fingertip frames), all computed only from the
    operator's landmarks in the current frame.  Retarget replies then carry a
    ``mapping`` object (finger scales, pinch closeness).  See the comment
    above ``GRASP_TERM_DEFAULTS``.

Failures are structured as ``{"ok": false, "error": {"code", "type",
"message"}}``.  No request can make the worker silently use a package other
than the wheel it was explicitly given during ``init``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, TextIO
import zipfile

import numpy as np


OFFICIAL_DEX_PACKAGE = "dex_retargeting"
OFFICIAL_DEX_VERSION = "0.4.6"


class WorkerError(RuntimeError):
    """Base class for failures that can be returned through the protocol."""

    code = "worker_error"


class InvalidRequest(WorkerError):
    code = "invalid_request"


class NotInitialized(WorkerError):
    code = "not_initialized"


class InvalidLandmarks(WorkerError):
    code = "invalid_landmarks"


class DexWheelError(WorkerError):
    code = "dex_wheel_error"


class DexInitializationError(WorkerError):
    code = "dex_initialization_error"


class RetargetingError(WorkerError):
    code = "retargeting_error"


@dataclass(frozen=True)
class WorkerState:
    """The one official solver owned by a worker process."""

    retargeting: Any
    wheel_path: Path
    config_path: Path
    urdf_root: Path
    wheel_sha256: str
    config_sha256: str
    human_indices: np.ndarray
    joint_names: tuple[str, ...]
    grasp_terms: dict[str, Any] | None = None


def _normalise_package_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(".", "_")


def _error_payload(request_id: Any, error: BaseException, *, code: str | None = None) -> dict[str, Any]:
    """Create the sole public error representation for malformed requests."""

    safe_id = request_id if isinstance(request_id, (str, int, float, bool)) or request_id is None else None
    return {
        "id": safe_id,
        "ok": False,
        "error": {
            "code": code or getattr(error, "code", "internal_error"),
            "type": type(error).__name__,
            "message": str(error) or type(error).__name__,
        },
    }


def _write_json(stream: TextIO, payload: dict[str, Any]) -> None:
    """Write exactly one JSON response line and never use stdout for logs."""

    stream.write(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


def _require_text_path(request: dict[str, Any], field: str, *, file_required: bool) -> Path:
    value = request.get(field)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise InvalidRequest(f"{field} must be a nonempty path string")
    try:
        path = Path(value).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise InvalidRequest(f"{field} cannot be resolved: {exc}") from exc
    if file_required and not path.is_file():
        raise InvalidRequest(f"{field} is not a readable file: {path}")
    if not file_required and not path.is_dir():
        raise InvalidRequest(f"{field} is not a readable directory: {path}")
    return path


def _read_wheel_metadata(wheel_path: Path) -> dict[str, str]:
    """Validate the supplied wheel before importing anything from it."""

    try:
        with zipfile.ZipFile(wheel_path) as archive:
            metadata_names = [
                name for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise DexWheelError("wheel must contain exactly one .dist-info/METADATA file")
            raw = archive.read(metadata_names[0]).decode("utf-8")
    except DexWheelError:
        raise
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile, KeyError) as exc:
        raise DexWheelError(f"cannot read dex wheel metadata: {type(exc).__name__}: {exc}") from exc

    metadata: dict[str, str] = {}
    for line in raw.splitlines():
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata.setdefault(key.strip().lower(), value.strip())
    package_name = metadata.get("name", "")
    version = metadata.get("version", "")
    if _normalise_package_name(package_name) != OFFICIAL_DEX_PACKAGE:
        raise DexWheelError(
            f"wheel package must be {OFFICIAL_DEX_PACKAGE!r}, found {package_name!r}"
        )
    if version != OFFICIAL_DEX_VERSION:
        raise DexWheelError(
            f"dex worker requires version {OFFICIAL_DEX_VERSION}, wheel contains {version or '<missing>'}"
        )
    return {"name": package_name, "version": version}


def _sha256(path: Path) -> str:
    """Hash source artifacts without putting their contents on the protocol."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise DexInitializationError(f"cannot hash {path}: {type(exc).__name__}: {exc}") from exc
    return digest.hexdigest()


def _import_exact_wheel(wheel_path: Path) -> Any:
    """Import dex only from the validated supplied wheel, ahead of site-packages."""

    wheel_text = str(wheel_path)
    normalised_wheel = os.path.normcase(os.path.normpath(wheel_text))
    sys.path[:] = [
        entry for entry in sys.path
        if os.path.normcase(os.path.normpath(entry)) != normalised_wheel
    ]
    sys.path.insert(0, wheel_text)
    # A long-lived worker may receive a second init.  Never retain the prior
    # package, because that would violate the supplied-wheel contract.
    for module_name in tuple(sys.modules):
        if module_name == OFFICIAL_DEX_PACKAGE or module_name.startswith(OFFICIAL_DEX_PACKAGE + "."):
            del sys.modules[module_name]
    importlib.invalidate_caches()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            package = importlib.import_module(OFFICIAL_DEX_PACKAGE)
    except Exception as exc:
        raise DexInitializationError(
            f"cannot import dex-retargeting from supplied wheel: {type(exc).__name__}: {exc}"
        ) from exc

    loaded_from = os.path.normcase(str(getattr(package, "__file__", "")))
    if normalised_wheel not in loaded_from:
        raise DexInitializationError(
            "dex import did not originate from the supplied wheel; refusing ambiguous package resolution"
        )
    package_version = str(getattr(package, "__version__", ""))
    if package_version != OFFICIAL_DEX_VERSION:
        raise DexInitializationError(
            f"dex package imported from wheel reports {package_version or '<missing>'}, "
            f"expected {OFFICIAL_DEX_VERSION}"
        )
    return package


def _build_state(request: dict[str, Any]) -> WorkerState:
    wheel_path = _require_text_path(request, "wheel_path", file_required=True)
    config_path = _require_text_path(request, "config_path", file_required=True)
    urdf_root = _require_text_path(request, "urdf_root", file_required=False)
    _read_wheel_metadata(wheel_path)
    wheel_sha256 = _sha256(wheel_path)
    config_sha256 = _sha256(config_path)
    _import_exact_wheel(wheel_path)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            module = importlib.import_module("dex_retargeting.retargeting_config")
            config_type = module.RetargetingConfig
            config_type.set_default_urdf_dir(str(urdf_root))
            retargeting = config_type.load_from_file(str(config_path)).build()
    except Exception as exc:
        raise DexInitializationError(
            f"official dex RetargetingConfig build failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        human_indices = np.asarray(retargeting.optimizer.target_link_human_indices, dtype=np.intp)
        joint_names = tuple(str(name) for name in retargeting.joint_names)
    except Exception as exc:
        raise DexInitializationError(
            f"official dex solver metadata is unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    # Hand-specific shapes (Allegro vector: 2x4 / 16 joints; Shadow DexPilot:
    # 2x15 / 24 joints) are enforced by the parent adapter for that hand.
    if (
        human_indices.ndim != 2
        or human_indices.shape[0] != 2
        or human_indices.shape[1] < 1
        or np.any(human_indices < 0)
        or np.any(human_indices >= 21)
    ):
        raise DexInitializationError(
            "official dex configuration must expose a 2xN human-index matrix within 21 landmarks"
        )
    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise DexInitializationError(
            "official dex configuration must expose distinct output joint names"
        )
    return WorkerState(
        retargeting=retargeting,
        wheel_path=wheel_path,
        config_path=config_path,
        urdf_root=urdf_root,
        wheel_sha256=wheel_sha256,
        config_sha256=config_sha256,
        human_indices=human_indices.copy(),
        joint_names=joint_names,
    )


# ---------------------------------------------------------------------------
# Project grasp terms (optional): official DexPilot with a re-weighted mapping.
#
# Everything here is a function of the operator's 21 landmarks in the current
# frame (no object, scene, policy, grasp preset or hand-shape memory), so the
# robot hand follows the operator with weight 1; only the camera-to-robot
# conversion differs from official DexPilot:
#
# * Posture: DexPilot matches fingertip vectors only, which leaves the thumb's
#   five joints and each finger's knuckle / middle-joint split to the warm
#   start (lateral grips and power wraps came out 25-45 deg off). The
#   operator's thumb segments and finger proximal / middle phalanges are added
#   as direction targets at the robot segment lengths.
# * Thumb lift gain: lifting the thumb off the palm (to bring it in front of
#   the fingers) moves it along the palm normal, which is the camera's depth
#   axis when the palm faces the camera, and monocular depth is compressed.
#   The thumb landmarks' offset from the operator's palm plane is multiplied
#   by ``thumb_lift_gain``; a thumb lying in the palm plane is unchanged.
# * Per-finger length: one scaling for all fingers gives a robot finger as
#   long as the others a short target when the human finger is short (the
#   little finger), so a straight hand still curled it. Each finger's palm
#   part (wrist to MCP) and finger part (MCP to tip) are instead scaled to the
#   robot's lengths, measured from the landmarks in the same frame, so the
#   robot finger is as straight as the operator's (``_per_finger_tips``).
# * Pinch: official DexPilot snaps a thumb-finger pair to contact below 3 cm
#   and releases it above 5 cm (a threshold with hysteresis). Here the robot
#   thumb-finger distance is always ``scaling * max(0, d - pinch_gap_m)``
#   along the operator's direction, where ``pinch_gap_m`` is the gap left
#   between the two tip landmarks when the finger pads touch. Its weight rises
#   smoothly from the official 1 to 200 between 5 and 3 cm, and the official
#   3 cm robot fingertip spacing for two fingers that both meet the thumb is
#   blended in by the same factor.
# * Pinch points (``pinch_points``): "tips" measures the robot thumb-finger
#   distance between the official fingertip frames, which sit inside the
#   fingertip about 5 mm above the pad surface (two tip frames together means
#   pads pressed ~1 cm into each other). "pads" measures it between points on
#   the pads (``_SHADOW_PAD_POINTS``), so distance 0 is pad-to-pad contact, as
#   in a human pinch. Only the four thumb-finger vectors change; the
#   wrist-to-tip and finger-finger vectors keep the fingertip frames.
# ---------------------------------------------------------------------------

GRASP_TERM_DEFAULTS: dict[str, Any] = {
    "thumb_segment_weights": [0.6, 0.6, 0.6],  # metacarpal, proximal, distal
    "finger_segment_weights": [0.3, 0.15],  # proximal, middle phalanx
    "thumb_lift_gain": 1.0,  # x the thumb's offset from the palm plane
    "finger_length_scaling": False,  # per-finger robot / operator chain length
    "pinch_gap_m": 0.01,  # tip-landmark distance that maps to robot contact
    "pinch_points": "tips",  # robot thumb-finger distance between "tips" or "pads"
}
THUMB_LIFT_GAIN_RANGE = (0.5, 4.0)
PINCH_GAP_RANGE_M = (0.0, 0.03)
PINCH_POINT_CHOICES = ("tips", "pads")
# Pad points of the Shadow distal phalanges: official tip frame -> (distal
# link, pad frame name, offset in the distal link frame in m). Each is on the
# palmar surface of the distal collision hull (MuJoCo collides the convex hull
# of f_distal_pst / th_distal_pst; the URDF and MJCF distal frames hold the
# same mesh unrotated) at ~0.69 of the segment from its joint: finger z 22 mm
# (tip frame 26 mm, mesh end 32 mm), thumb z 24 mm (tip frame 27.5, end 35).
# Flexion turns a distal link about +x toward -y, so the pad side is -y.
_SHADOW_PAD_POINTS = {
    "thtip": ("thdistal", "thpad", (0.0, -0.0070, 0.024)),
    "fftip": ("ffdistal", "ffpad", (0.0, -0.0061, 0.022)),
    "mftip": ("mfdistal", "mfpad", (0.0, -0.0061, 0.022)),
    "rftip": ("rfdistal", "rfpad", (0.0, -0.0061, 0.022)),
    "lftip": ("lfdistal", "lfpad", (0.0, -0.0061, 0.022)),
}
# Per-finger length scales outside this range mean a degenerate hand frame.
_FINGER_SCALE_RANGE = (0.5, 3.0)
_PALM_LANDMARKS = (0, 5, 9, 13, 17)
_THUMB_LANDMARKS = (1, 2, 3, 4)
# Operator chains from the wrist and the robot links they correspond to.
_FINGER_CHAINS = (
    ((0, 5, 6, 7, 8), ("ee_link", "ffproximal", "ffmiddle", "ffdistal", "fftip")),
    ((0, 9, 10, 11, 12), ("ee_link", "mfproximal", "mfmiddle", "mfdistal", "mftip")),
    ((0, 13, 14, 15, 16), ("ee_link", "rfproximal", "rfmiddle", "rfdistal", "rftip")),
    ((0, 17, 18, 19, 20), ("ee_link", "lfproximal", "lfmiddle", "lfdistal", "lftip")),
)
_TIP_LANDMARKS = (4, 8, 12, 16, 20)  # DexPilot finger order: thumb, ff, mf, rf, lf
# (robot origin link, robot task link, human from, human to, weight group, index in group)
_SHADOW_POSTURE_SEGMENTS = (
    ("thproximal", "thhub", 1, 2, "thumb", 0),
    ("thhub", "thdistal", 2, 3, "thumb", 1),
    ("thdistal", "thtip", 3, 4, "thumb", 2),
    ("ffproximal", "ffmiddle", 5, 6, "finger", 0), ("ffmiddle", "ffdistal", 6, 7, "finger", 1),
    ("mfproximal", "mfmiddle", 9, 10, "finger", 0), ("mfmiddle", "mfdistal", 10, 11, "finger", 1),
    ("rfproximal", "rfmiddle", 13, 14, "finger", 0), ("rfmiddle", "rfdistal", 14, 15, "finger", 1),
    ("lfproximal", "lfmiddle", 17, 18, "finger", 0), ("lfmiddle", "lfdistal", 18, 19, "finger", 1),
)
def _grasp_term_settings(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InvalidRequest("grasp_terms must be a JSON object")
    unknown = set(raw) - set(GRASP_TERM_DEFAULTS) - {"version", "description"}
    if unknown:
        raise InvalidRequest(f"unknown grasp_terms fields: {sorted(unknown)}")
    settings = {**GRASP_TERM_DEFAULTS, **{key: raw[key] for key in GRASP_TERM_DEFAULTS if key in raw}}
    thumb = np.asarray(settings["thumb_segment_weights"], dtype=np.float64)
    finger = np.asarray(settings["finger_segment_weights"], dtype=np.float64)
    if thumb.shape != (3,) or finger.shape != (2,) or np.any(thumb < 0) or np.any(finger < 0):
        raise InvalidRequest("grasp_terms weights must be 3 thumb and 2 finger nonnegative numbers")
    try:
        gain, gap = float(settings["thumb_lift_gain"]), float(settings["pinch_gap_m"])
    except (TypeError, ValueError) as exc:
        raise InvalidRequest("thumb_lift_gain and pinch_gap_m must be numbers") from exc
    if not THUMB_LIFT_GAIN_RANGE[0] <= gain <= THUMB_LIFT_GAIN_RANGE[1]:
        raise InvalidRequest(f"thumb_lift_gain must be within {THUMB_LIFT_GAIN_RANGE}")
    if not PINCH_GAP_RANGE_M[0] <= gap <= PINCH_GAP_RANGE_M[1]:
        raise InvalidRequest(f"pinch_gap_m must be within {PINCH_GAP_RANGE_M} m")
    if not isinstance(settings["finger_length_scaling"], bool):
        raise InvalidRequest("finger_length_scaling must be true or false")
    if settings["pinch_points"] not in PINCH_POINT_CHOICES:
        raise InvalidRequest(f"pinch_points must be one of {PINCH_POINT_CHOICES}")
    return {"thumb_segment_weights": thumb.tolist(), "finger_segment_weights": finger.tolist(),
            "thumb_lift_gain": gain, "finger_length_scaling": settings["finger_length_scaling"],
            "pinch_gap_m": gap, "pinch_points": settings["pinch_points"]}


def _add_pad_frames(robot: Any, pin: Any) -> None:
    """Add the ``_SHADOW_PAD_POINTS`` pad frames to the solver's Pinocchio model.

    They are BODY frames fixed to the distal links, so the official
    ``get_link_index`` / ``get_link_pose`` and the frame Jacobians treat them
    like the URDF's own fingertip frames. Existing frame ids do not change.
    """

    model = robot.model
    for distal, pad, offset in _SHADOW_PAD_POINTS.values():
        if not model.existFrame(distal, pin.BODY):
            raise DexInitializationError(f"pad points need the Shadow link {distal!r}")
        parent = model.getFrameId(distal, pin.BODY)
        frame = model.frames[parent]
        placement = frame.placement * pin.SE3(np.eye(3), np.asarray(offset, dtype=np.float64))
        model.addFrame(pin.Frame(pad, frame.parentJoint, parent, placement, pin.FrameType.BODY))
    robot.data = model.createData()


def _lift_thumb(landmarks: np.ndarray, gain: float) -> np.ndarray:
    """Multiply the thumb landmarks' offset from the operator's palm plane by ``gain``.

    The palm plane is the least-squares plane through the wrist and the four
    finger MCPs, so the result does not depend on how the hand is turned.
    """

    if gain == 1.0:
        return landmarks
    palm = landmarks[list(_PALM_LANDMARKS)]
    centre = palm.mean(axis=0)
    normal = np.linalg.svd(palm - centre)[2][2]
    thumb = list(_THUMB_LANDMARKS)
    lifted = landmarks.copy()
    lifted[thumb] += np.outer((gain - 1.0) * ((landmarks[thumb] - centre) @ normal), normal)
    return lifted


def _uniform_tips(landmarks: np.ndarray, scale: float) -> np.ndarray:
    """Official DexPilot: every fingertip scaled about the wrist by one factor."""

    return landmarks[0] + scale * (landmarks[list(_TIP_LANDMARKS)] - landmarks[0])


def _per_finger_tips(landmarks: np.ndarray, robot_lengths: np.ndarray,
                     thumb_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Robot fingertip targets with each finger's palm and finger part at robot length.

    For finger f the target is ``wrist + s_palm (MCP - wrist) + s_finger
    (tip - MCP)``, with ``s_palm`` = robot wrist-to-knuckle / operator
    wrist-to-MCP and ``s_finger`` = robot / operator phalanx length sum, all
    measured in this frame. The MCP-to-tip chord is then the robot finger
    length times the operator finger's own straightness, so a straight finger
    maps to a straight finger and a half-curled one to a proportionally curled
    one, whatever the proportions of the two hands. The thumb keeps
    ``thumb_scale``: it has no straight rest pose to preserve.
    Returns the five tips and the (palm, finger) scale of each finger.
    """

    wrist = landmarks[0]
    tips = np.empty((len(_TIP_LANDMARKS), 3))
    tips[0] = wrist + thumb_scale * (landmarks[_TIP_LANDMARKS[0]] - wrist)
    scales = np.empty((len(_FINGER_CHAINS), 2))
    for k, ((human, _robot), (palm_length, finger_length)) in enumerate(zip(_FINGER_CHAINS, robot_lengths)):
        chain = landmarks[list(human)]
        bones = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        scales[k] = palm_length / max(bones[0], 1e-9), finger_length / max(bones[1:].sum(), 1e-9)
        if np.any(scales[k] < _FINGER_SCALE_RANGE[0]) or np.any(scales[k] > _FINGER_SCALE_RANGE[1]):
            raise InvalidLandmarks(f"operator hand gives robot / human finger scales {scales[k].round(2).tolist()}")
        tips[k + 1] = wrist + scales[k, 0] * (chain[1] - wrist) + scales[k, 1] * (chain[-1] - chain[1])
    return tips, scales


def _scaled_targets(landmarks: np.ndarray, human_indices: np.ndarray, tips: np.ndarray) -> np.ndarray:
    """DexPilot's vectors (task - origin) with the fingertips replaced by robot targets."""

    scaled = landmarks.copy()
    scaled[list(_TIP_LANDMARKS)] = tips
    return scaled[human_indices[1]] - scaled[human_indices[0]]


def _pinch_closeness(distance: np.ndarray, near: float, far: float) -> np.ndarray:
    """1 at or below ``near``, 0 at or beyond ``far``, smoothstep in between."""

    t = np.clip((np.asarray(distance, dtype=np.float64) - near) / max(far - near, 1e-9), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def _pinch_distance(distance: np.ndarray, gap: float, scaling: float) -> np.ndarray:
    """Robot thumb-finger target distance: ``scaling * max(0, d - gap)``."""

    return scaling * np.maximum(np.asarray(distance, dtype=np.float64) - gap, 0.0)


def _install_grasp_terms(retargeting: Any, raw: Any) -> dict[str, Any]:
    """Give the official DexPilot optimizer instance the extended objective."""

    import types

    import torch

    settings = _grasp_term_settings(raw)
    optimizer = retargeting.optimizer
    if getattr(optimizer, "retargeting_type", "") != "DEXPILOT" or getattr(optimizer, "num_fingers", 0) != 5:
        raise DexInitializationError("grasp_terms need the five-finger DexPilot optimizer")
    robot = optimizer.robot
    pinocchio = importlib.import_module("pinocchio")
    origin_names = list(optimizer.origin_link_names)
    task_names = list(optimizer.task_link_names)
    if settings["pinch_points"] == "pads":
        _add_pad_frames(robot, pinocchio)
        # The first len_s1 DexPilot vectors are finger tip -> thumb tip.
        len_s1 = len(optimizer.projected) - len(optimizer.s2_project_index_task)
        for k in range(len_s1):
            if origin_names[k] not in _SHADOW_PAD_POINTS or task_names[k] not in _SHADOW_PAD_POINTS:
                raise DexInitializationError(
                    f"thumb-finger vector {k} links {origin_names[k]} -> {task_names[k]} have no pad point")
            origin_names[k] = _SHADOW_PAD_POINTS[origin_names[k]][1]
            task_names[k] = _SHADOW_PAD_POINTS[task_names[k]][1]
    posture_links = [name for segment in _SHADOW_POSTURE_SEGMENTS for name in segment[:2]]
    link_names = list(dict.fromkeys(list(optimizer.computed_link_names) + origin_names + task_names + posture_links))
    link_indices = optimizer.get_link_indices(link_names)
    robot.compute_forward_kinematics(np.zeros(robot.dof))
    at_zero = np.array([robot.get_link_pose(index)[:3, 3] for index in link_indices])
    lookup = {name: k for k, name in enumerate(link_names)}
    chain_points = [
        np.array([robot.get_link_pose(robot.get_link_index(name))[:3, 3] for name in robot_chain])
        for _human, robot_chain in _FINGER_CHAINS
    ]
    post_origin = [lookup[segment[0]] for segment in _SHADOW_POSTURE_SEGMENTS]
    post_task = [lookup[segment[1]] for segment in _SHADOW_POSTURE_SEGMENTS]
    groups = {"thumb": settings["thumb_segment_weights"], "finger": settings["finger_segment_weights"]}
    terms = types.SimpleNamespace(
        link_indices=link_indices,
        dex_origin=torch.tensor([lookup[name] for name in origin_names]),
        dex_task=torch.tensor([lookup[name] for name in task_names]),
        post_origin=torch.tensor(post_origin), post_task=torch.tensor(post_task),
        human_from=np.array([segment[2] for segment in _SHADOW_POSTURE_SEGMENTS]),
        human_to=np.array([segment[3] for segment in _SHADOW_POSTURE_SEGMENTS]),
        robot_lengths=np.linalg.norm(at_zero[post_task] - at_zero[post_origin], axis=1),
        weights=torch.tensor([groups[segment[4]][segment[5]] for segment in _SHADOW_POSTURE_SEGMENTS],
                             dtype=torch.float64),
        # (wrist-to-knuckle, phalanx sum) per finger at the robot's zero pose.
        robot_finger_lengths=np.array([
            (np.linalg.norm(points[1] - points[0]), np.linalg.norm(np.diff(points[1:], axis=0), axis=1).sum())
            for points in chain_points
        ]),
        pinch_gap=settings["pinch_gap_m"],
        last_mapping=None,
        pinocchio=pinocchio,
    )
    optimizer._grasp_terms = terms
    optimizer.get_objective_function = types.MethodType(_grasp_terms_objective, optimizer)
    return settings


def _grasp_terms_objective(self: Any, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray) -> Any:
    """Official 0.4.6 DexPilot objective (MIT, Yuzhe Qin) with the project mapping.

    ``target_vector`` (built by the worker) stacks the 15 DexPilot vectors
    scaled for the robot (``_scaled_targets``), the same 15 vectors as the
    camera measured them (unscaled, no thumb lift gain: a pinch is judged on
    what the camera saw touch), and the 21 operator landmarks after the thumb
    lift gain (MANO frame) for the posture terms.
    """

    import torch

    terms = self._grasp_terms
    count = len(self.origin_link_names)
    human = np.asarray(target_vector[count:2 * count], dtype=np.float64)
    landmarks = np.asarray(target_vector[2 * count:], dtype=np.float64)
    target_vector = np.asarray(target_vector[:count], dtype=np.float64)
    qpos = np.zeros(self.num_joints)
    qpos[self.idx_pin2fixed] = fixed_qpos

    len_proj = len(self.projected)
    len_s2 = len(self.s2_project_index_task)
    len_s1 = len_proj - len_s2
    # Continuous pinch (replaces the official snap / release thresholds):
    # thumb-finger pairs keep the operator's direction at scaling * (d - gap).
    thumb_dist = np.linalg.norm(human[:len_s1], axis=1)
    closeness = _pinch_closeness(thumb_dist, self.project_dist, self.escape_dist)
    both_close = np.minimum(closeness[self.s2_project_index_origin], closeness[self.s2_project_index_task])
    reference_vec = target_vector.copy()
    reference_vec[:len_s1] = (human[:len_s1] / (thumb_dist[:, None] + 1e-9)
                              * _pinch_distance(thumb_dist, terms.pinch_gap, self.scaling)[:, None])
    # Two fingers that both meet the thumb: official robot fingertip spacing
    # floor (3 cm), blended in by how close both are to the thumb.
    pairs = reference_vec[len_s1:len_proj]
    pair_dist = np.linalg.norm(pairs, axis=1)
    floor = self.projected_dist[len_s1:len_proj]
    reference_vec[len_s1:len_proj] = pairs / (pair_dist[:, None] + 1e-9) * (
        pair_dist + both_close * np.maximum(floor - pair_dist, 0.0))[:, None]
    weight = np.concatenate([1.0 + 199.0 * closeness, 1.0 + 399.0 * both_close,
                             np.full(self.num_fingers, float(len_proj + self.num_fingers))])
    weight = torch.from_numpy(weight.astype(np.float32))
    self.projected[:] = np.concatenate([closeness, both_close]) >= 0.5  # introspection only
    terms.last_mapping = {"pinch_closeness": closeness.round(4).tolist(),
                          "thumb_finger_distance_m": thumb_dist.round(5).tolist()}
    torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
    # Posture: the operator's segment direction at the robot segment's length.
    human_segments = landmarks[terms.human_to] - landmarks[terms.human_from]
    human_segments /= np.linalg.norm(human_segments, axis=1, keepdims=True) + 1e-9
    posture_target = torch.as_tensor(human_segments * terms.robot_lengths[:, None])

    def objective(x: np.ndarray, grad: np.ndarray) -> float:
        qpos[self.idx_pin2target] = x
        if self.adaptor is not None:
            qpos[:] = self.adaptor.forward_qpos(qpos)[:]
        self.robot.compute_forward_kinematics(qpos)
        link_poses = [self.robot.get_link_pose(index) for index in terms.link_indices]
        torch_body_pos = torch.as_tensor(np.array([pose[:3, 3] for pose in link_poses]))
        torch_body_pos.requires_grad_()
        robot_vec = torch_body_pos[terms.dex_task, :] - torch_body_pos[terms.dex_origin, :]
        vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
        loss = (self.huber_loss(vec_dist, torch.zeros_like(vec_dist)) * weight / robot_vec.shape[0]).sum()
        segments = torch_body_pos[terms.post_task, :] - torch_body_pos[terms.post_origin, :]
        segment_dist = torch.norm(segments - posture_target, dim=1)
        loss = loss + (self.huber_loss(segment_dist, torch.zeros_like(segment_dist)) * terms.weights).sum()
        result = loss.cpu().detach().item()
        if grad.size > 0:
            # Same LOCAL frame Jacobians as the official per-link
            # ``compute_single_link_local_jacobian``, from one kinematics pass
            # instead of one per link (22 links here).
            pin = terms.pinocchio
            pin.computeJointJacobians(self.robot.model, self.robot.data, qpos)
            jacobians = np.stack([
                pose[:3, :3] @ pin.getFrameJacobian(self.robot.model, self.robot.data, index, pin.ReferenceFrame.LOCAL)[:3, ...]
                for index, pose in zip(terms.link_indices, link_poses)
            ], axis=0)
            loss.backward()
            grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
            if self.adaptor is not None:
                jacobians = self.adaptor.backward_jacobian(jacobians)
            else:
                jacobians = jacobians[..., self.idx_pin2target]
            grad_qpos = np.matmul(grad_pos, np.array(jacobians)).mean(1).sum(0)
            grad_qpos += 2 * self.norm_delta * (x - last_qpos)
            grad[:] = grad_qpos[:]
        return result

    return objective


def _landmarks_from_request(request: dict[str, Any]) -> np.ndarray:
    # ``mano_landmarks`` is accepted as an explicit alias for callers that
    # distinguish their pre-converted MediaPipe coordinates in log schemas.
    has_landmarks = "landmarks" in request
    has_mano = "mano_landmarks" in request
    if has_landmarks == has_mano:
        raise InvalidLandmarks("retarget requires exactly one of landmarks or mano_landmarks")
    value = request["landmarks"] if has_landmarks else request["mano_landmarks"]
    try:
        landmarks = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidLandmarks("landmarks must be numeric 21x3 coordinates") from exc
    if landmarks.shape != (21, 3):
        raise InvalidLandmarks(f"landmarks must have shape [21, 3], received {list(landmarks.shape)}")
    if not np.all(np.isfinite(landmarks)):
        raise InvalidLandmarks("landmarks must contain only finite coordinates")
    return landmarks


def _reset_solver(retargeting: Any) -> None:
    """Clear official sequence and low-pass state for the main program's R key."""

    try:
        with contextlib.redirect_stdout(sys.stderr):
            retargeting.reset()
            filter_object = getattr(retargeting, "filter", None)
            reset_filter = getattr(filter_object, "reset", None)
            if callable(reset_filter):
                reset_filter()
    except Exception as exc:
        raise RetargetingError(f"official dex reset failed: {type(exc).__name__}: {exc}") from exc


class DexRetargetingWorker:
    """Stateful request dispatcher with no stdout side effects outside replies."""

    def __init__(self) -> None:
        self._state: WorkerState | None = None

    @property
    def state(self) -> WorkerState | None:
        """Expose immutable metadata for protocol tests; normal callers use JSON only."""

        return self._state

    def _require_state(self) -> WorkerState:
        if self._state is None:
            raise NotInitialized("send a successful init request before this operation")
        return self._state

    @staticmethod
    def _grasp_terms_input(state: WorkerState, landmarks: np.ndarray,
                           ref_value: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Solver input for the extended objective (see ``_grasp_terms_objective``)."""

        settings = {**GRASP_TERM_DEFAULTS, **state.grasp_terms}
        optimizer = state.retargeting.optimizer
        scaling = float(getattr(optimizer, "scaling", 1.0))
        lifted = _lift_thumb(landmarks, float(settings["thumb_lift_gain"]))
        terms = getattr(optimizer, "_grasp_terms", None)
        mapping: dict[str, Any] = {"thumb_lift_gain": float(settings["thumb_lift_gain"]), "thumb_scale": scaling}
        if settings["finger_length_scaling"] and terms is not None:
            tips, scales = _per_finger_tips(lifted, terms.robot_finger_lengths, scaling)
            mapping["finger_scales_palm_phalanges"] = scales.round(4).tolist()
        else:
            tips = _uniform_tips(lifted, scaling)
        targets = _scaled_targets(lifted, state.human_indices, tips)
        return np.concatenate((targets, ref_value, lifted), axis=0), mapping

    @staticmethod
    def _metadata(state: WorkerState) -> dict[str, Any]:
        return {
            "package": "dex-retargeting",
            "version": OFFICIAL_DEX_VERSION,
            "wheel_path": str(state.wheel_path),
            "wheel_sha256": state.wheel_sha256,
            "config_path": str(state.config_path),
            "config_sha256": state.config_sha256,
            "urdf_root": str(state.urdf_root),
            "human_indices": state.human_indices.tolist(),
            "joint_names": list(state.joint_names),
            "grasp_terms": state.grasp_terms,
        }

    def handle(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Dispatch one validated JSON object and report whether to keep serving."""

        request_id = request.get("id")
        operation = request.get("op")
        if not isinstance(operation, str) or not operation.strip():
            raise InvalidRequest("op must be a nonempty string")
        operation = operation.strip().lower()
        if operation == "init":
            state = _build_state(request)
            if request.get("grasp_terms") is not None:
                try:
                    with contextlib.redirect_stdout(sys.stderr):
                        settings = _install_grasp_terms(state.retargeting, request["grasp_terms"])
                except WorkerError:
                    raise
                except Exception as exc:
                    raise DexInitializationError(
                        f"grasp_terms could not be installed: {type(exc).__name__}: {exc}"
                    ) from exc
                state = replace(state, grasp_terms=settings)
            self._state = state
            return {"id": request_id, "ok": True, "result": self._metadata(self._state)}, True
        if operation == "retarget":
            state = self._require_state()
            landmarks = _landmarks_from_request(request)
            ref_value = landmarks[state.human_indices[1], :] - landmarks[state.human_indices[0], :]
            if ref_value.shape != (state.human_indices.shape[1], 3) or not np.all(np.isfinite(ref_value)):
                raise InvalidLandmarks("official vector reference computed from landmarks is invalid")
            solver_input, mapping = ref_value, None
            if state.grasp_terms is not None:
                solver_input, mapping = self._grasp_terms_input(state, landmarks, ref_value)
            try:
                # Joints a config leaves out of target_joint_names (Shadow's
                # SpaceMouse-owned wrist) are held at zero.
                fixed_count = len(getattr(state.retargeting.optimizer, "idx_pin2fixed", ()))
                with contextlib.redirect_stdout(sys.stderr):
                    if fixed_count:
                        solved = state.retargeting.retarget(solver_input, fixed_qpos=np.zeros(fixed_count))
                    else:
                        solved = state.retargeting.retarget(solver_input)
                    raw_qpos = np.asarray(solved, dtype=np.float64).reshape(-1)
            except Exception as exc:
                raise RetargetingError(
                    f"official dex retarget call failed: {type(exc).__name__}: {exc}"
                ) from exc
            if raw_qpos.shape != (len(state.joint_names),) or not np.all(np.isfinite(raw_qpos)):
                raise RetargetingError(
                    "official dex returned non-finite output or an unexpected number of joint values"
                )
            result = {
                # ``joint_targets_rad`` is the authoritative returned field.
                # ``raw_qpos`` is retained for existing telemetry terminology.
                "joint_targets_rad": raw_qpos.tolist(),
                "raw_qpos": raw_qpos.tolist(),
                "ref_value": ref_value.tolist(),
                "joint_names": list(state.joint_names),
            }
            if mapping is not None:
                terms = getattr(state.retargeting.optimizer, "_grasp_terms", None)
                mapping.update(getattr(terms, "last_mapping", None) or {})
                result["mapping"] = mapping
            return {"id": request_id, "ok": True, "result": result}, True
        if operation == "reset":
            state = self._require_state()
            _reset_solver(state.retargeting)
            return {"id": request_id, "ok": True, "result": {"reset": True}}, True
        if operation == "close":
            self._state = None
            return {"id": request_id, "ok": True, "result": {"closed": True}}, False
        raise InvalidRequest(f"unsupported op {operation!r}; expected init, retarget, reset, or close")

    def serve(self, input_stream: TextIO, output_stream: TextIO) -> int:
        """Run the newline-JSON loop.  Third-party output never reaches output_stream."""

        for line in input_stream:
            request_id: Any = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise InvalidRequest("each request line must be a JSON object")
                request_id = request.get("id")
                response, keep_serving = self.handle(request)
            except json.JSONDecodeError as exc:
                response = _error_payload(None, InvalidRequest(f"invalid JSON: {exc.msg}"), code="invalid_json")
                keep_serving = True
            except WorkerError as exc:
                response = _error_payload(request_id, exc)
                keep_serving = True
            except Exception as exc:  # Defensive protocol boundary: no trace on stdout.
                response = _error_payload(request_id, exc, code="internal_error")
                keep_serving = True
            _write_json(output_stream, response)
            if not keep_serving:
                return 0
        return 0


def main() -> int:
    """Entrypoint intentionally free of logging: stdout is the protocol channel."""

    return DexRetargetingWorker().serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
