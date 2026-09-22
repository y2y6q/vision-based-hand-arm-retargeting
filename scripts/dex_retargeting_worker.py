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

Failures are structured as ``{"ok": false, "error": {"code", "type",
"message"}}``.  No request can make the worker silently use a package other
than the wheel it was explicitly given during ``init``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
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
    if (
        human_indices.shape != (2, 4)
        or np.any(human_indices < 0)
        or np.any(human_indices >= 21)
    ):
        raise DexInitializationError(
            "official Allegro vector configuration must expose a finite 2x4 human-index matrix within 21 landmarks"
        )
    if len(joint_names) != 16 or len(set(joint_names)) != 16:
        raise DexInitializationError(
            "official Allegro vector configuration must expose 16 distinct output joint names"
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
        }

    def handle(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Dispatch one validated JSON object and report whether to keep serving."""

        request_id = request.get("id")
        operation = request.get("op")
        if not isinstance(operation, str) or not operation.strip():
            raise InvalidRequest("op must be a nonempty string")
        operation = operation.strip().lower()
        if operation == "init":
            self._state = _build_state(request)
            return {"id": request_id, "ok": True, "result": self._metadata(self._state)}, True
        if operation == "retarget":
            state = self._require_state()
            landmarks = _landmarks_from_request(request)
            ref_value = landmarks[state.human_indices[1], :] - landmarks[state.human_indices[0], :]
            if ref_value.shape != (4, 3) or not np.all(np.isfinite(ref_value)):
                raise InvalidLandmarks("official vector reference computed from landmarks is invalid")
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    raw_qpos = np.asarray(state.retargeting.retarget(ref_value), dtype=np.float64).reshape(-1)
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
