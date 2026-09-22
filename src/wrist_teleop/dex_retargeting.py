"""Official dex-retargeting boundary for the Panda + Allegro mainline.

The project deliberately keeps this module optional at import time.  The
``dex-retargeting`` package has a native Pinocchio dependency that is not
available as a Windows CPython 3.10 wheel in the checked-in environment.  A
request for dex mode therefore fails clearly instead of falling back to the
legacy curl mapper.  Once the pinned official dependency is available, this
module loads the unmodified 0.4.6 vector configuration and exposes only
validated, source-order Allegro targets to the existing action composer.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DEX_VERSION = "0.4.6"
OFFICIAL_CONFIG = PROJECT_ROOT / "configs" / "dex_retargeting" / "allegro_hand_right.yml"
OFFICIAL_CONFIG_SHA256 = "23cdb5385137dec4174f1f9211ffe4884a9222f5d3a98fcf7218c715555a5284"
OFFICIAL_WHEEL = (
    PROJECT_ROOT
    / "third_party"
    / "dex_retargeting_0_4_6"
    / "dex_retargeting-0.4.6-py3-none-any.whl"
)
OFFICIAL_WHEEL_SHA256 = "cf08b93e204af21b7146f12a83d55f0a0d227f991d202655c475535621bb62f7"
DEX_SIDECAR_WORKER = PROJECT_ROOT / "scripts" / "dex_retargeting_worker.py"
OFFICIAL_URDF_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "panda-gym"
    / "panda_gym"
    / "assets"
    / "robots"
    / "panda_allegro"
)

# These matrices are the official 0.4.6 dex-retargeting constants.  Keeping
# them here makes the MediaPipe coordinate conversion explicit before the
# optional native package is imported.
OPERATOR2MANO_RIGHT = np.array(((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
OPERATOR2MANO_LEFT = np.array(((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))


class DexRetargetingUnavailable(RuntimeError):
    """The explicit error for an unavailable or incompatible dex runtime."""


class DexFrameRejected(ValueError):
    """A camera frame cannot safely be used as a dex reference frame."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hand_label(label: str) -> str:
    value = str(label).strip().lower()
    if value not in {"left", "right"}:
        raise DexFrameRejected(f"unknown MediaPipe handedness {label!r}")
    return value


def _mirrored_label(label: str) -> str:
    return "left" if label == "right" else "right"


def mediapipe_world_to_mano(
    landmarks: Sequence[Any],
    *,
    handedness: str,
    expected_handedness: str = "Right",
    input_is_mirrored: bool = False,
) -> np.ndarray:
    """Convert one MediaPipe 21x3 world-landmark frame into dex MANO space.

    MediaPipe world landmarks are metre-valued but use camera/operator axes.
    The official dex vector config expects wrist-relative MANO coordinates.
    ``input_is_mirrored`` is intentionally explicit: when callers mirror a
    selfie image before detection, MediaPipe's reported left / right label is
    converted back to the physical hand before the right-Allegro gate.
    """

    values = np.asarray(
        [
            (point.x, point.y, point.z) if hasattr(point, "x") else point
            for point in landmarks
        ],
        dtype=np.float64,
    )
    if values.shape != (21, 3) or not np.all(np.isfinite(values)):
        raise DexFrameRejected("dex requires exactly 21 finite 3-D MediaPipe world landmarks")
    observed = _hand_label(handedness)
    if input_is_mirrored:
        observed = _mirrored_label(observed)
    expected = _hand_label(expected_handedness)
    if observed != expected:
        raise DexFrameRejected(
            f"expected {expected} hand for Allegro right but MediaPipe reported {observed}"
        )
    wrist_relative = values - values[0:1]
    transform = OPERATOR2MANO_RIGHT if expected == "right" else OPERATOR2MANO_LEFT
    mano = wrist_relative @ transform.T
    if not np.all(np.isfinite(mano)):
        raise DexFrameRejected("camera-to-MANO conversion produced non-finite landmarks")
    return mano


@dataclass(frozen=True)
class DexRetargetingResult:
    """One traceable official solver result in local Allegro joint order."""

    joint_targets_rad: np.ndarray
    raw_qpos: np.ndarray
    ref_value: np.ndarray
    human_indices: np.ndarray
    retargeting_joint_names: tuple[str, ...]


class _DexSidecar:
    """Serialize one official dex solver in a dedicated Python process.

    The robosuite interpreter never imports Pinocchio, NLopt, Torch, or the
    official dex wheel.  Only this child receives the native-library PATH and
    the temporary OpenMP compatibility switch required by the existing Conda
    runtime on this machine.
    """

    def __init__(self, python_path: Path) -> None:
        self.python_path = Path(python_path).expanduser().resolve()
        if not self.python_path.is_file():
            raise DexRetargetingUnavailable(
                f"dex sidecar Python is not an executable file: {self.python_path}"
            )
        if not DEX_SIDECAR_WORKER.is_file():
            raise DexRetargetingUnavailable(f"dex sidecar worker is missing: {DEX_SIDECAR_WORKER}")
        if not OFFICIAL_WHEEL.is_file() or _sha256(OFFICIAL_WHEEL) != OFFICIAL_WHEEL_SHA256:
            raise DexRetargetingUnavailable(
                "official dex-retargeting 0.4.6 wheel is missing or has an unexpected SHA-256"
            )

        environment = os.environ.copy()
        environment_root = self.python_path.parent
        native_prefixes = (
            environment_root / "Library" / "bin",
            environment_root / "Scripts",
            environment_root,
        )
        environment["PATH"] = os.pathsep.join(
            [*(str(path) for path in native_prefixes), environment.get("PATH", "")]
        )
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(OFFICIAL_WHEEL), *( [existing_pythonpath] if existing_pythonpath else [] )]
        )
        # The installed Conda Pinocchio stack and Torch load separate OpenMP
        # runtimes.  Restrict the compatibility override to this child; never
        # alter the robosuite / MuJoCo parent environment.
        environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(
                [str(self.python_path), "-u", str(DEX_SIDECAR_WORKER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=environment,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise DexRetargetingUnavailable(
                f"could not start dex sidecar with {self.python_path}: {type(exc).__name__}: {exc}"
            ) from exc

        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stdout_tail: deque[str] = deque(maxlen=16)
        self._stderr_tail: deque[str] = deque(maxlen=32)
        self._request_lock = threading.RLock()
        self._closed = False
        self._next_request_id = 0
        self._stdout_thread = threading.Thread(
            target=self._read_stdout, name="dex-sidecar-stdout", daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, name="dex-sidecar-stderr", daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        for line in stream:
            text = line.strip()
            if not text:
                continue
            try:
                response = json.loads(text)
            except json.JSONDecodeError:
                self._stdout_tail.append(text)
                continue
            if isinstance(response, dict):
                self._responses.put(response)
            else:
                self._stdout_tail.append(text)

    def _read_stderr(self) -> None:
        stream = self._process.stderr
        if stream is None:
            return
        for line in stream:
            text = line.strip()
            if text:
                self._stderr_tail.append(text)

    def _diagnostics(self) -> str:
        details: list[str] = []
        exit_code = self._process.poll()
        if exit_code is not None:
            details.append(f"sidecar exit code={exit_code}")
        if self._stdout_tail:
            details.append("unexpected stdout=" + " | ".join(self._stdout_tail))
        if self._stderr_tail:
            details.append("stderr=" + " | ".join(self._stderr_tail))
        return "; ".join(details) or "no sidecar diagnostics available"

    def _request_locked(self, operation: str, *, timeout_s: float, **payload: Any) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("dex sidecar is closed")
        if self._process.poll() is not None:
            raise RuntimeError(f"dex sidecar is not running: {self._diagnostics()}")
        self._next_request_id += 1
        request_id = self._next_request_id
        request = {"id": request_id, "op": operation, **payload}
        stream = self._process.stdin
        if stream is None:
            raise RuntimeError("dex sidecar stdin is unavailable")
        try:
            stream.write(json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"could not send {operation!r} to dex sidecar: {type(exc).__name__}: {exc}; "
                f"{self._diagnostics()}"
            ) from exc
        try:
            response = self._responses.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise RuntimeError(
                f"dex sidecar timed out after {timeout_s:.1f}s during {operation!r}; {self._diagnostics()}"
            ) from exc
        if response.get("id") != request_id:
            raise RuntimeError(
                f"dex sidecar response id mismatch for {operation!r}: expected {request_id}, "
                f"received {response.get('id')!r}; {self._diagnostics()}"
            )
        if response.get("ok") is not True:
            error = response.get("error")
            raise RuntimeError(f"dex sidecar {operation!r} failed: {error!r}; {self._diagnostics()}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"dex sidecar {operation!r} returned no JSON object result")
        return result

    def request(self, operation: str, *, timeout_s: float = 5.0, **payload: Any) -> dict[str, Any]:
        """Send one atomic request; retarget/reset/close cannot cross responses."""

        with self._request_lock:
            return self._request_locked(operation, timeout_s=timeout_s, **payload)

    def initialize(self, *, config_path: Path, urdf_root: Path) -> dict[str, Any]:
        return self.request(
            "init",
            timeout_s=30.0,
            wheel_path=str(OFFICIAL_WHEEL.resolve()),
            config_path=str(config_path.resolve()),
            urdf_root=str(urdf_root.resolve()),
        )

    def close(self) -> None:
        with self._request_lock:
            if self._closed:
                return
            try:
                if self._process.poll() is None:
                    self._request_locked("close", timeout_s=3.0)
            finally:
                self._closed = True
                stream = self._process.stdin
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
                if self._process.poll() is None:
                    try:
                        self._process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        self._process.terminate()
                        try:
                            self._process.wait(timeout=3.0)
                        except subprocess.TimeoutExpired:
                            self._process.kill()
                            self._process.wait(timeout=3.0)
                self._stdout_thread.join(timeout=1.0)
                self._stderr_thread.join(timeout=1.0)
                for owned_stream in (self._process.stdout, self._process.stderr):
                    if owned_stream is not None:
                        try:
                            owned_stream.close()
                        except OSError:
                            pass


def _discover_sidecar_python(explicit: str | Path | None) -> Path | None:
    """Resolve a configured sidecar without scanning arbitrary installations."""

    if explicit is not None:
        return Path(explicit).expanduser()
    configured = os.environ.get("DEX_RETARGETING_PYTHON")
    if configured:
        return Path(configured).expanduser()
    # The project venv is based on the sibling Conda environment
    # ``panda_retarget``.  The verified native dex environment is its direct
    # sibling.  A user can override this deterministic convention above.
    candidate = Path(sys.base_prefix).resolve().parent / "dexretarget" / "python.exe"
    return candidate if candidate.is_file() else None


class DexRetargetingAdapter:
    """Load official vector retargeting and map its named output to Allegro.

    ``runtime='auto'`` first selects the verified isolated official sidecar,
    then falls back only to a same-interpreter official dex installation. It
    never falls back to the legacy curl mapper.  ``runtime='native'`` is useful
    for a future complete native dependency install; ``runtime='sidecar'``
    requires the isolated process explicitly.
    """

    def __init__(
        self,
        *,
        allegro_joint_names: Sequence[str],
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        config_path: Path = OFFICIAL_CONFIG,
        runtime: str = "auto",
        sidecar_python: str | Path | None = None,
    ):
        self.allegro_joint_names = tuple(str(name) for name in allegro_joint_names)
        self.lower_limits = np.asarray(lower_limits, dtype=np.float64)
        self.upper_limits = np.asarray(upper_limits, dtype=np.float64)
        if (
            len(self.allegro_joint_names) != 16
            or len(set(self.allegro_joint_names)) != 16
            or self.lower_limits.shape != (16,)
            or self.upper_limits.shape != (16,)
            or not np.all(np.isfinite(self.lower_limits))
            or not np.all(np.isfinite(self.upper_limits))
            or not np.all(self.upper_limits > self.lower_limits)
        ):
            raise ValueError("Allegro dex mapping requires 16 distinct names and finite ordered limits")
        selected_runtime = str(runtime).strip().lower()
        if selected_runtime not in {"auto", "native", "sidecar"}:
            raise ValueError("dex runtime must be 'auto', 'native', or 'sidecar'")
        self.config_path = Path(config_path)
        if not self.config_path.is_file():
            raise DexRetargetingUnavailable(f"official dex config is missing: {self.config_path}")
        actual_sha = _sha256(self.config_path)
        if self.config_path.resolve() == OFFICIAL_CONFIG.resolve() and actual_sha != OFFICIAL_CONFIG_SHA256:
            raise DexRetargetingUnavailable(
                "official dex config hash does not match the recorded dex-retargeting 0.4.6 source"
            )
        if not OFFICIAL_URDF_ROOT.is_dir():
            raise DexRetargetingUnavailable(f"required Allegro URDF root is missing: {OFFICIAL_URDF_ROOT}")

        self._retargeting: Any | None = None
        self._sidecar: _DexSidecar | None = None
        self._solver_lock = threading.RLock()
        self.runtime: str | None = None
        self._runtime_details: dict[str, Any] = {}
        sidecar_error: DexRetargetingUnavailable | None = None
        if selected_runtime in {"auto", "sidecar"}:
            candidate = _discover_sidecar_python(sidecar_python)
            if candidate is None:
                sidecar_error = DexRetargetingUnavailable(
                    "no dex sidecar Python found; set dex_hand.sidecar_python or DEX_RETARGETING_PYTHON"
                )
            else:
                try:
                    self._start_sidecar(candidate)
                    return
                except DexRetargetingUnavailable as exc:
                    sidecar_error = exc
            if selected_runtime == "sidecar":
                raise sidecar_error

        if selected_runtime in {"auto", "native"}:
            try:
                self._start_native()
                return
            except DexRetargetingUnavailable as native_error:
                if sidecar_error is not None:
                    raise DexRetargetingUnavailable(
                        "no usable official dex runtime: sidecar="
                        f"{sidecar_error}; native={native_error}"
                    ) from native_error
                raise
        raise AssertionError("unreachable dex runtime selection")

    def _configure_layout(self, indices: Any, names: Sequence[Any]) -> None:
        validated_indices = np.asarray(indices, dtype=np.intp)
        if validated_indices.shape != (2, 4) or np.any(validated_indices < 0) or np.any(validated_indices >= 21):
            raise DexRetargetingUnavailable(
                f"official vector human-index matrix must be 2x4 inside 21 landmarks, got {validated_indices!r}"
            )
        validated_names = tuple(str(name) for name in names)
        if (
            len(validated_names) != 16
            or len(set(validated_names)) != 16
            or set(validated_names) != set(self.allegro_joint_names)
        ):
            raise DexRetargetingUnavailable(
                "official dex output joint names do not form a one-to-one match with local Allegro's 16 joints"
            )
        self.human_indices = validated_indices.copy()
        self.retargeting_joint_names = validated_names
        self._output_indices = np.asarray(
            [validated_names.index(name) for name in self.allegro_joint_names], dtype=np.intp
        )

    def _start_native(self) -> None:
        try:
            installed_version = importlib.metadata.version("dex-retargeting")
        except importlib.metadata.PackageNotFoundError as exc:
            raise DexRetargetingUnavailable(
                "dex-retargeting==0.4.6 is not installed in this interpreter; dex mode cannot fall back to legacy-curl. "
                "Install its pinned native dependencies (including pin>=2.7) before using runtime=native"
            ) from exc
        if installed_version != OFFICIAL_DEX_VERSION:
            raise DexRetargetingUnavailable(
                f"dex mode requires dex-retargeting=={OFFICIAL_DEX_VERSION}, found {installed_version}"
            )
        try:
            from dex_retargeting.retargeting_config import RetargetingConfig
        except Exception as exc:
            raise DexRetargetingUnavailable(
                "dex-retargeting import failed; install its pinned native dependencies before using runtime=native"
            ) from exc
        try:
            RetargetingConfig.set_default_urdf_dir(str(OFFICIAL_URDF_ROOT))
            solver = RetargetingConfig.load_from_file(str(self.config_path)).build()
        except Exception as exc:
            raise DexRetargetingUnavailable(
                f"official dex Allegro vector configuration failed to build: {type(exc).__name__}: {exc}"
            ) from exc
        self._configure_layout(solver.optimizer.target_link_human_indices, solver.joint_names)
        self._retargeting = solver
        self.runtime = "native"
        self._runtime_details = {"runtime_python": sys.executable}

    def _start_sidecar(self, python_path: Path) -> None:
        sidecar = _DexSidecar(python_path)
        try:
            metadata = sidecar.initialize(config_path=self.config_path, urdf_root=OFFICIAL_URDF_ROOT)
            if metadata.get("package") != "dex-retargeting" or metadata.get("version") != OFFICIAL_DEX_VERSION:
                raise DexRetargetingUnavailable("dex sidecar did not report official dex-retargeting 0.4.6")
            if metadata.get("wheel_sha256") != OFFICIAL_WHEEL_SHA256:
                raise DexRetargetingUnavailable("dex sidecar wheel SHA-256 does not match the checked-in official wheel")
            if metadata.get("config_sha256") != _sha256(self.config_path):
                raise DexRetargetingUnavailable("dex sidecar config SHA-256 does not match the requested config")
            self._configure_layout(metadata.get("human_indices"), metadata.get("joint_names", ()))
        except Exception as exc:
            try:
                sidecar.close()
            except Exception:
                pass
            if isinstance(exc, DexRetargetingUnavailable):
                raise
            raise DexRetargetingUnavailable(
                f"official dex sidecar failed to initialize: {type(exc).__name__}: {exc}"
            ) from exc
        self._sidecar = sidecar
        self.runtime = "sidecar"
        self._runtime_details = {
            "runtime_python": str(sidecar.python_path),
            "worker_path": str(DEX_SIDECAR_WORKER),
            "wheel_path": str(OFFICIAL_WHEEL),
            "wheel_sha256": OFFICIAL_WHEEL_SHA256,
            "openmp_duplicate_runtime_override": True,
        }

    def retarget(self, mano_landmarks_m: Sequence[Sequence[float]]) -> DexRetargetingResult:
        joints = np.asarray(mano_landmarks_m, dtype=np.float64)
        if joints.shape != (21, 3) or not np.all(np.isfinite(joints)):
            raise DexFrameRejected("dex retargeting requires a finite 21x3 MANO landmark array")
        ref_value = joints[self.human_indices[1], :] - joints[self.human_indices[0], :]
        if ref_value.shape != (4, 3) or not np.all(np.isfinite(ref_value)):
            raise DexFrameRejected("official dex vector reference is invalid")
        try:
            with self._solver_lock:
                if self._sidecar is not None:
                    result = self._sidecar.request("retarget", mano_landmarks=joints.tolist())
                    solver_ref = np.asarray(result.get("ref_value"), dtype=np.float64)
                    if solver_ref.shape != (4, 3) or not np.allclose(solver_ref, ref_value, rtol=0.0, atol=1e-12):
                        raise RuntimeError("dex sidecar returned a reference vector inconsistent with the requested MANO frame")
                    names = tuple(str(name) for name in result.get("joint_names", ()))
                    if names != self.retargeting_joint_names:
                        raise RuntimeError("dex sidecar changed output joint order after initialization")
                    raw = np.asarray(result.get("raw_qpos"), dtype=np.float64).reshape(-1)
                else:
                    if self._retargeting is None:
                        raise RuntimeError("dex runtime was not initialized")
                    raw = np.asarray(self._retargeting.retarget(ref_value), dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise DexFrameRejected(f"official dex retarget call failed: {type(exc).__name__}: {exc}") from exc
        if raw.shape != (16,) or not np.all(np.isfinite(raw)):
            raise DexFrameRejected("official dex returned non-finite or incorrectly sized joint output")
        mapped = raw[self._output_indices]
        if mapped.shape != (16,) or not np.all(np.isfinite(mapped)):
            raise DexFrameRejected("dex named joint mapping produced invalid local Allegro targets")
        clipped = np.clip(mapped, self.lower_limits, self.upper_limits)
        return DexRetargetingResult(
            joint_targets_rad=clipped,
            raw_qpos=raw.copy(),
            ref_value=ref_value.copy(),
            human_indices=self.human_indices.copy(),
            retargeting_joint_names=self.retargeting_joint_names,
        )

    def reset(self) -> None:
        """Clear both official solver history layers for an R-key reset."""

        try:
            with self._solver_lock:
                if self._sidecar is not None:
                    self._sidecar.request("reset")
                    return
                if self._retargeting is None:
                    raise RuntimeError("dex runtime was not initialized")
                self._retargeting.reset()
                filter_object = getattr(self._retargeting, "filter", None)
                reset_filter = getattr(filter_object, "reset", None)
                if callable(reset_filter):
                    reset_filter()
        except Exception as exc:
            raise RuntimeError(f"official dex reset failed: {type(exc).__name__}: {exc}") from exc

    def close(self) -> None:
        """Release the isolated solver after the camera worker has stopped."""

        with self._solver_lock:
            if self._sidecar is not None:
                self._sidecar.close()
                self._sidecar = None

    def metadata(self) -> dict[str, Any]:
        return {
            "package": "dex-retargeting",
            "version": OFFICIAL_DEX_VERSION,
            "runtime": self.runtime,
            "algorithm": "official vector retargeting",
            "config_path": str(self.config_path),
            "config_sha256": _sha256(self.config_path),
            "urdf_root": str(OFFICIAL_URDF_ROOT),
            "human_indices": self.human_indices.tolist(),
            "retargeting_joint_names": list(self.retargeting_joint_names),
            "allegro_joint_names": list(self.allegro_joint_names),
            "filtering": "official config low_pass_alpha only; no project output EMA",
            **self._runtime_details,
        }
