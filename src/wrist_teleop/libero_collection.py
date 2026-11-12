"""LIBERO data collection bookkeeping: success hold, attempts, success rate, demo files.

No simulator imports here. ``TaskSuccessMonitor`` only calls the LIBERO
environment it is given (``parsed_problem``, ``_eval_predicate``,
``_check_success``), so the rest of this module runs in any interpreter.

Output under ``outputs/libero_collect/<suite>/`` (see
docs/libero_data_collection_zh.md), nothing else:

* ``attempts.csv``: one row per attempt the operator started (success / failed / quit);
* ``videos/``: the video of each such attempt where the operator pressed S (from the press to the
  end), the operator's window with the time and the result (empty ``video`` column without one).

Only with ``--save-demos``, under ``outputs/libero_collect_demos/<suite>/``
(what ``scripts/build_libero_dataset.py`` reads):

* ``taskNN_<task_name>/demo_NNN.hdf5``: one raw demo (simulator states, actions) per success;
* ``attempts.jsonl``: one line per attempt, for the dataset summary.

The run logs (metadata, telemetry) go to ``outputs/libero/`` like the other entries'.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLLECT_ROOT = ROOT / "outputs" / "libero_collect"
DEMO_ROOT = ROOT / "outputs" / "libero_collect_demos"  # --save-demos only
RUN_LOG_ROOT = ROOT / "outputs" / "libero"
ATTEMPTS_CSV = "attempts.csv"
VIDEO_DIR = "videos"

# Tuning for the collection entry, all in one place.
DEMOS_PER_TASK = 10
# LIBERO's own collect_demonstration.py also waits for 10 successful steps in a row.
SUCCESS_HOLD_STEPS = 10
# A non-goal object counts as disturbed past either limit (from the first move).
DISTURB_MOVE_M = 0.03
DISTURB_TILT_DEG = 30.0
IMAGE_SIZE = 256
VIDEO_TASK_NAME_CHARS = 60

# v2: built on the planner entry; ``control/arm_owner`` says who moved the arm at each step.
DEMO_FORMAT = "libero_shadow_teleop_demo_v2"
# ``control/arm_owner`` codes (index = stored value).
ARM_OWNERS = ("spacemouse", "planner")
ATTEMPTS_FILE = "attempts.jsonl"
OUTCOMES = ("success", "failed", "quit", "not_started")
_DEMO_RE = re.compile(r"^demo_(\d{3,})\.hdf5$")

# Panel colours (RGB: the view window decorates the RGB frame before display).
WHITE = (240, 240, 240)
GREY = (160, 160, 160)
GREEN = (90, 225, 90)
AMBER = (255, 205, 0)
RED = (255, 80, 80)


# ------------------------------------------------------------------ naming
def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "unnamed"


def task_dir_name(task_id: int, task_name: str) -> str:
    return "task{:02d}_{}".format(int(task_id), safe_name(task_name))


def demo_file_name(index: int) -> str:
    return "demo_{:03d}.hdf5".format(int(index))


def video_file_name(suite: str, task_id: int, demo_index: int, mount: str, duration_s: float,
                    task_name: str, max_task_chars: int = VIDEO_TASK_NAME_CHARS) -> str:
    """The offline builder's name (``--save-demos`` data): ``libero_object_t00_demo003_forward_12.4s__<task>.mp4``."""

    return "{}_t{:02d}_demo{:03d}_{}_{:.1f}s__{}.mp4".format(
        safe_name(suite), int(task_id), int(demo_index), safe_name(mount), float(duration_s),
        safe_name(task_name)[:max(1, int(max_task_chars))].rstrip("_"))


def attempt_video_name(suite: str, task_id: int, attempt: int, result: str, time_s: Optional[float],
                       task_name: str, max_task_chars: int = VIDEO_TASK_NAME_CHARS) -> str:
    """The live video of one attempt: ``libero_object_t00_a012_SUCCESS_16.4s__<task>.mp4``."""

    return "{}_t{:02d}_a{:03d}_{}_{:.1f}s__{}.mp4".format(
        safe_name(suite), int(task_id), int(attempt), safe_name(result).upper(), float(time_s or 0.0),
        safe_name(task_name)[:max(1, int(max_task_chars))].rstrip("_"))


def ended_by_text(reason: str) -> str:
    """What ended an attempt, in words for the table and the video (a success passes its own text)."""

    reason = str(reason)
    if reason == "R":
        return "R: new layout"
    if reason == "M":
        return "M: hand mount switch"
    if reason.endswith(":Q"):
        return "window closed" if reason.startswith("views_window") else "Q: quit"
    if reason == "completed":
        return "time limit"
    return reason


def dataset_file_name(task_name: str, mount: str) -> str:
    return "{}__{}_demo.hdf5".format(safe_name(task_name), safe_name(mount))


def placement_seed(base_seed: int, task_id: int, placement_index: int) -> int:
    """The random-placement seed of one placement; unique per (base, task, index) for index < 10000."""

    return (int(base_seed) * 1_000_000 + int(task_id) * 10_000 + int(placement_index)) % (2 ** 32)


def parse_task_list(text: Optional[str], task_count: int) -> list[int]:
    """``None`` / ``"all"`` -> every task; ``"3"``, ``"0-4"``, ``"1,3,7-9"`` -> those ids in order."""

    if text is None or str(text).strip().lower() in ("", "all"):
        return list(range(int(task_count)))
    chosen: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        first, dash, last = part.partition("-")
        try:
            low = int(first)
            high = int(last) if dash else low
        except ValueError:
            raise ValueError("task list entries are N or N-M, got {!r}".format(part)) from None
        if low > high or low < 0 or high >= task_count:
            raise ValueError("tasks must be within 0-{}, got {!r}".format(task_count - 1, part))
        chosen.extend(task for task in range(low, high + 1) if task not in chosen)
    if not chosen:
        raise ValueError("no task selected")
    return chosen


# ---------------------------------------------------------- success rules
_PREDICATE_TEXT = {
    "in": "{0} in {1}", "on": "{0} on {1}", "open": "{0} open", "close": "{0} closed",
    "turnon": "{0} turned on", "turnoff": "{0} turned off", "up": "{0} lifted",
}


def describe_predicate(state: Sequence[Any]) -> str:
    """One BDDL goal condition as a short line, e.g. ``alphabet_soup_1 in basket_1_contain_region``."""

    name = str(state[0]).lower()
    args = [str(arg) for arg in state[1:]]
    template = _PREDICATE_TEXT.get(name)
    if template is not None and template.count("{") == len(args):
        return template.format(*args)
    return "{}({})".format(name, ", ".join(args))


def goal_object_names(goal_states: Iterable[Sequence[Any]], object_names: Iterable[str]) -> set[str]:
    """Objects a goal names directly or through one of their regions (``basket_1_contain_region``)."""

    args = [str(arg) for state in goal_states for arg in list(state)[1:]]
    return {str(name) for name in object_names
            if any(arg == str(name) or arg.startswith(str(name) + "_") for arg in args)}


class SuccessHold:
    """Success must hold ``hold_steps`` control steps in a row, counted only after the operator started.

    A goal that is already met when the attempt starts never counts: the
    goal has to be unmet at least once after the first SpaceMouse move.
    """

    def __init__(self, hold_steps: int = SUCCESS_HOLD_STEPS) -> None:
        if int(hold_steps) < 1:
            raise ValueError("hold steps must be at least 1")
        self.hold_steps = int(hold_steps)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.met_before_start = False
        self._seen_unmet = False

    def update(self, success: bool, started: bool) -> bool:
        success = bool(success)
        if not started:
            self.count = 0
            self.met_before_start = success
            self._seen_unmet = not success
            return False
        if not success:
            self._seen_unmet = True
            self.met_before_start = False
        self.count = self.count + 1 if success and self._seen_unmet else 0
        return self.confirmed

    @property
    def confirmed(self) -> bool:
        return self.count >= self.hold_steps


class TaskSuccessMonitor:
    """One task's success check, generated from that task's own BDDL goal.

    Every goal condition is evaluated on its own for the live display; the
    decision is LIBERO's official ``env._check_success()`` (the AND of the
    same conditions), so a collected success is a success in evaluation.
    """

    def __init__(self, goal_states: Sequence[Sequence[Any]], hold_steps: int = SUCCESS_HOLD_STEPS) -> None:
        self.goal_states = [list(state) for state in goal_states]
        if not self.goal_states:
            raise ValueError("the task has no goal conditions")
        self.labels = [describe_predicate(state) for state in self.goal_states]
        self.hold = SuccessHold(hold_steps)
        self.status = [False] * len(self.goal_states)
        self.success = False
        self.mismatches = 0

    @classmethod
    def for_env(cls, env: Any, hold_steps: int = SUCCESS_HOLD_STEPS) -> "TaskSuccessMonitor":
        return cls(env.parsed_problem["goal_state"], hold_steps)

    def reset(self) -> None:
        self.hold.reset()
        self.status = [False] * len(self.goal_states)
        self.success = False

    def evaluate(self, env: Any, started: bool) -> bool:
        """Check the current state; returns True once success has held long enough."""

        self.status = [bool(env._eval_predicate(state)) for state in self.goal_states]
        self.success = bool(env._check_success())
        if self.success != all(self.status):
            self.mismatches += 1
        return self.hold.update(self.success, started)


# ------------------------------------------------------------ disturbance
def tilt_deg(reference: np.ndarray, current: np.ndarray) -> float:
    """Angle between the object's up (z) axis then and now; spinning about the vertical is not tilt."""

    a = np.asarray(reference, dtype=np.float64).reshape(3, 3)[:, 2]
    b = np.asarray(current, dtype=np.float64).reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


class DisturbanceMonitor:
    """Non-goal objects that moved or tipped since the attempt started (BDDL success ignores them)."""

    def __init__(self, names: Iterable[str], move_m: float = DISTURB_MOVE_M, tilt_limit_deg: float = DISTURB_TILT_DEG):
        self.names = [str(name) for name in names]
        self.move_m, self.tilt_limit_deg = float(move_m), float(tilt_limit_deg)
        self.clear()

    def clear(self) -> None:
        self.reference: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None
        self._worst: dict[str, dict[str, float]] = {}

    @property
    def armed(self) -> bool:
        return self.reference is not None

    def set_reference(self, poses: dict[str, tuple[Any, Any]]) -> None:
        self.reference = {name: (np.asarray(pos, dtype=np.float64).copy(),
                                 np.asarray(mat, dtype=np.float64).reshape(3, 3).copy())
                          for name, (pos, mat) in poses.items() if name in self.names}
        self._worst = {}

    def update(self, poses: dict[str, tuple[Any, Any]]) -> list[dict[str, Any]]:
        """Objects past a limit right now (and remember the worst value of every flagged object)."""

        if self.reference is None:
            return []
        current = []
        for name, (ref_pos, ref_mat) in self.reference.items():
            if name not in poses:
                continue
            pos, mat = poses[name]
            moved = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - ref_pos))
            tilted = tilt_deg(ref_mat, mat)
            if moved > self.move_m or tilted > self.tilt_limit_deg:
                worst = self._worst.setdefault(name, {"moved_m": 0.0, "tilt_deg": 0.0})
                worst["moved_m"], worst["tilt_deg"] = max(worst["moved_m"], moved), max(worst["tilt_deg"], tilted)
                current.append({"object": name, "moved_m": moved, "tilt_deg": tilted})
        return current

    def flagged(self) -> list[dict[str, Any]]:
        """Every object that crossed a limit during the attempt, with its worst values."""

        return [{"object": name, "moved_m": round(values["moved_m"], 4), "tilt_deg": round(values["tilt_deg"], 1)}
                for name, values in sorted(self._worst.items())]


def describe_disturbance(entry: dict[str, Any]) -> str:
    return "{} ({:.0f} cm, tilted {:.0f} deg)".format(entry["object"], 100.0 * entry["moved_m"], entry["tilt_deg"])


# ------------------------------------------------------------- recording
class DemoRecorder:
    """Per-step arrays of one attempt, from the placement to the confirmed success.

    ``states[t]`` is the simulator state before ``actions[t]`` (as in
    LIBERO's demos); ``success`` / ``goal_status`` are checked after it.
    ``control/arm_owner[t]`` is who set the arm part of ``actions[t]``
    (``ARM_OWNERS``: the SpaceMouse, or the planner after a number key).
    """

    def __init__(self, goal_count: int, hand_dof: int = 20) -> None:
        self.goal_count, self.hand_dof = int(goal_count), int(hand_dof)
        self.clear()

    def clear(self) -> None:
        self._rows: dict[str, list[Any]] = {key: [] for key in (
            "actions", "states", "success", "goal_status", "human/spacemouse_axes", "human/spacemouse_buttons",
            "human/arm_delta", "human/hand_targets", "human/hand_valid", "human/mano_landmarks",
            "control/arm_owner", "time/wall_s", "time/sim_s")}
        self.first_input_step: Optional[int] = None
        self.first_input_wall_s: Optional[float] = None

    @property
    def steps(self) -> int:
        return len(self._rows["actions"])

    @property
    def started(self) -> bool:
        return self.first_input_step is not None

    @property
    def planner_steps(self) -> int:
        return sum(1 for owner in self._rows["control/arm_owner"] if owner == ARM_OWNERS.index("planner"))

    def append(self, *, state: Any, action: Any, success: bool, goal_status: Sequence[bool], moving: bool,
               spacemouse_axes: Any, spacemouse_buttons: Sequence[bool], arm_delta: Any,
               hand_targets: Any = None, hand_valid: bool = False, mano_landmarks: Any = None,
               arm_owner: str = "spacemouse", wall_s: float, sim_s: float) -> None:
        # Checked before anything is appended: a rejected step leaves no partial row.
        if arm_owner not in ARM_OWNERS:
            raise ValueError("arm owner must be one of {}".format(", ".join(ARM_OWNERS)))
        status = [bool(value) for value in goal_status]
        if len(status) != self.goal_count:
            raise ValueError("goal status has {} entries, the task has {}".format(len(status), self.goal_count))
        rows = self._rows
        rows["actions"].append(np.asarray(action, dtype=np.float64).copy())
        rows["states"].append(np.asarray(state, dtype=np.float64).copy())
        rows["success"].append(bool(success))
        rows["goal_status"].append(status)
        rows["human/spacemouse_axes"].append(_fixed(spacemouse_axes, (6,)))
        buttons = [bool(value) for value in spacemouse_buttons][:8]
        rows["human/spacemouse_buttons"].append(buttons + [False] * (8 - len(buttons)))
        rows["human/arm_delta"].append(_fixed(arm_delta, (6,)))
        rows["human/hand_targets"].append(_fixed(hand_targets, (self.hand_dof,)))
        rows["human/hand_valid"].append(bool(hand_valid))
        rows["human/mano_landmarks"].append(_fixed(mano_landmarks, (21, 3)))
        rows["control/arm_owner"].append(ARM_OWNERS.index(arm_owner))
        rows["time/wall_s"].append(float(wall_s))
        rows["time/sim_s"].append(float(sim_s))
        if moving and self.first_input_step is None:
            self.first_input_step, self.first_input_wall_s = self.steps - 1, float(wall_s)

    def elapsed_s(self, now_wall_s: float) -> float:
        """Seconds since the first move (0 before it)."""

        return 0.0 if self.first_input_wall_s is None else max(0.0, float(now_wall_s) - self.first_input_wall_s)

    def arrays(self, final_state: Any) -> dict[str, np.ndarray]:
        if not self.steps:
            raise ValueError("nothing recorded")
        out: dict[str, np.ndarray] = {
            "actions": np.stack(self._rows["actions"]),
            "states": np.stack(self._rows["states"]),
            "final_state": np.asarray(final_state, dtype=np.float64).copy(),
            "success": np.asarray(self._rows["success"], dtype=bool),
            "goal_status": np.asarray(self._rows["goal_status"], dtype=bool).reshape(self.steps, self.goal_count),
            "human/spacemouse_axes": np.asarray(self._rows["human/spacemouse_axes"], dtype=np.float32),
            "human/spacemouse_buttons": np.asarray(self._rows["human/spacemouse_buttons"], dtype=bool),
            "human/arm_delta": np.asarray(self._rows["human/arm_delta"], dtype=np.float32),
            "human/hand_targets": np.asarray(self._rows["human/hand_targets"], dtype=np.float32),
            "human/hand_valid": np.asarray(self._rows["human/hand_valid"], dtype=bool),
            "human/mano_landmarks": np.asarray(self._rows["human/mano_landmarks"], dtype=np.float32),
            "control/arm_owner": np.asarray(self._rows["control/arm_owner"], dtype=np.uint8),
            "time/wall_s": np.asarray(self._rows["time/wall_s"], dtype=np.float64),
            "time/sim_s": np.asarray(self._rows["time/sim_s"], dtype=np.float64),
        }
        return out


def _fixed(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """``value`` as float32 of ``shape``, or NaN of that shape when missing / malformed."""

    try:
        array = np.asarray(value, dtype=np.float32)
        if array.shape == shape:
            return array.copy()
    except (TypeError, ValueError):
        pass
    return np.full(shape, np.nan, dtype=np.float32)


# ---------------------------------------------------------- demo files
def _attr(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("cannot serialize {!r}".format(type(value)))


def write_demo_file(path: Path, arrays: dict[str, np.ndarray], attrs: dict[str, Any], model_xml: str) -> Path:
    """Write one demo atomically: a ``.tmp`` file first, renamed when complete."""

    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    json_keys = sorted(key for key, value in attrs.items() if not isinstance(_attr(value), (bool, int, float))
                       and not isinstance(value, str))
    with h5py.File(temporary, "w") as handle:
        for key, value in arrays.items():
            value = np.asarray(value)
            options = {"compression": "gzip", "compression_opts": 4, "shuffle": True} if value.size > 64 else {}
            handle.create_dataset(key, data=value, **options)
        handle.create_dataset("model_xml", data=np.frombuffer(model_xml.encode("utf-8"), dtype=np.uint8),
                              compression="gzip", compression_opts=4)
        for key, value in attrs.items():
            handle.attrs[key] = _attr(value)
        handle.attrs["json_attrs"] = json.dumps(json_keys)
    os.replace(temporary, path)
    return path


def _decode_attrs(raw: Any) -> dict[str, Any]:
    attrs = {key: (value.decode("utf-8") if isinstance(value, bytes) else value) for key, value in raw.items()}
    for key in json.loads(attrs.pop("json_attrs", "[]")):
        if key in attrs:
            attrs[key] = json.loads(attrs[key])
    for key, value in list(attrs.items()):
        if isinstance(value, np.generic):
            attrs[key] = value.item()
    return attrs


def read_demo_attrs(path: Path) -> dict[str, Any]:
    """Only the attrs of one raw demo (JSON values decoded)."""

    import h5py

    with h5py.File(path, "r") as handle:
        return _decode_attrs(handle.attrs)


def read_demo_file(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    """(arrays, attrs with JSON values decoded, model XML) of one raw demo."""

    import h5py

    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        def visit(name: str, item: Any) -> None:
            if isinstance(item, h5py.Dataset) and name != "model_xml":
                arrays[name] = item[()]

        handle.visititems(visit)
        xml = bytes(handle["model_xml"][()]).decode("utf-8")
        attrs = _decode_attrs(handle.attrs)
    return arrays, attrs, xml


# ------------------------------------------------------- attempts / stats
@dataclass
class AttemptStats:
    success: int = 0
    failed: int = 0
    quit: int = 0
    not_started: int = 0
    durations_s: list[float] = field(default_factory=list)

    @property
    def counted(self) -> int:
        return self.success + self.failed

    @property
    def rate(self) -> Optional[float]:
        return self.success / self.counted if self.counted else None

    def describe(self) -> str:
        if not self.counted:
            return "-"
        return "{}/{} ({:.0f}%)".format(self.success, self.counted, 100.0 * float(self.rate))

    @property
    def median_s(self) -> Optional[float]:
        return float(np.median(self.durations_s)) if self.durations_s else None


def stats_from_records(records: Iterable[dict[str, Any]], task_id: Optional[int] = None,
                       mount: Optional[str] = None) -> AttemptStats:
    stats = AttemptStats()
    for record in records:
        if task_id is not None and int(record.get("task_id", -1)) != int(task_id):
            continue
        if mount is not None and record.get("hand_mount") != mount:
            continue
        outcome = record.get("outcome")
        if outcome == "success":
            stats.success += 1
            if record.get("duration_s") is not None:
                stats.durations_s.append(float(record["duration_s"]))
        elif outcome == "failed":
            stats.failed += 1
        elif outcome == "quit":
            stats.quit += 1
        elif outcome == "not_started":
            stats.not_started += 1
    return stats


def attempt_outcome(started: bool, reason: str) -> str:
    """R / M after the first move = failed; Q (or an error) after it = quit; anything before it = not_started."""

    if not started:
        return "not_started"
    return "failed" if reason in ("R", "M") else "quit"


class DemoStore:
    """The demo files and attempt log of one suite (resume = count what is already there)."""

    def __init__(self, suite_root: Path, task_names: dict[int, str]) -> None:
        self.root = Path(suite_root)
        self.task_names = {int(task): str(name) for task, name in task_names.items()}
        self.attempts_path = self.root / ATTEMPTS_FILE
        self._records = self._read_records()

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.attempts_path.is_file():
            return []
        records = []
        for line in self.attempts_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue  # a line cut by a crash: skip it, keep the rest
        return records

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def task_dir(self, task_id: int) -> Path:
        return self.root / task_dir_name(task_id, self.task_names[int(task_id)])

    def demo_paths(self, task_id: int) -> list[Path]:
        folder = self.task_dir(task_id)
        if not folder.is_dir():
            return []
        found = [(int(match.group(1)), path) for path in folder.iterdir()
                 for match in [_DEMO_RE.match(path.name)] if match and path.is_file()]
        return [path for _index, path in sorted(found)]

    def demo_count(self, task_id: int) -> int:
        return len(self.demo_paths(task_id))

    def next_demo_index(self, task_id: int) -> int:
        """One past the highest existing index (a deleted demo is never overwritten)."""

        indices = [int(_DEMO_RE.match(path.name).group(1)) for path in self.demo_paths(task_id)]
        return max(indices) + 1 if indices else 0

    def next_attempt_number(self) -> int:
        return 1 + max((int(record.get("attempt", 0)) for record in self._records), default=0)

    def next_placement_index(self, task_id: int) -> int:
        used = [int(record["placement_index"]) for record in self._records
                if int(record.get("task_id", -1)) == int(task_id) and record.get("placement_index") is not None]
        return max(used) + 1 if used else 0

    def stats(self, task_id: Optional[int] = None, mount: Optional[str] = None) -> AttemptStats:
        return stats_from_records(self._records, task_id, mount)

    def log_attempt(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=_json_default, separators=(",", ":"))
        self.root.mkdir(parents=True, exist_ok=True)
        with self.attempts_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._records.append(json.loads(line))

    def save_demo(self, task_id: int, arrays: dict[str, np.ndarray], attrs: dict[str, Any], model_xml: str) -> Path:
        index = self.next_demo_index(task_id)
        attrs = dict(attrs, demo_index=index)
        return write_demo_file(self.task_dir(task_id) / demo_file_name(index), arrays, attrs, model_xml)


# --------------------------------------------------------- attempts.csv
CSV_FIELDS = ("attempt", "started_at", "task_id", "task", "hand_mount", "layout", "result", "time_s", "ended_by",
              "moved_objects", "video", "operator")


class AttemptTable:
    """``attempts.csv``: one row per attempt the operator started (success / failed / quit).

    The collection's only bookkeeping: successes per task (resume), the
    success rate, the next attempt number and layout. A row that cannot be
    written because the file is open in another program (Excel locks it) is
    kept and written with the next row or by ``flush`` at exit.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.fields: tuple[str, ...] = CSV_FIELDS
        self._rows: list[dict[str, str]] = self._read()
        self._pending: list[dict[str, str]] = []
        self.blocked: Optional[str] = None  # why the last write failed

    def _read(self) -> list[dict[str, str]]:
        if not self.path.is_file():
            return []
        with self.path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames:
                self.fields = tuple(reader.fieldnames)  # keep the file's own columns when appending
            # A row cut by a crash has no result: skip it, keep the rest.
            return [dict(row) for row in reader if row.get("attempt") and row.get("result")]

    @property
    def rows(self) -> list[dict[str, str]]:
        return [dict(row) for row in self._rows]

    @property
    def pending(self) -> int:
        return len(self._pending)

    @staticmethod
    def _int(value: Any, default: int = -1) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    def _task_rows(self, task_id: int) -> list[dict[str, str]]:
        return [row for row in self._rows if self._int(row.get("task_id")) == int(task_id)]

    def success_count(self, task_id: int) -> int:
        return sum(1 for row in self._task_rows(task_id) if row.get("result") == "success")

    def stats(self, task_id: Optional[int] = None, mount: Optional[str] = None) -> "AttemptStats":
        records = []
        for row in self._rows:
            try:
                duration = float(row.get("time_s") or "")
            except ValueError:
                duration = None
            records.append({"task_id": self._int(row.get("task_id")), "hand_mount": row.get("hand_mount"),
                            "outcome": row.get("result"), "duration_s": duration})
        return stats_from_records(records, task_id, mount)

    def next_attempt_number(self) -> int:
        return 1 + max((self._int(row.get("attempt"), 0) for row in self._rows), default=0)

    def next_layout(self, task_id: int) -> int:
        used = [self._int(row.get("layout")) for row in self._task_rows(task_id)]
        return max(used) + 1 if used and max(used) >= 0 else 0

    def append(self, row: dict[str, Any]) -> bool:
        """Add one attempt; False when the file is locked (the row waits in memory, it still counts)."""

        clean = {key: "" if row.get(key) is None else str(row.get(key)) for key in CSV_FIELDS}
        self._rows.append(clean)
        self._pending.append(clean)
        return self.flush()

    def flush(self) -> bool:
        if not self._pending:
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.is_file() or self.path.stat().st_size == 0
            cut = False
            if not new:  # a last line cut by a crash: the next row starts on a line of its own
                with self.path.open("rb") as stream:
                    stream.seek(-1, os.SEEK_END)
                    cut = stream.read(1) not in (b"\n", b"\r")
            # A BOM in a new file so Excel reads it as UTF-8; appended rows go after it as plain UTF-8.
            with self.path.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as stream:
                if cut:
                    stream.write("\r\n")
                writer = csv.DictWriter(stream, fieldnames=list(self.fields), extrasaction="ignore", restval="")
                if new:
                    writer.writeheader()
                writer.writerows(self._pending)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            self.blocked = "{}: {}".format(type(exc).__name__, exc)
            return False
        self._pending.clear()
        self.blocked = None
        return True

    def save_unwritten(self, stamp: str) -> Optional[Path]:
        """At exit, if the table is still locked: the whole table to ``attempts_unsaved_<stamp>.csv``."""

        if not self._pending:
            return None
        path = self.path.with_name("attempts_unsaved_{}.csv".format(stamp))
        try:
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(CSV_FIELDS), extrasaction="ignore", restval="")
                writer.writeheader()
                writer.writerows(self._rows)
        except OSError:
            return None
        return path


def share_collection(suite_root: Path, share_dir: Path, suite: str) -> list[Path]:
    """Copy attempts.csv and the videos into ``share_dir/<suite>/`` (videos already there are skipped)."""

    suite_root, target_root = Path(suite_root), Path(share_dir) / suite
    sources = [suite_root / ATTEMPTS_CSV] + sorted((suite_root / VIDEO_DIR).glob("*.mp4"))
    copied = []
    for source in sources:
        if not source.is_file() or source.name.startswith("_recording_"):
            continue
        target = target_root / source.relative_to(suite_root)
        if source.suffix == ".mp4" and target.is_file() and target.stat().st_size == source.stat().st_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


# --------------------------------------------------------------- display
def collection_panel(*, task_id: int, language: str, labels: Sequence[str], status: Sequence[bool],
                     started: bool, met_before_start: bool, hold_count: int, hold_steps: int, elapsed_s: float,
                     disturbed: Sequence[dict[str, Any]], saved: int, target: int, rate_text: str,
                     recording: Optional[bool] = None) -> list[tuple[str, tuple[int, int, int]]]:
    """The lines of the live collection panel, each with its RGB colour.

    ``recording``: whether S started a video of this attempt (no line when
    ``None``).
    """

    lines: list[tuple[str, tuple[int, int, int]]] = [("TASK {}: {}".format(task_id, language), WHITE)]
    for label, ok in zip(labels, status):
        lines.append(("[OK] " + label if ok else "[--] " + label, GREEN if ok else GREY))
    if not started:
        if met_before_start:
            lines.append(("goal already met before the first move - press R", AMBER))
        else:
            lines.append(("waiting for the SpaceMouse / 1-9 - the timer starts on the first move", AMBER))
    elif hold_count > 0:
        lines.append(("SUCCESS {}/{} - hold still   {:.1f} s".format(hold_count, hold_steps, elapsed_s), GREEN))
    else:
        lines.append(("running   {:.1f} s".format(elapsed_s), WHITE))
    if recording is not None:
        lines.append(("REC  recording, saved when this attempt ends   S: cancel", RED) if recording
                     else ("not recording   S: record this attempt from now", GREY))
    for entry in list(disturbed)[:2]:
        lines.append(("MOVED: " + describe_disturbance(entry), AMBER))
    lines.append(("successes {}/{}   success rate {}".format(saved, target, rate_text), WHITE))
    lines.append(("1-9: approach a target   0: cancel   L: rays   M: hand mount", GREY))
    lines.append(("R: new layout (a failure once you moved)   Q: quit", GREY))
    return lines


def format_summary(rows: Sequence[dict[str, Any]]) -> str:
    """Console table: task, successes / target, success rate, median time."""

    lines = ["task  successes  success rate      median   name"]
    for row in rows:
        median = row["stats"].median_s
        lines.append("{:>4}  {:>4}/{:<4}  {:<16}  {:>6}   {}".format(
            row["task_id"], row["demos"], row["target"], row["stats"].describe(),
            "-" if median is None else "{:.1f} s".format(median), row["task_name"]))
    return "\n".join(lines)
