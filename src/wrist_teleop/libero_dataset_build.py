"""Offline: raw collection demos -> videos, LIBERO-format datasets, summary, share-folder copy.

Images are rendered from the saved simulator states with the saved model, the
way robosuite renders them (collision geoms hidden). Checked 2026-11-08: the
offline agentview / eye-in-hand images matched robosuite's own renders of
the same state within 0.11-0.18 / 255 per pixel on average. The dataset
images keep LIBERO's image convention (upside down, like the environment's
camera observations); the videos are upright.

Runs in .venv-libero (MuJoCo, h5py, imageio-ffmpeg). No LIBERO environment
is built; the mesh files named in the saved model must exist on this machine.
"""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .libero_collection import (
    ARM_OWNERS,
    ATTEMPTS_FILE,
    DEMOS_PER_TASK,
    IMAGE_SIZE,
    _DEMO_RE,
    dataset_file_name,
    read_demo_attrs,
    read_demo_file,
    stats_from_records,
    video_file_name,
)

VIDEO_FPS = 20  # = control rate: real time
VIDEO_MAIN = (768, 576)  # operator main view, width x height
VIDEO_SIDE = 256
VIDEO_TITLE_H = 64
VIDEO_SIZE = (VIDEO_MAIN[0] + VIDEO_SIDE, VIDEO_TITLE_H + VIDEO_MAIN[1])  # 1024 x 640
VIDEO_PREROLL_STEPS = 10  # 0.5 s before the first move
VIDEO_TAIL_FRAMES = 10  # the success state stays 0.5 s
VIDEO_QUALITY = 7  # imageio-ffmpeg 0-10
DATASET_CAMERAS = (("agentview", "agentview_rgb"), ("robot0_eye_in_hand", "eye_in_hand_rgb"))
IMAGE_CONVENTION = "opengl"  # as LIBERO's datasets: rows bottom-up (upside down)
SUMMARY_NAME, CSV_NAME, SLACK_NAME = "SUMMARY.md", "summary.csv", "share_message.txt"
_WHITE, _GREY, _GREEN, _AMBER = (240, 240, 240), (170, 170, 170), (90, 225, 90), (255, 205, 0)


def quat_wxyz_to_xyzw(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    return np.array([x, y, z, w], dtype=np.float64)


def quat2axisangle(quat_xyzw: Sequence[float]) -> np.ndarray:
    """robosuite's ``transform_utils.quat2axisangle`` (x, y, z, w)."""

    quat = np.asarray(quat_xyzw, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * 2.0 * np.arccos(quat[3]) / den


class SceneRenderer:
    """One saved model, rendered at saved states with robosuite's visible groups (no collision geoms)."""

    def __init__(self, model_xml: str, image_size: int = IMAGE_SIZE, main_size: tuple[int, int] = VIDEO_MAIN) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), int(main_size[0]), int(image_size))
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), int(main_size[1]), int(image_size))
        self.data = mujoco.MjData(self.model)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[0] = 0  # robosuite: render_collision_mesh=False
        self.image_size = int(image_size)
        self.small = mujoco.Renderer(self.model, self.image_size, self.image_size)
        self.main = mujoco.Renderer(self.model, int(main_size[1]), int(main_size[0]))

    def check_state_size(self, size: int) -> None:
        expected = 1 + self.model.nq + self.model.nv
        if size not in (expected, expected + self.model.na):
            raise ValueError("saved states have {} values, the saved model needs {} (1 + nq + nv)".format(size, expected))

    def set_state(self, state: np.ndarray) -> None:
        model, data = self.model, self.data
        nq, nv = model.nq, model.nv
        data.time = float(state[0])
        data.qpos[:] = state[1:1 + nq]
        data.qvel[:] = state[1 + nq:1 + nq + nv]
        if model.na and len(state) >= 1 + nq + nv + model.na:
            data.act[:] = state[1 + nq + nv:1 + nq + nv + model.na]
        self._mujoco.mj_forward(model, data)

    def set_camera(self, name: str, pos: Sequence[float], quat_wxyz: Sequence[float], fovy_deg: float) -> None:
        camera = int(self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_CAMERA, name))
        if camera < 0:
            raise ValueError("camera {!r} is not in the saved model".format(name))
        self.model.cam_pos[camera] = np.asarray(pos, dtype=np.float64)
        self.model.cam_quat[camera] = np.asarray(quat_wxyz, dtype=np.float64)
        self.model.cam_fovy[camera] = float(fovy_deg)
        self._mujoco.mj_forward(self.model, self.data)

    def render(self, camera: str, main: bool = False) -> np.ndarray:
        """Upright RGB uint8 (``mujoco.Renderer`` already flips OpenGL's rows)."""

        renderer = self.main if main else self.small
        renderer.update_scene(self.data, camera=camera, scene_option=self.option)
        return np.array(renderer.render(), dtype=np.uint8)

    def robot_obs(self, attrs: dict[str, Any]) -> dict[str, np.ndarray]:
        """LIBERO's low-dim obs from the current state: joints, hand joints, end effector."""

        mujoco = self._mujoco
        qpos = self.data.qpos
        site = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, attrs["eef_site"]))
        body = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, attrs["eef_body"]))
        if site < 0 or body < 0:
            raise ValueError("end-effector site / body missing from the saved model")
        ee_pos = np.array(self.data.site_xpos[site], dtype=np.float64)
        ee_quat = quat_wxyz_to_xyzw(self.data.xquat[body])  # robosuite robot0_eef_quat
        ee_ori = quat2axisangle(ee_quat)
        return {
            "joint_states": np.array(qpos[list(attrs["arm_qpos_index"])], dtype=np.float64),
            "gripper_states": np.array(qpos[list(attrs["hand_qpos_index"])], dtype=np.float64),
            "ee_pos": ee_pos, "ee_ori": ee_ori, "ee_states": np.concatenate((ee_pos, ee_ori)), "ee_quat": ee_quat,
        }

    def close(self) -> None:
        for renderer in (self.small, self.main):
            try:
                renderer.close()
            except Exception:
                pass


class _RendererCache:
    """One renderer at a time, reused while consecutive demos share the same model."""

    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)
        self._key: Optional[str] = None
        self._renderer: Optional[SceneRenderer] = None

    def get(self, model_xml: str) -> SceneRenderer:
        key = hashlib.sha1(model_xml.encode("utf-8")).hexdigest()
        if key != self._key:
            self.close()
            self._renderer, self._key = SceneRenderer(model_xml, self.image_size), key
        assert self._renderer is not None
        return self._renderer

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._renderer, self._key = None, None


# ------------------------------------------------------------- video
def _put(cv2: Any, frame: np.ndarray, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int],
         max_width: Optional[int] = None) -> None:
    thickness = 1 if scale < 0.6 else 2
    if max_width is not None:
        (width, _h), _b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if width > max_width:
            scale *= max_width / float(width)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def compose_video_frame(main: np.ndarray, agent: np.ndarray, wrist: np.ndarray, *, title: Sequence[str],
                        clock_text: str, success: bool, note: str = "") -> np.ndarray:
    """1024 x 640 RGB: title on top, operator view left, agentview + wrist camera right."""

    import cv2

    width, height = VIDEO_SIZE
    frame = np.full((height, width, 3), 28, dtype=np.uint8)
    main_w, main_h = VIDEO_MAIN
    frame[VIDEO_TITLE_H:VIDEO_TITLE_H + main_h, :main_w] = main
    side = [cv2.resize(image, (VIDEO_SIDE, VIDEO_SIDE), interpolation=cv2.INTER_AREA)
            if image.shape[:2] != (VIDEO_SIDE, VIDEO_SIDE) else image for image in (agent, wrist)]
    frame[VIDEO_TITLE_H:VIDEO_TITLE_H + VIDEO_SIDE, main_w:] = side[0]
    frame[VIDEO_TITLE_H + VIDEO_SIDE:VIDEO_TITLE_H + 2 * VIDEO_SIDE, main_w:] = side[1]
    for index, line in enumerate(title[:2]):
        _put(cv2, frame, line, (10, 26 + 28 * index), 0.62 if index == 0 else 0.55, _WHITE if index == 0 else _GREY,
             max_width=width - 20)
    _put(cv2, frame, "operator view", (10, VIDEO_TITLE_H + 22), 0.5, _WHITE)
    _put(cv2, frame, "agentview", (main_w + 8, VIDEO_TITLE_H + 20), 0.45, _WHITE)
    _put(cv2, frame, "wrist camera", (main_w + 8, VIDEO_TITLE_H + VIDEO_SIDE + 20), 0.45, _WHITE)
    footer_y = VIDEO_TITLE_H + 2 * VIDEO_SIDE
    _put(cv2, frame, clock_text, (main_w + 10, footer_y + 26), 0.6, _WHITE)
    if note:
        _put(cv2, frame, note, (10, height - 12), 0.5, _AMBER, max_width=main_w - 20)
    if success:
        _put(cv2, frame, "SUCCESS", (main_w + 10, footer_y + 56), 0.75, _GREEN)
        cv2.rectangle(frame, (2, VIDEO_TITLE_H + 2), (main_w - 3, height - 3), _GREEN, 3)
    return frame


class _VideoWriter:
    def __init__(self, path: Path) -> None:
        import imageio_ffmpeg

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(self.path.stem + ".tmp.mp4")
        self._writer = imageio_ffmpeg.write_frames(
            str(self.temporary), VIDEO_SIZE, fps=VIDEO_FPS, codec="libx264", pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p", quality=VIDEO_QUALITY, macro_block_size=16, ffmpeg_log_level="error",
        )
        self._writer.send(None)
        self.frames = 0

    def add(self, frame: np.ndarray) -> None:
        self._writer.send(np.ascontiguousarray(frame, dtype=np.uint8))
        self.frames += 1

    def close(self) -> Path:
        self._writer.close()
        os.replace(self.temporary, self.path)
        return self.path

    def abort(self) -> None:
        try:
            self._writer.close()
        except Exception:
            pass
        try:
            self.temporary.unlink()
        except OSError:
            pass


# ------------------------------------------------------------ one demo
class DemoInfo:
    """A raw demo on disk: attrs at once, arrays and model only while it is being rendered."""

    def __init__(self, path: Path, suite_root: Path) -> None:
        self.path = Path(path)
        self.relative = self.path.relative_to(suite_root).as_posix()
        self.attrs = read_demo_attrs(self.path)
        self._arrays: Optional[dict[str, np.ndarray]] = None
        self._xml: Optional[str] = None
        self.index = int(_DEMO_RE.match(self.path.name).group(1))
        self.mount = str(self.attrs.get("hand_mount", "unknown"))
        self.duration_s = float(self.attrs.get("duration_s", 0.0))
        self.first = int(self.attrs.get("first_input_step") or 0)
        self.disturbed = list(self.attrs.get("disturbed_objects") or [])

    def _load(self) -> None:
        if self._arrays is None:
            self._arrays, _attrs, self._xml = read_demo_file(self.path)

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        self._load()
        assert self._arrays is not None
        return self._arrays

    @property
    def model_xml(self) -> str:
        self._load()
        assert self._xml is not None
        return self._xml

    def release(self) -> None:
        self._arrays, self._xml = None, None

    @property
    def video_name(self) -> str:
        return video_file_name(self.attrs["suite"], self.attrs["task_id"], self.index, self.mount, self.duration_s,
                               self.attrs["task_name"])

    def stamp(self) -> list[Any]:
        stat = self.path.stat()
        return [self.relative, int(stat.st_size), int(stat.st_mtime_ns)]


def _moved_note(demo: DemoInfo) -> str:
    if not demo.disturbed:
        return ""
    return "moved other objects: " + ", ".join(
        "{} ({:.0f} cm, {:.0f} deg)".format(entry["object"], 100.0 * entry["moved_m"], entry["tilt_deg"])
        for entry in demo.disturbed)


def process_demo(demo: DemoInfo, cache: _RendererCache, *, video_path: Optional[Path],
                 want_obs: bool) -> Optional[dict[str, np.ndarray]]:
    """One rendering pass: the video (if ``video_path``) and the dataset obs from the first move (if ``want_obs``)."""

    attrs, arrays = demo.attrs, demo.arrays
    states = np.asarray(arrays["states"], dtype=np.float64)
    renderer = cache.get(demo.model_xml)
    renderer.check_state_size(states.shape[1])
    steps = states.shape[0]
    start_obs = min(max(demo.first, 0), steps - 1)
    start_video = max(0, start_obs - VIDEO_PREROLL_STEPS)
    writer = _VideoWriter(video_path) if video_path is not None else None
    obs: dict[str, list[np.ndarray]] = {key: [] for _cam, key in DATASET_CAMERAS}
    low: dict[str, list[np.ndarray]] = {}
    title = [
        "{} task {} | demo {:03d} | {} mount | {:.1f} s | {}".format(
            attrs["suite"], attrs["task_id"], demo.index, demo.mount, demo.duration_s,
            str(attrs.get("collected_at", ""))[:10]),
        str(attrs.get("language", "")),
    ]
    note = _moved_note(demo)
    # Steps the planner drove are tagged in the video (demos before format v2 have no owner record).
    owners = np.asarray(arrays.get("control/arm_owner", np.zeros(steps)), dtype=np.int64)
    planner_code = ARM_OWNERS.index("planner")
    main_view = str(attrs["operator_camera"]["name"])
    try:
        if writer is not None:
            pose = attrs["operator_camera"]
            renderer.set_camera(main_view, pose["pos"], pose["quat_wxyz"], pose["fovy_deg"])
        control_hz = float(attrs.get("control_freq_hz", VIDEO_FPS))
        for index in range(start_video if writer is not None else start_obs, steps):
            renderer.set_state(states[index])
            upright = {name: renderer.render(name) for name, _key in DATASET_CAMERAS}
            if want_obs and index >= start_obs:
                for name, key in DATASET_CAMERAS:
                    obs[key].append(upright[name][::-1].copy())  # LIBERO / robosuite "opengl" rows
                for key, value in renderer.robot_obs(attrs).items():
                    low.setdefault(key, []).append(value)
            if writer is not None and index >= start_video:
                seconds = (index - start_obs) / control_hz
                clock = "t = {:.1f} s".format(seconds) if seconds >= 0 else "ready"
                if owners[index] == planner_code:
                    clock += "  planner"
                writer.add(compose_video_frame(
                    renderer.render(main_view, main=True), upright["agentview"], upright["robot0_eye_in_hand"],
                    title=title, clock_text=clock, success=False, note=note))
        if writer is not None:
            renderer.set_state(np.asarray(arrays["final_state"], dtype=np.float64))
            final = compose_video_frame(
                renderer.render(main_view, main=True), renderer.render("agentview"),
                renderer.render("robot0_eye_in_hand"), title=title,
                clock_text="t = {:.1f} s".format(demo.duration_s), success=True, note=note)
            for _ in range(VIDEO_TAIL_FRAMES):
                writer.add(final)
            writer.close()
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        if not want_obs:
            demo.release()
    if not want_obs:
        return None
    out = {key: np.stack(values) for key, values in obs.items()}
    out.update({key: np.stack(values) for key, values in low.items()})
    out["start"] = np.asarray(start_obs)
    return out


# ----------------------------------------------------------- datasets
def write_task_dataset(path: Path, demos: Sequence[DemoInfo], cache: _RendererCache, *,
                       videos: dict[str, Path]) -> Path:
    """LIBERO layout (``data/demo_i/{actions, states, rewards, dones, robot_states, obs/...}``), written atomically."""

    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    first = demos[0].attrs
    total = 0
    try:
        with h5py.File(temporary, "w") as handle:
            data = handle.create_group("data")
            for number, demo in enumerate(demos):
                obs = process_demo(demo, cache, video_path=videos.get(demo.relative), want_obs=True)
                assert obs is not None
                start = int(obs.pop("start"))
                actions = np.asarray(demo.arrays["actions"], dtype=np.float64)[start:]
                states = np.asarray(demo.arrays["states"], dtype=np.float64)[start:]
                steps = actions.shape[0]
                group = data.create_group("demo_{}".format(number))
                group.create_dataset("actions", data=actions)
                group.create_dataset("states", data=states)
                ends = np.zeros(steps, dtype=np.uint8)
                ends[-1] = 1
                group.create_dataset("rewards", data=ends)
                group.create_dataset("dones", data=ends)
                group.create_dataset("robot_states", data=np.concatenate(
                    (obs["gripper_states"], obs["ee_pos"], obs["ee_quat"]), axis=1))
                if "control/arm_owner" in demo.arrays:  # who set the arm action: ARM_OWNERS index
                    group.create_dataset("arm_owner", data=np.asarray(
                        demo.arrays["control/arm_owner"], dtype=np.uint8)[start:])
                obs_group = group.create_group("obs")
                for _camera, key in DATASET_CAMERAS:
                    images = obs.pop(key)
                    obs_group.create_dataset(key, data=images, chunks=(1,) + images.shape[1:],
                                             compression="gzip", compression_opts=4)
                obs.pop("ee_quat")
                for key, value in obs.items():
                    obs_group.create_dataset(key, data=value)
                group.attrs["num_samples"] = steps
                group.attrs["model_file"] = demo.model_xml
                group.attrs["init_state"] = states[0]
                group.attrs["source_demo"] = demo.relative
                group.attrs["trimmed_leading_steps"] = start
                group.attrs["duration_s"] = demo.duration_s
                group.attrs["hand_mount"] = demo.mount
                group.attrs["disturbed_objects"] = json.dumps(demo.disturbed)
                demo.release()
                total += steps
            env_kwargs = dict(first.get("env_kwargs") or {})
            env_kwargs.update({"camera_names": [camera for camera, _key in DATASET_CAMERAS],
                               "camera_heights": cache.image_size, "camera_widths": cache.image_size})
            problem = dict(first.get("problem_info") or {})
            data.attrs["env_name"] = str(problem.get("problem_name", ""))
            data.attrs["problem_info"] = json.dumps(problem)
            data.attrs["bddl_file_name"] = str(env_kwargs.get("bddl_file_name", first.get("bddl_file", "")))
            data.attrs["env_args"] = json.dumps({"type": 1, "env_name": problem.get("problem_name", ""),
                                                 "problem_name": problem.get("problem_name", ""),
                                                 "bddl_file": data.attrs["bddl_file_name"], "env_kwargs": env_kwargs})
            data.attrs["macros_image_convention"] = IMAGE_CONVENTION
            data.attrs["num_demos"] = len(demos)
            data.attrs["total"] = total
            data.attrs["tag"] = "libero_shadow_teleop_dataset_v2"
            data.attrs["arm_owner_codes"] = json.dumps(list(ARM_OWNERS))
            data.attrs["hand_mount"] = demos[0].mount
            data.attrs["action_layout"] = json.dumps(first.get("action_layout"))
            data.attrs["language"] = str(first.get("language", ""))
            data.attrs["success_rule"] = json.dumps(first.get("success_rule"))
            data.attrs["source_demos"] = json.dumps([demo.stamp() for demo in demos])
            data.attrs["image_size"] = cache.image_size
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return path


def _dataset_is_current(path: Path, demos: Sequence[DemoInfo], image_size: int) -> bool:
    if not path.is_file():
        return False
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            attrs = handle["data"].attrs
            return (json.loads(attrs["source_demos"]) == [demo.stamp() for demo in demos]
                    and int(attrs["image_size"]) == int(image_size))
    except Exception:
        return False


# ------------------------------------------------------------- summary
def _demo_paths(suite_root: Path) -> dict[int, list[Path]]:
    found: dict[int, list[Path]] = {}
    for folder in sorted(suite_root.glob("task[0-9][0-9]_*")):
        if not folder.is_dir():
            continue
        task = int(folder.name[4:6])
        paths = sorted((int(match.group(1)), path) for path in folder.iterdir()
                       for match in [_DEMO_RE.match(path.name)] if match and path.is_file())
        found[task] = [path for _index, path in paths]
    return found


def _records(suite_root: Path) -> list[dict[str, Any]]:
    path = suite_root / ATTEMPTS_FILE
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def write_summary(suite_root: Path, suite: str, demos_by_task: dict[int, list[DemoInfo]],
                  records: list[dict[str, Any]], demos_per_task: int, share_url: Optional[str]) -> dict[str, Any]:
    """SUMMARY.md, summary.csv and the Slack message; returns the totals."""

    tasks = sorted(set(demos_by_task) | {int(r["task_id"]) for r in records if "task_id" in r})
    names = {int(r["task_id"]): r.get("task_name", "") for r in records if "task_id" in r}
    language: dict[int, str] = {}
    for task, demos in demos_by_task.items():
        if demos:
            names[task] = demos[0].attrs["task_name"]
            language[task] = demos[0].attrs.get("language", "")
    rows = []
    all_stats = stats_from_records(records)
    total_demos = sum(len(demos) for demos in demos_by_task.values())
    lines = [
        "# {} teleoperation demos (Panda + Shadow hand)".format(suite),
        "",
        "Updated {}. Success = the task's own BDDL goal (LIBERO `_check_success()`) held 10 control steps "
        "(0.5 s). Only successful demos are kept. Success rate counts every try after the first move "
        "(a reset or hand-mount switch after that = failure; quitting mid-try is not counted). The arm is the "
        "operator's SpaceMouse, or the approach planner after a number key (SpaceMouse input stops it); the "
        "planner's steps are tagged in the videos and stored as `arm_owner`.".format(
            datetime.now().strftime("%Y-%m-%d %H:%M")),
        "",
        "| Task | Instruction | Demos | Success rate | Median time | Hand mount | Moved other objects |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task in tasks:
        demos = demos_by_task.get(task, [])
        stats = stats_from_records(records, task)
        mounts = sorted({demo.mount for demo in demos})
        moved = sum(1 for demo in demos if demo.disturbed)
        median = np.median([demo.duration_s for demo in demos]) if demos else None
        lines.append("| {} | {} | {}/{} | {} | {} | {} | {} |".format(
            task, language.get(task) or names.get(task, "").replace("_", " "), len(demos), demos_per_task,
            stats.describe(), "-" if median is None else "{:.1f} s".format(median), ", ".join(mounts) or "-",
            moved or "-"))
        for demo in demos:
            planner = demo.attrs.get("planner") or {}
            rows.append({"suite": suite, "task_id": task, "task_name": names.get(task, ""), "demo": demo.index,
                         "file": demo.relative, "hand_mount": demo.mount, "duration_s": demo.duration_s,
                         "steps": int(demo.attrs.get("num_steps", 0)), "first_input_step": demo.first,
                         "planner_requests": int(planner.get("requests", 0)),
                         "planner_steps": int(planner.get("steps", 0)),
                         "disturbed_objects": "; ".join(entry["object"] for entry in demo.disturbed),
                         "collected_at": demo.attrs.get("collected_at", ""), "operator": demo.attrs.get("operator", ""),
                         "video": "videos/" + demo.video_name})
    durations = [demo.duration_s for demos in demos_by_task.values() for demo in demos]
    lines.append("| **All** | | {} | {} | {} | | {} |".format(
        total_demos, all_stats.describe(), "{:.1f} s".format(np.median(durations)) if durations else "-",
        sum(1 for demos in demos_by_task.values() for demo in demos if demo.disturbed) or "-"))
    lines += ["", "## Videos", ""]
    for task in tasks:
        demos = demos_by_task.get(task, [])
        if not demos:
            continue
        lines.append("### Task {}: {}".format(task, language.get(task) or names.get(task, "")))
        lines.append("")
        for demo in demos:
            planner_steps = int((demo.attrs.get("planner") or {}).get("steps", 0))
            timed_steps = max(1, int(demo.attrs.get("num_steps", 0)) - demo.first)  # from the first move on
            lines.append("- [demo {:03d} ({}, {:.1f} s{}{})](videos/{})".format(
                demo.index, demo.mount, demo.duration_s,
                ", planner {:.0f}% of steps".format(100.0 * planner_steps / timed_steps) if planner_steps else "",
                ", moved other objects" if demo.disturbed else "", demo.video_name))
        lines.append("")
    (suite_root / SUMMARY_NAME).write_text("\n".join(lines), encoding="utf-8")
    with (suite_root / CSV_NAME).open("w", encoding="utf-8", newline="") as stream:
        fields = ["suite", "task_id", "task_name", "demo", "file", "hand_mount", "duration_s", "steps",
                  "first_input_step", "planner_requests", "planner_steps", "disturbed_objects", "collected_at",
                  "operator", "video"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = sum(1 for task in tasks if len(demos_by_task.get(task, [])) >= demos_per_task)
    message = "{} teleop demos (Panda + Shadow): {} demo{}, {} of {} tasks complete, success rate {}, median {} per demo.".format(
        suite, total_demos, "" if total_demos == 1 else "s", complete, len(tasks), all_stats.describe(),
        "{:.1f} s".format(np.median(durations)) if durations else "-")
    if share_url:
        message += " Videos + summary: {}".format(share_url)
    (suite_root / SLACK_NAME).write_text(message + "\n", encoding="utf-8")
    return {"demos": total_demos, "tasks": len(tasks), "complete": complete, "rate": all_stats.describe(),
            "message": message}


def copy_to_share(suite_root: Path, share_dir: Path, suite: str, *, datasets: bool) -> list[Path]:
    """Copy videos, SUMMARY.md, summary.csv (and datasets) into ``share_dir/<suite>``; unchanged files are skipped."""

    target_root = Path(share_dir) / suite
    copied = []
    sources = [suite_root / SUMMARY_NAME, suite_root / CSV_NAME, suite_root / SLACK_NAME]
    sources += sorted((suite_root / "videos").glob("*.mp4"))
    if datasets:
        sources += sorted((suite_root / "datasets").glob("*.hdf5"))
    for source in sources:
        if not source.is_file():
            continue
        target = target_root / source.relative_to(suite_root)
        if target.is_file() and target.stat().st_size == source.stat().st_size and \
                target.stat().st_mtime >= source.stat().st_mtime:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


# ---------------------------------------------------------------- all
def build_suite(output_root: Path, suite: str, *, tasks: Optional[Iterable[int]] = None,
                demos_per_task: int = DEMOS_PER_TASK, image_size: int = IMAGE_SIZE, videos: bool = True,
                datasets: bool = True, partial: bool = False, exclude_disturbed: bool = False, force: bool = False,
                share_dir: Optional[Path] = None, share_url: Optional[str] = None, share_datasets: bool = False,
                log: Any = print) -> dict[str, Any]:
    """Videos for demos without one, datasets for complete tasks (``partial``: every task with demos), summary, share."""

    suite_root = Path(output_root) / suite
    if not suite_root.is_dir():
        raise FileNotFoundError("no collection data at {}".format(suite_root))
    selected = None if tasks is None else {int(task) for task in tasks}
    paths = _demo_paths(suite_root)
    demos_by_task = {task: [DemoInfo(path, suite_root) for path in task_paths] for task, task_paths in paths.items()}
    cache = _RendererCache(image_size)
    made_videos: list[Path] = []
    made_datasets: list[Path] = []
    try:
        for task, demos in sorted(demos_by_task.items()):
            if selected is not None and task not in selected or not demos:
                continue
            wanted: dict[str, Path] = {}
            if videos:
                for demo in demos:
                    target = suite_root / "videos" / demo.video_name
                    if force or not target.is_file():
                        wanted[demo.relative] = target
            groups: dict[str, list[DemoInfo]] = {}
            for demo in demos:
                if not (exclude_disturbed and demo.disturbed):
                    groups.setdefault(demo.mount, []).append(demo)
            done_with_dataset: set[str] = set()
            if datasets and (partial or len(demos) >= demos_per_task):
                for mount, group in sorted(groups.items()):
                    target = suite_root / "datasets" / dataset_file_name(group[0].attrs["task_name"], mount)
                    if not force and _dataset_is_current(target, group, image_size):
                        continue
                    log("task {} ({}): dataset {} demos -> {}".format(task, mount, len(group), target.name))
                    write_task_dataset(target, group, cache, videos=wanted)
                    made_datasets.append(target)
                    done_with_dataset.update(demo.relative for demo in group if demo.relative in wanted)
            for demo in demos:  # videos not made during a dataset pass
                if demo.relative in wanted and demo.relative not in done_with_dataset:
                    log("task {}: video {}".format(task, demo.video_name))
                    process_demo(demo, cache, video_path=wanted[demo.relative], want_obs=False)
            made_videos.extend(wanted.values())
    finally:
        cache.close()
    totals = write_summary(suite_root, suite, demos_by_task, _records(suite_root), demos_per_task, share_url)
    shared: list[Path] = []
    if share_dir:
        shared = copy_to_share(suite_root, Path(share_dir), suite, datasets=share_datasets)
    return {"suite_root": suite_root, "videos": made_videos, "datasets": made_datasets, "shared": shared,
            "share_dir": None if not share_dir else Path(share_dir) / suite, **totals}
