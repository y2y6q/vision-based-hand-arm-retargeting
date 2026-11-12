"""LIBERO data collection (on the planner entry): bookkeeping, every task's success rule, the loop and the offline build.

The bookkeeping tests run in any interpreter with numpy (h5py for the file
tests); the LIBERO tests need .venv-libero (robosuite 1.4):

    .\\.venv-libero\\Scripts\\python.exe -m pytest -q .\\tests\\test_libero_collection.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.libero_collection import (  # noqa: E402
    AMBER,
    ARM_OWNERS,
    CSV_FIELDS,
    GREEN,
    GREY,
    RED,
    AttemptTable,
    DemoRecorder,
    DemoStore,
    DisturbanceMonitor,
    SuccessHold,
    TaskSuccessMonitor,
    attempt_outcome,
    attempt_video_name,
    collection_panel,
    dataset_file_name,
    describe_predicate,
    ended_by_text,
    goal_object_names,
    parse_task_list,
    placement_seed,
    read_demo_file,
    share_collection,
    stats_from_records,
    task_dir_name,
    tilt_deg,
    video_file_name,
)

try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:  # pragma: no cover
    HAVE_H5PY = False

try:
    import cv2  # noqa: F401
    import imageio_ffmpeg  # noqa: F401

    HAVE_VIDEO = True
except ImportError:  # pragma: no cover
    HAVE_VIDEO = False


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _video_frames(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    try:
        return [np.asarray(frame) for frame in reader]
    finally:
        reader.close()


def _close(pixel, rgb, tolerance: int = 40) -> bool:
    """Near a colour (H.264 / yuv420p shifts colours a little)."""
    return bool(np.all(np.abs(np.asarray(pixel, dtype=np.int32)[:3] - np.asarray(rgb, dtype=np.int32)) <= tolerance))

try:
    import robosuite

    LEGACY_ROBOSUITE = robosuite.__version__.startswith("1.4")
except ImportError:  # pragma: no cover
    LEGACY_ROBOSUITE = False


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class NamingTests(unittest.TestCase):
    def test_file_names(self):
        self.assertEqual(task_dir_name(3, "pick up/the bbq sauce"), "task03_pick_up_the_bbq_sauce")
        name = video_file_name("libero_object", 0, 3, "forward", 12.44,
                               "pick_up_the_alphabet_soup_and_place_it_in_the_basket")
        self.assertEqual(name, "libero_object_t00_demo003_forward_12.4s__"
                               "pick_up_the_alphabet_soup_and_place_it_in_the_basket.mp4")
        long_name = video_file_name("libero_10", 4, 0, "down", 61.0, "x" * 200)
        self.assertLess(len(long_name), 110)
        self.assertEqual(dataset_file_name("put_the_bowl_on_the_plate", "down"), "put_the_bowl_on_the_plate__down_demo.hdf5")
        live = attempt_video_name("libero_object", 0, 12, "success", 16.44,
                                  "pick_up_the_alphabet_soup_and_place_it_in_the_basket")
        self.assertEqual(live, "libero_object_t00_a012_SUCCESS_16.4s__"
                               "pick_up_the_alphabet_soup_and_place_it_in_the_basket.mp4")
        self.assertTrue(attempt_video_name("libero_spatial", 9, 3, "quit", None, "x" * 200).startswith(
            "libero_spatial_t09_a003_QUIT_0.0s__"))
        self.assertLess(len(attempt_video_name("libero_spatial", 9, 3, "failed", 5.0, "x" * 200)), 110)

    def test_what_ended_an_attempt_in_words(self):
        self.assertEqual(ended_by_text("R"), "R: new layout")
        self.assertEqual(ended_by_text("M"), "M: hand mount switch")
        self.assertEqual(ended_by_text("views:Q"), "Q: quit")
        self.assertEqual(ended_by_text("camera:Q"), "Q: quit")
        self.assertEqual(ended_by_text("views_window:Q"), "window closed")
        self.assertEqual(ended_by_text("completed"), "time limit")
        self.assertEqual(ended_by_text("Ctrl+C"), "Ctrl+C")

    def test_placement_seeds_do_not_repeat(self):
        seeds = {placement_seed(7, task, index) for task in range(90) for index in range(200)}
        self.assertEqual(len(seeds), 90 * 200)
        self.assertNotEqual(placement_seed(7, 0, 0), placement_seed(8, 0, 0))

    def test_task_lists(self):
        self.assertEqual(parse_task_list("all", 4), [0, 1, 2, 3])
        self.assertEqual(parse_task_list(None, 2), [0, 1])
        self.assertEqual(parse_task_list("3", 10), [3])
        self.assertEqual(parse_task_list("1,3,7-9,3", 10), [1, 3, 7, 8, 9])
        for bad in ("10", "4-2", "a", ","):
            with self.assertRaises(ValueError):
                parse_task_list(bad, 10)


class SuccessRuleTests(unittest.TestCase):
    def test_predicate_text_and_goal_objects(self):
        self.assertEqual(describe_predicate(["in", "alphabet_soup_1", "basket_1_contain_region"]),
                         "alphabet_soup_1 in basket_1_contain_region")
        self.assertEqual(describe_predicate(["turnon", "flat_stove_1"]), "flat_stove_1 turned on")
        self.assertEqual(describe_predicate(["stack", "a", "b"]), "stack(a, b)")
        goal = [["in", "alphabet_soup_1", "basket_1_contain_region"]]
        objects = ["alphabet_soup_1", "basket_1", "basket_10", "milk_1"]
        self.assertEqual(goal_object_names(goal, objects), {"alphabet_soup_1", "basket_1"})

    def test_hold_counts_only_after_the_start_and_restarts_when_broken(self):
        hold = SuccessHold(3)
        self.assertFalse(hold.update(True, started=False))  # met before the first move: never counts
        self.assertTrue(hold.met_before_start)
        self.assertFalse(hold.update(True, started=True))
        self.assertEqual(hold.count, 0)  # still met since before the start
        hold.update(False, started=True)
        self.assertFalse(hold.met_before_start)
        self.assertFalse(hold.update(True, started=True))
        self.assertFalse(hold.update(True, started=True))
        hold.update(False, started=True)  # bounced out: count again from zero
        self.assertEqual(hold.count, 0)
        results = [hold.update(True, started=True) for _ in range(3)]
        self.assertEqual(results, [False, False, True])
        hold.reset()
        self.assertEqual((hold.count, hold.met_before_start), (0, False))

    def test_monitor_uses_the_official_check_and_reports_each_condition(self):
        env = SimpleNamespace(
            parsed_problem={"goal_state": [["in", "a_1", "box_1_region"], ["turnon", "stove_1"]]},
            truth={"in": False, "turnon": True},
        )
        env._eval_predicate = lambda state: env.truth[state[0]]
        env._check_success = lambda: all(env.truth.values())
        monitor = TaskSuccessMonitor.for_env(env, hold_steps=2)
        self.assertEqual(monitor.labels, ["a_1 in box_1_region", "stove_1 turned on"])
        self.assertFalse(monitor.evaluate(env, started=True))
        self.assertEqual(monitor.status, [False, True])
        env.truth["in"] = True
        self.assertFalse(monitor.evaluate(env, started=True))
        self.assertTrue(monitor.evaluate(env, started=True))
        self.assertEqual(monitor.mismatches, 0)
        env._check_success = lambda: False  # an official result that disagrees is counted, and wins
        monitor.evaluate(env, started=True)
        self.assertEqual((monitor.success, monitor.mismatches, monitor.hold.count), (False, 1, 0))


class DisturbanceTests(unittest.TestCase):
    def test_moved_and_tipped_objects_are_flagged_with_their_worst_values(self):
        monitor = DisturbanceMonitor(["bottle_1", "milk_1"], move_m=0.03, tilt_limit_deg=30.0)
        start = {"bottle_1": (np.zeros(3), np.eye(3)), "milk_1": (np.ones(3), np.eye(3))}
        self.assertEqual(monitor.update(start), [])  # no reference yet: nothing to compare
        monitor.set_reference(start)
        nudged = {"bottle_1": (np.array([0.02, 0.0, 0.0]), _rot_z(1.0)), "milk_1": start["milk_1"]}
        self.assertEqual(monitor.update(nudged), [])  # 2 cm and a spin about the vertical: fine
        tipped = {"bottle_1": (np.array([0.05, 0.0, -0.03]), _rot_x(np.pi / 2)), "milk_1": start["milk_1"]}
        flagged = monitor.update(tipped)
        self.assertEqual([entry["object"] for entry in flagged], ["bottle_1"])
        self.assertAlmostEqual(flagged[0]["tilt_deg"], 90.0, places=5)
        monitor.update(nudged)  # put back: still remembered, with the worst values
        worst = monitor.flagged()
        self.assertEqual(len(worst), 1)
        self.assertAlmostEqual(worst[0]["moved_m"], np.hypot(0.05, 0.03), places=4)
        self.assertAlmostEqual(tilt_deg(np.eye(3), _rot_x(np.radians(40))), 40.0, places=5)


class PanelTests(unittest.TestCase):
    def _panel(self, **overrides):
        values = dict(task_id=0, language="pick up the soup", labels=["soup in basket"], status=[False],
                      started=True, met_before_start=False, hold_count=0, hold_steps=10, elapsed_s=3.25,
                      disturbed=[], saved=2, target=10, rate_text="2/3 (67%)")
        values.update(overrides)
        return collection_panel(**values)

    def test_panel_states(self):
        waiting = self._panel(started=False)
        self.assertIn("waiting for the SpaceMouse", waiting[2][0])
        self.assertEqual(waiting[2][1], AMBER)
        self.assertTrue(waiting[-2][0].startswith("1-9: approach a target"))  # the planner entry's keys
        self.assertTrue(waiting[-1][0].startswith("R: new layout"))
        self.assertTrue(waiting[-1][0].endswith("Q: quit"))
        self.assertIn("press R", self._panel(started=False, met_before_start=True)[2][0])
        self.assertEqual(self._panel()[2][0], "running   3.2 s")
        # S: whether this attempt is being recorded, right under the timer (before and after the first move).
        self.assertEqual(self._panel(recording=False)[3], ("not recording   S: record this attempt from now", GREY))
        self.assertEqual(self._panel(started=False, recording=True)[3],
                         ("REC  recording, saved when this attempt ends   S: cancel", RED))
        self.assertEqual(len(self._panel(recording=True)), len(self._panel()) + 1)  # no line when not given
        holding = self._panel(status=[True], hold_count=4)
        self.assertEqual(holding[1], ("[OK] soup in basket", GREEN))
        self.assertTrue(holding[2][0].startswith("SUCCESS 4/10"))
        moved = self._panel(disturbed=[{"object": "bottle_1", "moved_m": 0.06, "tilt_deg": 80.0}])
        self.assertIn("MOVED: bottle_1 (6 cm, tilted 80 deg)", [line for line, _color in moved])
        self.assertIn("successes 2/10   success rate 2/3 (67%)", [line for line, _color in moved])


class AttemptTests(unittest.TestCase):
    def test_outcomes_and_rate(self):
        self.assertEqual(attempt_outcome(True, "R"), "failed")
        self.assertEqual(attempt_outcome(True, "M"), "failed")
        self.assertEqual(attempt_outcome(True, "views:Q"), "quit")
        self.assertEqual(attempt_outcome(False, "R"), "not_started")
        records = [{"task_id": 0, "hand_mount": "forward", "outcome": o, "duration_s": d}
                   for o, d in (("success", 10.0), ("failed", None), ("success", 14.0), ("quit", None),
                                ("not_started", None))]
        records.append({"task_id": 1, "hand_mount": "down", "outcome": "success", "duration_s": 30.0})
        stats = stats_from_records(records, 0)
        self.assertEqual((stats.success, stats.failed, stats.quit, stats.not_started), (2, 1, 1, 1))
        self.assertEqual(stats.describe(), "2/3 (67%)")
        self.assertEqual(stats.median_s, 12.0)
        self.assertEqual(stats_from_records(records, mount="down").describe(), "1/1 (100%)")
        self.assertEqual(stats_from_records([], 0).describe(), "-")


def _row(attempt: int, task_id: int, result: str, layout: int, time_s: str = "", **extra) -> dict:
    row = {"attempt": attempt, "started_at": "2026-11-09 10:00:00", "task_id": task_id, "task": "pick up the soup",
           "hand_mount": "down", "layout": layout, "result": result, "time_s": time_s, "ended_by": "R: new layout",
           "moved_objects": "", "video": "v{}.mp4".format(attempt), "operator": ""}
    row.update(extra)
    return row


class AttemptTableTests(unittest.TestCase):
    def test_rows_counts_resume_and_a_cut_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "libero_object" / "attempts.csv"
            table = AttemptTable(path)
            self.assertEqual((table.success_count(0), table.next_attempt_number(), table.next_layout(0)), (0, 1, 0))
            self.assertEqual(table.stats(0).describe(), "-")
            self.assertFalse(path.exists())  # nothing is created until a row is written
            self.assertTrue(table.append(_row(1, 0, "failed", 0, "4.20")))
            self.assertTrue(table.append(_row(2, 0, "success", 1, "12.30", ended_by="goal held 0.5 s",
                                              moved_objects="milk_1 (6 cm, tilted 80 deg)")))
            self.assertTrue(table.append(_row(3, 1, "quit", 0, "2.00", ended_by="Q: quit")))
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))  # Excel opens it as UTF-8
            rows = _read_csv(path)
            self.assertEqual(list(rows[0]), list(CSV_FIELDS))
            self.assertEqual([(r["attempt"], r["result"], r["time_s"]) for r in rows],
                             [("1", "failed", "4.20"), ("2", "success", "12.30"), ("3", "quit", "2.00")])
            self.assertEqual(rows[1]["moved_objects"], "milk_1 (6 cm, tilted 80 deg)")
            with path.open("a", encoding="utf-8", newline="") as stream:
                stream.write("4,2026-11-09 10:05:00,0,pick up")  # cut by a crash: no result, no newline
            again = AttemptTable(path)
            self.assertEqual(len(again.rows), 3)  # the cut row is skipped
            self.assertEqual((again.success_count(0), again.success_count(1)), (1, 0))
            self.assertEqual(again.stats(0).describe(), "1/2 (50%)")  # quit is not counted
            self.assertEqual(again.stats(0).median_s, 12.3)
            self.assertEqual((again.next_attempt_number(), again.next_layout(0), again.next_layout(1)), (4, 2, 1))
            self.assertTrue(again.append(_row(4, 0, "success", 2, "9.00")))
            rows = _read_csv(path)
            self.assertEqual([r["attempt"] for r in rows if r["result"]], ["1", "2", "3", "4"])  # on its own line
            self.assertEqual(AttemptTable(path).success_count(0), 2)

    def test_a_locked_table_keeps_its_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "attempts.csv"
            path.mkdir()  # cannot be opened as a file, like a CSV Excel holds open
            table = AttemptTable(path)
            self.assertFalse(table.append(_row(1, 0, "success", 0, "8.00")))
            self.assertIsNotNone(table.blocked)
            self.assertEqual((table.pending, table.success_count(0), table.next_attempt_number()), (1, 1, 2))
            unsaved = table.save_unwritten("20261109_100000")
            self.assertEqual(unsaved.name, "attempts_unsaved_20261109_100000.csv")
            self.assertEqual([r["result"] for r in _read_csv(unsaved)], ["success"])
            path.rmdir()  # closed in Excel: the next write gets everything
            self.assertTrue(table.append(_row(2, 0, "failed", 1, "3.00")))
            self.assertEqual((table.pending, table.blocked), (0, None))
            self.assertEqual([r["attempt"] for r in _read_csv(path)], ["1", "2"])

    def test_share_copies_the_table_and_finished_videos(self):
        with tempfile.TemporaryDirectory() as folder:
            suite_root = Path(folder) / "collect" / "libero_object"
            (suite_root / "videos").mkdir(parents=True)
            (suite_root / "attempts.csv").write_text("attempt,result\n1,success\n", encoding="utf-8")
            (suite_root / "videos" / "a.mp4").write_bytes(b"1234")
            (suite_root / "videos" / "_recording_1_1.mp4").write_bytes(b"12")  # still being written
            share = Path(folder) / "share"
            copied = share_collection(suite_root, share, "libero_object")
            self.assertEqual(sorted(p.relative_to(share).as_posix() for p in copied),
                             ["libero_object/attempts.csv", "libero_object/videos/a.mp4"])
            again = share_collection(suite_root, share, "libero_object")
            self.assertEqual([p.name for p in again], ["attempts.csv"])  # the video is there already


@unittest.skipUnless(HAVE_VIDEO, "needs OpenCV and imageio-ffmpeg")
class AttemptVideoTests(unittest.TestCase):
    """The live video: the window frames as pushed, a timer header, the result held at the end."""

    def _frames(self, count: int, height: int = 360, width: int = 640) -> list[np.ndarray]:
        frames = []
        for index in range(count):
            frame = np.full((height, width, 3), 90, dtype=np.uint8)
            frame[:, :width // 2] = (200, 30, 30)  # left half red: the frame is not flipped
            frame[height // 2:height // 2 + 20, (index * 7) % (width - 20):(index * 7) % (width - 20) + 20] = 255
            frames.append(frame)
        return frames

    def test_preroll_timer_result_and_stale_frame(self):
        from wrist_teleop.libero_attempt_video import HEADER_BG, RESULT_STYLE, AttemptVideo, video_geometry

        self.assertEqual(video_geometry(2544, 1334), (1920, 1006, 82))  # the real window -> 1920 x 1088
        self.assertEqual(video_geometry(640, 360, 320), (320, 180, 44))
        with tempfile.TemporaryDirectory() as folder:
            frames = self._frames(36)
            stale = frames[0]
            video = AttemptVideo(Path(folder), fps=20.0, title="TASK 0: pick up the soup", subtitle="test | attempt 1",
                                 width=320, preroll_s=0.5, hold_s=0.5, queue_frames=200, stale_frame=stale)
            video.push(stale, None)  # the previous layout's picture: skipped
            for frame in frames[1:16]:
                video.push(frame, None)  # before the first move: only the last 0.5 s is kept
            for index, frame in enumerate(frames[16:36]):
                video.push(frame, index / 20.0)
            video.finish("success", 1.0, "goal held 0.5 s", "clip_SUCCESS_1.0s.mp4")
            path = video.wait(60)
            self.assertIsNone(video.error)
            self.assertEqual(path, Path(folder) / "clip_SUCCESS_1.0s.mp4")
            self.assertEqual((video.frames, video.dropped), (10 + 20 + 10, 0))
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), [path.name])  # no temporary file left
            out = _video_frames(path)
            self.assertEqual(len(out), 40)
            self.assertEqual(out[0].shape, (224, 320, 3))
            self.assertTrue(_close(out[0][3, 3], HEADER_BG))  # READY header before the first move
            self.assertTrue(_close(out[15][3, 3], HEADER_BG))  # timer header while recording
            self.assertTrue(_close(out[15][44 + 40, 40], (200, 30, 30)))  # the window, not flipped
            self.assertTrue(_close(out[15][44 + 40, 280], (90, 90, 90)))
            self.assertTrue(_close(out[-1][3, 3], RESULT_STYLE["success"][1]))  # SUCCESS header held at the end
            self.assertTrue(_close(out[-1][-2, 160], RESULT_STYLE["success"][1]))  # and the coloured border

    def test_discard_leaves_nothing_and_an_early_finish_still_writes(self):
        from wrist_teleop.libero_attempt_video import RESULT_STYLE, AttemptVideo

        with tempfile.TemporaryDirectory() as folder:
            frames = self._frames(6)
            dropped = AttemptVideo(Path(folder), fps=20.0, title="t", subtitle="s", width=320, queue_frames=50)
            for index, frame in enumerate(frames):
                dropped.push(frame, index / 20.0)
            dropped.discard()
            self.assertIsNone(dropped.wait(60))
            self.assertEqual(list(Path(folder).iterdir()), [])
            # The first move's frame had not reached the video yet: the kept frames and the result still make one.
            early = AttemptVideo(Path(folder), fps=20.0, title="t", subtitle="s", width=320, hold_s=0.25,
                                 queue_frames=50)
            for frame in frames[:3]:
                early.push(frame, None)
            early.finish("failed", 0.0, "R: new layout", "clip_FAILED_0.0s.mp4")
            path = early.wait(60)
            self.assertIsNone(early.error)
            out = _video_frames(path)
            self.assertEqual(len(out), 3 + 5)
            self.assertTrue(_close(out[-1][3, 3], RESULT_STYLE["failed"][1]))

    def test_an_unfinished_recording_stays_out_of_the_videos_folder(self):
        from wrist_teleop.libero_attempt_video import AttemptVideo

        with tempfile.TemporaryDirectory() as root:
            videos, work = Path(root) / "videos", Path(root) / "run_log"
            video = AttemptVideo(videos, fps=20.0, title="t", subtitle="s", width=320, queue_frames=50, work_dir=work)
            for index, frame in enumerate(self._frames(10)):
                video.push(frame, index / 20.0)
            deadline = time.monotonic() + 20.0
            while not (work.is_dir() and any(work.iterdir())) and time.monotonic() < deadline:
                time.sleep(0.02)
            # Recording: the file is in the run log, nothing is in the videos folder yet.
            self.assertEqual([p.suffix for p in work.iterdir()], [".mp4"])
            self.assertFalse(videos.exists())
            video.finish("quit", 0.5, "Q: quit", "clip_QUIT_0.5s.mp4")
            path = video.wait(60)
            self.assertIsNone(video.error)
            self.assertEqual(path, videos / "clip_QUIT_0.5s.mp4")
            self.assertEqual(list(work.iterdir()), [])  # moved, not copied
            self.assertEqual([p.name for p in videos.iterdir()], [path.name])
            self.assertEqual(len(_video_frames(path)), 10 + 40)  # 10 frames + the 2 s result hold

    def test_a_cancelled_recording_is_deleted_from_the_run_log(self):
        from wrist_teleop.libero_attempt_video import AttemptVideo

        with tempfile.TemporaryDirectory() as root:
            videos, work = Path(root) / "videos", Path(root) / "run_log"
            video = AttemptVideo(videos, fps=20.0, title="t", subtitle="s", width=320, queue_frames=50, work_dir=work)
            for index, frame in enumerate(self._frames(10)):
                video.push(frame, index / 20.0)
            deadline = time.monotonic() + 20.0
            while not (work.is_dir() and any(work.iterdir())) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual([p.suffix for p in work.iterdir()], [".mp4"])  # it was recording
            video.discard()  # S pressed again: cancelled
            self.assertIsNone(video.wait(60))
            self.assertIsNone(video.error)
            self.assertTrue(video.started)
            self.assertEqual(list(work.iterdir()), [])  # ffmpeg stopped, the file deleted
            self.assertFalse(videos.exists())


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class DemoFileTests(unittest.TestCase):
    def _recorder(self, steps: int = 5) -> DemoRecorder:
        recorder = DemoRecorder(goal_count=1, hand_dof=20)
        for step in range(steps):
            recorder.append(
                state=np.arange(7.0) + step, action=np.full(26, step, dtype=np.float64), success=step >= 3,
                goal_status=[step >= 3], moving=step >= 1, spacemouse_axes=np.ones(6), spacemouse_buttons=(True,),
                arm_delta=np.zeros(6), hand_targets=None if step == 0 else np.zeros(20),
                hand_valid=step > 0, mano_landmarks=np.zeros((21, 3)) if step == 2 else None,
                arm_owner="planner" if step == 4 else "spacemouse", wall_s=0.05 * step, sim_s=0.05 * step)
        return recorder

    def test_recorder_round_trip_through_the_store(self):
        recorder = self._recorder()
        self.assertEqual((recorder.steps, recorder.first_input_step), (5, 1))
        self.assertAlmostEqual(recorder.elapsed_s(0.25), 0.2)
        self.assertEqual(recorder.planner_steps, 1)
        with self.assertRaises(ValueError):
            recorder.append(state=np.zeros(7), action=np.zeros(26), success=False, goal_status=[False], moving=False,
                            spacemouse_axes=np.zeros(6), spacemouse_buttons=(), arm_delta=np.zeros(6),
                            arm_owner="policy", wall_s=0.0, sim_s=0.0)
        self.assertEqual(recorder.steps, 5)  # the rejected step left no partial row
        with tempfile.TemporaryDirectory() as folder:
            store = DemoStore(Path(folder) / "libero_object", {0: "pick_up_the_soup"})
            self.assertFalse(store.root.exists())  # nothing is created until something is written
            attrs = {"task_id": 0, "hand_mount": "forward", "duration_s": 0.2, "success_rule": [["in", "a", "b"]],
                     "disturbed_objects": [], "language": "pick up the soup", "first_input_step": 1, "note": None}
            first = store.save_demo(0, recorder.arrays(np.arange(7.0) + 9), attrs, "<mujoco/>")
            second = store.save_demo(0, recorder.arrays(np.arange(7.0)), attrs, "<mujoco/>")
            self.assertEqual([p.name for p in store.demo_paths(0)], ["demo_000.hdf5", "demo_001.hdf5"])
            self.assertFalse(list(store.task_dir(0).glob("*.tmp")))
            arrays, read_attrs, xml = read_demo_file(first)
            self.assertEqual(xml, "<mujoco/>")
            self.assertEqual(arrays["actions"].shape, (5, 26))
            self.assertEqual(arrays["states"].shape, (5, 7))
            self.assertEqual(arrays["final_state"][0], 9.0)
            self.assertEqual(arrays["goal_status"].shape, (5, 1))
            self.assertTrue(np.isnan(arrays["human/hand_targets"][0]).all())
            self.assertTrue(np.isnan(arrays["human/mano_landmarks"][0]).all())
            self.assertFalse(np.isnan(arrays["human/mano_landmarks"][2]).any())
            self.assertEqual(arrays["human/spacemouse_buttons"].shape, (5, 8))
            self.assertEqual(arrays["control/arm_owner"].tolist(), [0, 0, 0, 0, 1])  # ARM_OWNERS index
            self.assertEqual(read_attrs["success_rule"], [["in", "a", "b"]])
            self.assertEqual(read_attrs["demo_index"], 0)
            self.assertIsNone(read_attrs["note"])
            second.unlink()  # a deleted demo's number is not reused while a higher one exists
            store.save_demo(0, recorder.arrays(np.zeros(7)), attrs, "<mujoco/>")
            first.unlink()
            self.assertEqual(store.next_demo_index(0), 2)
            store.log_attempt({"attempt": 1, "task_id": 0, "outcome": "failed", "placement_index": 0})
            store.log_attempt({"attempt": 2, "task_id": 0, "outcome": "success", "placement_index": 1, "duration_s": 3.0})
            with store.attempts_path.open("a", encoding="utf-8") as stream:
                stream.write('{"attempt": 3, "task_id": 0, "outc')  # cut by a crash
            again = DemoStore(store.root, {0: "pick_up_the_soup"})
            self.assertEqual(again.next_attempt_number(), 3)
            self.assertEqual(again.next_placement_index(0), 2)
            self.assertEqual(again.next_placement_index(5), 0)
            self.assertEqual(again.stats(0).describe(), "1/2 (50%)")


@unittest.skipUnless(LEGACY_ROBOSUITE, "LIBERO tests require .venv-libero (robosuite 1.4)")
class ViewTests(unittest.TestCase):
    def test_close_up_is_the_planner_close_up_after_two_v_presses_fixed_and_mirrored(self):
        from dataclasses import replace

        import cv2

        from wrist_teleop.config import WristConfig
        from wrist_teleop.libero_collect_teleop import CLOSE_UP_LABEL, COLLECT_DETAIL_FOLLOW, CollectionViewWindow
        from wrist_teleop.libero_planner_teleop import LABEL_RGB, PlannerViewWindow, _AnnotatedStatus
        from wrist_teleop.libero_spatial_teleop import DETAIL_FOLLOW
        from wrist_teleop.libero_views import Panel

        config = WristConfig.load(ROOT / "configs" / "spacemouse_wrist.json").tri_view
        planner_window = PlannerViewWindow(
            config, (Panel("front", "", camera="frontview"), Panel("wrist", "CLOSE-UP", follow=DETAIL_FOLLOW),
                     Panel("side", "3D", follow=DETAIL_FOLLOW)), size=(960, 540), show_window=False)
        planner_window.cycle_detail_view()
        planner_window.cycle_detail_view()  # V twice in the planner entry
        # The camera the planner entry renders then; every other close-up setting is unchanged.
        self.assertEqual(COLLECT_DETAIL_FOLLOW.azimuth_deg, DETAIL_FOLLOW.azimuth_deg + planner_window.orbit_deg)
        self.assertEqual(COLLECT_DETAIL_FOLLOW.azimuth_deg, 270.0)  # on the robot's left, looking at the hand
        self.assertEqual(replace(COLLECT_DETAIL_FOLLOW, azimuth_deg=DETAIL_FOLLOW.azimuth_deg, orbit=True), DETAIL_FOLLOW)
        self.assertFalse(COLLECT_DETAIL_FOLLOW.orbit)  # fixed: V does not turn it
        panels = (Panel("front", "", camera="frontview"), Panel("wrist", CLOSE_UP_LABEL, follow=COLLECT_DETAIL_FOLLOW),
                  Panel("side", "3D", follow=COLLECT_DETAIL_FOLLOW))
        window = CollectionViewWindow(config, panels, size=(960, 540), show_window=False)
        self.assertIsInstance(window, PlannerViewWindow)  # fingertip rays and target badges as in the planner entry
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        x, y, width, height = window.rects["wrist"]
        frame[y:y + height, x:x + width // 2] = (200, 10, 10)  # left half red, right half blue
        frame[y:y + height, x + width // 2:x + width] = (10, 10, 200)
        front_x, front_y, front_w, front_h = window.rects["front"]
        badge = (front_x + 60, front_y + 90)
        window.panel_lines = (("TASK 0: test", (240, 240, 240)), ("[OK] a in b", (90, 225, 90)))
        window._decorate(cv2, frame, _AnnotatedStatus("status", [(badge[0], badge[1], "3")]))
        bottom = y + height - 5
        self.assertEqual(tuple(frame[bottom, x + 5]), (10, 10, 200))  # the image is mirrored: left is now blue
        self.assertEqual(tuple(frame[bottom, x + width - 5]), (200, 10, 10))
        patch = frame[badge[1] - 6:badge[1] + 7, badge[0] - 6:badge[0] + 7]
        self.assertTrue(np.all(patch == np.asarray(LABEL_RGB, dtype=np.uint8), axis=-1).any())  # target badge
        self.assertTrue(frame[front_y + front_h - 60:front_y + front_h, front_x + front_w // 2:front_x + front_w].any())


def _teleport_into_goal(env) -> None:
    """Put each goal object 3-5 cm above its goal region / object, at rest."""

    for state in env.parsed_problem["goal_state"]:
        name, target = state[1], state[2]
        joint = env.get_object(name).joints[-1]
        if target in env.object_sites_dict:
            place = np.array(env.sim.data.get_site_xpos(target), dtype=np.float64) + (0.0, 0.0, 0.03)
        else:
            place = np.array(env.sim.data.body_xpos[env.obj_body_id[target]], dtype=np.float64) + (0.0, 0.0, 0.05)
        qpos = np.array(env.sim.data.get_joint_qpos(joint), dtype=np.float64)
        qpos[:3] = place
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.data.set_joint_qvel(joint, np.zeros(6))
    env.sim.forward()


def _tip_over(env, name: str) -> None:
    joint = env.get_object(name).joints[-1]
    qpos = np.array(env.sim.data.get_joint_qpos(joint), dtype=np.float64)
    qpos[2] += 0.02
    qpos[3:] = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)  # lying on its side
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()


@unittest.skipUnless(LEGACY_ROBOSUITE, "LIBERO tests require .venv-libero (robosuite 1.4)")
class EveryTaskSuccessRuleTests(unittest.TestCase):
    """Each libero_object / libero_spatial task: the monitor is that task's BDDL goal and agrees with LIBERO."""

    def _check_suite(self, suite: str) -> None:
        from wrist_teleop.config import WristConfig
        from wrist_teleop.libero_collect_teleop import CollectionSession, suite_task_table
        from wrist_teleop.libero_spatial_teleop import LiberoActionComposer, shadow_spec

        config = WristConfig.load(ROOT / "configs" / "spacemouse_wrist.json")
        table = suite_task_table(suite)
        self.assertEqual(len(table), 10)
        for row in table:
            with self.subTest(suite=suite, task=row["task_id"]):
                session = CollectionSession(shadow_spec("forward"), config, suite=suite, task_id=row["task_id"], seed=7)
                try:
                    session.open()
                    session.reset_to(placement_seed(7, row["task_id"], 0))
                    env = session.env
                    monitor = TaskSuccessMonitor.for_env(env, hold_steps=10)
                    self.assertEqual(monitor.goal_states, row["goal"])
                    composer = LiberoActionComposer(env, shadow_spec("forward"))
                    composer.initialize_goal()
                    hold = np.zeros(6)
                    env.step(composer.compose(hold, None))
                    self.assertFalse(monitor.evaluate(env, started=True))
                    self.assertFalse(monitor.success, "goal met right after the random placement")
                    _teleport_into_goal(env)
                    confirmed = False
                    for _ in range(40):
                        env.step(composer.compose(hold, None))
                        confirmed = monitor.evaluate(env, started=True) or confirmed
                    self.assertTrue(monitor.success, "official _check_success() after placing the object")
                    self.assertEqual(monitor.status, [True] * len(monitor.goal_states))
                    self.assertTrue(confirmed)
                    self.assertEqual(monitor.mismatches, 0)
                finally:
                    session.close()

    def test_libero_object(self):
        self._check_suite("libero_object")

    def test_libero_spatial(self):
        self._check_suite("libero_spatial")


class _ScriptedDevice:
    """A SpaceMouse that pushes forward for the first 3 polls after each reset, then rests."""

    def __init__(self, config) -> None:
        self.raw = 0.6 * float(np.asarray(config.raw_ranges, dtype=np.float64)[0])
        self.polls = 0

    def open(self):
        return {"product_string": "scripted-test-device"}

    def reset_input_state(self) -> None:
        self.polls = 0

    def poll(self):
        from wrist_teleop.device import MotionSample

        self.polls += 1
        axes = np.zeros(6, dtype=np.float64)
        if self.polls <= 3:
            axes[0] = self.raw
        return MotionSample(axes.copy(), (False,) * 8, True, True, time.monotonic(), buttons_valid=True,
                            calibrated_axes=axes.copy(), zero_bias_axes=np.zeros(6), neutral_locked=True,
                            calibration_state="scripted")

    def close(self) -> None:
        return None


@unittest.skipUnless(LEGACY_ROBOSUITE, "LIBERO tests require .venv-libero (robosuite 1.4)")
class CollectionLoopTests(unittest.TestCase):
    def test_r_m_success_next_layout_task_done_then_build(self):
        from wrist_teleop.config import WristConfig
        from wrist_teleop.libero_attempt_video import RESULT_STYLE
        from wrist_teleop.libero_collect_teleop import SETTLE_CALM_STEPS, CollectionSession, run_libero_collect
        from wrist_teleop.libero_dataset_build import VIDEO_PREROLL_STEPS, VIDEO_TAIL_FRAMES, build_suite
        from wrist_teleop.libero_spatial_teleop import shadow_spec

        config = WristConfig.load(ROOT / "configs" / "spacemouse_wrist.json")
        tipped: list[str] = []

        def hook(env, info):
            if info.attempt == 1 and info.step == 1:
                return "s"  # record this attempt from now on
            if info.attempt == 1 and info.step == 5:
                return "r"  # moved already: a failure
            if info.attempt == 2 and info.step == 5:
                return "m"  # moved already: a failure (no S: nothing recorded), then the other mount, same layout
            if info.attempt == 3 and info.step == 0:
                return "2"  # asked while the SpaceMouse still pushes: the SpaceMouse keeps the arm
            if info.attempt == 3 and info.step in (3, 4):
                return "s"  # S held down (auto-repeat, 50 ms apart): it toggles once, the recording goes on
            if info.attempt == 3 and info.step == 8:
                tipped.append(sorted(set(env.objects_dict) - {"alphabet_soup_1", "basket_1"})[0])
                _tip_over(env, tipped[-1])  # knock a bystander over: flagged, still a success
            if info.attempt == 4 and info.step == 1:
                return "s"
            if info.attempt == 4 and info.step == 4:
                return "1"  # SpaceMouse at rest: the planner drives toward target 1 ...
            if info.attempt == 4 and info.step == 9:
                return "0"  # ... until 0 cancels it (or it has arrived)
            if info.attempt == 4 and info.step == 12:
                return "s"  # S again: the recording is cancelled, this success gets no video
            if info.attempt in (3, 4) and info.step == 15:
                _teleport_into_goal(env)
            return None

        with tempfile.TemporaryDirectory() as folder:
            root, demo_root, log_root = Path(folder) / "collect", Path(folder) / "demos", Path(folder) / "logs"
            result = run_libero_collect(
                config, suite="libero_object", tasks=[0], demos_per_task=2, hand_mount="forward", seed=7,
                headless=True, duration_s=90.0, camera_display=False, hand_control="off", view_size=(960, 540),
                planner_backend="straight", output_root=root, operator="test", save_demos=True,
                demo_root=demo_root, log_root=log_root, device=_ScriptedDevice(config), step_hook=hook,
            )
            self.assertEqual((result.exit_code, result.reason), (0, "all tasks done"))
            self.assertEqual((result.successes, len(result.demos), len(result.videos)), (2, 2, 2))
            suite_root = root / "libero_object"
            # The collection folder holds the table and the videos, nothing else; the run log is elsewhere.
            self.assertEqual(sorted(p.name for p in suite_root.iterdir()), ["attempts.csv", "videos"])
            # Every attempt has its row; a video only where S started one (and it was not cancelled).
            events = [json.loads(line) for line in (result.run_dir / "telemetry.jsonl").read_text(
                encoding="utf-8").splitlines()]
            presses = [e for e in events if e.get("event") == "video_record"]
            self.assertEqual([(e["attempt"], e["recording"], e["step"]) for e in presses],
                             [(1, True, 2), (3, True, 4), (4, True, 2), (4, False, 13)])
            exits = [e for e in events if e.get("event") == "exit"]
            self.assertEqual(exits[0]["recordings_cancelled"], 1)
            self.assertEqual(list(log_root.glob("*/*.mp4")), [])  # the cancelled recording was deleted
            self.assertEqual(result.run_dir.parent, log_root)
            self.assertTrue((result.run_dir / "metadata.json").is_file())
            rows = _read_csv(suite_root / "attempts.csv")
            self.assertEqual([(r["result"], r["ended_by"], r["hand_mount"]) for r in rows], [
                ("failed", "R: new layout", "forward"), ("failed", "M: hand mount switch", "forward"),
                ("success", "goal held 0.5 s", "down"), ("success", "goal held 0.5 s", "down")])
            self.assertEqual([(r["attempt"], r["layout"]) for r in rows], [("1", "0"), ("2", "1"), ("3", "1"), ("4", "2")])
            self.assertEqual({(r["task_id"], r["operator"]) for r in rows}, {("0", "test")})
            self.assertTrue(all(r["task"].startswith("pick up the alphabet soup") for r in rows))
            self.assertIn(tipped[0], rows[2]["moved_objects"])
            self.assertEqual(rows[3]["moved_objects"], "")
            self.assertEqual([bool(r["video"]) for r in rows], [True, False, True, False])
            kept = [r for r in rows if r["video"]]
            videos = sorted((suite_root / "videos").iterdir())
            self.assertEqual([p.name for p in videos], sorted(r["video"] for r in kept))
            self.assertEqual(videos, sorted(result.videos))
            for row in kept:
                self.assertTrue(row["video"].startswith("libero_object_t00_a{:03d}_{}_".format(
                    int(row["attempt"]), row["result"].upper())), row["video"])
                self.assertIn("s__pick_up_the_alphabet_soup", row["video"])
            # A video is the window as drawn (960 x 540 here) under a 36-pixel header; the result is held at the end.
            success_clip = _video_frames(suite_root / "videos" / rows[2]["video"])
            self.assertEqual(success_clip[0].shape, (576, 960, 3))
            self.assertGreater(len(success_clip), 40 + 15)  # the 2 s result after the recorded ticks
            ends = [e for e in events if e.get("event") == "attempt_end"]
            # From the S press (handled before step 4) to the end, then the 2 s result: not the whole attempt.
            self.assertAlmostEqual(len(success_clip), ends[2]["steps"] - 4 + 40, delta=3)
            self.assertTrue(_close(success_clip[-1][3, 3], RESULT_STYLE["success"][1]))
            self.assertTrue(_close(_video_frames(suite_root / "videos" / rows[0]["video"])[-1][3, 3],
                                   RESULT_STYLE["failed"][1]))
            live_share = share_collection(suite_root, Path(folder) / "share_live", "libero_object")
            self.assertEqual(len(live_share), 3)  # the table and the two kept videos

            # --save-demos: the raw demos and their attempt list, in their own folder.
            demo_suite = demo_root / "libero_object"
            records = [json.loads(line) for line in (demo_suite / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([(r["outcome"], r["reason"], r["hand_mount"]) for r in records], [
                ("failed", "R", "forward"), ("failed", "M", "forward"),
                ("success", "goal held 0.5 s", "down"), ("success", "goal held 0.5 s", "down")])
            self.assertEqual([r["video"] or "" for r in records], [r["video"] for r in rows])
            self.assertEqual([r["demo_file"] is None for r in records], [True, True, False, False])
            self.assertEqual(records[1]["placement_seed"], records[2]["placement_seed"])  # M keeps the layout
            self.assertNotEqual(records[2]["placement_seed"], records[3]["placement_seed"])
            self.assertEqual([r["placement_index"] for r in records], [0, 1, 1, 2])
            self.assertIn(tipped[0], [o["object"] for o in records[2]["disturbed_objects"]])
            self.assertEqual(records[3]["disturbed_objects"], [])
            self.assertTrue(all(r["first_input_step"] == 0 for r in records))
            self.assertEqual([r["planner"]["requests"] for r in records], [0, 0, 1, 1])
            self.assertEqual((records[2]["planner"]["cancelled"], records[2]["planner"]["steps"]), (1, 0))
            self.assertGreater(records[3]["planner"]["steps"], 0)

            # Attempt 3: a number key while the SpaceMouse pushes. One owner per tick: the planner is
            # stopped in that same tick and never moves the arm.
            _arrays3, attrs3, _xml3 = read_demo_file(result.demos[0])
            events3 = attrs3["planner_events"]
            self.assertEqual([(e["event"], e["step"]) for e in events3], [("request", 1), ("cancel", 1)])
            self.assertEqual((events3[0]["number"], events3[1]["reason"]), (2, "spacemouse"))
            self.assertFalse(_arrays3["control/arm_owner"].any())
            # Attempt 4: the planner drove a contiguous stretch with the SpaceMouse at rest, then 0 (or arrival).
            arrays4, attrs4, _xml4 = read_demo_file(result.demos[1])
            events4 = attrs4["planner_events"]
            self.assertEqual([e["event"] for e in events4[:2]], ["request", "result"])
            self.assertEqual((events4[0]["step"], events4[0]["number"]), (5, 1))
            self.assertIn(events4[2]["event"], ("cancel", "arrived"))
            if events4[2]["event"] == "cancel":
                self.assertEqual((events4[2]["step"], events4[2]["reason"]), (10, "key 0"))
            owner = arrays4["control/arm_owner"]
            driven = np.flatnonzero(owner == ARM_OWNERS.index("planner"))
            self.assertEqual(len(driven), attrs4["planner"]["steps"])
            self.assertEqual(len(driven), records[3]["planner"]["steps"])
            self.assertTrue(np.all(np.diff(driven) == 1))
            self.assertEqual(int(driven[0]), events4[1]["step"])
            self.assertLessEqual(int(driven[-1]), events4[2]["step"] - 1)
            self.assertFalse(arrays4["human/arm_delta"][driven].any())  # the SpaceMouse was idle meanwhile
            self.assertTrue(np.abs(arrays4["actions"][driven, :6]).max() > 0.0)  # and the arm did move
            self.assertEqual(attrs4["planner"]["backend"], "straight")
            self.assertEqual(attrs4["arm_owner_codes"], list(ARM_OWNERS))

            arrays, attrs, xml = read_demo_file(result.demos[0])
            steps = arrays["actions"].shape[0]
            self.assertEqual(arrays["actions"].shape, (steps, 26))
            self.assertEqual(arrays["states"].shape, (steps, 1 + attrs["state_layout"]["qpos"] + attrs["state_layout"]["qvel"]))
            self.assertTrue(arrays["success"][-10:].all())
            self.assertFalse(arrays["success"][:15].any())
            self.assertEqual(attrs["hand_mount"], "down")
            self.assertEqual(attrs["first_input_step"], 0)
            self.assertTrue(attrs["settle"]["settled"])  # the dropped objects had landed before the first move
            self.assertGreaterEqual(attrs["settle"]["steps"], SETTLE_CALM_STEPS)
            self.assertEqual(attrs["success_rule"], [["in", "alphabet_soup_1", "basket_1_contain_region"]])
            self.assertAlmostEqual(attrs["duration_s"], steps / 20.0)
            # The CSV time is the timer the operator watched (wall clock since the first move).
            self.assertAlmostEqual(float(rows[2]["time_s"]), attrs["wall_duration_s"], delta=0.006)
            self.assertAlmostEqual(records[2]["duration_s"], attrs["wall_duration_s"], places=3)
            self.assertEqual(attrs["format"], "libero_shadow_teleop_demo_v2")
            self.assertEqual(len(attrs["operator_camera"]["pos"]), 3)
            self.assertIn("mujoco", xml)
            self.assertTrue(np.all(np.isnan(arrays["human/mano_landmarks"])))  # hand control off: no landmarks
            self.assertTrue(np.any(arrays["human/arm_delta"][:3] != 0.0))  # the scripted push is recorded
            hand_joints = len(attrs["hand_qpos_index"])

            share = Path(folder) / "share"
            built = build_suite(demo_root, "libero_object", demos_per_task=2, share_dir=share,
                                share_url="https://example.invalid/folder", log=lambda *_: None)
            self.assertEqual(len(built["videos"]), 2)
            self.assertEqual(len(built["datasets"]), 1)
            dataset = built["datasets"][0]
            self.assertTrue(dataset.name.endswith("__down_demo.hdf5"))
            import h5py
            import imageio.v2 as imageio

            with h5py.File(dataset, "r") as handle:
                data = handle["data"]
                self.assertEqual(int(data.attrs["num_demos"]), 2)
                self.assertEqual(data.attrs["macros_image_convention"], "opengl")
                demo = data["demo_0"]
                count = int(demo.attrs["num_samples"])
                self.assertEqual(count, steps)  # first input at step 0: nothing trimmed
                self.assertEqual(demo["obs/agentview_rgb"].shape, (count, 256, 256, 3))
                self.assertEqual(demo["obs/eye_in_hand_rgb"].shape, (count, 256, 256, 3))
                self.assertEqual(demo["obs/joint_states"].shape, (count, 7))
                self.assertEqual(demo["obs/gripper_states"].shape, (count, hand_joints))
                self.assertEqual(demo["obs/ee_states"].shape, (count, 6))
                self.assertEqual(demo["robot_states"].shape, (count, hand_joints + 7))
                self.assertEqual(demo["rewards"][-1], 1)
                self.assertEqual(int(np.sum(demo["dones"][()])), 1)
                self.assertEqual(data.attrs["tag"], "libero_shadow_teleop_dataset_v2")
                self.assertFalse(demo["arm_owner"][()].any())
                np.testing.assert_array_equal(data["demo_1"]["arm_owner"][()], owner)  # nothing trimmed: step 0 moved
                state0 = demo["states"][0]
                image0 = demo["obs/agentview_rgb"][0]
            offline = sorted((demo_suite / "videos").glob("*.mp4"))
            self.assertEqual(len(offline), 2)
            self.assertTrue(offline[0].name.startswith("libero_object_t00_demo000_down_"))
            reader = imageio.get_reader(str(offline[0]))
            try:
                frames = reader.count_frames()
                first_frame = reader.get_data(0)
            finally:
                reader.close()
            self.assertEqual(frames, steps + VIDEO_TAIL_FRAMES)  # pre-roll is empty when the first move is step 0
            self.assertEqual(first_frame.shape, (640, 1024, 3))
            self.assertEqual(VIDEO_PREROLL_STEPS, 10)
            summary = (demo_suite / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("2/2", summary)
            self.assertIn("2/4 (50%)", summary)
            self.assertIn("% of steps", summary)  # demo 001's line says how much the planner drove
            with (demo_suite / "summary.csv").open(encoding="utf-8", newline="") as stream:
                summary_rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["planner_steps"]) for row in summary_rows], [0, len(driven)])
            self.assertEqual([int(row["planner_requests"]) for row in summary_rows], [1, 1])
            self.assertIn("https://example.invalid/folder", (demo_suite / "share_message.txt").read_text(encoding="utf-8"))
            self.assertTrue((share / "libero_object" / "SUMMARY.md").is_file())
            self.assertEqual(len(list((share / "libero_object" / "videos").glob("*.mp4"))), 2)
            again = build_suite(demo_root, "libero_object", demos_per_task=2, log=lambda *_: None)
            self.assertEqual((again["videos"], again["datasets"]), ([], []))  # nothing changed: nothing remade

            # The dataset image is what robosuite itself renders at that state (LIBERO's upside-down rows).
            session = CollectionSession(shadow_spec("down"), config, suite="libero_object", task_id=0, seed=7)
            try:
                session.open()
                session.reset_to(records[2]["placement_seed"])
                session.env.sim.set_state_from_flattened(state0)
                session.env.sim.forward()
                live = np.asarray(session.env.sim.render(camera_name="agentview", width=256, height=256))
            finally:
                session.close()
            difference = float(np.abs(live.astype(np.int32) - image0.astype(np.int32)).mean())
            self.assertLess(difference, 1.0)

    def test_default_run_writes_only_the_table_and_videos_and_resumes(self):
        from wrist_teleop.config import WristConfig
        from wrist_teleop.libero_attempt_video import RESULT_STYLE
        from wrist_teleop.libero_collect_teleop import run_libero_collect

        config = WristConfig.load(ROOT / "configs" / "spacemouse_wrist.json")
        with tempfile.TemporaryDirectory() as folder:
            root, demo_root, log_root = Path(folder) / "collect", Path(folder) / "demos", Path(folder) / "logs"
            common = dict(suite="libero_object", tasks=[1], demos_per_task=1, hand_mount="down", seed=7, headless=True,
                          camera_display=False, hand_control="off", view_size=(960, 540), planner_backend="straight",
                          output_root=root, demo_root=demo_root, log_root=log_root)
            first = run_libero_collect(config, duration_s=3.0, device=_ScriptedDevice(config), **common)
            self.assertEqual((first.exit_code, first.reason, first.successes, first.demos), (0, "completed", 0, []))
            self.assertFalse(demo_root.exists())  # no raw demos without save_demos
            suite_root = root / "libero_object"
            # S never pressed: the attempt has its row, but no video (nothing was recorded).
            self.assertEqual(first.videos, [])
            self.assertEqual(sorted(p.name for p in suite_root.iterdir()), ["attempts.csv"])
            self.assertEqual(list(log_root.glob("*/*.mp4")), [])
            rows = _read_csv(suite_root / "attempts.csv")
            self.assertEqual([(r["attempt"], r["task_id"], r["layout"], r["result"], r["ended_by"], r["video"])
                              for r in rows], [("1", "1", "0", "quit", "time limit", "")])
            self.assertGreater(float(rows[0]["time_s"]), 2.0)

            # The same command again, S pressed 1.5 s in: numbering and layouts continue from the table, and
            # this attempt is recorded from the press to the end.
            second = run_libero_collect(config, duration_s=4.0, device=_ScriptedDevice(config),
                                        step_hook=lambda env, info: "s" if info.step == 30 else None, **common)
            self.assertEqual(second.exit_code, 0)
            rows = _read_csv(suite_root / "attempts.csv")
            self.assertEqual([(r["attempt"], r["layout"]) for r in rows], [("1", "0"), ("2", "1")])
            self.assertTrue(rows[1]["video"].startswith("libero_object_t01_a002_QUIT_"))
            self.assertEqual(sorted(p.name for p in suite_root.iterdir()), ["attempts.csv", "videos"])
            self.assertEqual(second.videos, [suite_root / "videos" / rows[1]["video"]])
            quit_clip = _video_frames(suite_root / "videos" / rows[1]["video"])
            self.assertTrue(_close(quit_clip[-1][3, 3], RESULT_STYLE["quit"][1]))
            self.assertTrue(_close(quit_clip[0][3, 3], (24, 24, 24)))  # the header while it ran
            ended = [json.loads(line) for line in (second.run_dir / "telemetry.jsonl").read_text(
                encoding="utf-8").splitlines() if '"attempt_end"' in line]
            self.assertEqual(len(ended), 1)
            self.assertGreater(ended[0]["steps"], 40)
            # One frame per step from the press (handled before step 31) plus the 2 s result.
            self.assertAlmostEqual(len(quit_clip), ended[0]["steps"] - 31 + 40, delta=3)
            self.assertEqual([e["recording"] for e in map(json.loads, (second.run_dir / "telemetry.jsonl").read_text(
                encoding="utf-8").splitlines()) if e.get("event") == "video_record"], [True])
            # An attempt that never started (no SpaceMouse input, no number key) leaves no row and no video.
            third = run_libero_collect(config, duration_s=2.0, synthetic_input=True, **common)
            self.assertEqual((third.exit_code, third.videos), (0, []))
            self.assertEqual(len(_read_csv(suite_root / "attempts.csv")), 2)
            self.assertEqual([p.name for p in (suite_root / "videos").iterdir()], [rows[1]["video"]])
            self.assertFalse(demo_root.exists())
            # The one recording was written in its run's log folder and moved out when the attempt ended.
            self.assertEqual(list(log_root.glob("*/*.mp4")), [])
            self.assertEqual(sorted(p.parent for p in log_root.glob("*/telemetry.jsonl")),
                             sorted({first.run_dir, second.run_dir, third.run_dir}))

    @unittest.skipUnless(sys.platform == "win32", "locks the file the way WPS / Excel do (Windows share modes)")
    def test_a_success_while_attempts_csv_is_locked_keeps_collecting(self):
        """2026-11-10: attempts.csv open in WPS at a success used to end the run (see ConsoleStderrTests)."""
        import contextlib
        import ctypes
        import io

        from wrist_teleop.config import WristConfig
        from wrist_teleop.libero_collect_teleop import run_libero_collect

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.restype = ctypes.c_void_p
        kernel32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                                         ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        config = WristConfig.load(ROOT / "configs" / "spacemouse_wrist.json")
        with tempfile.TemporaryDirectory() as folder:
            root, log_root = Path(folder) / "collect", Path(folder) / "logs"
            table_path = root / "libero_object" / "attempts.csv"
            table_path.parent.mkdir(parents=True)
            with table_path.open("w", encoding="utf-8-sig", newline="") as stream:
                csv.writer(stream).writerow(CSV_FIELDS)
            # Read access, others may only read: what WPS holds while the table is open in it.
            lock = [kernel32.CreateFileW(str(table_path), 0x80000000, 0x1, None, 3, 0x80, None)]
            self.assertNotIn(lock[0], (None, ctypes.c_void_p(-1).value))

            def unlock():
                if lock[0]:
                    kernel32.CloseHandle(lock[0])
                    lock[0] = None

            def hook(env, info):
                if info.attempt == 2 and info.step == 3:
                    unlock()  # the table closed in WPS during attempt 2
                if info.step == 15:
                    _teleport_into_goal(env)
                return None

            warnings = io.StringIO()
            try:
                with contextlib.redirect_stderr(warnings):
                    result = run_libero_collect(
                        config, suite="libero_object", tasks=[0], demos_per_task=2, hand_mount="forward", seed=7,
                        headless=True, duration_s=60.0, camera_display=False, hand_control="off",
                        view_size=(960, 540), planner_backend="straight", output_root=root, log_root=log_root,
                        device=_ScriptedDevice(config), step_hook=hook)
            finally:
                unlock()
            self.assertEqual((result.exit_code, result.reason, result.successes), (0, "all tasks done", 2))
            self.assertEqual(warnings.getvalue().count("attempts.csv is open in another program"), 1)
            # The first success waited in memory and was written with the second once the file was free.
            rows = _read_csv(table_path)
            self.assertEqual([(r["attempt"], r["layout"], r["result"]) for r in rows],
                             [("1", "0", "success"), ("2", "1", "success")])
            self.assertEqual(list(table_path.parent.glob("attempts_unsaved_*.csv")), [])
            events = [json.loads(line) for line in (result.run_dir / "telemetry.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual([e["event"] for e in events if e.get("event") in ("exception", "exit")], ["exit"])


@unittest.skipUnless(sys.platform == "win32" and LEGACY_ROBOSUITE, "a Windows console and .venv-libero (mediapipe)")
class ConsoleStderrTests(unittest.TestCase):
    """2026-11-10, in a real console: after mediapipe (the camera thread) imported sounddevice, every
    ``print(..., file=sys.stderr)`` raised ``OSError: [WinError 6]``; the collection entry opens stderr again."""

    CHILD = "\n".join([
        "import json, sys",
        "report = {'raw': type(sys.stderr.buffer.raw).__name__}",
        "if sys.argv[2] == 'entry':",
        "    sys.path.insert(0, sys.argv[3])",
        "    import run_libero_collect as entry",
        "    report['reopened'] = entry.STDERR_REOPENED",
        "import mediapipe",
        "try:",
        "    print('stderr check', file=sys.stderr, flush=True)",
        "    report['print'] = 'ok'",
        "except OSError as exc:",
        "    report['print'] = '{}: {}'.format(type(exc).__name__, exc)",
        "with open(sys.argv[1], 'w', encoding='utf-8') as stream:",
        "    json.dump(report, stream)",
        "",
    ])

    def _in_a_console(self, how: str) -> tuple[int, dict]:
        import subprocess

        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0  # hidden, but a console of its own (stdout / stderr are console handles)
        with tempfile.TemporaryDirectory() as folder:
            child, report = Path(folder) / "child.py", Path(folder) / "report.json"
            child.write_text(self.CHILD, encoding="utf-8")
            done = subprocess.run([sys.executable, str(child), str(report), how, str(ROOT / "scripts")],
                                  cwd=str(ROOT), creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup,
                                  timeout=300, check=False)
            return done.returncode, json.loads(report.read_text(encoding="utf-8"))

    def test_stderr_still_prints_after_mediapipe_with_the_entry_loaded_first(self):
        code, plain = self._in_a_console("plain")
        self.assertEqual(plain["raw"], "_WindowsConsoleIO")  # a real console, as when run from PowerShell
        # The bug itself (and exit code 120: stderr cannot be flushed at exit). If this ever prints fine,
        # sounddevice / Python fixed it and the entry's reopen is no longer needed.
        self.assertIn("WinError 6", plain["print"])
        self.assertEqual(code, 120)
        code, entry = self._in_a_console("entry")
        self.assertEqual((code, entry["raw"], entry["reopened"], entry["print"]), (0, "_WindowsConsoleIO", True, "ok"))


if __name__ == "__main__":
    unittest.main()
