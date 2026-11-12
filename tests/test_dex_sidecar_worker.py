"""Protocol-only tests for the standalone dex subprocess worker.

These tests deliberately never import dex-retargeting or robosuite.  The
actual external interpreter performs that native integration; here we protect
the JSON contract, landmark validation, and reset/close semantics with a
solver shaped like the official 0.4.6 API.
"""

from __future__ import annotations

from io import StringIO
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "scripts" / "dex_retargeting_worker.py"
SPEC = importlib.util.spec_from_file_location("dex_retargeting_worker_test_module", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


class _FakeFilter:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeRetargeting:
    def __init__(self) -> None:
        self.optimizer = type(
            "Optimizer",
            (), {"target_link_human_indices": np.array(((0, 0, 0, 0), (4, 8, 12, 16)))},
        )()
        self.joint_names = tuple(f"joint_{index}.0" for index in range(16))
        self.filter = _FakeFilter()
        self.references: list[np.ndarray] = []
        self.reset_calls = 0

    def retarget(self, reference: np.ndarray) -> np.ndarray:
        self.references.append(np.asarray(reference).copy())
        return np.arange(16, dtype=np.float64)

    def reset(self) -> None:
        self.reset_calls += 1


def _state(fake: _FakeRetargeting) -> object:
    return worker.WorkerState(
        retargeting=fake,
        wheel_path=ROOT / "official-dex-0.4.6.whl",
        config_path=ROOT / "configs" / "dex_retargeting" / "allegro_hand_right.yml",
        urdf_root=ROOT / "third_party" / "panda-gym",
        wheel_sha256="wheel-sha",
        config_sha256="config-sha",
        human_indices=np.array(((0, 0, 0, 0), (4, 8, 12, 16))),
        joint_names=fake.joint_names,
    )


class DexSidecarWorkerTests(unittest.TestCase):
    def _serve(self, lines: list[str], server: object | None = None) -> list[dict]:
        server = worker.DexRetargetingWorker() if server is None else server
        output = StringIO()
        result = server.serve(StringIO("".join(line + "\n" for line in lines)), output)
        self.assertEqual(result, 0)
        return [worker.json.loads(line) for line in output.getvalue().splitlines()]

    def test_non_json_and_preinit_operations_return_structured_errors(self):
        replies = self._serve([
            "not-json",
            worker.json.dumps({"id": "before", "op": "retarget", "landmarks": []}),
        ])
        self.assertEqual(replies[0]["error"]["code"], "invalid_json")
        self.assertEqual(replies[1]["id"], "before")
        self.assertEqual(replies[1]["error"]["code"], "not_initialized")

    def test_retarget_uses_official_indices_and_validates_finite_21x3_input(self):
        fake = _FakeRetargeting()
        server = worker.DexRetargetingWorker()
        server._state = _state(fake)
        landmarks = (np.arange(63, dtype=np.float64).reshape(21, 3) / 100.0).tolist()
        replies = self._serve([
            worker.json.dumps({"id": 1, "op": "retarget", "landmarks": landmarks}),
            worker.json.dumps({"id": 2, "op": "retarget", "landmarks": landmarks[:20]}),
            worker.json.dumps({"id": 3, "op": "retarget", "landmarks": [[float("nan"), 0, 0]] * 21}),
        ], server)
        self.assertTrue(replies[0]["ok"])
        np.testing.assert_allclose(
            fake.references[0],
            np.asarray(landmarks)[[4, 8, 12, 16]] - np.asarray(landmarks)[[0, 0, 0, 0]],
        )
        self.assertEqual(replies[0]["result"]["joint_targets_rad"], list(range(16)))
        self.assertEqual(replies[1]["error"]["code"], "invalid_landmarks")
        self.assertEqual(replies[2]["error"]["code"], "invalid_landmarks")

    def test_reset_clears_sequence_and_filter_then_close_stops_the_loop(self):
        fake = _FakeRetargeting()
        server = worker.DexRetargetingWorker()
        server._state = _state(fake)
        replies = self._serve([
            worker.json.dumps({"id": "reset", "op": "reset"}),
            worker.json.dumps({"id": "close", "op": "close"}),
            worker.json.dumps({"id": "ignored", "op": "reset"}),
        ], server)
        self.assertEqual([reply["id"] for reply in replies], ["reset", "close"])
        self.assertEqual(fake.reset_calls, 1)
        self.assertEqual(fake.filter.reset_calls, 1)
        self.assertIsNone(server.state)

    def test_init_metadata_requires_the_official_wheel_version(self):
        with self.assertRaises(worker.DexWheelError):
            worker._read_wheel_metadata(ROOT / "does-not-exist.whl")

    def test_grasp_terms_settings_are_validated(self):
        settings = worker._grasp_term_settings(
            worker.json.loads((ROOT / "configs" / "dex_retargeting" / "shadow_grasp_terms.json").read_text()))
        self.assertEqual(len(settings["thumb_segment_weights"]), 3)
        self.assertGreater(settings["thumb_lift_gain"], 1.0)
        self.assertTrue(settings["finger_length_scaling"])
        self.assertEqual(settings["pinch_points"], "pads")
        self.assertEqual(worker._grasp_term_settings({})["pinch_points"], "tips")  # default stays official
        for bad in ({"thumb_segment_weights": [1, 1]}, {"thumb_lift_gain": 9.0}, {"thumb_lift_gain": "x"},
                    {"pinch_gap_m": 0.2}, {"finger_length_scaling": 1}, {"tip_contact_enter": 0.8},
                    {"pinch_points": "nails"}, {"pinch_points": None}, {"unknown": 1}, [1, 2]):
            with self.assertRaises(worker.InvalidRequest):
                worker._grasp_term_settings(bad)

    def test_thumb_lift_gain_scales_only_the_thumb_offset_from_the_palm_plane(self):
        hand = _flat_hand()
        hand[[2, 3, 4], 2] = (-0.01, -0.02, -0.03)  # thumb lifted off the palm (z = palm normal)
        lifted = worker._lift_thumb(hand, 1.5)
        np.testing.assert_allclose(lifted[[2, 3, 4], 2], (-0.015, -0.03, -0.045), atol=1e-12)
        np.testing.assert_allclose(lifted[:, :2], hand[:, :2], atol=1e-12)
        np.testing.assert_allclose(np.delete(lifted, [1, 2, 3, 4], axis=0), np.delete(hand, [1, 2, 3, 4], axis=0))
        flat = _flat_hand()  # thumb in the palm plane: no lift to amplify
        np.testing.assert_allclose(worker._lift_thumb(flat, 3.0), flat, atol=1e-12)

    def test_per_finger_scale_maps_a_straight_finger_to_a_straight_robot_finger(self):
        robot_lengths = np.array(((0.1006, 0.096), (0.0996, 0.096), (0.0956, 0.096), (0.0927, 0.096)))
        for bend in (0.0, 0.6):
            hand = _flat_hand()
            if bend:  # curl the little finger: MCP / PIP / DIP each flexed by ``bend`` rad
                mcp, total = hand[17].copy(), 0.0
                lengths = np.linalg.norm(np.diff(hand[17:21], axis=0), axis=1)
                for k, length in enumerate(lengths):
                    total += bend
                    hand[18 + k] = hand[17 + k] + length * np.array((0.0, np.cos(total), -np.sin(total)))
            tips, scales = worker._per_finger_tips(hand, robot_lengths, 1.2)
            np.testing.assert_allclose(tips[0], 1.2 * hand[4])
            for k, mcp_index in enumerate((5, 9, 13, 17)):
                chain = hand[mcp_index:mcp_index + 4]
                straightness = np.linalg.norm(chain[-1] - chain[0]) / np.linalg.norm(np.diff(chain, axis=0), axis=1).sum()
                knuckle = scales[k, 0] * hand[mcp_index]
                np.testing.assert_allclose(np.linalg.norm(knuckle), robot_lengths[k, 0], rtol=1e-9)
                # MCP-to-tip chord = robot finger length x the operator finger's own straightness.
                np.testing.assert_allclose(np.linalg.norm(tips[k + 1] - knuckle), robot_lengths[k, 1] * straightness, rtol=1e-9)
            if not bend:
                np.testing.assert_allclose(np.linalg.norm(tips[1:] - scales[:, :1] * hand[[5, 9, 13, 17]], axis=1),
                                           robot_lengths[:, 1], rtol=1e-9)
            # The short little finger gets a larger phalanx scale than the middle finger.
            self.assertGreater(scales[3, 1], scales[1, 1])
        indices = np.array(((0, 0, 0, 0, 0), (4, 8, 12, 16, 20)))
        np.testing.assert_allclose(worker._scaled_targets(hand, indices, tips), tips - hand[0])
        with self.assertRaises(worker.InvalidLandmarks):
            worker._per_finger_tips(np.zeros((21, 3)), robot_lengths, 1.2)

    def test_pinch_mapping_is_continuous_with_no_snap(self):
        d = np.linspace(0.0, 0.1, 2001)
        target = worker._pinch_distance(d, 0.01, 1.2)
        self.assertTrue(np.all(target[d <= 0.01] == 0.0))
        np.testing.assert_allclose(target[d >= 0.01], 1.2 * (d[d >= 0.01] - 0.01))
        self.assertLessEqual(np.max(np.diff(target)), 1.2 * (d[1] - d[0]) + 1e-12)
        weight = worker._pinch_closeness(d, 0.03, 0.05)
        self.assertTrue(np.all(weight[d <= 0.03] == 1.0) and np.all(weight[d >= 0.05] == 0.0))
        self.assertTrue(np.all(np.diff(weight) <= 0.0))
        self.assertLess(np.max(np.abs(np.diff(weight))), 0.01)

    def test_grasp_terms_input_stacks_targets_measured_vectors_and_landmarks(self):
        fake = _FakeRetargeting()
        server = worker.DexRetargetingWorker()
        server._state = worker.replace(_state(fake), grasp_terms={"thumb_segment_weights": [1, 1, 1]})
        landmarks = (np.arange(63, dtype=np.float64).reshape(21, 3) / 100.0)
        replies = self._serve([worker.json.dumps({"id": 1, "op": "retarget", "landmarks": landmarks.tolist()})], server)
        self.assertTrue(replies[0]["ok"])
        measured = landmarks[[4, 8, 12, 16]] - landmarks[[0, 0, 0, 0]]
        np.testing.assert_allclose(fake.references[0][:4], measured)  # gain 1, scale 1 (fake has no scaling)
        np.testing.assert_allclose(fake.references[0][4:8], measured)
        np.testing.assert_allclose(fake.references[0][8:], landmarks)
        self.assertEqual(replies[0]["result"]["mapping"]["thumb_lift_gain"], 1.0)


def _flat_hand() -> np.ndarray:
    """Right hand in the z = 0 plane: wrist at the origin, straight fingers along +y."""

    points = [(0.0, 0.0, 0.0), (0.025, 0.025, 0.0), (0.045, 0.05, 0.0), (0.06, 0.075, 0.0), (0.07, 0.1, 0.0)]
    for x, bones in ((0.022, (0.040, 0.024, 0.020)), (0.002, (0.044, 0.028, 0.021)),
                     (-0.017, (0.041, 0.026, 0.020)), (-0.034, (0.032, 0.019, 0.018))):
        y = 0.09
        points.append((x, y, 0.0))
        for bone in bones:
            y += bone
            points.append((x, y, 0.0))
    return np.asarray(points)


if __name__ == "__main__":
    unittest.main()
