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


if __name__ == "__main__":
    unittest.main()
