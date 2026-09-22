"""Entry-level Q and Ctrl+C cleanup routing without a physical device."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import inspect

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop import WristConfig  # noqa: E402
from wrist_teleop.runtime_controls import KeyboardCommandRouter  # noqa: E402


SPEC = importlib.util.spec_from_file_location(
    "spacemouse_integrated_entry_for_test",
    ROOT / "scripts" / "run_spacemouse_panda_wrist_teleop.py",
)
assert SPEC is not None and SPEC.loader is not None
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


class MemoryLog:
    instances: list["MemoryLog"] = []

    def __init__(self, *_args):
        self.events = []
        self.metadata_contents = None
        self.path = Path("memory-log")
        self.closed = False
        self.__class__.instances.append(self)

    def metadata(self, contents):
        self.metadata_contents = contents

    def event(self, contents):
        self.events.append(contents)

    def close(self):
        self.closed = True


class FakeDevice:
    def __init__(self, _config):
        self.poll_result = None

    def open(self):
        return {"product_string": "fake", "vendor_id": 0x256F, "product_id": 0xC62E}

    def poll(self):
        if isinstance(self.poll_result, BaseException):
            raise self.poll_result
        return self.poll_result


class FakeMapper:
    def __init__(self, _config):
        pass

    def reset(self):
        pass

    def describe_control_frame(self):
        return {"control_frame": "test"}


class FakeHandInput:
    smoothing_alpha = 0.45
    stale_timeout_s = 0.5
    calibration_state = "not_required_rule_based_fingers"


class FakeDexRetargeter:
    def __init__(self):
        self.closed = False

    @staticmethod
    def metadata():
        return {
            "package": "dex-retargeting",
            "version": "0.4.6",
            "runtime": "sidecar",
            "config_path": "test.yml",
            "config_sha256": "test-sha",
            "human_indices": [[0, 0, 0, 0], [4, 8, 12, 16]],
            "retargeting_joint_names": [f"joint_{index}.0" for index in range(16)],
        }

    def close(self):
        self.closed = True


class FakeCamera:
    def __init__(self, *_args, **_kwargs):
        self.error = None
        self.running = False

    def start(self):
        self.running = True

    def close(self):
        self.running = False


class FakeComposer:
    arm_slice = slice(0, 6)
    hand_slice = slice(6, 22)

    def describe(self):
        return {"action_dim": 22, "arm_slice": [0, 6], "hand_slice": [6, 22]}


class FakeGestures:
    gesture_targets = {"safe_open/hold": np.zeros(16)}


class IntegratedEntryShutdownTests(unittest.TestCase):
    def setUp(self):
        MemoryLog.instances.clear()
        self.config = WristConfig()
        self.close_calls = []

    def _reset_result(self, *_args, **_kwargs):
        return (
            FakeComposer(),
            FakeGestures(),
            SimpleNamespace(source="safe_open/hold", active_override=None),
            np.zeros(22),
        )

    def _close_result(self, **kwargs):
        self.close_calls.append(kwargs)
        return np.zeros(22)

    def _patches(self, device):
        return (
            patch.object(ENTRY, "RunLog", MemoryLog),
            patch.object(ENTRY, "SixDofMapper", FakeMapper),
            patch.object(ENTRY, "SpaceMouseHid", lambda _config: device),
            patch.object(ENTRY, "HandInputAdapter", FakeHandInput),
            patch.object(ENTRY, "CameraHandWorker", FakeCamera),
            patch.object(ENTRY, "make_allegro_panda_env", lambda *_args, **_kwargs: object()),
            patch.object(ENTRY, "reset_integrated_environment", self._reset_result),
            patch.object(ENTRY, "close_integrated_resources", self._close_result),
        )

    def test_ctrl_c_runs_the_same_final_zero_and_resource_cleanup(self):
        device = FakeDevice(self.config)
        device.poll_result = KeyboardInterrupt()
        with self._patches(device)[0], self._patches(device)[1], self._patches(device)[2], self._patches(device)[3], \
             self._patches(device)[4], self._patches(device)[5], self._patches(device)[6], self._patches(device)[7]:
            result = ENTRY.run_integrated(self.config, True, 1.0, 0, False, hand_control="legacy-curl")
        self.assertEqual(result, 0)
        self.assertEqual(len(self.close_calls), 1)
        self.assertEqual(MemoryLog.instances[-1].events[-1]["reason"], "Ctrl+C")
        np.testing.assert_allclose(MemoryLog.instances[-1].events[-1]["last_zero_wrist_action"][:6], np.zeros(6))

    def test_integrated_api_defaults_to_dex_not_legacy_curl(self):
        self.assertEqual(inspect.signature(ENTRY.run_integrated).parameters["hand_control"].default, "dex")

    def test_dex_sidecar_is_closed_after_unified_resource_cleanup(self):
        device = FakeDevice(self.config)
        device.poll_result = KeyboardInterrupt()
        retargeter = FakeDexRetargeter()
        patches = self._patches(device) + (
            patch.object(ENTRY, "_make_hand_input", return_value=(FakeHandInput(), retargeter)),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            result = ENTRY.run_integrated(self.config, True, 1.0, 0, False, hand_control="dex")
        self.assertEqual(result, 0)
        self.assertEqual(len(self.close_calls), 1)
        self.assertTrue(retargeter.closed)

    def test_q_routes_to_unified_cleanup_and_exits_zero(self):
        device = FakeDevice(self.config)
        router = KeyboardCommandRouter()
        router.feed("q", source="test", timestamp_s=1.0)
        patches = self._patches(device) + (patch.object(ENTRY, "KeyboardCommandRouter", lambda: router),)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            result = ENTRY.run_integrated(self.config, True, 1.0, 0, False, hand_control="legacy-curl")
        self.assertEqual(result, 0)
        self.assertEqual(len(self.close_calls), 1)
        self.assertEqual(MemoryLog.instances[-1].events[-1]["reason"], "test:Q")


if __name__ == "__main__":
    unittest.main()
