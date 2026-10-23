"""Shadow DexPilot mapping, camera adapter, worker generality, and rigid TCP."""

from __future__ import annotations

import importlib.util
from io import StringIO
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.shadow_dex import (  # noqa: E402
    SHADOW_DEX_CONFIG,
    SHADOW_DEX_JOINT_NAMES,
    SHADOW_DEX_URDF_ROOT,
    ShadowDexHandInput,
    shadow_targets_from_dex,
)
from wrist_teleop.shadow_gripper import (  # noqa: E402
    SHADOW_ACTUATOR_NAMES,
    ShadowHandRight,
    ShadowHandRightRigidTcp,
    _robosuite_tree,
)

SPEC = importlib.util.spec_from_file_location("dex_worker_shadow_test", ROOT / "scripts" / "dex_retargeting_worker.py")
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)

# Source order reported by the real 0.4.6 solver for the dex-urdf Shadow.
DEX_ORDER = (
    "WRJ2", "WRJ1", "FFJ4", "FFJ3", "FFJ2", "FFJ1", "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
    "MFJ4", "MFJ3", "MFJ2", "MFJ1", "RFJ4", "RFJ3", "RFJ2", "RFJ1", "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
)
LOWER = np.full(20, -10.0)
UPPER = np.full(20, 10.0)


def _landmarks() -> np.ndarray:
    points = [(0.0, 0.0, 0.0)]
    points += [(0.025, -0.02, -0.01), (0.04, -0.045, -0.03), (0.035, -0.07, -0.045), (0.03, -0.095, -0.055)]
    for x in (0.03, 0.01, -0.01, -0.03):
        points += [(x, -0.09, 0.0), (x, -0.13, -0.01), (x, -0.155, -0.02), (x, -0.175, -0.03)]
    return np.asarray(points)


class _FakeShadowRetargeter:
    joint_names = DEX_ORDER

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []
        self.reset_calls = 0

    def retarget(self, mano: np.ndarray) -> np.ndarray:
        self.inputs.append(np.asarray(mano).copy())
        return np.linspace(0.1, 2.4, len(DEX_ORDER))

    def reset(self) -> None:
        self.reset_calls += 1


class ShadowDexMappingTests(unittest.TestCase):
    def test_assets_and_names_match_the_real_solver_contract(self):
        self.assertTrue(SHADOW_DEX_CONFIG.is_file())
        self.assertTrue((SHADOW_DEX_URDF_ROOT / "shadow_hand" / "shadow_hand_right.urdf").is_file())
        self.assertEqual(set(DEX_ORDER), SHADOW_DEX_JOINT_NAMES)
        self.assertIn("target_joint_names", SHADOW_DEX_CONFIG.read_text(encoding="utf-8"))

    def test_named_mapping_locks_wrist_and_sums_distal_tendons(self):
        values = {name: 0.01 * (index + 1) for index, name in enumerate(DEX_ORDER)}
        targets = shadow_targets_from_dex(DEX_ORDER, [values[n] for n in DEX_ORDER], LOWER, UPPER)
        by_name = dict(zip(SHADOW_ACTUATOR_NAMES, targets))
        self.assertEqual(by_name["rh_A_WRJ2"], 0.0)
        self.assertEqual(by_name["rh_A_WRJ1"], 0.0)
        for finger in ("FF", "MF", "RF", "LF"):
            self.assertAlmostEqual(by_name[f"rh_A_{finger}J0"], values[f"{finger}J2"] + values[f"{finger}J1"])
            self.assertAlmostEqual(by_name[f"rh_A_{finger}J3"], values[f"{finger}J3"])
        for joint in ("THJ5", "THJ4", "THJ3", "THJ2", "THJ1", "LFJ5"):
            self.assertAlmostEqual(by_name[f"rh_A_{joint}"], values[joint])

    def test_mapping_clips_to_actuator_limits(self):
        lower, upper = np.zeros(20), np.full(20, 0.5)
        targets = shadow_targets_from_dex(DEX_ORDER, np.full(24, 3.0), lower, upper)
        self.assertTrue(np.all(targets <= upper) and np.all(targets >= lower))


class ShadowDexHandInputTests(unittest.TestCase):
    def test_unmirror_restores_unflipped_geometry_and_tracks(self):
        flipped = _landmarks() * np.array((-1.0, 1.0, 1.0))
        outputs = {}
        for unmirror in (True, False):
            fake = _FakeShadowRetargeter()
            adapter = ShadowDexHandInput(fake, LOWER, UPPER, unmirror_world_x=unmirror)
            command = adapter.update_world_landmarks(flipped, "Right", timestamp_s=10.0)
            self.assertTrue(command.valid)
            self.assertEqual(command.joint_targets.shape, (20,))
            outputs[unmirror] = fake.inputs[-1]
        # Un-mirroring is a reflection: the MANO frames differ, not just by noise.
        self.assertFalse(np.allclose(outputs[True], outputs[False]))
        reference = ShadowDexHandInput(_FakeShadowRetargeter(), LOWER, UPPER, unmirror_world_x=False)
        reference.update_world_landmarks(_landmarks(), "Right", timestamp_s=1.0)
        np.testing.assert_allclose(outputs[True], reference.retargeter.inputs[-1])

    def test_left_hand_rejected_hold_then_stale_and_reset(self):
        fake = _FakeShadowRetargeter()
        adapter = ShadowDexHandInput(fake, LOWER, UPPER, stale_timeout_s=0.5)
        self.assertTrue(adapter.update_world_landmarks(_landmarks(), "Right", timestamp_s=1.0).valid)
        held = adapter.update_world_landmarks(_landmarks(), "Left", timestamp_s=1.2)
        self.assertTrue(held.valid)
        self.assertIn("rejected", held.error)
        self.assertFalse(adapter.latest(2.0).valid)
        reset = adapter.reset(3.0)
        self.assertFalse(reset.valid)
        self.assertEqual(fake.reset_calls, 1)
        np.testing.assert_allclose(reset.joint_targets, np.clip(np.zeros(20), LOWER, UPPER))

    def test_normalized_image_landmarks_are_rejected(self):
        adapter = ShadowDexHandInput(_FakeShadowRetargeter(), LOWER, UPPER)
        self.assertFalse(adapter.update_landmarks(_landmarks(), timestamp_s=1.0).valid)


class _FakeDexPilot:
    def __init__(self) -> None:
        self.optimizer = type("Optimizer", (), {"idx_pin2fixed": np.array([0, 1])})()
        self.joint_names = DEX_ORDER
        self.fixed: list[np.ndarray] = []

    def retarget(self, reference: np.ndarray, fixed_qpos: np.ndarray = np.array([])) -> np.ndarray:
        self.fixed.append(np.asarray(fixed_qpos).copy())
        return np.zeros(len(DEX_ORDER))

    def reset(self) -> None:
        pass


class WorkerGeneralityTests(unittest.TestCase):
    def test_worker_serves_2x15_dexpilot_and_fixes_wrist_at_zero(self):
        indices = np.zeros((2, 15), dtype=np.intp)
        indices[1] = (4, 8, 12, 16, 20, 4, 4, 4, 4, 8, 8, 8, 12, 12, 16)
        fake = _FakeDexPilot()
        server = worker.DexRetargetingWorker()
        server._state = worker.WorkerState(
            retargeting=fake, wheel_path=ROOT, config_path=SHADOW_DEX_CONFIG, urdf_root=SHADOW_DEX_URDF_ROOT,
            wheel_sha256="w", config_sha256="c", human_indices=indices, joint_names=DEX_ORDER,
        )
        output = StringIO()
        request = worker.json.dumps({"id": 1, "op": "retarget", "mano_landmarks": _landmarks().tolist()})
        server.serve(StringIO(request + "\n"), output)
        reply = worker.json.loads(output.getvalue())
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(len(reply["result"]["ref_value"]), 15)
        np.testing.assert_allclose(fake.fixed[-1], np.zeros(2))


class RigidTcpModelTests(unittest.TestCase):
    def test_rigid_tcp_sits_on_joint_free_mount_with_same_neutral_pose(self):
        palm_tree, rigid_tree = _robosuite_tree(False), _robosuite_tree(True)
        self.assertIsNotNone(palm_tree.getroot().find(".//body[@name='rh_palm']/body[@name='eef']"))
        roots = rigid_tree.getroot().findall("./worldbody/body")
        self.assertEqual([body.get("name") for body in roots], ["rh_mount"])
        mount = roots[0]
        self.assertIsNone(mount.find("joint"))
        self.assertIsNotNone(mount.find("./body[@name='eef']/site[@name='grip_site']"))
        self.assertIsNotNone(mount.find("./body[@name='rh_wrist']"))
        self.assertTrue(issubclass(ShadowHandRightRigidTcp, ShadowHandRight))
        self.assertFalse(ShadowHandRight.rigid_tcp)

    def test_rigid_tcp_matches_palm_tcp_at_zero_wrist_and_ignores_wrist_motion(self):
        import mujoco

        positions = {}
        for rigid in (False, True):
            xml = _robosuite_tree(rigid)
            path = ROOT / "tmp" / f"shadow_tcp_test_{int(rigid)}.xml"
            path.parent.mkdir(exist_ok=True)
            xml.write(path, encoding="unicode")
            try:
                model = mujoco.MjModel.from_xml_path(str(path))
            finally:
                path.unlink()
            data = mujoco.MjData(model)
            site = model.site("grip_site").id
            mujoco.mj_kinematics(model, data)
            neutral = data.site_xpos[site].copy()
            data.qpos[model.joint("rh_WRJ1").qposadr[0]] = 0.4
            mujoco.mj_kinematics(model, data)
            positions[rigid] = (neutral, data.site_xpos[site].copy())
        np.testing.assert_allclose(positions[True][0], positions[False][0], atol=1e-9)
        self.assertGreater(np.linalg.norm(positions[False][1] - positions[False][0]), 0.01)
        np.testing.assert_allclose(positions[True][1], positions[True][0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
