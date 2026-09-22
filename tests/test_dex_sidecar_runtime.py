"""Local integration proof for the isolated official dex-retargeting runtime.

This test is intentionally skipped on hosts that lack the separately managed
native dex environment. It never substitutes a fake solver: when it runs, it
loads the checked-in official 0.4.6 wheel through the worker process, executes
``SeqRetargeting``, and inserts its output into a real Panda + Allegro action.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop import PandaAllegroActionComposer, WristConfig  # noqa: E402
from wrist_teleop.allegro_model import ALLEGRO_JOINT_NAMES, make_allegro_panda_env, source_joint_specs  # noqa: E402
from wrist_teleop.dex_retargeting import (  # noqa: E402
    OFFICIAL_CONFIG_SHA256,
    OFFICIAL_DEX_VERSION,
    DexRetargetingAdapter,
)
from wrist_teleop.hand_input import DexHandInputAdapter  # noqa: E402


def _sidecar_python() -> Path:
    return Path(sys.base_prefix).resolve().parent / "dexretarget" / "python.exe"


@unittest.skipUnless(_sidecar_python().is_file(), "verified local dex sidecar Python is unavailable")
class DexSidecarRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        specs = source_joint_specs()
        self.lower = np.asarray([spec.lower for spec in specs], dtype=np.float64)
        self.upper = np.asarray([spec.upper for spec in specs], dtype=np.float64)
        self.adapter = DexRetargetingAdapter(
            allegro_joint_names=ALLEGRO_JOINT_NAMES,
            lower_limits=self.lower,
            upper_limits=self.upper,
            runtime="sidecar",
            sidecar_python=_sidecar_python(),
        )

    def tearDown(self) -> None:
        self.adapter.close()

    @staticmethod
    def _world_landmarks() -> np.ndarray:
        return np.asarray(
            [[index * 0.004, index * 0.002, -index * 0.003] for index in range(21)],
            dtype=np.float64,
        )

    def test_official_solver_retargets_and_resets(self) -> None:
        metadata = self.adapter.metadata()
        self.assertEqual(metadata["runtime"], "sidecar")
        self.assertEqual(metadata["version"], OFFICIAL_DEX_VERSION)
        self.assertEqual(metadata["config_sha256"], OFFICIAL_CONFIG_SHA256)
        self.assertEqual(metadata["human_indices"], [[0, 0, 0, 0], [4, 8, 12, 16]])
        self.assertEqual(set(metadata["retargeting_joint_names"]), set(ALLEGRO_JOINT_NAMES))

        result = self.adapter.retarget(self._world_landmarks())
        self.assertEqual(result.joint_targets_rad.shape, (16,))
        self.assertTrue(np.all(np.isfinite(result.joint_targets_rad)))
        self.assertTrue(np.all(result.joint_targets_rad >= self.lower))
        self.assertTrue(np.all(result.joint_targets_rad <= self.upper))
        self.adapter.reset()
        repeated = self.adapter.retarget(self._world_landmarks())
        self.assertTrue(np.all(np.isfinite(repeated.joint_targets_rad)))

    def test_dex_camera_path_writes_only_the_real_hand_slice(self) -> None:
        camera_adapter = DexHandInputAdapter(
            self.adapter,
            lower_limits=self.lower,
            upper_limits=self.upper,
            joint_names=ALLEGRO_JOINT_NAMES,
        )
        command = camera_adapter.update_world_landmarks(
            self._world_landmarks(), "Right", timestamp_s=10.0
        )
        self.assertTrue(command.valid, command.error)

        env = make_allegro_panda_env(WristConfig(smoothing_alpha=None), has_renderer=False)
        try:
            env.reset()
            composer = PandaAllegroActionComposer(env)
            composer.initialize_goal()
            action = composer.compose(np.zeros(6, dtype=np.float64), command.joint_targets)
            np.testing.assert_array_equal(action[composer.arm_slice], np.zeros(6))
            self.assertEqual(action.shape, (22,))
            self.assertTrue(np.all(np.isfinite(action[composer.hand_slice])))
            env.step(action)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
