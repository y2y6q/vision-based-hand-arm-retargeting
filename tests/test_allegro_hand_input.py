"""Shadow-style Allegro curl-pinch mapping and the palm-down teleop mount."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop.allegro_hand_input import (  # noqa: E402
    AllegroCurlPinchHandInput,
    curls_to_allegro_pinch_targets,
)
from wrist_teleop.allegro_model import (  # noqa: E402
    ALLEGRO_PALM_DOWN_GRASP_CENTER_LOCAL,
    ALLEGRO_PALM_DOWN_NORMAL_LOCAL,
    make_allegro_panda_env,
)
from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.hand_input import (  # noqa: E402
    DEFAULT_ALLEGRO_LOWER_LIMITS as LOWER,
    DEFAULT_ALLEGRO_UPPER_LIMITS as UPPER,
    curls_to_allegro_targets,
)
from wrist_teleop.integrated import PandaAllegroActionComposer  # noqa: E402


def _hand(pinch: str | None = None) -> np.ndarray:
    """Synthetic 21-point hand: open fingers, thumb tip optionally on a tip."""
    points = [(0.0, 0.0, 0.0), (0.03, -0.03, 0.0), (0.05, -0.06, 0.0), (0.065, -0.085, 0.0), (0.08, -0.105, 0.0)]
    for x in (0.03, 0.01, -0.01, -0.03):
        points += [(x, -0.09, 0.0), (x, -0.13, 0.0), (x, -0.155, 0.0), (x, -0.175, 0.0)]
    points = np.asarray(points)
    if pinch is not None:
        tip = {"index": 8, "middle": 12}[pinch]
        points[4] = points[tip] + (0.004, 0.0, 0.0)
    return points


class CurlPinchMappingTests(unittest.TestCase):
    def test_no_pinch_matches_legacy_curl_mapping(self):
        curls = np.array([0.2, 0.4, 0.6, 0.3])
        targets, profile = curls_to_allegro_pinch_targets(curls, LOWER, UPPER, thumb_pinch_strengths=np.zeros(4))
        self.assertIsNone(profile)
        np.testing.assert_allclose(targets, curls_to_allegro_targets(curls, LOWER, UPPER))

    def test_pinch_closes_partner_and_selects_profile(self):
        curls = np.zeros(4)
        index, profile = curls_to_allegro_pinch_targets(curls, LOWER, UPPER, thumb_pinch_strengths=[1, 0, 0, 0])
        self.assertEqual(profile, "index")
        self.assertGreater(index[1], 0.5)  # index MCP flexes to the pinch floor
        middle, profile = curls_to_allegro_pinch_targets(curls, LOWER, UPPER, thumb_pinch_strengths=[0, 1, 0, 0])
        self.assertEqual(profile, "middle")
        self.assertGreater(middle[5], 0.5)
        ring, profile = curls_to_allegro_pinch_targets(curls, LOWER, UPPER, thumb_pinch_strengths=[0, 0, 0, 1])
        self.assertEqual(profile, "ring")
        self.assertGreater(ring[9], 0.5)
        self.assertFalse(np.allclose(index[12:16], middle[12:16]))
        for targets in (index, middle, ring):
            self.assertTrue(np.all(targets >= LOWER) and np.all(targets <= UPPER))

    def test_adapter_reports_pinch_and_resets(self):
        adapter = AllegroCurlPinchHandInput(smoothing_alpha=1.0)
        command = adapter.update_landmarks(_hand("index"), timestamp_s=1.0)
        self.assertTrue(command.valid)
        self.assertEqual(command.details["thumb_pinch_profile"], "index")
        self.assertIn("index", command.details["thumb_pinch_partners"])
        open_command = adapter.update_landmarks(_hand(), timestamp_s=1.1)
        self.assertIsNone(open_command.details["thumb_pinch_profile"])
        reset = adapter.reset(2.0)
        self.assertFalse(reset.valid)
        self.assertIsNone(reset.details["thumb_pinch_profile"])


class ThumbOppositionTests(unittest.TestCase):
    def test_opposition_cue_tracks_thumb_crossing_the_palm(self):
        from wrist_teleop.allegro_hand_input import estimate_thumb_opposition

        open_hand = _hand()
        across = _hand()
        across[4] = across[13] + (0.0, 0.01, 0.0)  # thumb tip at the ring MCP
        self.assertLess(estimate_thumb_opposition(open_hand), 0.2)
        self.assertGreater(estimate_thumb_opposition(across), 0.9)

    def test_opposition_moves_thumb_joints_12_13_and_keeps_limits(self):
        curls = np.array([0.0, 0.0, 0.0, 0.3])
        low, _ = curls_to_allegro_pinch_targets(curls, LOWER, UPPER, thumb_opposition=0.0)
        high, _ = curls_to_allegro_pinch_targets(curls, LOWER, UPPER, thumb_opposition=1.0)
        self.assertTrue(np.all(high[12:14] > low[12:14] + 0.3))
        np.testing.assert_allclose(high[:12], low[:12])
        self.assertTrue(np.all(high >= LOWER) and np.all(high <= UPPER))


class PalmDownAllegroPhysicsTests(unittest.TestCase):
    config = WristConfig(smoothing_alpha=None)

    def _env(self, palm_down: bool):
        env = make_allegro_panda_env(self.config, has_renderer=False, palm_down=palm_down)
        self.addCleanup(env.close)
        env.reset()
        return env

    def test_palm_down_mount_faces_table_and_tcp_sits_at_grasp_centre(self):
        default, palm_down = self._env(False), self._env(True)
        self.assertEqual(default.robots[0].gripper["right"].describe()["gripper_type"], "AllegroRightHand")
        m, d = default.sim.model, default.sim.data  # Gate 2 model unchanged: TCP on the flange
        np.testing.assert_allclose(
            d.site_xpos[m.site_name2id("gripper0_right_grip_site")],
            d.body_xpos[m.body_name2id("robot0_right_hand")], atol=1e-9,
        )
        m, d = palm_down.sim.model, palm_down.sim.data
        root = m.body_name2id("gripper0_right_allegro_right_hand")
        grip_local = d.body_xmat[root].reshape(3, 3).T @ (
            d.site_xpos[m.site_name2id("gripper0_right_grip_site")] - d.body_xpos[root]
        )
        np.testing.assert_allclose(grip_local, ALLEGRO_PALM_DOWN_GRASP_CENTER_LOCAL, atol=1e-9)
        site = m.site_name2id("gripper0_right_palm_laser_site")
        normal = d.site_xmat[site].reshape(3, 3) @ np.asarray(ALLEGRO_PALM_DOWN_NORMAL_LOCAL)
        fingers = d.body_xpos[m.body_name2id("gripper0_right_link_7.0_tip")] - d.site_xpos[site]
        self.assertLess(normal[2], -0.9)  # palm faces the table
        self.assertLess(abs(fingers[2]) / np.linalg.norm(fingers), 0.3)  # fingers roughly horizontal

    def test_index_pinch_brings_thumb_to_index_in_mujoco(self):
        gaps = {}
        for label, strengths in (("curl", None), ("pinch", [1.0, 0.0, 0.0, 0.0])):
            env = self._env(True)
            composer = PandaAllegroActionComposer(env)
            composer.initialize_goal()
            curls = np.array([0.0, 0.0, 0.0, 1.0])  # full legacy thumb curl
            targets, _ = curls_to_allegro_pinch_targets(curls, LOWER, UPPER, thumb_pinch_strengths=strengths)
            if strengths is None:
                targets[0:4] = curls_to_allegro_targets([0.8, 0, 0, 0], LOWER, UPPER)[0:4]
            for _ in range(30):
                env.step(composer.compose(np.zeros(6), targets))
            m, d = env.sim.model, env.sim.data
            thumb = d.body_xpos[m.body_name2id("gripper0_right_link_15.0_tip")]
            index = d.body_xpos[m.body_name2id("gripper0_right_link_3.0_tip")]
            gaps[label] = float(np.linalg.norm(thumb - index))
        self.assertLess(gaps["pinch"], 0.022)
        self.assertGreater(gaps["curl"], gaps["pinch"] + 0.008)


if __name__ == "__main__":
    unittest.main()
