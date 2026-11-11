"""LIBERO operator views, palm-down Shadow mount and object mass scale.

Run with .venv-libero (robosuite 1.4):

    .\\.venv-libero\\Scripts\\python.exe -m pytest -q .\\tests\\test_libero_views_and_options.py

The layout test runs anywhere; the rest skip outside the LIBERO interpreter.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import robosuite

    LEGACY_ROBOSUITE = robosuite.__version__.startswith("1.4")
except ImportError:  # pragma: no cover
    LEGACY_ROBOSUITE = False


class ViewLayoutTests(unittest.TestCase):
    def test_panels_tile_the_window_without_overlap(self):
        from wrist_teleop.libero_views import GUTTER_PX, layout

        rects = layout(2544, 1334)
        x, y, w, h = rects["front"]
        self.assertEqual((x, y, h), (0, 0, 1334))
        right = [rects["wrist"], rects["side"]]
        self.assertTrue(all(r[0] == w + GUTTER_PX for r in right))
        self.assertEqual(right[0][0] + right[0][2], 2544)
        self.assertEqual(right[1][1] + right[1][3], 1334)
        self.assertEqual(right[0][1] + right[0][3] + GUTTER_PX, right[1][1])

    def test_view_size_parsing(self):
        from wrist_teleop.libero_views import parse_view_size

        self.assertEqual(parse_view_size("1920x1080"), (1920, 1080))
        with self.assertRaises(ValueError):
            parse_view_size("100x100")


@unittest.skipUnless(LEGACY_ROBOSUITE, "LIBERO tests require .venv-libero (robosuite 1.4)")
class LiberoOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from wrist_teleop.config import WristConfig
        from wrist_teleop import libero_spatial_teleop as L

        cls.L = L
        cls.config = WristConfig.load(ROOT / "configs" / "spacemouse_wrist.json")

    def _session(self, spec, **kwargs):
        session = self.L.OfficialLiberoSpatialSession(
            spec, self.config, task_id=0, task_order=0, init_state_id=0, seed=7,
            has_offscreen_renderer=True, suite="libero_object", **kwargs,
        )
        session.open()
        self.addCleanup(session.close)
        return session

    def test_object_mass_scale_scales_every_object_once_per_reset(self):
        session = self._session(self.L.PANDA_SHADOW_SPEC, object_mass_scale=10.0)
        first = {entry["object"]: entry for entry in session.object_masses}
        self.assertIn("alphabet_soup_1", first)
        for entry in first.values():
            self.assertAlmostEqual(entry["mass_kg"], 10.0 * entry["official_mass_kg"], places=9)
        session.reset()  # robosuite 1.4 reloads the model: the scale must not compound
        again = {entry["object"]: entry["mass_kg"] for entry in session.object_masses}
        for name, entry in first.items():
            self.assertAlmostEqual(again[name], entry["mass_kg"], places=9)
        sim = session.env.sim
        body = sim.model.body_name2id(session.env.objects_dict["alphabet_soup_1"].root_body)
        self.assertAlmostEqual(float(sim.model.body_mass[body]),
                               10.0 * first["alphabet_soup_1"]["official_mass_kg"], places=6)
        with self.assertRaises(ValueError):
            self.L.OfficialLiberoSpatialSession(
                self.L.PANDA_SHADOW_SPEC, self.config, task_id=0, task_order=0, init_state_id=0, seed=7,
                has_offscreen_renderer=False, object_mass_scale=0.0)

    def test_object_mass_parsing(self):
        parse = self.L.parse_object_mass_overrides
        self.assertEqual(parse(["AkitaBlackBowl=300", "butter_2=110.5"]),
                         {"AkitaBlackBowl": 0.3, "butter_2": 0.1105})
        self.assertEqual(parse([]), {})
        for bad in ("AkitaBlackBowl", "=300", "AkitaBlackBowl=heavy", "AkitaBlackBowl=0", "AkitaBlackBowl=50000",
                    "AkitaBlackBowl=nan"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse([bad])

    def test_object_mass_sets_named_objects_and_scales_the_rest(self):
        # Class name (any case / underscores) and instance name; the rest x scale.
        session = self._session(self.L.PANDA_SHADOW_SPEC, object_mass_scale=2.0,
                                object_masses_kg={"AlphabetSoup": 0.4, "basket_1": 0.5})
        first = {entry["object"]: entry for entry in session.object_masses}
        self.assertAlmostEqual(first["alphabet_soup_1"]["mass_kg"], 0.4, places=9)
        self.assertAlmostEqual(first["basket_1"]["mass_kg"], 0.5, places=9)
        self.assertEqual(first["alphabet_soup_1"]["set_by"], "object_mass")
        others = [entry for name, entry in first.items() if name not in ("alphabet_soup_1", "basket_1")]
        self.assertTrue(others)
        for entry in others:
            self.assertEqual(entry["set_by"], "scale")
            self.assertAlmostEqual(entry["mass_kg"], 2.0 * entry["official_mass_kg"], places=9)
        # The model holds the set mass, inertia scaled by the same factor, and a reset does not compound.
        sim = session.env.sim
        body = sim.model.body_name2id(session.env.objects_dict["alphabet_soup_1"].root_body)
        factor = 0.4 / first["alphabet_soup_1"]["official_mass_kg"]
        session.reset()
        self.assertAlmostEqual(float(session.env.sim.model.body_mass[body]), 0.4, places=6)
        self.assertIn("alphabet_soup_1*", self.L.describe_object_masses(session))
        official = self.L.OfficialLiberoSpatialSession(
            self.L.PANDA_SHADOW_SPEC, self.config, task_id=0, task_order=0, init_state_id=0, seed=7,
            has_offscreen_renderer=False, suite="libero_object")
        official.open()
        self.addCleanup(official.close)
        np.testing.assert_allclose(np.asarray(session.env.sim.model.body_inertia[body]),
                                   factor * np.asarray(official.env.sim.model.body_inertia[body]), rtol=1e-6)
        self.assertEqual(session.describe()["object_mass_overrides_g"], {"AlphabetSoup": 400.0, "basket_1": 500.0})

    def test_object_mass_name_not_in_the_task_is_an_error(self):
        session = self.L.OfficialLiberoSpatialSession(
            self.L.PANDA_SHADOW_SPEC, self.config, task_id=0, task_order=0, init_state_id=0, seed=7,
            has_offscreen_renderer=False, suite="libero_object", object_masses_kg={"WineBottle": 1.0})
        self.addCleanup(session.close)
        with self.assertRaisesRegex(ValueError, "WineBottle.*alphabet_soup_1"):
            session.open()
        with self.assertRaises(ValueError):
            self.L.OfficialLiberoSpatialSession(
                self.L.PANDA_SHADOW_SPEC, self.config, task_id=0, task_order=0, init_state_id=0, seed=7,
                has_offscreen_renderer=False, object_masses_kg={"Plate": 0.0})

    def test_down_mount_points_the_fingers_down_without_initial_contact(self):
        spec = self.L.shadow_spec("down")
        session = self._session(spec)
        composer = self.L.LiberoActionComposer(session.env, spec)
        composer.initialize_goal()
        self.assertEqual(composer.gripper.describe()["hand_mount"], "down")
        sim = session.env.sim
        palm = sim.model.body_name2id("gripper0_rh_palm")
        fingers = sim.data.body_xmat[palm].reshape(3, 3) @ np.array((0.0, 0.0, 1.0))
        self.assertLess(fingers[2], -0.95)  # straight down at the official start pose
        self.assertEqual(tuple(composer.gripper.laser_direction_local), (0.0, 0.0, 1.0))
        start = composer.ee_pose()[0]
        for _ in range(20):
            session.env.step(composer.compose(np.zeros(6), None))
        self.assertLess(float(np.linalg.norm(composer.ee_pose()[0] - start)), 0.005)

    def test_view_window_renders_follow_views_without_blocking_the_step(self):
        from wrist_teleop.visualization import LaserPointer

        spec = self.L.PANDA_SHADOW_SPEC
        session = self._session(spec)
        env = session.env
        composer = self.L.LiberoActionComposer(env, spec)
        composer.initialize_goal()
        tri = self.L.follow_robot_base(self.L._shadow_tri_view_config(self.config), env)
        laser = LaserPointer(env, self.L._shadow_laser_config(self.config, composer.gripper))
        window = self.L.operator_view_window(
            tri, env, composer, headless=True, view_size=(960, 540), key_handler=None, on_close=None,
            laser=laser, preview=None)
        self.addCleanup(window.close)
        records = []
        for _ in range(10):
            env.step(composer.compose(np.array((0.004, 0.0, 0.0, 0.0, 0.0, 0.0)), None))
            laser.update()
            records.append(window.update(env, "test"))
        deadline = time.monotonic() + 10.0
        while window.last_frame is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(window.last_frame)
        self.assertEqual(window.last_frame.shape, (540, 960, 3))
        self.assertTrue(all(record["rendered"] for record in records))
        self.assertLess(float(np.median([record["capture_ms"] for record in records])), 20.0)
        # Two snapshots per control step: one mid-step, one after it.
        time.sleep(0.3)
        self.assertGreaterEqual(window.display_stats()["frames"], 10)
        x, y, w, h = window.rects["wrist"]
        self.assertGreater(float(window.last_frame[y:y + h, x:x + w].std()), 5.0)  # the close-up is not blank


if __name__ == "__main__":
    unittest.main()
