"""Focused nonphysical tests for the shared tri-view and TCP laser helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop.config import LaserConfig, TriViewConfig, WristConfig  # noqa: E402
from wrist_teleop.visualization import LaserPointer, TriViewCompositor  # noqa: E402
from wrist_teleop.allegro_model import make_allegro_panda_env  # noqa: E402
from wrist_teleop.integrated import PandaAllegroActionComposer  # noqa: E402


class _RenderedSim:
    """A same-state render fake that makes camera placement easy to assert."""

    def __init__(self) -> None:
        self.data = SimpleNamespace(time=12.25)
        self.requests: list[tuple[int, int, str]] = []
        self._values = {"front_cam": 40, "wrist_cam": 120, "side_cam": 220}

    def render(self, *, width: int, height: int, camera_name: str) -> np.ndarray:
        self.requests.append((width, height, camera_name))
        return np.full((height, width, 3), self._values[camera_name], dtype=np.uint8)


class _NativeSim:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)


class _GuiCv2:
    """Small HighGUI fake for testing R/Q routing without a desktop window."""

    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 0
    INTER_AREA = 0
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0

    def __init__(self, keys: list[int], visible: float = 1.0):
        self.keys = list(keys)
        self.visible = visible

    @staticmethod
    def resize(image, size, interpolation=0):
        return np.resize(image, (int(size[1]), int(size[0]), 3))

    @staticmethod
    def putText(image, *_args, **_kwargs):
        return image

    def namedWindow(self, *_args):
        return None

    def resizeWindow(self, *_args):
        return None

    def imshow(self, *_args):
        return None

    def waitKey(self, _delay):
        return self.keys.pop(0) if self.keys else -1

    def getWindowProperty(self, *_args):
        return self.visible

    def destroyWindow(self, *_args):
        return None


def _laser_sim() -> _NativeSim:
    """A robot-root/site/table model with a self hit before the table."""
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="table" type="box" pos="0 0 -0.05" size="1 1 0.05"/>
            <body name="robot" pos="0 0 1">
              <geom name="robot_shell" type="sphere" pos="0 0 -0.15" size="0.06"/>
              <site name="grip_site" type="sphere" size="0.002"/>
              <site name="grip_site_cylinder" type="cylinder" size="0.003 0.08" rgba="0 1 0 0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return _NativeSim(model, data)


class VisualConfigTests(unittest.TestCase):
    def test_visual_settings_are_nested_in_the_single_wrist_config(self):
        config = WristConfig.from_dict(
            {
                "tri_view": {
                    "front_camera_name": "front_cam",
                    "wrist_camera_name": "wrist_cam",
                    "side_camera_name": "side_cam",
                    "image_width": 320,
                    "image_height": 240,
                    "render_hz": 8.0,
                    "layout": "main_top",
                },
                "laser": {
                    "color_rgba": [0.2, 0.3, 0.4, 0.5],
                    "local_direction": [0.0, 0.0, 3.0],
                    "visible_views": ["front", "side"],
                },
            }
        )
        self.assertIsInstance(config.tri_view, TriViewConfig)
        self.assertIsInstance(config.laser, LaserConfig)
        self.assertEqual(config.tri_view.camera_names, ("front_cam", "wrist_cam", "side_cam"))
        self.assertEqual(config.laser.local_direction, (0.0, 0.0, 1.0))
        self.assertEqual(config.laser.visible_views, ("front", "side"))
        self.assertEqual(config.to_dict()["tri_view"]["layout"], "main_top")

    def test_invalid_visual_config_is_rejected(self):
        with self.assertRaises(ValueError):
            TriViewConfig(front_camera_name="same", wrist_camera_name="same")
        with self.assertRaises(ValueError):
            TriViewConfig(image_width=True)
        with self.assertRaises(ValueError):
            LaserConfig(local_direction=(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            LaserConfig(direction_frame="camera")
        with self.assertRaises(ValueError):
            LaserConfig(world_direction=(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            LaserConfig(visible_views=("front", "invalid"))
        with self.assertRaises(ValueError):
            WristConfig.from_dict({"laser": {"width_m": 0.0}})
        with self.assertRaises(ValueError):
            TriViewConfig(display_width=1600, display_height=None)

    def test_default_layout_is_native_1920_by_1080_composite(self):
        config = TriViewConfig()
        self.assertEqual(config.composite_size, (1920, 1080))
        self.assertEqual(config.panel_sizes, {
            "front": (1530, 1080),
            "wrist": (380, 535),
            "side": (380, 535),
        })


class TriViewCompositorTests(unittest.TestCase):
    def test_three_images_share_one_state_and_main_left_layout(self):
        now = [0.0]
        sim = _RenderedSim()
        config = TriViewConfig(
            front_camera_name="front_cam",
            wrist_camera_name="wrist_cam",
            side_camera_name="side_cam",
            image_width=80,
            image_height=60,
            render_hz=10.0,
            thumbnail_scale=0.5,
            gutter_px=2,
            show_window=False,
        )
        compositor = TriViewCompositor(config, clock=lambda: now[0])
        record = compositor.update(sim, status_text="state=hold")
        self.assertTrue(record["rendered"])
        self.assertTrue(record["same_state"])
        self.assertEqual(record["sync_time_s"], 12.25)
        self.assertEqual(
            sim.requests,
            [(80, 60, "front_cam"), (40, 29, "wrist_cam"), (40, 29, "side_cam")],
        )
        self.assertEqual(record["composite_shape"], [60, 122, 3])
        self.assertEqual(int(compositor.last_frame[40, 40, 0]), 40)
        self.assertEqual(int(compositor.last_frames["wrist"][0, 0, 0]), 120)
        self.assertEqual(int(compositor.last_frames["side"][0, 0, 0]), 220)
        self.assertEqual(record["native_panel_resolutions"], {
            "front": [80, 60], "wrist": [40, 29], "side": [40, 29],
        })
        self.assertEqual(record["resize_operations"], 0)

        now[0] = 0.01
        limited = compositor.update(sim)
        self.assertFalse(limited["rendered"])
        self.assertTrue(limited["rate_limited"])
        self.assertEqual(len(sim.requests), 3)
        now[0] = 0.11
        self.assertTrue(compositor.update(sim)["rendered"])
        self.assertEqual(len(sim.requests), 6)
        compositor.close()
        self.assertTrue(compositor.closed)

    def test_main_top_layout_does_not_require_a_gui(self):
        sim = _RenderedSim()
        config = TriViewConfig(
            front_camera_name="front_cam",
            wrist_camera_name="wrist_cam",
            side_camera_name="side_cam",
            image_width=80,
            image_height=60,
            layout="main_top",
            thumbnail_scale=0.5,
            gutter_px=2,
            show_window=False,
        )
        record = TriViewCompositor(config).update(sim)
        self.assertTrue(record["rendered"])
        self.assertEqual(record["composite_shape"], [92, 80, 3])
        self.assertIsNone(record["key"])

    def test_composite_routes_key_and_window_close_through_callbacks(self):
        sim = _RenderedSim()
        keys: list[int] = []
        closed: list[bool] = []
        gui = _GuiCv2([ord("r")])
        compositor = TriViewCompositor(
            TriViewConfig(
                front_camera_name="front_cam",
                wrist_camera_name="wrist_cam",
                side_camera_name="side_cam",
                image_width=80,
                image_height=60,
                show_window=True,
            ),
            key_handler=keys.append,
            on_close=lambda: closed.append(True),
            cv2_module=gui,
        )
        first = compositor.update(sim)
        self.assertEqual(first["key"], ord("r"))
        self.assertEqual(keys, [ord("r")])
        gui.visible = 0.0
        compositor._last_render_at = None  # force the fake next GUI poll
        second = compositor.update(sim)
        self.assertTrue(second["closed"])
        self.assertEqual(closed, [True])


class LaserPointerTests(unittest.TestCase):
    def test_laser_skips_robot_hit_tracks_tcp_and_restores_without_physics_side_effects(self):
        sim = _laser_sim()
        start_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        visual_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site_cylinder")
        original = {
            "pos": sim.model.site_pos[visual_id].copy(),
            "quat": sim.model.site_quat[visual_id].copy(),
            "size": sim.model.site_size[visual_id].copy(),
            "rgba": sim.model.site_rgba[visual_id].copy(),
            "sameframe": int(sim.model.site_sameframe[visual_id]),
        }
        qpos_before, qvel_before, contacts_before = sim.data.qpos.copy(), sim.data.qvel.copy(), sim.data.ncon
        pointer = LaserPointer(sim, LaserConfig(local_direction=(0.0, 0.0, -1.0), max_length_m=2.0))
        record = pointer.update()
        self.assertTrue(pointer.available)
        self.assertTrue(record["active"])
        self.assertEqual(record["hit_geom"], "table")
        self.assertGreater(len(record["skipped_geom_ids"]), 0)
        np.testing.assert_array_equal(sim.data.qpos, qpos_before)
        np.testing.assert_array_equal(sim.data.qvel, qvel_before)
        self.assertEqual(sim.data.ncon, contacts_before)

        origin = sim.data.site_xpos[start_id]
        direction = np.asarray(record["direction_world"])
        expected_center = origin + direction * (record["length_m"] / 2.0)
        np.testing.assert_allclose(sim.data.site_xpos[visual_id], expected_center, atol=1.0e-10)
        np.testing.assert_allclose(
            sim.data.site_xmat[visual_id].reshape(3, 3)[:, 2], direction, atol=1.0e-10
        )
        pointer.close()
        np.testing.assert_array_equal(sim.model.site_pos[visual_id], original["pos"])
        np.testing.assert_array_equal(sim.model.site_quat[visual_id], original["quat"])
        np.testing.assert_array_equal(sim.model.site_size[visual_id], original["size"])
        np.testing.assert_array_equal(sim.model.site_rgba[visual_id], original["rgba"])
        self.assertEqual(int(sim.model.site_sameframe[visual_id]), original["sameframe"])

    def test_laser_direction_uses_start_site_orientation_and_unavailable_sites_disable_safely(self):
        sim = _laser_sim()
        # Rotate the robot root so the test checks the site frame rather than a
        # hard-coded world direction.
        body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        sim.model.body_quat[body_id] = np.array([0.9238795, 0.0, 0.3826834, 0.0])
        sim.forward()
        pointer = LaserPointer(sim, LaserConfig(local_direction=(0.0, 0.0, 1.0), max_length_m=0.4))
        record = pointer.update()
        start_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        expected = sim.data.site_xmat[start_id].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(record["direction_world"], expected, atol=1.0e-8)
        np.testing.assert_allclose(record["visual_axis_world"], expected, atol=1.0e-8)
        pointer.close()

        missing = LaserPointer(sim, LaserConfig(start_site="does_not_exist"))
        self.assertFalse(missing.available)
        self.assertFalse(missing.update()["active"])
        self.assertIn("missing_start_site", missing.update()["reason"])

    def test_world_frame_laser_direction_stays_vertical_when_start_site_rotates(self):
        sim = _laser_sim()
        body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        sim.model.body_quat[body_id] = np.array([0.9238795, 0.0, 0.3826834, 0.0])
        sim.forward()
        pointer = LaserPointer(
            sim,
            LaserConfig(
                direction_frame="world",
                world_direction=(0.0, 0.0, -1.0),
                max_length_m=2.0,
            ),
        )
        record = pointer.update()
        np.testing.assert_allclose(record["direction_world"], [0.0, 0.0, -1.0], atol=1.0e-12)
        self.assertEqual(record["direction_frame"], "world")
        pointer.close()

    def test_laser_visual_updates_leave_real_allegro_physics_bitwise_equivalent(self):
        config = WristConfig(smoothing_alpha=None)
        plain = make_allegro_panda_env(
            config, has_renderer=False, has_offscreen_renderer=False,
            initialization_noise=None, seed=19,
        )
        visual = make_allegro_panda_env(
            config, has_renderer=False, has_offscreen_renderer=True,
            initialization_noise=None, seed=19,
        )
        self.addCleanup(plain.close)
        self.addCleanup(visual.close)
        plain.reset()
        visual.reset()
        plain_composer = PandaAllegroActionComposer(plain)
        visual_composer = PandaAllegroActionComposer(visual)
        plain_composer.initialize_goal()
        visual_composer.initialize_goal()
        pointer = LaserPointer(visual, config.laser)
        self.addCleanup(pointer.close)
        for _ in range(3):
            plain.step(plain_composer.compose(np.zeros(6), None))
            pointer.update()
            visual.step(visual_composer.compose(np.zeros(6), None))
            np.testing.assert_array_equal(plain.sim.data.qpos, visual.sim.data.qpos)
            np.testing.assert_array_equal(plain.sim.data.qvel, visual.sim.data.qvel)
            self.assertEqual(plain.sim.data.ncon, visual.sim.data.ncon)


if __name__ == "__main__":
    unittest.main()
