"""Focused regressions for the isolated Panda + Allegro grasp gates."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop.grasp_validation import (  # noqa: E402
    ArtifactWriter,
    ContactSummary,
    DEFAULT_CONFIG,
    Gate2Candidate,
    GATE2_FAILURES,
    GATE3_FAILURES,
    GraspValidationRunner,
    TemporaryCubeSupport,
    _cube_to_hand_normal,
    _cube_handles,
    _cube_state,
    _step,
    aggregate_protocol,
    canonical_sha256,
    classify_gate2,
    classify_gate3,
    evaluate_cube_hand_opposition,
    load_locked_gate2_candidate,
)


class _FakeSim:
    def __init__(self) -> None:
        self.requests: list[tuple[int, int, str]] = []

    def render(self, *, width: int, height: int, camera_name: str) -> np.ndarray:
        self.last_request = (width, height, camera_name)
        self.requests.append(self.last_request)
        return np.zeros((height, width, 3), dtype=np.uint8)


class _FakeEnvironment:
    def __init__(self) -> None:
        self.sim = _FakeSim()


class GraspValidationUnitTests(unittest.TestCase):
    def _gate2_success_metrics(self) -> dict[str, object]:
        return {
            "model_collision": False,
            "max_penetration_m": 0.0,
            "actuator_saturation_s": 0.0,
            "joint_limit_s": 0.0,
            "hand_target_error_rad": 0.0,
            "table_contact": False,
            "object_ejected": False,
            "cube_contacts": 12,
            "cube_fingers": ["thumb", "index", "middle"],
            "opposition_seen": True,
            "opposition_hold_s": DEFAULT_CONFIG.static_hold_s,
            "slip_distance_m": 0.0,
            "contact_lost_after_opposition": False,
            "support_released": True,
            "gravity_enabled": True,
            "hold_duration_s": DEFAULT_CONFIG.static_hold_s,
        }

    def _gate3_success_metrics(self) -> dict[str, object]:
        return {
            "pregrasp_pose_error": False,
            "approach_collision": False,
            "osc_tracking_error": False,
            "timed_out": False,
            "cube_contacts": 12,
            "hand_closed": True,
            "contact_lost_during_lift": False,
            "slip_distance_m": 0.0,
            "lift_m": DEFAULT_CONFIG.lift_success_m,
            "hold_duration_s": DEFAULT_CONFIG.lift_hold_s,
            "table_contact_after_lift": False,
        }

    def test_config_hash_is_stable_and_tracks_physical_values(self):
        first = DEFAULT_CONFIG.sha256
        self.assertEqual(first, DEFAULT_CONFIG.sha256)
        self.assertEqual(first, canonical_sha256(DEFAULT_CONFIG.to_dict()))
        self.assertEqual(len(DEFAULT_CONFIG.panda_initial_joint_positions_rad), 7)
        self.assertEqual(len(DEFAULT_CONFIG.static_close_curls), 4)

    def test_visual_replay_loads_only_a_matching_qualified_lock(self):
        candidate = Gate2Candidate(
            candidate_id="locked",
            stage="geometry_xz",
            cube_center_tcp_m=(0.08, 0.02, 0.11),
            final_curls=(0.50, 0.50, 0.50, 0.85),
            close_schedule=((1.2, (0.50, 0.50, 0.50, 0.85)),),
            rationale="test lock",
        )
        candidate_sha = canonical_sha256({"gate2_base_config": DEFAULT_CONFIG.to_dict(), "candidate": candidate.to_dict()})
        payload = {
            "gate": 2,
            "base_config_sha256": DEFAULT_CONFIG.sha256,
            "candidate": candidate.to_dict(),
            "candidate_sha256": candidate_sha,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "locked_gate2_configuration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_locked_gate2_candidate(path)
            self.assertEqual(loaded.to_dict(), candidate.to_dict())

            payload["candidate_sha256"] = "wrong"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_locked_gate2_candidate(path)

    def test_gate2_classification_requires_continuous_opposition_and_tracking(self):
        metrics = self._gate2_success_metrics()
        self.assertEqual(classify_gate2(metrics), (True, None))

        metrics["opposition_seen"] = False
        passed, failure = classify_gate2(metrics)
        self.assertFalse(passed)
        self.assertEqual(failure, "insufficient_opposition")

        metrics = self._gate2_success_metrics()
        metrics["opposition_hold_s"] = DEFAULT_CONFIG.static_hold_s - 1.0 / DEFAULT_CONFIG.control_hz
        self.assertEqual(classify_gate2(metrics), (False, "insufficient_opposition"))

        metrics = self._gate2_success_metrics()
        metrics["model_collision"] = True
        self.assertEqual(classify_gate2(metrics), (False, "model_collision"))

        metrics = self._gate2_success_metrics()
        metrics["hand_target_error_rad"] = DEFAULT_CONFIG.joint_target_error_rad + 1e-6
        self.assertEqual(classify_gate2(metrics), (False, "joint_tracking_error"))
        self.assertIn("joint_tracking_error", GATE2_FAILURES)

    def test_gate3_classification_has_one_stable_failure(self):
        metrics = self._gate3_success_metrics()
        self.assertEqual(classify_gate3(metrics), (True, None))

        metrics["osc_tracking_error"] = True
        passed, failure = classify_gate3(metrics)
        self.assertFalse(passed)
        self.assertEqual(failure, "osc_tracking_error")
        self.assertIn(failure, GATE3_FAILURES)

    def test_protocol_requires_matching_hash_and_8_of_10_after_qualification(self):
        config_sha = DEFAULT_CONFIG.sha256
        trials: list[dict[str, object]] = [{"protocol_phase": "single", "success": True, "config_sha256": config_sha}]
        trials.extend({"protocol_phase": "qualification", "success": True, "config_sha256": config_sha} for _ in range(3))
        trials.extend(
            {"protocol_phase": "formal", "success": index < 8, "config_sha256": config_sha}
            for index in range(10)
        )
        result = aggregate_protocol(trials, config_sha256=config_sha)
        self.assertTrue(result["passed"])
        self.assertEqual(result["formal_successes"], 8)

        trials[-1]["config_sha256"] = "different"
        changed = aggregate_protocol(trials, config_sha256=config_sha)
        self.assertFalse(changed["passed"])
        self.assertEqual(changed["failure_category"], "unknown")

    def test_artifact_writer_records_jsonl_and_optional_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            writer = ArtifactWriter(root, config_sha256="abc", frame_every_s=0.1, record_frames=True)
            try:
                writer.event({"event": "start"})
                writer.trial({"success": False, "failure_category": "no_contact"})
                writer.contacts([{"event": "contact"}])
                frame = writer.capture(_FakeEnvironment(), "fake", sim_time=0.0, force=True)
                self.assertEqual(frame, "frames/0000_fake.png")
                self.assertTrue((root / str(frame)).is_file())
                self.assertEqual((root / "trials.jsonl").read_text(encoding="utf-8").count("abc"), 1)
                views = writer.capture_validation_views(_FakeEnvironment(), "views", sim_time=0.0)
                self.assertEqual(len(views), 3)
                self.assertTrue(all((root / item).is_file() for item in views))
            finally:
                writer.close()

            disabled_root = Path(temporary) / "disabled"
            disabled = ArtifactWriter(disabled_root, config_sha256="abc", frame_every_s=0.1, record_frames=False)
            try:
                self.assertIsNone(disabled.capture(object(), "ignored", sim_time=0.0, force=True))
                self.assertEqual(disabled.frame_paths, [])
                self.assertIsNone(disabled.frame_error)
            finally:
                disabled.close()

    def test_cube_normal_is_canonical_independent_of_geom_order(self):
        raw = np.array([0.6, -0.8, 0.0])
        np.testing.assert_allclose(_cube_to_hand_normal(raw, cube_is_geom1=True), raw)
        np.testing.assert_allclose(_cube_to_hand_normal(-raw, cube_is_geom1=False), raw)

    def test_normal_opposition_requires_thumb_and_two_distinct_nonthumb_fingers(self):
        summary = ContactSummary(cube_fingers={"thumb", "index", "middle"})
        summary.cube_hand_contacts = [
            {"finger": "thumb", "hand_geom": "thumb_tip", "cube_to_hand_normal_world": [1.0, 0.0, 0.0], "normal_force_abs_n": 0.2},
            {"finger": "index", "hand_geom": "index_tip", "cube_to_hand_normal_world": [-1.0, 0.0, 0.0], "normal_force_abs_n": 0.2},
            {"finger": "middle", "hand_geom": "middle_tip", "cube_to_hand_normal_world": [-0.8, 0.6, 0.0], "normal_force_abs_n": 0.2},
        ]
        opposed = evaluate_cube_hand_opposition(summary, dot_max=-0.5, minimum_normal_force_n=0.05)
        self.assertTrue(opposed["valid"])
        self.assertEqual(opposed["opposed_nonthumb_fingers"], ["index", "middle"])

        summary.cube_hand_contacts[-1]["cube_to_hand_normal_world"] = [1.0, 0.0, 0.0]
        not_opposed = evaluate_cube_hand_opposition(summary, dot_max=-0.5, minimum_normal_force_n=0.05)
        self.assertFalse(not_opposed["valid"])
        self.assertEqual(not_opposed["opposed_nonthumb_fingers"], ["index"])


class GraspValidationSimulatorTests(unittest.TestCase):
    def test_seed_repeats_cube_parameters_across_fresh_environments(self):
        snapshots = []
        for _ in range(2):
            runner = GraspValidationRunner(record_frames=False)
            env = runner._new_environment()
            try:
                env.reset()
                cube = _cube_state(env, _cube_handles(env))
                snapshots.append((cube["geom_half_size_m"], cube["mass_kg"], cube["friction"]))
            finally:
                env.close()
        self.assertEqual(snapshots[0], snapshots[1])

    def test_fixed_panda_pose_survives_zero_delta_hand_steps(self):
        """Regression for OSC's cached null-space pose after a scripted reset."""

        runner = GraspValidationRunner(record_frames=False)
        env = runner._new_environment()
        try:
            composer = runner._reset(env)
            qpos_ids = np.asarray(composer.robot._ref_joint_pos_indexes, dtype=np.intp)
            targets = composer.hand_qpos()
            action = composer.compose(np.zeros(6), targets)
            self.assertEqual((composer.arm_slice.start, composer.arm_slice.stop), (0, 6))
            self.assertEqual((composer.hand_slice.start, composer.hand_slice.stop), (6, 22))
            for _ in range(16):
                env.step(action)
            np.testing.assert_allclose(
                env.sim.data.qpos[qpos_ids],
                DEFAULT_CONFIG.panda_initial_joint_positions_rad,
                atol=2e-4,
                rtol=0.0,
            )
            np.testing.assert_array_equal(action[composer.arm_slice], np.zeros(6))
            first_cube = _cube_state(env, _cube_handles(env))
            runner._reset(env)
            second_cube = _cube_state(env, _cube_handles(env))
            np.testing.assert_allclose(first_cube["geom_half_size_m"], second_cube["geom_half_size_m"])
            self.assertEqual(first_cube["mass_kg"], second_cube["mass_kg"])
        finally:
            env.close()

    def test_fixture_updates_once_per_mujoco_substep_and_release_does_not_write_state(self):
        runner = GraspValidationRunner(record_frames=False)
        env = runner._new_environment()
        with tempfile.TemporaryDirectory() as temporary:
            writer = ArtifactWriter(
                Path(temporary) / "artifacts",
                config_sha256=runner.config.sha256,
                frame_every_s=runner.config.frame_every_s,
                record_frames=False,
            )
            try:
                composer = runner._reset(env)
                handles = _cube_handles(env)
                target = np.asarray(_cube_state(env, handles)["position_m"], dtype=np.float64)
                support = TemporaryCubeSupport(env, handles, target, runner.config)
                _summary, event = _step(
                    env,
                    composer,
                    hand_targets=composer.hand_qpos(),
                    wrist_delta=np.zeros(6),
                    gate=2,
                    trial_id="fixture_substep_test",
                    phase="fixture_test",
                    writer=writer,
                    config=runner.config,
                    support=support,
                )
                self.assertEqual(event["fixture"]["substep_count"], 25)
                self.assertTrue(event["fixture"]["substep_count_matches_expected"])
                qpos_start, qvel_start = handles["qpos_start"], handles["qvel_start"]
                before_qpos = env.sim.data.qpos[qpos_start:qpos_start + 7].copy()
                before_qvel = env.sim.data.qvel[qvel_start:qvel_start + 6].copy()
                support.release()
                np.testing.assert_allclose(env.sim.data.qpos[qpos_start:qpos_start + 7], before_qpos)
                np.testing.assert_allclose(env.sim.data.qvel[qvel_start:qvel_start + 6], before_qvel)
                self.assertEqual(float(np.linalg.norm(env.sim.data.xfrc_applied[handles["body_id"]])), 0.0)
            finally:
                writer.close()
                env.close()


if __name__ == "__main__":
    unittest.main()
