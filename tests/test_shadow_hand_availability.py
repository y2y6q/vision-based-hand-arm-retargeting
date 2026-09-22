"""Regression tests for the project-local Panda + Shadow Hand preflight."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop.shadow_hand import (  # noqa: E402
    ShadowHandUnavailable,
    audit_shadow_hand_support,
    require_shadow_hand_support,
)
from wrist_teleop.shadow_gripper import GRIPPER_TYPE, ShadowHandRight  # noqa: E402


class ShadowHandAvailabilityTests(unittest.TestCase):
    def test_missing_factory_or_asset_never_reports_ready(self):
        audit = audit_shadow_hand_support(
            gripper_mapping={"PandaGripper": object, "AllegroRightHand": object},
            find_spec=lambda _name: None,
            local_model_candidates=(),
        )
        self.assertFalse(audit.ready)
        self.assertEqual(audit.shadow_gripper_names, ())
        self.assertIn("GRIPPER_MAPPING", audit.blocker or "")
        self.assertIn("MJCF", audit.blocker or "")
        self.assertEqual(audit.existing_local_model_candidates, ())

    def test_current_project_adapter_is_registered_and_ready_without_simulation(self):
        audit = audit_shadow_hand_support()
        self.assertTrue(audit.ready, audit.blocker)
        self.assertIn(GRIPPER_TYPE, audit.shadow_gripper_names)
        self.assertEqual(audit.adapter["class"], ShadowHandRight.__name__)
        self.assertEqual(audit.adapter["registered_type"], GRIPPER_TYPE)
        self.assertTrue(audit.adapter["source_meshes_complete"])
        self.assertIn(audit.adapter["source_mjcf"], audit.existing_local_model_candidates)

    def test_audit_records_actual_24_joint_20_actuator_contract_and_no_dex(self):
        audit = audit_shadow_hand_support()
        self.assertEqual(audit.physical_joint_count, 24)
        self.assertEqual(audit.action_dof, 20)
        self.assertEqual(len(audit.adapter["physical_joint_order"]), 24)
        self.assertEqual(len(audit.adapter["actuator_order"]), 20)
        self.assertTrue(audit.dex_not_integrated)
        self.assertIn("not integrated", " ".join(audit.required_integration_contract))
        self.assertEqual(audit.official_model_reference["license"], "Apache-2.0")

    def test_require_returns_ready_audit_and_still_rejects_explicit_failure_audit(self):
        ready = require_shadow_hand_support()
        self.assertTrue(ready.ready)
        bad = audit_shadow_hand_support(
            gripper_mapping={}, find_spec=lambda _name: None, local_model_candidates=()
        )
        with self.assertRaises(ShadowHandUnavailable):
            require_shadow_hand_support(bad)


if __name__ == "__main__":
    unittest.main()
