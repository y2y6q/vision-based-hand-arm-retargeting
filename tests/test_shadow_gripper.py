import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop.config import WristConfig
from wrist_teleop.hand_gestures import (
    CAMERA_SOURCE,
    FOUR_FINGER_PINCH,
    SAFE_OPEN_HOLD,
    THUMB_INDEX_PINCH,
    HandGestureConfig,
)
from wrist_teleop.shadow_gripper import (
    GRIPPER_TYPE,
    SHADOW_ACTUATOR_NAMES,
    SHADOW_JOINT_NAMES,
    ShadowHandRight,
    make_shadow_panda_env,
)
from wrist_teleop.shadow_hand_input import (
    ShadowHandGestureController,
    ShadowLegacyHandInput,
    curls_to_shadow_targets,
)
from wrist_teleop.shadow_integrated import PandaShadowActionComposer


def test_shadow_gripper_contract_and_normalized_round_trip():
    hand = ShadowHandRight(idn="unit")
    assert hand.dof == 20
    assert tuple(name.removeprefix("gripperunit_") for name in hand.joints) == SHADOW_JOINT_NAMES
    assert tuple(name.removeprefix("gripperunit_") for name in hand.actuators) == SHADOW_ACTUATOR_NAMES
    assert hand.init_qpos.shape == (24,)
    assert hand.initial_action.shape == (20,)
    assert np.allclose(hand.action_to_targets(hand.targets_to_action(hand.lower_limits)), hand.lower_limits)
    assert np.allclose(hand.action_to_targets(hand.targets_to_action(hand.upper_limits)), hand.upper_limits)
    assert set(hand.important_sites) >= {"grip_site", "grip_cylinder", "ee", "ee_x", "ee_y", "ee_z"}


def test_shadow_legacy_curl_is_bounded_and_keeps_wrist_neutral():
    hand = ShadowHandRight(idn="unit")
    targets = curls_to_shadow_targets([1.0, 0.5, 0.25, 0.75], hand.lower_limits, hand.upper_limits)
    assert targets.shape == (20,)
    assert np.all(targets >= hand.lower_limits)
    assert np.all(targets <= hand.upper_limits)
    assert np.allclose(targets[:2], np.clip(0.0, hand.lower_limits[:2], hand.upper_limits[:2]))
    source = ShadowLegacyHandInput(hand.lower_limits, hand.upper_limits)
    reset = source.reset(timestamp_s=2.0)
    assert reset.joint_targets.shape == (20,)
    assert not reset.valid


def test_shadow_button_gestures_toggle_on_edges_and_keep_targets_bounded():
    hand = ShadowHandRight(idn="unit")
    controller = ShadowHandGestureController(
        HandGestureConfig(interpolation_duration_s=0.20),
        lower_limits=hand.lower_limits,
        upper_limits=hand.upper_limits,
    )
    camera = (controller.lower_limits + controller.upper_limits) * 0.5

    tracked = controller.update((False, False), camera, camera_valid=True, timestamp_s=1.0)
    assert tracked.source == CAMERA_SOURCE
    tracked = controller.update((False, False), camera, camera_valid=True, timestamp_s=1.21)
    assert not tracked.transition_active
    np.testing.assert_allclose(tracked.joint_targets, camera)

    targets = controller.gesture_targets
    for target in targets.values():
        assert target.shape == (20,)
        assert np.all(np.isfinite(target))
        assert np.all(target >= controller.lower_limits)
        assert np.all(target <= controller.upper_limits)
    # Shadow actuator order is wrist, thumb, index, middle, ring, little.
    open_target = targets[SAFE_OPEN_HOLD]
    thumb_index = targets[THUMB_INDEX_PINCH]
    four_finger = targets[FOUR_FINGER_PINCH]
    assert np.allclose(thumb_index[:2], open_target[:2])
    assert np.any(thumb_index[2:7] > open_target[2:7])
    assert np.any(thumb_index[7:10] > open_target[7:10])
    assert np.allclose(thumb_index[10:20], open_target[10:20])
    assert np.any(four_finger[2:20] > open_target[2:20])

    clicked = controller.update((True, False), camera, camera_valid=True, timestamp_s=1.30)
    assert clicked.source == THUMB_INDEX_PINCH
    assert clicked.active_override == THUMB_INDEX_PINCH
    assert clicked.button_rising_edges == (0,)
    assert clicked.event == "activate_thumb_index_pinch"
    # Interpolation begins at the current camera pose to avoid an impulse.
    np.testing.assert_allclose(clicked.joint_targets, camera)

    held = controller.update((True, False), camera, camera_valid=True, timestamp_s=1.40)
    assert held.active_override == THUMB_INDEX_PINCH
    assert held.button_rising_edges == ()
    assert held.event is None
    settled = controller.update((True, False), camera, camera_valid=True, timestamp_s=1.51)
    assert not settled.transition_active
    np.testing.assert_allclose(settled.joint_targets, thumb_index)

    controller.update((False, False), camera, camera_valid=True, timestamp_s=1.52)
    cancelled = controller.update((True, False), camera, camera_valid=True, timestamp_s=1.53)
    assert cancelled.active_override is None
    assert cancelled.source == CAMERA_SOURCE
    assert cancelled.event == "deactivate_thumb_index_pinch"

    controller.update((False, False), camera, camera_valid=True, timestamp_s=1.54)
    reactivated = controller.update((True, False), camera, camera_valid=True, timestamp_s=1.55)
    assert reactivated.active_override == THUMB_INDEX_PINCH
    controller.update((False, False), camera, camera_valid=True, timestamp_s=1.56)
    switched = controller.update((False, True), camera, camera_valid=True, timestamp_s=1.57)
    assert switched.active_override == FOUR_FINGER_PINCH
    assert switched.source == FOUR_FINGER_PINCH
    assert switched.event == "switch_thumb_index_pinch_to_four_finger_pinch"

    invalid = controller.update((False, False), None, camera_valid=False, timestamp_s=2.0)
    # An active explicit gesture remains held until its next click.
    assert invalid.source == FOUR_FINGER_PINCH
    deactivated = controller.update((False, True), None, camera_valid=False, timestamp_s=2.1)
    assert deactivated.active_override is None
    assert deactivated.source == SAFE_OPEN_HOLD
    safe = controller.update((False, True), None, camera_valid=False, timestamp_s=2.31)
    np.testing.assert_allclose(safe.joint_targets, open_target)

    reset = controller.reset(buttons=(True, False), timestamp_s=3.0)
    assert reset.source == SAFE_OPEN_HOLD
    assert reset.active_override is None
    # A held button through reset is deliberately not another rising edge.
    seeded = controller.update((True, False), None, camera_valid=False, timestamp_s=3.1)
    assert seeded.button_rising_edges == ()
    assert seeded.active_override is None


def test_shadow_environment_reset_step_and_slice_isolation():
    env = make_shadow_panda_env(WristConfig(), has_renderer=False)
    try:
        env.reset()
        composer = PandaShadowActionComposer(env)
        assert env.action_dim == 26
        assert composer.arm_slice == slice(0, 6)
        assert composer.hand_slice == slice(6, 26)
        before = composer.hold_action()
        targets = composer.hand_upper_limits.copy()
        action = composer.compose(np.zeros(6), targets, before)
        assert np.allclose(action[composer.arm_slice], 0.0)
        assert not np.allclose(action[composer.hand_slice], before[composer.hand_slice])
        env.step(action)

        # A button-selected Shadow pose changes only the 20-dimension hand
        # slice.  The OSC wrist command is identical before and after click.
        gesture_controller = ShadowHandGestureController(
            HandGestureConfig(interpolation_duration_s=0.0),
            lower_limits=composer.hand_lower_limits,
            upper_limits=composer.hand_upper_limits,
        )
        camera = (composer.hand_lower_limits + composer.hand_upper_limits) * 0.5
        wrist = np.array([0.01, -0.01, 0.0, 0.02, 0.0, -0.02])
        camera_command = gesture_controller.update(
            (False, False), camera, camera_valid=True, timestamp_s=3.0
        )
        button_command = gesture_controller.update(
            (True, False), camera, camera_valid=True, timestamp_s=3.1
        )
        camera_action = composer.compose(wrist, camera_command.joint_targets, before)
        button_action = composer.compose(wrist, button_command.joint_targets, before)
        np.testing.assert_allclose(
            camera_action[composer.arm_slice], button_action[composer.arm_slice]
        )
        assert not np.allclose(
            camera_action[composer.hand_slice], button_action[composer.hand_slice]
        )
    finally:
        env.close()
        # The production runner intentionally retains its local factory
        # registration across hard resets.  Restore the installed-factory
        # baseline after this integration test so the independent audit tests
        # can still assert their pre-adapter boundary.
        from robosuite.models.grippers import GRIPPER_MAPPING
        GRIPPER_MAPPING.pop(GRIPPER_TYPE, None)
