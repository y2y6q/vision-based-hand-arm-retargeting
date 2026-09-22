# Vision-Based Hand-Arm Retargeting

This repository contains a Windows robosuite and MuJoCo mainline for vision-based teleoperation. The current system uses OSC_POSE to control a Panda wrist with a SpaceMouse. Camera hand input controls a dexterous hand or gripper without changing the arm command.

## Current status

| Area | Status | Evidence |
| --- | --- | --- |
| Panda + Allegro wrist and hand control | Verified | Real 22-D action contract, headless smoke tests, and synchronized three-view smoke tests. |
| Camera-relative SpaceMouse mapping | Verified | The six-axis command is converted through the main camera basis before composing the Panda OSC action. |
| Three-view visualization and palm laser | Verified | Native 1920 x 1080 composite rendering; the laser is visual only and does not affect physics. |
| Panda + PandaGripper | Verified | Real 7-D action contract, open/close behavior, headless smoke, and viewer smoke. |
| Jaco + JacoThreeFingerGripper | Verified | Real 7-D action contract, continuous hold-to-open / hold-to-close behavior, headless smoke, and viewer smoke. |
| Gate 2 fixed-Panda static grasp | Verified | The independent regression for the locked static-grasp configuration continues to pass. |
| Panda + Shadow Hand | Audit only | The fourth entry point audits the local model and robosuite registration, then refuses normal startup. It never substitutes Allegro for Shadow Hand. |
| Official dex-retargeting sidecar | Verified | The isolated worker loads dex-retargeting 0.4.6, creates SeqRetargeting, runs retarget/reset, and writes only bounded named Allegro targets. |
| Live SpaceMouse direction, camera retargeting, and grasping | Manual verification required | Automated synthetic input cannot validate physical axes, camera quality, human motion, or live teleoperation. |

## Latest progress+�u���T 2026-09-22

Panda + Allegro now defaults to official dex hand control. The main robosuite process launches an isolated sidecar, avoiding direct native dex dependency imports in the MuJoCo process. The worker loads the bundled dex-retargeting 0.4.6 wheel, constructs the official Allegro vector solver, and returns named hand targets through newline-delimited JSON.

| Item | Result |
| --- | --- |
| Official wheel SHA-256 | cf08b93e204af21b7146f12a83d55f0a0d227f991d202655c475535621bb62f7 |
| Official configuration SHA-256 | 23cdb5385137dec4174f1f9211ffe4884a9222f5d3a98fcf7218c715555a5284 |
| Solver configuration | configs/dex_retargeting/allegro_hand_right.yml |
| Wheel provenance | third_party/dex_retargeting_0_4_6/PROVENANCE.md |
| Focused regression tests | 23 passed: worker protocol, official solver, real Panda step, slice isolation, reset, and shutdown |
| Frozen reference baseline | 172/173 hashes match; the pre-existing mismatch is src/teleop/teleop_factory.py, outside the current dex work |

The real Panda + Allegro environment has a 22-D action vector:

| Slice | Meaning |
| --- | --- |
| [0:6] | Panda OSC_POSE increment: Tx, Ty, Tz, Rx, Ry, Rz |
| [6:22] | 16 named Allegro joint targets |

The dex target is reordered by joint name, clamped to the real Allegro limits, and written only to [6:22]. Camera input cannot overwrite the Panda arm slice. The latest hardware-free dex smoke selected runtime=sidecar and exited safely.

### Required manual checks

1. Start the default Panda + Allegro command and confirm the terminal reports hand mode=dex, runtime=sidecar, and dex-retargeting=0.4.6.
2. Confirm that the live camera image, hand landmarks, and DEX ACTIVE / dex-update state appear. Right-hand motion must change only the Allegro hand.
3. Confirm that SpaceMouse axes move only the Panda wrist. Confirm that SpaceMouse buttons change Allegro gestures only.
4. Confirm physical push/pull direction and neutral behavior. Releasing the SpaceMouse must produce a zero wrist command with no continuing drift.
5. Confirm R, Q, and Ctrl+C cleanup from both the simulation and camera windows.
6. Run a longer live session before removing the sidecar-only KMP_DUPLICATE_LIB_OK compatibility setting used for the current Conda OpenMP DLL conflict.

Generated logs are stored locally in outputs/ and are deliberately excluded from GitHub.

## Programs

| Entry point | Robot and end effector | Action layout | Inputs |
| --- | --- | --- | --- |
| scripts/run_spacemouse_panda_wrist_teleop.py | Panda + right Allegro hand | [0:6] arm, [6:22] 16 Allegro joints | SpaceMouse, camera, and button gestures |
| scripts/run_spacemouse_panda_jaw_gripper_teleop.py | Panda + PandaGripper | [0:6] arm, [6:7] parallel gripper | SpaceMouse; HID bit 0 closes and bit 1 opens |
| scripts/run_spacemouse_jaco_three_finger_teleop.py | Jaco + JacoThreeFingerGripper | [0:6] arm, [6:7] shared three-finger command | SpaceMouse; hold HID bit 0 to close and bit 1 to open |
| scripts/run_spacemouse_panda_shadow_hand_teleop.py | Panda + Shadow Hand | No runtime action layout | Audit only; ordinary startup intentionally refuses |

All active action values are in [-1, 1]. The OSC controllers receive world-coordinate increments. SpaceMouse commands are interpreted in the main camera frame, using screen-right, viewing depth, and world-up, then transformed into world coordinates.

## Visualization, laser, and telemetry

The default viewer combines a shoulder-back main view, robot0_eye_in_hand, and a side view. The synchronized composite is rendered at native 1920 x 1080: a 1530 x 1080 main image plus two 380 x 535 auxiliary images. All views render after the same env.step call. Camera and rendering settings are centralized in configs/spacemouse_wrist.json under tri_view.

Panda + Allegro opens a live camera status window by default. It reuses the existing capture and detection thread. The window shows the color image, hand landmarks, detected hands, current hand mode, detection status, frame rate, and dex update state. R and Q work in both windows; closing either window triggers safe shutdown. Use --headless for no windows or --no-camera-display to hide only the camera status window.

For Panda + Allegro, the red laser starts at palm_laser_site and points vertically down along world -Z. It skips robot geometry when locating a scene hit and draws only a transparent visual cylinder. It does not add collision, mass, contact, constraints, qpos/qvel writes, or actions.

The interactive telemetry records cube height, table contact, hand contact, penetration, unsupported hold time, and outcome. This is teleoperation evidence, not an autonomous grasp benchmark. Gate 2 uses its own locked static-grasp protocol.

## SpaceMouse and keyboard controls

The input pipeline applies initial neutral-offset estimation, hysteretic deadzones, axis order, inversion, EMA smoothing, scale, clamp, and camera-frame mapping before composing the arm action.

| Parameter | Value |
| --- | --- |
| Translation and rotation enter deadzone | 0.08 |
| Translation and rotation exit deadzone | 0.06 |
| Consecutive neutral frames before exact-zero lock | 3 |
| Translation scale per step | 0.003 m |
| Rotation scale per step | 0.025 rad |
| Maximum translation / rotation increment | 0.004 m / 0.040 rad |
| Control rate | 20 Hz |
| EMA smoothing factor | 0.35 |

Offset estimation uses only the first eight centered frames and never learns a new offset during normal motion. The unattended 30-second record received no SpaceMouse motion reports, so it validates only the stale/no-input route: final OSC command was exactly zero, end-effector translation was 1.55e-10 m, and orientation change was 4.68e-6 degrees. Real hardware behavior must still be confirmed manually.

| Control | Panda + Allegro | Panda / Jaco grippers |
| --- | --- | --- |
| Left button, HID bit 0 | Rising edge toggles thumb-index pinch | Panda closes; Jaco closes while held |
| Right button, HID bit 1 | Rising edge toggles four-finger pinch | Panda opens; Jaco opens while held |
| B | Enlarges the interactive Lift cube once to 1.5x default size | No effect |
| R | Resets environment, OSC, inputs, gestures, camera buffer, and dex history | Resets environment, OSC, input state, and gripper state |
| Q or Ctrl+C | Sends a final zero arm command and safely exits | Same behavior |

Button gestures never alter the Panda arm slice. B applies only to Panda + Allegro integrated mode; R restores the default cube size. HID-bit-to-physical-button mapping must be checked with the diagnostic command. Jaco uses real elapsed dt while held, stops immediately on release, and remains still if both buttons are pressed.

## Quick start

Read Run Instructions.txt for dependency setup, diagnostics, smoke tests, Gate 2, and the frozen-baseline check.

~~~powershell
Set-Location "E:\hand_arm_retargeting"
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_wrist_teleop.py integrated
~~~

This is the main interactive Panda + Allegro entry point. It starts official dex sidecar control by default.

~~~powershell
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_wrist_teleop.py integrated --headless --dex-smoke --duration 2
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_wrist_teleop.py integrated --hand-control legacy-curl
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_jaw_gripper_teleop.py
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_jaco_three_finger_teleop.py
& .\.venv-robosuite\Scripts\python.exe .\scripts\run_spacemouse_panda_shadow_hand_teleop.py --audit
~~~

The dex smoke command does not need a SpaceMouse. legacy-curl is available only when explicitly selected. The Shadow Hand audit does not start HID, camera, simulation, or a viewer. Removing --audit makes the script refuse clearly because the installed robosuite version has no verified native Shadow Hand runtime.

## Project map

| Path | Purpose |
| --- | --- |
| src/wrist_teleop/visualization.py | Shared three-view rendering, screenshots, and non-physical laser visualization |
| src/wrist_teleop/hand_input.py | Single camera capture, hand detection, overlay, and Panda + Allegro input state |
| src/wrist_teleop/gripper_teleop.py | Shared Panda / Jaco gripper control loop |
| src/wrist_teleop/grasp_telemetry.py | Shared read-only interactive grasp telemetry |
| src/wrist_teleop/dex_retargeting.py | Dex configuration, named mapping, sidecar client, and integrity checks |
| scripts/dex_retargeting_worker.py | Official dex 0.4.6 solver in an isolated interpreter |
| third_party/dex_retargeting_0_4_6 | Checked official wheel and provenance |
| configs/spacemouse_wrist.json | Source of control, camera, laser, dex-loss, and telemetry parameters |
| outputs/spacemouse_wrist/<timestamp>_integrated | Local Panda + Allegro metadata, telemetry, and screenshots |
| outputs/gripper_teleop/<timestamp>_* | Local Panda / Jaco metadata, telemetry, and screenshots |

See docs/robosuite_spacemouse_wrist.md for lower-level API and configuration details.

## Rollback and frozen baseline

Pre-change copies and their SHA-256 manifest are stored locally under outputs/weekly_grasp_ux/20260916T153432Z/prechange/. Restore those files to revert this iteration. Set laser.enabled or tri_view.enabled to false in configs/spacemouse_wrist.json to disable the corresponding visual feature. Use --hand-control legacy-curl only when an explicit fallback is required.
