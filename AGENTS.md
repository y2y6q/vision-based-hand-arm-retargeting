# AGENTS.md

## Project

Repository: `E:\hand_arm_retargeting`

This repository is the implementation workspace for a dexterous teleoperation / retargeting project.

The current main development path is:

**robosuite + MuJoCo + Panda + OSC_POSE**

The older **PyBullet / panda-gym Panda + Allegro** implementation is a verified reference baseline and should be treated as frozen unless a regression fix is explicitly required.

---

## Source of Truth

Follow these priorities when making engineering decisions:

1. The current repository and installed package versions.
2. This `AGENTS.md`.
3. Current task instructions from the user.
4. Verified official robosuite / MuJoCo APIs and local installed source.
5. The frozen PyBullet implementation only as a behavioral reference.

Do not assume APIs from memory when they can be inspected locally.

---

## Project Workstreams

The project is organized conceptually into three workstreams.

### 00 | Project Control and Stage Acceptance

Purpose:
- Long-term goals
- Stage roadmap
- Task decomposition
- Key decisions
- Risks
- Cross-task coordination
- Stage acceptance

Do not turn normal implementation work into project-management work unless the task explicitly requires it.

### 01 | Robosuite Mainline Development

This is the default workstream for code changes in this repository.

Responsibilities:
- robosuite / MuJoCo integration
- Panda control
- `OSC_POSE`
- SpaceMouse integration
- Camera input
- Future dexterous-hand integration
- Environment setup
- Running code
- Debugging
- Automated checks
- Regression testing

Rules:
- Prefer the robosuite / MuJoCo mainline.
- Do not add new features to the old PyBullet / panda-gym backend unless explicitly requested.
- Do not assume robosuite natively supports Allegro Hand; verify first.
- For arm pose control, prefer `OSC_POSE` rather than rebuilding the previous IK pipeline unless there is a clear technical reason.

### 02 | IK Feasibility and Trajectory Experiments

This workstream is for research / diagnostic questions rather than routine integration.

Keep these three questions separate:

1. Does a static target pose have a solution?
2. Can the robot move from the current joint state into that pose?
3. Can the complete trajectory be executed continuously?

Do not infer static unreachability from a dynamic continuity failure.

Historical warning:
- The old PyBullet experiment with `9987/10000` branch-jump rejections mixed a dynamic continuity condition into static sampling.
- That result must not be treated as a theoretical reachability conclusion.

Use the robosuite / MuJoCo system and the current robot morphology when redefining feasibility.

Escalate to this workstream when the mainline encounters questions such as:
- repeated pose failures
- discontinuous joint branches
- trajectory-level infeasibility
- motion-planning alternatives
- robomimic trajectory replay diagnostics

---

## Current Mainline Baseline

Known verified legacy baseline:
- PyBullet / panda-gym
- Panda arm + Allegro Hand
- 23 movable joints
- camera pipeline
- unified parameter factory
- MediaPipe-based rule-style finger curl mapping
- static workspace point-cloud experiments

Known unresolved legacy issues:
- continuous IK rejection
- side-palm rotation robustness
- end-to-end grasp reliability

Current mainline:
- robosuite
- MuJoCo
- Panda
- `OSC_POSE`

The PyBullet implementation is a reference, not the target architecture.

---

## Current SpaceMouse Task

Current task:

**Use a 3Dconnexion SpaceMouse to control the Panda robot arm in robosuite.**

Target control pipeline:

`SpaceMouse -> 6DoF incremental input -> OSC_POSE -> Panda end effector`

Scope:
- Panda arm only
- robosuite / MuJoCo only
- SpaceMouse translation and rotation
- reliable real-time teleoperation
- Windows local machine

Out of scope for this task:
- Allegro Hand integration
- custom inverse-kinematics pipeline
- robomimic training
- motion-planning research
- modifying the frozen PyBullet baseline

Before implementing:
1. Inspect the repository.
2. Inspect the installed robosuite version and source.
3. Check whether that version already contains SpaceMouse / device support.
4. Check how `OSC_POSE` actions are represented in the installed version.
5. Identify the smallest set of files that need to change.

Do not begin with a speculative rewrite.

---

## SpaceMouse Implementation Requirements

Prefer reusing verified robosuite device support if it works with the installed version and Windows environment.

If custom device input is necessary, first create a minimal hardware diagnostic.

The diagnostic should verify:
- device detection
- six raw motion axes
- button states
- physical-axis mapping
- clean disconnect / failure behavior

The main control implementation should separate:

1. device input acquisition
2. normalization / deadzone
3. filtering and sensitivity
4. axis / sign mapping
5. coordinate-frame mapping
6. robosuite action generation
7. teleoperation loop

Do not scatter tuning constants throughout the code.

Centralize:
- translation sensitivity
- rotation sensitivity
- translation deadzone
- rotation deadzone
- axis inversion
- control frequency
- maximum translation delta
- maximum rotation delta
- optional smoothing

Safety behavior:
- centered device must produce zero command
- neutral noise must not move the robot
- clamp large commands
- do not allow target drift when input returns to zero
- handle disconnects safely
- avoid malformed action vectors
- exit cleanly

Coordinate frame must be explicit and documented.

Start with a simple intuitive mapping, preferably robot-base / world-aligned control, unless the existing codebase already establishes another convention.

---

## Engineering Rules

### Inspect Before Editing

For any non-trivial task:
- inspect relevant files first
- identify existing entry points
- identify controller configuration
- identify reusable utilities
- inspect installed APIs where uncertain

Before making broad changes, report:
- relevant existing files
- observed architecture
- proposed plan
- files to create / modify

### Minimal Changes

Prefer the smallest correct change.

Do not:
- rewrite unrelated modules
- rename large directory trees without need
- replace working dependencies casually
- introduce multiple competing device libraries
- duplicate functionality already provided by robosuite

### Dependency Discipline

Do not randomly install libraries.

Before adding a dependency:
1. verify whether the repository already has an implementation
2. verify whether robosuite already provides the required functionality
3. identify the minimal appropriate package
4. explain why it is needed

### Version-Aware Development

robosuite APIs can differ across versions.

Always verify against:
- the installed version
- local package source
- actual runtime behavior

Do not invent API names.

### Preserve the Frozen Baseline

Do not modify legacy PyBullet / panda-gym code unless:
- the task explicitly requires it, or
- a regression fix is necessary and justified

If legacy code is used, treat it as a reference for behavior or mapping.

---

## Testing and Validation

Do not consider a feature complete just because the code imports.

For SpaceMouse arm control, acceptance criteria are:

1. A standalone diagnostic reads all six SpaceMouse axes.
2. robosuite starts a Panda environment successfully.
3. SpaceMouse translation produces smooth end-effector translation.
4. SpaceMouse rotation produces smooth end-effector orientation changes.
5. Releasing the device stops commanded motion.
6. Neutral noise does not move the robot.
7. Large commands are clamped.
8. The process exits cleanly.
9. Existing frozen PyBullet functionality is not broken.
10. Exact run commands are documented.

When possible, run:
- syntax / import checks
- focused unit tests
- the smallest integration test
- the actual teleoperation entry point

If hardware prevents full validation, clearly distinguish:
- code verified
- simulator verified
- hardware not yet verified

Never report a hardware test as passed if it was not actually run.

---

## Debugging Rules

When something fails:
1. reproduce the failure
2. identify the smallest failing layer
3. inspect logs / exceptions
4. test that layer independently
5. fix the root cause
6. rerun the smallest relevant regression test

For control failures, distinguish:
- device input problem
- mapping problem
- controller/action-format problem
- coordinate-frame problem
- simulator problem
- static feasibility problem
- transition / trajectory feasibility problem

Do not label all pose-control failures as "IK failure."

---

## Completion Report

At the end of an implementation task, report:

1. files created
2. files modified
3. what changed
4. exact commands to run
5. tests actually executed
6. test results
7. tunable parameters
8. known limitations
9. recommended next step

Keep claims tied to what was actually inspected or tested.

---

## Default Codex Behavior

For a new task, use this workflow:

`inspect -> plan -> implement -> run -> debug -> retest -> summarize`

For substantial or risky changes, pause after the inspection / plan phase if the user explicitly asks for approval before edits.

Otherwise, proceed through implementation and validation without unnecessary interruption.

The priority is a working, testable robosuite mainline with minimal speculative changes.
