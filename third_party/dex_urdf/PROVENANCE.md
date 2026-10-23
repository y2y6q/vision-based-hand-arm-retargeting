# dex-urdf Shadow Hand (kinematics for the dex-retargeting sidecar)

- Source: dexsuite/dex-retargeting `assets/robots/hands/shadow_hand` (dex-urdf), local clone `E:/robot_projects/dex-retargeting` at commit `3f56141bc8bd2760d5e452e382937269554ebb21`.
- Files: `shadow_hand/shadow_hand_right.urdf`, `LICENSE` (Apache-2.0), `README.md`, copied unchanged.
- `shadow_hand_right.urdf` SHA-256: `bdd12dbde89f71b8f427ecc973ee6dcac48643b42b14d401bee8631f6569f87e`
- Meshes are intentionally omitted: the solver only builds Pinocchio kinematics, so missing-mesh warnings on sidecar stderr are expected.
- Verified 2026-10-22: fingertip pairwise distances match the MuJoCo Menagerie Shadow (`assets/shadow_hand/right_hand.xml`) within 0.1 mm for random joint samples; joint names, signs and limits agree.
