# Third-Party Notices

This repository retains the license files and provenance notes that accompany
the third-party software and robot assets used by the project.

## panda-gym

- Source: <https://github.com/qgallouedec/panda-gym>
- Included base: 3.0.8
- License: MIT
- Local path: [`third_party/panda-gym`](third_party/panda-gym)
- License copy: [`third_party/panda-gym/LICENSE`](third_party/panda-gym/LICENSE)

The directory is the frozen legacy reference baseline. The robosuite mainline
reads its Panda-Allegro URDF assets but does not modify the baseline code.

## dex-retargeting

- Source: <https://pypi.org/project/dex-retargeting/0.4.6/>
- Version: 0.4.6
- License: MIT
- Verified wheel and SHA-256: [`third_party/dex_retargeting_0_4_6/PROVENANCE.md`](third_party/dex_retargeting_0_4_6/PROVENANCE.md)

The official wheel runs only inside the isolated dex sidecar. The robosuite /
MuJoCo process does not import its native dependencies.

## dex-urdf Shadow Hand (kinematics for the dex-retargeting sidecar)

- Source: dex-urdf Shadow Hand as distributed with dexsuite/dex-retargeting
- License: Apache-2.0
- Local path and provenance: [`third_party/dex_urdf/PROVENANCE.md`](third_party/dex_urdf/PROVENANCE.md)
- License copy: [`third_party/dex_urdf/shadow_hand/LICENSE`](third_party/dex_urdf/shadow_hand/LICENSE)

Only the URDF is included (meshes omitted); it is used for kinematics in the
isolated dex sidecar.

## LIBERO

- Source: <https://github.com/Lifelong-Robot-Learning/LIBERO>, commit `8f1084e`
- License: MIT

LIBERO is installed from a local clone in `third_party/LIBERO` inside the
isolated `.venv-libero`; it is not redistributed in this repository.

## Shadow Hand assets

The locally audited Shadow Hand assets retain their source license at
[`assets/shadow_hand/LICENSE-MUJOCOMENAGERIE.txt`](assets/shadow_hand/LICENSE-MUJOCOMENAGERIE.txt).

## Project license

A root license for project-specific source has not yet been selected. The
third-party licenses above apply only to their respective components.
