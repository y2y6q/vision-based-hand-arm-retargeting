# Third-Party Notices

This repository contains third-party software and robot assets required by the customized Panda-Allegro simulation environment. The original copyright notices and license files are retained.

## panda-gym

- Source: <https://github.com/qgallouedec/panda-gym>
- Included version base: 3.0.8
- License: MIT
- License file: [`vendor/panda-gym/LICENSE`](vendor/panda-gym/LICENSE)
- Local changes: Panda-Allegro environment registration, robot implementation, pick-and-place task, and required model assets.

## dex-urdf

- Source: <https://github.com/dexsuite/dex-urdf>
- Referenced revision: `f5e7132f22108164577fea4c25ef99b5cc0e1900`
- License: MIT
- License copy: [`vendor/licenses/dex-urdf-LICENSE`](vendor/licenses/dex-urdf-LICENSE)
- Citation metadata: [`vendor/licenses/dex-urdf-CITATION.cff`](vendor/licenses/dex-urdf-CITATION.cff)

The Allegro Hand model files used by the customized environment were derived from the dex-urdf asset set. Their model-specific license is retained at:

[`vendor/panda-gym/panda_gym/assets/robots/panda_allegro/allegro_hand/LICENSE`](vendor/panda-gym/panda_gym/assets/robots/panda_allegro/allegro_hand/LICENSE)

## Franka Panda model

The Franka Panda model included with the customized environment is distributed under the Apache License 2.0. Its license is retained at:

[`vendor/panda-gym/panda_gym/assets/robots/panda_allegro/franka_panda/LICENSE.txt`](vendor/panda-gym/panda_gym/assets/robots/panda_allegro/franka_panda/LICENSE.txt)

## Project license

A root license for the project-specific source code has not yet been selected. The third-party licenses above apply only to their respective components.
