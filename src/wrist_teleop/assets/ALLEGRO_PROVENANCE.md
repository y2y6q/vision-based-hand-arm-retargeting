# Allegro right-hand source and license

This robosuite adapter does not copy, modify, or vend a second Allegro mesh set. At runtime it reads the existing frozen source asset:

`third_party/panda-gym/panda_gym/assets/robots/panda_allegro/allegro_hand/allegro_hand_right.urdf`

and its sibling `meshes/` directory. The adapter translates that URDF's link hierarchy, visual OBJ mesh references, collision boxes / fingertip collision mesh, masses, 16 revolute joint names, and joint limits into a temporary MJCF consumed by robosuite's `GripperModel`.

The asset directory's [`LICENSE`](../../../third_party/panda-gym/panda_gym/assets/robots/panda_allegro/allegro_hand/LICENSE) identifies the original copyright as SimLab (2016), describes modifications by the `dex_urdf` authors, and supplies the BSD-style redistribution conditions. The containing Panda-gym repository is MIT-licensed in [`third_party/panda-gym/LICENSE`](../../../third_party/panda-gym/LICENSE).

This adapter adds the attachment transform documented in `allegro_model.py`: it derives the relative transform from the frozen combined Panda-Allegro URDF's `panda_allegro_mount_joint` and the installed robosuite Panda `right_hand` frame. It does not alter either source model.
