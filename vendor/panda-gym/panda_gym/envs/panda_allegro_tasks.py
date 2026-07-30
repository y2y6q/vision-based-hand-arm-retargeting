from typing import Optional

import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda_allegro import PandaAllegro
from panda_gym.envs.tasks.pick_and_place import PickAndPlace
from panda_gym.pybullet import PyBullet


class PandaAllegroPickAndPlaceEnv(RobotTaskEnv):
    """Panda arm + Allegro Hand PickAndPlace environment.

    This environment reuses the original panda-gym PickAndPlace task,
    but replaces the original Panda two-finger gripper robot with PandaAllegro.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)

        robot = PandaAllegro(
            sim=sim,
            block_gripper=False,
            base_position=np.array([-0.6, 0.0, 0.0]),
            control_type=control_type,
        )

        task = PickAndPlace(
            sim=sim,
            reward_type=reward_type,
        )

        super().__init__(
            robot=robot,
            task=task,
        )