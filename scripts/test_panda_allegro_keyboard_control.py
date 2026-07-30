import time

import numpy as np
import pybullet as p

from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_allegro import PandaAllegro


def is_key_down(keys, key):
    return key in keys and keys[key] & p.KEY_IS_DOWN


def is_any_key_down(keys, key_list):
    for key in key_list:
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            return True
    return False


def main():
    sim = PyBullet(render_mode="human")

    robot = PandaAllegro(
        sim=sim,
        block_gripper=False,
        base_position=None,
        control_type="ee",
    )

    robot.reset()

    print("=" * 80)
    print("PandaAllegro keyboard control")
    print("=" * 80)
    print("W / S : +X / -X")
    print("D / A : +Y / -Y")
    print("Q / E : +Z / -Z")
    print("O / 0 : open / reverse Allegro fingers")
    print("P     : close / curl Allegro fingers")
    print("R     : reset")
    print("X     : exit")
    print("ESC   : exit, if supported by current PyBullet build")
    print("=" * 80)
    print("Click the PyBullet window first, then press keys.")

    action = np.zeros(19, dtype=np.float32)

    # 不同 PyBullet / Windows 版本里 ESC key code 可能不同
    esc_key_codes = [27, 65307]
    if hasattr(p, "B3G_ESCAPE"):
        esc_key_codes.append(p.B3G_ESCAPE)

    try:
        while True:
            keys = p.getKeyboardEvents()

            action[:] = 0.0

            # Arm control
            if is_key_down(keys, ord("w")) or is_key_down(keys, ord("W")):
                action[0] = 0.4
            if is_key_down(keys, ord("s")) or is_key_down(keys, ord("S")):
                action[0] = -0.4

            if is_key_down(keys, ord("d")) or is_key_down(keys, ord("D")):
                action[1] = 0.4
            if is_key_down(keys, ord("a")) or is_key_down(keys, ord("A")):
                action[1] = -0.4

            if is_key_down(keys, ord("q")) or is_key_down(keys, ord("Q")):
                action[2] = 0.4
            if is_key_down(keys, ord("e")) or is_key_down(keys, ord("E")):
                action[2] = -0.4

            # Allegro hand control
            if (
                is_key_down(keys, ord("p"))
                or is_key_down(keys, ord("P"))
            ):
                action[3:] = 0.5

            if (
                is_key_down(keys, ord("o"))
                or is_key_down(keys, ord("O"))
                or is_key_down(keys, ord("0"))
            ):
                action[3:] = -0.5

            # Reset
            if is_key_down(keys, ord("r")) or is_key_down(keys, ord("R")):
                print("Reset robot.")
                robot.reset()
                time.sleep(0.3)

            # Exit
            if (
                is_key_down(keys, ord("x"))
                or is_key_down(keys, ord("X"))
                or is_any_key_down(keys, esc_key_codes)
            ):
                print("Exit.")
                break

            robot.set_action(action)
            sim.step()

            time.sleep(1.0 / 240.0)

    finally:
        sim.close()


if __name__ == "__main__":
    main()