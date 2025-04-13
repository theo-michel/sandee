from lerobot.common.robot_devices.robots.configs import LeKiwiRobotConfig
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.robots.mobile_manipulator import MobileManipulator
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.configs import FeetechMotorsBusConfig
import time
import numpy as np
import argparse
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect


from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from lerobot.common.robot_devices.control_utils import log_control_info

def execute_command(robot: MobileManipulator, command: str, duration: float = 2.0):
    """
    Execute a movement command on the robot.
    
    Args:
        robot: LeKiwi robot instance
        command: Movement command ("forward", "backward", "left", "right", "rotate_left", "rotate_right")
        duration: Duration of the movement in seconds
    """

    if command == "forward":
        robot.pressed_keys["forward"] = True
    elif command == "backward":
        robot.pressed_keys["backward"] = True
    elif command == "left":
        robot.pressed_keys["left"] = True
    elif command == "right":
        robot.pressed_keys["right"] = True
    elif command == "rotate_left":
        robot.pressed_keys["rotate_left"] = True
    elif command == "rotate_right":
        robot.pressed_keys["rotate_right"] = True

    robot.teleop_step()

    time.sleep(duration)

    robot.pressed_keys["forward"] = False
    robot.pressed_keys["backward"] = False
    robot.pressed_keys["left"] = False
    robot.pressed_keys["right"] = False
    robot.pressed_keys["rotate_left"] = False
    robot.pressed_keys["rotate_right"] = False
    robot.teleop_step()
    
def replay_commands(robot: MobileManipulator, move: str, duration: float = 2.0,):
    dataset = LeRobotDataset(f"theo-michel/{move}", root=None, episodes=[0])
    actions = dataset.hf_dataset.select_columns("action")
    if not robot.is_connected:
        robot.connect()
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / 30 - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=30)

def main():

    lekiwi_config = LeKiwiRobotConfig()
    mobile_manipulator = make_robot_from_config(lekiwi_config)

    mobile_manipulator.connect()

    commands = ["rotate_left", "forward", "rotate_right", "forward"]
    duration = 2.0

    try:
        for command in commands:
            print(f"Executing command: {command}")
            execute_command(mobile_manipulator, command, duration)
        replay_commands(mobile_manipulator, commands, duration)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        mobile_manipulator.disconnect()

if __name__ == "__main__":
    main()
