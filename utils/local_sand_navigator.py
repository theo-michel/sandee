#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import numpy as np
import torch

from lerobot.common.robot_devices.robots.configs import FeetechMotorsBusConfig 
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.mobile_manipulator import LeKiwi


def direct_wheel_command(robot: LeKiwi, command: str, duration: float = 2.0):
    """
    Directly control the robot's wheels without using network communication.
    
    Args:
        robot: LeKiwi instance
        command: Movement command ("forward", "backward", "left", "right", "rotate_left", "rotate_right")
        duration: Duration of the movement in seconds
    """
    # Movement parameters in body frame
    xy_speed = 0.2  # m/s
    theta_speed = 60  # deg/s
    
    # Initialize movement values
    x_cmd = 0.0  # m/s lateral
    y_cmd = 0.0  # m/s forward/backward
    theta_cmd = 0.0  # deg/s rotation
    
    # Set movement based on command
    if command == "forward":
        y_cmd = xy_speed
        print(f"Moving FORWARD at {y_cmd} m/s")
    elif command == "backward":
        y_cmd = -xy_speed
        print(f"Moving BACKWARD at {-y_cmd} m/s")
    elif command == "left":
        x_cmd = xy_speed
        print(f"Moving LEFT at {x_cmd} m/s")
    elif command == "right":
        x_cmd = -xy_speed
        print(f"Moving RIGHT at {-x_cmd} m/s")
    elif command == "rotate_left":
        theta_cmd = theta_speed
        print(f"Rotating LEFT at {theta_cmd} deg/s")
    elif command == "rotate_right":
        theta_cmd = -theta_speed
        print(f"Rotating RIGHT at {-theta_cmd} deg/s")
    else:
        print(f"Unknown command: {command}")
        return
    
    # Convert body motion to wheel velocities using kinematics
    # Wheel mounting angles (defined from y axis cw): 300°, 180°, 60°
    angles = np.radians(np.array([300, 180, 60]))
    
    # Base radius (m) - distance from robot center to wheels
    base_radius = 0.125
    
    # Wheel radius (m)
    wheel_radius = 0.05
    
    # Convert theta from deg/s to rad/s
    theta_rad = theta_cmd * (np.pi / 180.0)
    
    # Velocity vector [x, y, theta]
    velocity_vector = np.array([x_cmd, y_cmd, theta_rad])
    
    # Kinematic matrix: maps body velocities to wheel linear speeds
    kin_matrix = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
    
    # Calculate wheel linear and angular speeds
    wheel_linear_speeds = kin_matrix.dot(velocity_vector)
    wheel_angular_speeds_rad = wheel_linear_speeds / wheel_radius
    
    # Convert to deg/s
    wheel_angular_speeds_deg = wheel_angular_speeds_rad * (180.0 / np.pi)
    
    # Convert to Feetech motor command format
    steps_per_deg = 4096.0 / 360.0
    raw_speeds = []
    
    for speed_deg in wheel_angular_speeds_deg:
        speed_steps = abs(speed_deg) * steps_per_deg
        speed_int = int(round(speed_steps))
        
        # Cap to max value
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF
            
        # Add direction bit
        if speed_deg < 0:
            speed_int = speed_int | 0x8000
        else:
            speed_int = speed_int & 0x7FFF
            
        raw_speeds.append(speed_int)
    
    print(f"Wheel speeds (raw): Left={raw_speeds[0]}, Back={raw_speeds[1]}, Right={raw_speeds[2]}")
    
    # Set velocity directly
    robot.set_velocity(raw_speeds)
    
    # Run for duration
    print(f"Running for {duration} seconds...", end="", flush=True)
    time.sleep(duration)
    print(" Done.")
    
    # Stop the robot
    print("Stopping robot")
    robot.stop()


def execute_command_sequence(robot: LeKiwi, commands: list, duration_per_command: float = 2.0):
    """Execute a sequence of movement commands"""
    print("\n===== Starting command sequence =====")
    
    for cmd in commands:
        print(f"\n----- Executing command: {cmd} -----")
        direct_wheel_command(robot, cmd, duration_per_command)
        time.sleep(0.5)  # Pause between commands
    
    print("\n===== Command sequence completed =====")


def main():
    try:
        # Create motors bus configuration for the wheels
        motors_config = {
            "wheels": FeetechMotorsBusConfig(
                port="/dev/ttyACM0",  # Raspberry Pi port
                motors={
                    "left_wheel": (7, "sts3215"),
                    "back_wheel": (8, "sts3215"),
                    "right_wheel": (9, "sts3215"),
                }
            )
        }
        
        # Create the motors bus - note the function is plural (buses)
        print("Creating motors bus...")
        motors_buses = make_motors_buses_from_configs(motors_config)
        motor_bus = motors_buses["wheels"]
        motor_bus.connect()
        
        # Create LeKiwi robot controller
        print("Initializing LeKiwi controller...")
        robot = LeKiwi(motor_bus)
        
        # Execute test sequence
        test_sequence = ["forward", "forward", "backward", "rotate_right"]
        execute_command_sequence(robot, test_sequence, duration_per_command=3.0)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop the robot if we have a robot instance
        motor_bus.disconnect()
        if 'robot' in locals():
            print("Ensuring robot is stopped")
            robot.stop()


if __name__ == "__main__":
    main()
