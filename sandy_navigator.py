#!/usr/bin/env python

import time
import numpy as np
import cv2
import torch
import signal
import argparse
import os
import sys
import datetime

from lerobot.common.robot_devices.robots.configs import FeetechMotorsBusConfig 
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.mobile_manipulator import LeKiwi

import mistralai

import os
from PIL import Image
import base64
from io import BytesIO
from mistralai import Mistral

client = mistralai.Mistral(
    api_key = "")

def analyze_image_in_context(image, client: Mistral, model_name="mistral-small-latest"):
    try:
        # Calculate new dimensions while preserving aspect ratio
        width, height = image.size
        max_dim = 1024
        
        if width > max_dim or height > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        print(f"Loaded image with size: {image.size}")
        
        # Convert the image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Load few-shot learning example images
        examples = []
        few_shot_dir = "/home/raspberrypi/lerobot/lerobot/images_vlm"
        example_images = [
            {"path": "image_0_1.jpg", "label": 1},
            {"path": "image_1_0.jpg", "label": 0},
            {"path": "image_2_1.jpg", "label": 1},
            {"path": "image_3_0.jpg", "label": 0}
        ]
        
        for example in example_images:
            try:
                example_path = os.path.join(few_shot_dir, example["path"])
                example_img = Image.open(example_path)
                
                # Resize example images if needed
                width, height = example_img.size
                max_dim = 1024
                if width > max_dim or height > max_dim:
                    if width > height:
                        new_width = max_dim
                        new_height = int(height * (max_dim / width))
                    else:
                        new_height = max_dim
                        new_width = int(width * (max_dim / height))
                    example_img = example_img.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert example image to base64
                example_buffered = BytesIO()
                example_img.save(example_buffered, format="JPEG")
                example_img_str = base64.b64encode(example_buffered.getvalue()).decode()
                
                examples.append({
                    "image": f"data:image/jpeg;base64,{example_img_str}",
                    "label": "yes" if example["label"] == 1 else "no"
                })
            except Exception as e:
                print(f"Error loading example image {example['path']}: {str(e)}")

        # Refined System Prompt: Focus on the specific task and output format
        system_prompt = (
            "Your task is to determine if a specific object, a grey/blue metallic RedBull can, "
            "is the closest object to the camera in the provided image. "
            "Ignore natural objects like seaweed, shells, crabs, or rocks when determining closeness. "
            "Respond ONLY with 'yes' or 'no' in lowercase without punctuation."
        )
        
        # Refined User Prompt: Structure examples clearly and ask the specific question
        user_content = [
            {"type": "text", "text": "Analyze the following examples to understand the task, then evaluate the final image."}
        ]
        
        # Add structured examples to user content
        for idx, example in enumerate(examples):
            # Provide reasoning specific to the task
            reasoning = "The RedBull can (grey/blue metallic object) is visible and closest." if example['label'] == 'yes' else "The RedBull can is either not visible or not the closest object."
            user_content.extend([
                {"type": "text", "text": f"--- Example {idx+1} ---"},
                {"type": "image_url", "image_url": {"url": example["image"]}},
                {"type": "text", "text": f"Reasoning: {reasoning}"},
                {"type": "text", "text": f"Answer: {example['label']}"}
            ])
        
        # Add the final query image and question
        user_content.extend([
             {"type": "text", "text": "--- End of Examples ---"},
             {"type": "text", "text": "Now, analyze this new image:"},
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}},
             {"type": "text", "text": "Is the RedBull can the closest object to the camera in this image?"}
        ])
        
        # Make API call with refined prompts
        chat_response = client.chat.complete(
            model=model_name,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,  # Adjust temperature for creativity
        )
        
        # Extract response
        response = chat_response.choices[0].message.content
        return response
        
    except Exception as e:
        return f"Error processing image: {str(e)}"


def direct_wheel_command(robot: LeKiwi, command: str, duration: float = 0.2):
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


def analyze_image_for_sand(frame, save_debug=False):
    """
    Analyze the image to detect sand (yellow pixels) and determine the ratio in each half.
    The image is split horizontally into left and right halves.
    First identifies the predominant shade of yellow in the bottom 20% of the image (floor),
    then detects similar colors throughout the image.
    
    Args:
        frame: The video frame to analyze
        save_debug: Whether to save debug images
        
    Returns:
        tuple: (left_ratio, right_ratio, decision)
    """
    # Make sure the frame is valid, if not no action
    if frame is None or frame.size == 0:
        print("Empty frame received!")
        return 0, 0, "right"

    # # if image is too bright, i.e. most of the pixels are white, stop running the script
    if np.mean(frame) > 240:
        print("Image is too bright, stopping...")
        return 0, 0, "bright"
    
    
    # Create debug directory if needed
    debug_dir = "debug_images"
    if save_debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    # Generate timestamp for debug images
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Extract the bottom 20% of the image (floor area)
    floor_region_start = int(height * 0.8)
    floor_region = frame[floor_region_start:, :]
    
    # Save floor region if debugging
    if save_debug:
        cv2.imwrite(f"{debug_dir}/floor_region_{timestamp}.jpg", floor_region)
    
    # Analyze floor region to determine predominant yellow shade
    # Convert to HSV for better color analysis
    floor_hsv = cv2.cvtColor(floor_region, cv2.COLOR_BGR2HSV)
    
    # Define a broad yellow range to start with
    lower_yellow_broad = np.array([15, 50, 50])
    upper_yellow_broad = np.array([35, 255, 255])
    
    # Create a mask for yellow pixels in the floor
    yellow_mask_floor = cv2.inRange(floor_hsv, lower_yellow_broad, upper_yellow_broad)
    
    # Check if we have any yellow pixels at all
    yellow_pixel_count = np.sum(yellow_mask_floor > 0)
    if yellow_pixel_count == 0:
        print("No yellow pixels found in floor region!")
        if save_debug:
            cv2.imwrite(f"{debug_dir}/floor_yellow_mask_{timestamp}.jpg", yellow_mask_floor)
        return 0, 0, "right"  # Default to moving right
    
    # Extract the HSV values of yellow pixels
    yellow_pixels_hsv = floor_hsv[yellow_mask_floor > 0]
    
    # Calculate the average HSV values for yellow pixels
    avg_h = np.mean(yellow_pixels_hsv[:, 0])
    avg_s = np.mean(yellow_pixels_hsv[:, 1])
    avg_v = np.mean(yellow_pixels_hsv[:, 2])
    
    # Define a more specific yellow range around the average values
    h_range = 5  # Allow some variation in hue
    s_range = 50  # Allow more variation in saturation
    v_range = 50  # Allow more variation in value
    
    lower_yellow = np.array([max(0, avg_h - h_range), max(0, avg_s - s_range), max(0, avg_v - v_range)])
    upper_yellow = np.array([min(180, avg_h + h_range), min(255, avg_s + s_range), min(255, avg_v + v_range)])
    
    print(f"Detected predominant yellow in floor: HSV=({avg_h:.1f}, {avg_s:.1f}, {avg_v:.1f})")
    print(f"Using HSV range: {lower_yellow} to {upper_yellow}")
    
    # Save floor yellow detection if debugging
    if save_debug:
        # Visualize detected yellow in floor
        detected_yellow = np.zeros_like(floor_region)
        detected_yellow[yellow_mask_floor > 0] = floor_region[yellow_mask_floor > 0]
        cv2.imwrite(f"{debug_dir}/floor_yellow_detected_{timestamp}.jpg", detected_yellow)
    
    # Now split the image into left and right halves
    left_half = frame[:, :width//2]
    right_half = frame[:, width//2:]
    
    # Save original halves if debugging
    if save_debug:
        cv2.imwrite(f"{debug_dir}/orig_full_{timestamp}.jpg", frame)
        cv2.imwrite(f"{debug_dir}/orig_left_{timestamp}.jpg", left_half)
        cv2.imwrite(f"{debug_dir}/orig_right_{timestamp}.jpg", right_half)
    
    # Process each half to detect the specific yellow shade
    def process_half(half_img, name):
        # Convert to HSV
        half_hsv = cv2.cvtColor(half_img, cv2.COLOR_BGR2HSV)

        # Create mask using the specific yellow range determined from the floor
        yellow_mask = cv2.inRange(half_hsv, lower_yellow, upper_yellow)
        
        # Count yellow pixels
        yellow_pixels = np.sum(yellow_mask > 0)
        total_pixels = half_img.shape[0] * half_img.shape[1]
        
        # Calculate ratio
        ratio = yellow_pixels / total_pixels if total_pixels > 0 else 0
        
        # Save debug images
        if save_debug:
            # Save the mask
            cv2.imwrite(f"{debug_dir}/{name}_yellow_mask_{timestamp}.jpg", yellow_mask)
            
            # Create visual representation of detected yellow
            detected_yellow = np.zeros_like(half_img)
            detected_yellow[yellow_mask > 0] = half_img[yellow_mask > 0]
            cv2.imwrite(f"{debug_dir}/{name}_yellow_detected_{timestamp}.jpg", detected_yellow)
            
            # Save with additional info text
            info_img = half_img.copy()
            text = f"Yellow pixels: {yellow_pixels}, Total: {total_pixels}, Ratio: {ratio:.4f}"
            cv2.putText(info_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(f"{debug_dir}/{name}_with_info_{timestamp}.jpg", info_img)
            
            # Print detailed info
            print(f"{name} half - Yellow pixels: {yellow_pixels}, Total pixels: {total_pixels}, Ratio: {ratio:.4f}")
        
        return ratio
    
    # Process both halves
    left_ratio = process_half(left_half, "left")
    right_ratio = process_half(right_half, "right")
    
    # Make decision based on sand ratios with more robust logic:
    # 1. If right side has almost no yellow (below minimum threshold) AND left has some, turn left
    # 2. If left side has significantly more yellow than right side, turn left
    # 3. If both sides have minimal sand content, turn left
    # 4. Otherwise, go right
    
    # Parameters for decision
    min_yellow_threshold = 0.02  # Minimum ratio to consider side has any meaningful yellow
    relative_ratio_threshold = 1.2  # Left needs to be this times more yellow than right
    
    # Decision logic
    if (right_ratio < min_yellow_threshold and left_ratio > min_yellow_threshold):
        decision = "rotate_left"
        print(f"RIGHT HAS ALMOST NO YELLOW ({right_ratio:.4f}) - LEFT HAS SOME ({left_ratio:.4f}) - Rotating left")
    elif left_ratio > right_ratio * relative_ratio_threshold:
        decision = "rotate_left"
        print(f"LEFT HAS SIGNIFICANTLY MORE YELLOW ({left_ratio:.4f} vs {right_ratio:.4f}) - Rotating left")
    elif (left_ratio < min_yellow_threshold and right_ratio < min_yellow_threshold):
        decision = "rotate_left"
        print(f"BOTH SIDES HAVE MINIMAL SAND ({left_ratio:.4f}, {right_ratio:.4f}) - Rotating left")
    else:
        decision = "right"
        print(f"Going right - Left: {left_ratio:.4f}, Right: {right_ratio:.4f}")
    
    # Save a decision visualization if debugging
    if save_debug:
        decision_img = frame.copy()
        if decision == "rotate_left":
            color = (0, 0, 255)  # Red for rotate
            text = "ROTATING LEFT"
        else:
            color = (0, 255, 0)  # Green for moving right
            text = "MOVING RIGHT"
        
        # Add detailed ratio information
        cv2.putText(decision_img, text, (width//2-100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(decision_img, f"Left: {left_ratio:.4f}, Right: {right_ratio:.4f}", 
                   (width//2-150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Add colored boxes showing ratio on each side
        left_box_height = int(height * min(left_ratio * 5, 0.9))  # Scale ratio for visualization
        right_box_height = int(height * min(right_ratio * 5, 0.9))
        cv2.rectangle(decision_img, (10, height-left_box_height), (width//4, height-10), (255, 0, 0), -1)
        cv2.rectangle(decision_img, (width-width//4, height-right_box_height), (width-10, height-10), (255, 0, 0), -1)
        
        # Save the decision visualization
        cv2.imwrite(f"{debug_dir}/decision_{timestamp}.jpg", decision_img)
    
    return left_ratio, right_ratio, decision


# Setup signal handling for clean exit with Ctrl+C
running = True
def signal_handler(sig, frame):
    global running
    print("Signal received, stopping...")
    running = False
signal.signal(signal.SIGINT, signal_handler)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sand Navigator for LeKiwi Robot')
    parser.add_argument('--camera', type=str, default=None, help='Camera device (e.g., /dev/video0, /dev/video1)')
    parser.add_argument('--camera-index', type=int, default=None, help='Camera index (e.g., 0, 1)')
    parser.add_argument('--duration', type=float, default=0.5, help='Duration of each movement command')
    parser.add_argument('--interval', type=int, default=15, help='Frame interval for making decisions')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0', help='Serial port for motor bus')
    parser.add_argument('--debug-only', action='store_true', help='Only process one frame and save debug images, then exit')
    args = parser.parse_args()
    
    cap = None
    motor_bus = None
    robot = None

    # image_history = []
    # action_history = []
    # action_history_size = 4

    
    try:
        # Initialize video capture from camera
        print("Initializing camera...")
        
        # Try camera initialization methods in this order:
        # 1. Use specified device path if provided
        # 2. Use specified camera index if provided
        # 3. Try common device paths
        # 4. Try common indices
        
        # Method 1: Use specified device path
        if args.camera:
            print(f"Trying specified camera device: {args.camera}")
            if os.path.exists(args.camera):
                cap = cv2.VideoCapture(args.camera)
                if cap.isOpened():
                    print(f"Successfully opened camera: {args.camera}")
                else:
                    print(f"Failed to open specified camera device: {args.camera}")
                    cap = None
            else:
                print(f"Specified camera device does not exist: {args.camera}")
        
        # Method 2: Use specified camera index
        if cap is None and args.camera_index is not None:
            print(f"Trying camera index: {args.camera_index}")
            cap = cv2.VideoCapture(args.camera_index)
            if cap.isOpened():
                print(f"Successfully opened camera at index: {args.camera_index}")
            else:
                print(f"Failed to open camera at index: {args.camera_index}")
                cap = None
        
        # Method 3: Try common device paths
        if cap is None:
            common_devices = ['/dev/video0', '/dev/video1', '/dev/video2', 
                             '/dev/v4l/by-id/usb-Raspberry_Pi_Camera_1-video-index0']
            for device in common_devices:
                if os.path.exists(device):
                    print(f"Trying camera device: {device}")
                    cap = cv2.VideoCapture(device)
                    if cap.isOpened():
                        print(f"Successfully opened camera: {device}")
                        break
                    else:
                        print(f"Failed to open camera device: {device}")
                        cap = None
        
        # Method 4: Try common indices
        if cap is None:
            for idx in range(3):  # Try indices 0, 1, 2
                print(f"Trying camera index: {idx}")
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    print(f"Successfully opened camera at index: {idx}")
                    break
                else:
                    print(f"Failed to open camera at index: {idx}")
                    cap = None
        
        # If still no camera, raise exception
        if cap is None or not cap.isOpened():
            # Print diagnostic information
            import subprocess
            print("\nDiagnostic information:")
            try:
                print("\nAvailable video devices:")
                result = subprocess.run(['ls', '-la', '/dev/video*'], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE,
                                       text=True)
                print(result.stdout)
                print(result.stderr)
            except Exception as e:
                print(f"Could not list video devices: {e}")
            
            try:
                print("\nVideo4Linux device information:")
                result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE,
                                       text=True)
                print(result.stdout)
                print(result.stderr)
            except Exception as e:
                print(f"Could not get V4L2 device information: {e}")
                
            raise Exception("Could not open any video source")
        
        # Test camera by reading a frame
        ret, test_frame = cap.read()
        if not ret or test_frame is None or test_frame.size == 0:
            raise Exception("Camera opened but could not read frame")
        else:
            print(f"Successfully read test frame of size {test_frame.shape}")
        
        # Create motors bus configuration for the wheels
        motors_config = {
            "wheels": FeetechMotorsBusConfig(
                port=args.port,  # Use port from command line args
                motors={
                    "left_wheel": (7, "sts3215"),
                    "back_wheel": (8, "sts3215"),
                    "right_wheel": (9, "sts3215"),
                }
            )
        }
        
        # Create the motors bus
        print("Creating motors bus...")
        motors_buses = make_motors_buses_from_configs(motors_config)
        motor_bus = motors_buses["wheels"]
        motor_bus.connect()
        
        # Create LeKiwi robot controller
        print("Initializing LeKiwi controller...")
        robot = LeKiwi(motor_bus)

        def finish():
            for i in range(5):
                decision = "rotate_left"
                direct_wheel_command(robot, decision, duration=0.5)
            robot.stop()
            print("Found the trash!")

        
        # Parameters for control logic
        frame_count = 0
        vlm_interval = 8
        decision_interval = args.interval  # From command line args
        movement_duration = args.duration  # From command line args
        
        print("Starting video processing loop...")
        print("Press Ctrl+C to stop")
        
        # If debug-only mode, just process one frame and exit
        if args.debug_only:
            print("Running in debug-only mode...")
            ret, frame = cap.read()
            if ret:
                analyze_image_for_sand(frame, save_debug=True)
                print("Debug images saved. Exiting.")
                return
            else:
                print("Failed to capture frame for debugging.")
                return
        # use a fixed exposure
        cap.set(cv2.CAP_PROP_EXPOSURE, -4)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        # cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 6000) 
        global running
        while running: 
            frame_count+=1
            ret, frame = cap.read()
            if not ret:
                print("Failed to receive frame. Retrying...")
                # check if the camera is not connected
                if not cap.isOpened():
                    print("Camera is not connected. Exiting...")
                    break
            
            try:
                # Process frame and make decision every N frames
                if frame_count % decision_interval == 0:
                    if (frame_count // decision_interval) % vlm_interval == 0:
                        print("Frame Count")
                        # Convert frame to PIL image for VLM processing
                        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        
                        # Process the image with history context
                        response = analyze_image_in_context(pil_image, client)
                        print(f"VLM Response: {response}")

                        if response == "Yes":
                            print("Found the trash!")
                            finish()
                            break

                    # Only save debug images occasionally to avoid filling storage
                    save_debug = (frame_count % (decision_interval * 100) == 0)
                    _, _, decision = analyze_image_for_sand(frame, save_debug=save_debug)

                    if decision =="kill":
                        print("Received kill command, stopping...")
                        break

                    direct_wheel_command(robot, decision, duration=movement_duration)
                

            except Exception as e:
                print(f"Error processing frame: {e}")
            
            # Explicitly wait between frames to match camera frame rate 
            # (typically 30fps = ~33ms per frame)
            time.sleep(0.03)
                
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if cap is not None and cap.isOpened():
            cap.release()
        
        if motor_bus is not None:
            motor_bus.disconnect()
        
        if robot is not None:
            print("Ensuring robot is stopped")
            robot.stop()


            


if __name__ == "__main__":
    main()
