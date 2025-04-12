import time

# Import functions from other modules
from perception import speech_to_text, detect_trash
from communication import process_with_llm, text_to_speech
from robot_control import move_robot
# Assume necessary imports for LeRobot policy

def run_conversation_phase():
    """Handles the initial conversation with the user."""
    print("--- Conversation Phase ---")
    while True:
        user_text = speech_to_text()
        if not user_text:
            continue
        llm_response, start_cleaning = process_with_llm(user_text)
        text_to_speech(llm_response)
        if start_cleaning:
            print("Cleaning phase triggered!")
            break
    print("--- End Conversation Phase ---")

def run_object_detection_phase():
    """Looks for trash using the segmentation model."""
    print("--- Object Detection Phase ---")
    trash_coords = None
    while trash_coords is None:
        # Move the robot in a search pattern if needed
        move_robot(action="search")
        trash_coords = detect_trash()
        if trash_coords:
            break
        # Add a small delay or other logic if needed
        time.sleep(0.5)
    print("--- End Object Detection Phase ---")
    return trash_coords

def run_approach_trash_phase(trash_coords):
    """Moves the robot to the detected trash."""
    print("--- Approach Trash Phase ---")
    while True:
        reached = move_robot(target_location=trash_coords, action="approach")
        if reached:
            print("Successfully approached trash.")
            break
        # Optional: Add logic if approach fails or needs adjustment
        time.sleep(0.5)
        # Update trash_coords if necessary (e.g., re-detect)
        # trash_coords = detect_trash()
        # if not trash_coords: # Trash lost?
        #    print("Lost sight of trash, going back to detection.")
        #    return False # Indicate failure/need to re-detect
    print("--- End Approach Trash Phase ---")
    return True # Indicate success

def run_arm_preparation_phase():
    """Moves the arm to the position required for pickup."""
    print("--- Arm Preparation Phase ---")
    move_robot(action="pickup_ready_position")
    print("Arm is in position for pickup.")
    print("--- End Arm Preparation Phase ---")

def run_pickup_phase():
    """Executes the LeRobot policy to pick up the trash."""
    print("--- Pickup Phase ---")
    print("Triggering LeRobot pickup policy...")
    # Placeholder: Execute LeRobot policy
    time.sleep(3)
    pickup_successful = True # Simulate result
    if pickup_successful:
        print("LeRobot policy executed successfully. Trash picked up.")
    else:
        print("LeRobot policy failed.")
    print("--- End Pickup Phase ---")
    return pickup_successful

def run_dispose_phase():
    """Moves the arm to dispose of the trash in the basket."""
    print("--- Dispose Phase ---")
    move_robot(action="dispose_trash")
    print("Trash disposed.")
    print("--- End Dispose Phase ---")

def run_reset_phase():
    """Returns the robot and arm to initial positions."""
    print("--- Reset Phase ---")
    # Move arm back first
    move_robot(action="initial_arm_position")
    # Then potentially move the base if needed (user mentioned central position)
    # move_robot(action="return_to_center") # Optional depending on desired flow
    print("Robot reset to initial state for next cycle.")
    print("--- End Reset Phase ---") 