

# Import phase functions
from sandee.phases import (
    run_conversation_phase,
    run_object_detection_phase,
    run_approach_trash_phase,
    run_arm_preparation_phase,
    run_pickup_phase,
    run_dispose_phase,
    run_reset_phase
)
# Import utility functions if needed directly (e.g., for initialization outside phases)
from sandee.robot_control import move_robot # Keep if move_robot needed directly in main

# Assume necessary imports for Kiwi, ElevenLabs, Mistral, Segmentation Model, LeRobot, Camera, Robot Control

# Placeholder function for external APIs/hardware - replace with actual implementations
def initialize_systems():
    """Initialize all required components."""
    print("Initializing systems...")
    # Initialize ElevenLabs client
    # Initialize Mistral client
    # Initialize Camera
    # Initialize Robot Control (Kiwi)
    # Load Segmentation Model
    # Load LeRobot Policy
    print("Systems Initialized.")
    return True # Indicate success

def main():
    """Main function to run the robot's operations."""
    if not initialize_systems():
        print("Failed to initialize systems. Exiting.")
        return

    # Start with conversation
    run_conversation_phase()

    # Main cleaning loop
    while True:
        print("\n=== Starting New Cleaning Cycle ===")

        # 1. Detect Trash
        trash_location = run_object_detection_phase()
        if not trash_location: # Should not happen if detection phase loops, but as safeguard
             print("Error: Detection phase finished without finding trash.")
             break # Or implement recovery logic

        # 2. Approach Trash
        success = run_approach_trash_phase(trash_location)
        if not success:
            print("Failed to approach trash, restarting cycle.")
            continue # Go back to detection

        # 3. Prepare Arm
        run_arm_preparation_phase()

        # 4. Pick up Trash
        picked_up = run_pickup_phase()
        if not picked_up:
            print("Failed to pick up trash, resetting arm and restarting cycle.")
            # Explicitly call move_robot here if needed before restarting cycle
            move_robot(action="initial_arm_position") # Ensure arm is reset
            continue # Go back to detection

        # 5. Dispose Trash
        run_dispose_phase()

        # 6. Reset for next cycle
        run_reset_phase()

        # Optional: Add a condition to stop looping (e.g., battery low, user command, timer)
        # should_continue = check_continue_condition()
        # if not should_continue:
        #    print("Stopping cleaning cycles.")
        #    break

if __name__ == "__main__":
    main()




