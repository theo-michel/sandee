# Disclaimer:
This is the code for the Voice and grasping of the can.
We are running the policy running a bash file that we put in our Lerobot folder because we couldn't make it work in code in time.

The movement policy was running on the raspberry pie in local, we will add it to the folder soon. There also was a lot of modifications to make it work on the LeRobot folder to make it work.

The robot is also scanning for vulnerabilities in the network as we where thinking that on the beach there are a lot of vulnerabilities, but we didn't put in on the demo.

I don't think the project is runnable easily in it's entirety, please reach out if you need help or want to try. We will at least push all the used code to the folder today or tomorow. And then try to make it runnable all together one day maybe (we don't have the robot at home so hard to test)

# SANDEE - Beach Cleaning Robot

SANDEE (Smart Autonomous Navigation and Debris Extraction Entity) is a beach-cleaning robot built for a hackathon. It combines speech recognition, natural language processing, and text-to-speech capabilities to create an interactive experience while cleaning beaches.

## Features

- Voice interaction using OpenAI's Whisper for speech recognition and ElevenLabs for text-to-speech
- Natural conversation using Mistral AI's language models
- Automatic switching between conversation mode and cleaning mode
- Object detection to identify and collect trash on beaches

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up API keys as environment variables:
   ```
   export OPENAI_API_KEY=your_openai_api_key
   export ELEVEN_LABS_API_KEY=your_elevenlabs_api_key
   export MISTRAL_API_KEY=your_mistral_api_key
   ```
   
   Alternatively, you can pass the API keys directly when creating the agent.

## Usage

Run the beach robot agent:

```
python beach_robot_agent.py
```

The robot will:
1. Greet you and engage in conversation
2. Listen for your voice input
3. Process your input using Mistral AI
4. Respond through ElevenLabs text-to-speech
5. Eventually transition to cleaning mode when appropriate
6. Detect and collect trash on the beach

### Run just the navigator (movement)

You must first install lerobot and place the sandy_navigator.py script inside the scripts folder.

```
python lerobot/scripts/sandy_navigator.py --duration 0.25 --camera "/dev/device2/"
```

You can find the code & algorithm for the movement inside the sandy_navigator.py file in the analyze_image_for_sand(frame) function.


## Components

- `speech_to_text.py`: Client for OpenAI's Whisper API for speech recognition
- `eleven_labs_client.py`: Client for ElevenLabs text-to-speech API
- `mistral_client.py`: Client for Mistral AI's language models
- `beach_robot_agent.py`: Main agent integrating all components
- `sandy_navigator.py`: Handles the movement and object detection.


### Movement Control
Here is a brief overview of the movement algorithm:
1. First identifies the predominant shade of yellow in the bottom 20% of the image (floor), then detects similar colors throughout the image.
2. The image is split horizontally into left and right halves.
3. Calculate the ratio of these pixels in each half.
4. Compare ratios and make decision

Ratio Decision Making:
- If right side has almost no yellow (below minimum threshold) AND left has some, rotato left
- If left side has significantly more yellow than right side, rotate left
- If both sides have minimal sand content, rotate left
- Otherwise, go right

## Extending the Project

To extend the project with object detection and robot control:

1. Implement the object detection module using a segmentation model
2. Integrate with the robot's movement controls
3. Implement the trash pickup policy and arm movement
4. Connect everything through the `start_cleaning()` method in the `BeachRobotAgent` class

## License

This project is open source and available under the MIT License. 
