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

### Run just the navigator

You must first install lerobot and place the sandy_navigator.py script inside the scripts folder.

```
python lerobot/scripts/sandy_navigator.py --duration 0.25 --camera "/dev/device2/"
```


## Components

- `speech_to_text.py`: Client for OpenAI's Whisper API for speech recognition
- `eleven_labs_client.py`: Client for ElevenLabs text-to-speech API
- `mistral_client.py`: Client for Mistral AI's language models
- `beach_robot_agent.py`: Main agent integrating all components

## Extending the Project

To extend the project with object detection and robot control:

1. Implement the object detection module using a segmentation model
2. Integrate with the robot's movement controls
3. Implement the trash pickup policy and arm movement
4. Connect everything through the `start_cleaning()` method in the `BeachRobotAgent` class

## License

This project is open source and available under the MIT License. 
