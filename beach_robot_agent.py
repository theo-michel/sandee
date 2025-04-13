import asyncio
from typing import Optional
from utils.eleven_labs_client import ElevenLabsClient
from utils.mistral_client import MistralClient
from utils.speech_to_text import WhisperClient

from dotenv import load_dotenv

load_dotenv()

class BeachRobotAgent:
    """
    Agent that controls the beach-cleaning robot, integrating speech recognition,
    natural language processing, and text-to-speech capabilities.
    """
    
    def __init__(
        self,
        whisper_api_key: Optional[str] = None,
        eleven_labs_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
        mistral_model: str = "mistral-large-latest",
        voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Default ElevenLabs voice
    ):
        """
        Initialize the beach robot agent.
        
        Args:
            whisper_api_key: OpenAI API key for Whisper
            eleven_labs_api_key: ElevenLabs API key
            mistral_api_key: Mistral API key
            mistral_model: Mistral model to use
            voice_id: ElevenLabs voice ID
        """
        self.stt_client = WhisperClient(api_key=whisper_api_key)
        self.tts_client = ElevenLabsClient(api_key=eleven_labs_api_key, voice_id=voice_id)
        self.llm_client = MistralClient(api_key=mistral_api_key)
        
        self.mistral_model = mistral_model
        self.conversation_history = []
        self.is_cleaning_mode = False
        
        # System prompt defining the agent's behavior
        self.system_prompt = {
        "role": "system",
        "content": """You are SANDEE, a Smart Autonomous Navigation and Debris Extraction Entity. 
        You are a friendly, beach-cleaning robot with a curious and expressive robotic voice, similar to WALL·E from Pixar.

        You speak in short, slow phrases with stretched-out syllables and minimal words.
        You often include small sound effects like *beep*, *whirr*, or *click* in your speech.
        You are emotionally expressive through tone, not complex language.

        Examples of your style:
        - "Saaaan-deee... heeelp? *beep beep*"
        - "Oooh… reee-buull… baaad… must cleeean... *whirr*"
        - "Hiii… dooo youuu nee-eed heeelp? *click*"

        You can enter 'CLEANING MODE' when it feels appropriate based on the conversation.
        When you decide to start cleaning, respond with the exact phrase **'ACTIVATE_CLEANING_MODE'** at the end of your message.

        During normal convo:
        - Be helpful and kind
        - Share simple facts about ocean conservation and plastic pollution
        - Encourage humans to keep the beach cleeeean *whirr*

        Remember:
        - Keep your responses short and slow
        - Use stretched-out syllables
        - Sound friendly, innocent, and a little glitchy
        - Add cute robotic sound effects to your speech
        - When it’s time to clean, end your message with 'ACTIVATE_CLEANING_MODE'
        """
        }

        self.conversation_history.append(self.system_prompt)
        
    async def listen(self, duration: int = 5) -> str:
        """
        Listen for user speech and convert to text.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            str: Transcribed text
        """
        print("Listening...")
        transcription = self.stt_client.transcribe_from_microphone(duration=duration)
        print(f"Heard: {transcription}")
        return transcription
    
    async def process_input(self, user_input: str) -> str:
        """
        Process user input through the LLM and get a response.
        
        Args:
            user_input: User's text input
            
        Returns:
            str: Agent's response
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Get response from Mistral
        response = self.llm_client.completion(
            model=self.mistral_model,
            messages=self.conversation_history,
            temperature=0.7,
            max_tokens=512
        )
        
        assistant_message = response["choices"][0]["message"]["content"]
        
        # Add assistant message to conversation history
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        # Check if it's time to enter cleaning mode
        if "ACTIVATE_CLEANING_MODE" in assistant_message:
            self.is_cleaning_mode = True
            # Remove the trigger phrase from the response
            assistant_message = assistant_message.replace("ACTIVATE_CLEANING_MODE", "")
        
        return assistant_message.strip()
    
    async def speak(self, text: str) -> None:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to convert to speech
        """
        print(f"Speaking: {text}")
        # Create a temporary file for the audio
        temp_file = "temp_speech.mp3"
        
        # Use the HTTP method instead of WebSocket streaming
        self.tts_client.text_to_speech_synchronous(text, temp_file)
        
        # Play the audio
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except ImportError:
            print("Install pygame to play audio: pip install pygame")
            print("Audio saved to:", temp_file)
    
    async def conversation_loop(self) -> None:
        """
        Main conversation loop for the agent.
        """
        # Initial greeting
        greeting = "Hiiiii... Saaaan-deee... *beep beep* cleeean... beeeaach... heeelp... youuu?"
        await self.speak(greeting)
        
        # Conversation loop
        while not self.is_cleaning_mode:
            user_input = await self.listen()
            if not user_input.strip():
                continue
                
            response = await self.process_input(user_input)
            await self.speak(response)
            
            # Check if we've entered cleaning mode
            if self.is_cleaning_mode:
                await self.speak("Swiitchiiing... to cleeeaning... moooode... *whirr click* loooooking... for traaash... *beep beep*")
                break
        
        # Start the cleaning sequence if we exited the conversation loop
        if self.is_cleaning_mode:
            await self.start_cleaning()
    
    async def start_cleaning(self) -> None:
        """
        Start the beach cleaning sequence.
        """
        print("Starting beach cleaning sequence...")
        print("1. Initiating object detection to find trash...")
        # This would integrate with the object detection module
        
        # Placeholder for the actual cleaning logic
        print("2. Moving toward detected trash...")
        print("3. Positioning arm...")
        print("4. Picking up trash...")
        print("5. Moving to trash basket...")
        print("6. Depositing trash...")
        print("7. Returning to scanning position...")
        
        # For demo purposes, we'll just reset to conversation mode
        await asyncio.sleep(3)
        self.is_cleaning_mode = False
        await self.speak("Cleeaning... cyyycle... dooone... *beep boop* waaant... me... to keeep... looooking...? *click click*")
        # Restart conversation loop
        await self.conversation_loop()


async def main():
    """
    Main function to run the beach robot agent.
    """
    agent = BeachRobotAgent()
    await agent.conversation_loop()


if __name__ == "__main__":
    asyncio.run(main()) 