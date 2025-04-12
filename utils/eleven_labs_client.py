import os
import json
import asyncio
import websockets
import base64
from elevenlabs import ElevenLabs

class ElevenLabsClient:
    """
    Client for ElevenLabs text-to-speech streaming API using WebSockets.
    """
    
    def __init__(self, api_key=None, voice_id="21m00Tcm4TlvDq8ikWAM"):
        """
        Initialize the ElevenLabs client.
        
        Args:
            api_key (str): Your ElevenLabs API key. If None, reads from ELEVEN_LABS_API_KEY env variable.
            voice_id (str): The voice ID to use for text-to-speech.
        """
        self.api_key = api_key or os.environ.get("ELEVEN_LABS_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as ELEVEN_LABS_API_KEY environment variable")
        
        self.voice_id = voice_id
        self.base_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input"
        self.voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "speed": 1.0
        }
        
    async def text_to_speech(self, text, output_file=None, model_id="eleven_monolingual_v1"):
        """
        Convert text to speech asynchronously.
        
        Args:
            text (str): The text to convert to speech
            output_file (str, optional): Path to save the audio. If None, returns audio data.
            model_id (str): The model ID to use for synthesis
            
        Returns:
            bytes: The audio data if output_file is None
        """
        url = f"{self.base_url}?model_id={model_id}"
        audio_chunks = []
        
        async with websockets.connect(
            url,
            extra_headers={"xi-api-key": self.api_key}
        ) as websocket:
            # Initialize connection with voice settings
            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": self.voice_settings
            }))
            
            # Send the text
            await websocket.send(json.dumps({
                "text": text,
                "try_trigger_generation": True
            }))
            
            # Send empty text to indicate end of input
            await websocket.send(json.dumps({"text": ""}))
            
            # Receive audio chunks
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    if "audio" in data:
                        audio_chunk = base64.b64decode(data["audio"])
                        audio_chunks.append(audio_chunk)
                    elif "isFinal" in data:
                        # End of stream
                        break
                except websockets.exceptions.ConnectionClosed:
                    break
        
        # Combine all audio chunks
        audio_data = b''.join(audio_chunks)
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            return output_file
        
        return audio_data
    
    def text_to_speech_sync(self, text, output_file=None, model_id="eleven_monolingual_v1"):
        """
        Synchronous wrapper for text_to_speech.
        
        Args:
            text (str): The text to convert to speech
            output_file (str, optional): Path to save the audio. If None, returns audio data.
            model_id (str): The model ID to use for synthesis
            
        Returns:
            bytes: The audio data if output_file is None
        """
        return asyncio.run(self.text_to_speech(text, output_file, model_id))

    def text_to_speech_synchronous(self, text, output_file=None, model_id="eleven_monolingual_v1", output_format="mp3_44100_128"):
        """
        Convert text to speech using the ElevenLabs SDK.
        
        Args:
            text (str): The text to convert to speech
            output_file (str, optional): Path to save the audio. If None, returns audio data.
            model_id (str): The model ID to use for synthesis
            output_format (str): Output format of the generated audio
            
        Returns:
            bytes: The audio data if output_file is None, otherwise the path to the output file
        """
        # Initialize the SDK client
        client = ElevenLabs(api_key=self.api_key)
        
        # Generate audio - this returns a generator
        audio_generator = client.text_to_speech.convert(
            voice_id=self.voice_id,
            output_format=output_format,
            text=text,
            model_id=model_id,
        )
        
        # Convert generator to bytes
        audio_data = b''.join(chunk for chunk in audio_generator)
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            return output_file
        
        return audio_data


# Example usage
if __name__ == "__main__":
    client = ElevenLabsClient()
    
    # Synchronous usage
    result = client.text_to_speech_sync("Hello, this is a test of the ElevenLabs text-to-speech API.", "output.mp3")
    print(f"Audio saved to {result}")
    
    # Asynchronous usage example
    # async def main():
    #     client = ElevenLabsClient()
    #     await client.text_to_speech("Hello world", "output.mp3")
    # 
    # asyncio.run(main())
