import os
import httpx
import tempfile
from typing import Optional, Union, BinaryIO


class WhisperClient:
    """
    Client for OpenAI's Whisper speech-to-text API.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the Whisper client.
        
        Args:
            api_key (str): Your OpenAI API key. If None, reads from OPENAI_API_KEY env variable.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.base_url = "https://api.openai.com/v1/audio/transcriptions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def transcribe(self, 
                   audio_file: Union[str, BinaryIO], 
                   model: str = "whisper-1",
                   language: Optional[str] = None,
                   prompt: Optional[str] = None) -> str:
        """
        Transcribe speech in an audio file to text.
        
        Args:
            audio_file: Either a path to an audio file or a file-like object
            model: Model to use for transcription
            language: Language of the audio (optional)
            prompt: Optional text to guide the model's style or continue a previous audio segment
            
        Returns:
            str: The transcribed text
        """
        # Prepare the form data
        files = {}
        data = {
            "model": model
        }
        

        data["language"] = "en"
        
        if prompt:
            data["prompt"] = prompt
        
        # Handle file path vs file object
        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                file_content = f.read()
        else:
            # If it's a file-like object, read its content
            audio_file.seek(0)
            file_content = audio_file.read()
            
        # Create a temporary file for the request
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
            
        try:
            with open(temp_path, "rb") as f:
                files = {"file": f}
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        self.base_url,
                        headers=self.headers,
                        data=data,
                        files=files
                    )
                    response.raise_for_status()
                    result = response.json()
                    return result.get("text", "")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def transcribe_from_microphone(self, 
                                   duration: int = 5, 
                                   model: str = "whisper-1",
                                   language: Optional[str] = None,
                                   prompt: Optional[str] = None) -> str:
        """
        Record audio from microphone and transcribe it.
        
        Args:
            duration: Recording duration in seconds
            model: Model to use for transcription
            language: Language of the audio (optional)
            prompt: Optional text to guide the model's style
            
        Returns:
            str: The transcribed text
        """
        try:
            import sounddevice as sd
            import scipy.io.wavfile as wav
        except ImportError:
            raise ImportError("Please install sounddevice and scipy: pip install sounddevice scipy")
        
        # Record audio
        print(f"Recording for {duration} seconds...")
        fs = 44100  # Sample rate
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            wav.write(temp_path, fs, recording)
            return self.transcribe(temp_path, model, language, prompt)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)


# Example usage
if __name__ == "__main__":
    client = WhisperClient()
    
    # Transcribe from file
    # result = client.transcribe("audio_file.wav")
    # print(f"Transcription: {result}")
    
    # Transcribe from microphone
    # result = client.transcribe_from_microphone(duration=5)
    # print(f"Transcription: {result}")
