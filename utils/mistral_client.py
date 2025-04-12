import httpx
import os
import json
from typing import Dict, List, Optional, Union, Iterator, Any


class MistralClient:
    """Client for the Mistral AI API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.mistral.ai"):
        """Initialize the Mistral client.
        
        Args:
            api_key: Mistral API key. If not provided, will look for MISTRAL_API_KEY environment variable.
            base_url: Base URL for the Mistral API.
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Either pass it as an argument or set the MISTRAL_API_KEY environment variable.")
        
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        random_seed: Optional[int] = None,
        safe_prompt: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a chat completion.
        
        Args:
            model: ID of the model to use.
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: What sampling temperature to use. Higher values make output more random.
            top_p: Nucleus sampling parameter. Model considers tokens with top_p probability mass.
            max_tokens: Maximum number of tokens to generate.
            stream: If False, returns full response. If True, use stream_completion() instead.
            stop: Stop generation if this token(s) is detected.
            random_seed: Seed for random sampling.
            safe_prompt: Whether to inject a safety prompt before all conversations.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            The API response as a dictionary.
        """
        if stream:
            raise ValueError("For streaming responses, use the stream_completion method instead.")
        
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "safe_prompt": safe_prompt,
        }
        
        # Add optional parameters if provided
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        if random_seed is not None:
            payload["random_seed"] = random_seed
            
        # Add any other parameters provided
        payload.update(kwargs)
        
        with httpx.Client() as client:
            response = client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
    
    def stream_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        random_seed: Optional[int] = None,
        safe_prompt: bool = False,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Create a streaming chat completion.
        
        Args:
            model: ID of the model to use.
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: What sampling temperature to use. Higher values make output more random.
            top_p: Nucleus sampling parameter. Model considers tokens with top_p probability mass.
            max_tokens: Maximum number of tokens to generate.
            stop: Stop generation if this token(s) is detected.
            random_seed: Seed for random sampling.
            safe_prompt: Whether to inject a safety prompt before all conversations.
            **kwargs: Additional parameters to pass to the API.
            
        Yields:
            Streamed response chunks from the API.
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "safe_prompt": safe_prompt,
        }
        
        # Add optional parameters if provided
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        if random_seed is not None:
            payload["random_seed"] = random_seed
            
        # Add any other parameters provided
        payload.update(kwargs)
        
        with httpx.Client() as client:
            with client.stream("POST", url, headers=self.headers, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        if line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix
                            if line == "[DONE]":
                                break
                            try:
                                chunk = json.loads(line)
                                yield chunk
                            except json.JSONDecodeError:
                                pass
