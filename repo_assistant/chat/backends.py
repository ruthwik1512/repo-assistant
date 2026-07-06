"""
backends.py

Defines the LLM integration layer using the Strategy Pattern.
"""

from abc import ABC, abstractmethod
from typing import List

import requests

from .models import ChatMessage


class LLMBackend(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    def generate(self, messages: List[ChatMessage]) -> str:
        """
        Takes a conversation history and generates the next response.
        
        Args:
            messages: List of ChatMessage objects (system, user, assistant).
            
        Returns:
            The generated text string from the LLM.
        """
        pass

    @abstractmethod
    def generate_stream(self, messages: List[ChatMessage]):
        """
        Takes a conversation history and yields the response token by token.
        """
        pass


class OllamaBackend(LLMBackend):
    """
    Communicates with a locally running Ollama instance via its REST API.
    By default, expects Ollama to be running on localhost:11434.
    """

    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434") -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate(self, messages: List[ChatMessage]) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "stream": False
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Please ensure the Ollama app is installed and running."
            )
        except requests.exceptions.RequestException as e:
            # e.g., 404 if the model isn't pulled yet
            if response is not None and response.status_code == 404:
                raise RuntimeError(
                    f"Model '{self.model_name}' not found. "
                    f"Try running: ollama run {self.model_name}"
                )
            raise RuntimeError(f"Ollama generation failed: {e}")

    def generate_stream(self, messages: List[ChatMessage]):
        import json
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "stream": True
        }

        try:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    yield data["message"]["content"]
                    if data.get("done"):
                        break
        except Exception as e:
            yield f"\n[Error connecting to Ollama: {e}]"
