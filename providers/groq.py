"""
Groq provider implementation.
Supports Llama 3.2 (1B, 3B), Llama 3.1 (8B), Gemma 2 (9B).
Free tier: 1000 requests/day.
"""

from typing import Optional

from groq import Groq

from .base import ModelProvider


class GroqProvider(ModelProvider):
    """
    Groq API provider for fast SLM inference.

    Supported models:
    - llama-3.2-1b-preview (SLM)
    - llama-3.2-3b-preview
    - llama-3.1-8b-instant
    - gemma2-9b-it
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "llama-3.2-1b-preview",
    ):
        """
        Initialize Groq provider.

        Args:
            api_key: Groq API key
            model_id: Model identifier (default: llama-3.2-1b-preview)
        """
        self.client = Groq(api_key=api_key)
        self._model_id = model_id

    @property
    def name(self) -> str:
        return f"Groq:{self._model_id}"

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text using Groq API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""
