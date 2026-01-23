"""
Google Gemini provider implementation.
"""

from typing import Optional

from google import genai

from .base import ModelProvider


class GeminiProvider(ModelProvider):
    """
    Google Gemini API provider.
    Default model: gemini-2.0-flash
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-2.0-flash",
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key
            model_id: Model identifier (default: gemini-2.0-flash)
        """
        self.client = genai.Client(api_key=api_key)
        self._model_id = model_id

    @property
    def name(self) -> str:
        return f"Gemini:{self._model_id}"

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
        """Generate text using Gemini API."""
        config = genai.types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        if system_prompt:
            config.system_instruction = system_prompt

        response = self.client.models.generate_content(
            model=self._model_id,
            contents=prompt,
            config=config,
        )

        return response.text
