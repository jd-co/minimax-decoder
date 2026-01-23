"""
Abstract base class for model providers.
All providers must implement this interface.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Optional


class ModelProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        pass

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> dict:
        """
        Generate JSON response with parsing.

        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: Optional system instruction
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Parsed JSON dict, or empty dict on failure
        """
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self._parse_json(response)

    def _parse_json(self, text: str) -> dict:
        """
        Parse JSON from response text, handling common issues.

        Args:
            text: Raw response text

        Returns:
            Parsed dict or empty dict on failure
        """
        text = text.strip()

        # Remove markdown code blocks
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                # Remove language identifier
                if text.startswith("json"):
                    text = text[4:]
                elif text.startswith("\n"):
                    text = text[1:]
            text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try to find JSON array
        array_match = re.search(r"\[[\s\S]*\]", text)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass

        return {}
