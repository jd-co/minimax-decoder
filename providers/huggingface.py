"""
HuggingFace Inference API provider implementation.
Supports SmolLM2 (360M, 1.7B) and Qwen 2.5 (0.5B, 1.5B, 3B).
"""

from typing import Optional

from huggingface_hub import InferenceClient

from .base import ModelProvider


class HuggingFaceProvider(ModelProvider):
    """
    HuggingFace Inference API provider.

    Supported models:
    - HuggingFaceTB/SmolLM2-360M-Instruct (SLM)
    - HuggingFaceTB/SmolLM2-1.7B-Instruct (SLM)
    - Qwen/Qwen2.5-0.5B-Instruct (SLM)
    - Qwen/Qwen2.5-1.5B-Instruct (SLM)
    - Qwen/Qwen2.5-3B-Instruct
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
    ):
        """
        Initialize HuggingFace provider.

        Args:
            api_key: HuggingFace API token
            model_id: Full model identifier (e.g., "HuggingFaceTB/SmolLM2-360M-Instruct")
        """
        self.client = InferenceClient(token=api_key)
        self._model_id = model_id

    @property
    def name(self) -> str:
        # Return short name from full model ID
        short_name = self._model_id.split("/")[-1]
        return f"HF:{short_name}"

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
        """Generate text using HuggingFace Inference API."""
        # Format prompt with system instruction if provided
        if system_prompt:
            # Use chat format for instruction-tuned models
            full_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        else:
            full_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"

        try:
            # Try chat completion first (for chat-optimized models)
            response = self.client.chat_completion(
                model=self._model_id,
                messages=[
                    {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception:
            # Fallback to text generation
            try:
                response = self.client.text_generation(
                    prompt=full_prompt,
                    model=self._model_id,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                )
                return response
            except Exception as e:
                # Last resort: simple text generation
                response = self.client.text_generation(
                    prompt=prompt,
                    model=self._model_id,
                    max_new_tokens=max_tokens,
                )
                return response
