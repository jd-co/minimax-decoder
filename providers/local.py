"""
Local model provider using HuggingFace Transformers.
Runs models locally on CPU/GPU without API calls.
Ideal for SmolLM2, Qwen, and other small models.
"""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base import ModelProvider


class LocalProvider(ModelProvider):
    """
    Local inference provider using HuggingFace Transformers.

    Supported models:
    - HuggingFaceTB/SmolLM2-360M-Instruct (360M params, ~720MB)
    - HuggingFaceTB/SmolLM2-1.7B-Instruct (1.7B params, ~3.4GB)
    - Qwen/Qwen2.5-0.5B-Instruct (500M params, ~1GB)
    - Qwen/Qwen2.5-1.5B-Instruct (1.5B params, ~3GB)

    Memory requirements (approximate):
    - 360M model: ~1GB RAM
    - 500M model: ~1.5GB RAM
    - 1.5B model: ~4GB RAM
    - 1.7B model: ~5GB RAM
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize local model provider.

        Args:
            model_id: HuggingFace model ID
            device: Device to use ("auto", "cpu", "cuda", "mps")
            torch_dtype: Torch dtype ("auto", "float16", "bfloat16", "float32")
            load_in_8bit: Load model in 8-bit quantization (requires bitsandbytes)
            load_in_4bit: Load model in 4-bit quantization (requires bitsandbytes)
        """
        self._model_id = model_id
        self._device = device

        print(f"Loading model: {model_id}...")
        print(f"Device: {device}, Dtype: {torch_dtype}")

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Determine dtype
        if torch_dtype == "auto":
            if device == "cpu":
                dtype = torch.float32  # CPU works best with float32
            else:
                dtype = torch.float16
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate settings
        model_kwargs = {
            "dtype": dtype,
            "device_map": device if device != "cpu" else None,
            "low_cpu_mem_usage": True,
        }

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs
        )

        # Move to device if not using device_map
        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()
        self._actual_device = device

        print(f"Model loaded on {device}")

    @property
    def name(self) -> str:
        short_name = self._model_id.split("/")[-1]
        return f"Local:{short_name}"

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> str:
        """Generate text using local model."""
        start_time = time.time()

        # Format messages for chat models
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without chat template
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                formatted_prompt = f"User: {prompt}\nAssistant:"

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        elapsed = time.time() - start_time

        # Log SLM response and time
        if verbose:
            response_preview = response.strip()[:200] + "..." if len(response.strip()) > 200 else response.strip()
            print(f"\n   [SLM] Time: {elapsed:.2f}s | Response: {response_preview}")

        return response.strip()


def get_local_model_info() -> dict[str, dict]:
    """Get information about recommended local models."""
    return {
        "smollm2-360m-local": {
            "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "params": "360M",
            "memory": "~1GB",
            "speed": "Fast on CPU",
        },
        "smollm2-1.7b-local": {
            "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "params": "1.7B",
            "memory": "~5GB",
            "speed": "Moderate on CPU",
        },
        "qwen2.5-0.5b-local": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "params": "500M",
            "memory": "~1.5GB",
            "speed": "Fast on CPU",
        },
        "qwen2.5-1.5b-local": {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "params": "1.5B",
            "memory": "~4GB",
            "speed": "Moderate on CPU",
        },
    }
