"""
Model configurations and presets for the Minimax Decoder.
Supports Gemini, Groq (Llama), and HuggingFace (SmolLM2, Qwen) models.
"""

import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    """Supported model providers."""

    GEMINI = "gemini"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    provider: ProviderType = Field(description="Model provider type")
    model_id: str = Field(description="Model identifier for the API")
    display_name: str = Field(description="Human-readable name")
    params_billions: float = Field(description="Model size in billions of parameters")
    is_slm: bool = Field(
        default=False, description="Whether this is a Small Language Model (<3B params)"
    )


# Predefined model configurations
MODELS: dict[str, ModelConfig] = {
    # ========================================
    # Gemini Models (Google)
    # ========================================
    "gemini-flash": ModelConfig(
        provider=ProviderType.GEMINI,
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        params_billions=0,  # Unknown/proprietary
        is_slm=False,
    ),
    "gemini-flash-lite": ModelConfig(
        provider=ProviderType.GEMINI,
        model_id="gemini-2.0-flash-lite",
        display_name="Gemini 2.0 Flash Lite",
        params_billions=0,
        is_slm=False,
    ),
    # ========================================
    # Groq Models (Free tier: 1000 req/day)
    # ========================================
    "llama-3.2-1b": ModelConfig(
        provider=ProviderType.GROQ,
        model_id="llama-3.2-1b-preview",
        display_name="Llama 3.2 1B",
        params_billions=1.0,
        is_slm=True,
    ),
    "llama-3.2-3b": ModelConfig(
        provider=ProviderType.GROQ,
        model_id="llama-3.2-3b-preview",
        display_name="Llama 3.2 3B",
        params_billions=3.0,
        is_slm=False,
    ),
    "llama-3.1-8b": ModelConfig(
        provider=ProviderType.GROQ,
        model_id="llama-3.1-8b-instant",
        display_name="Llama 3.1 8B",
        params_billions=8.0,
        is_slm=False,
    ),
    "gemma2-9b": ModelConfig(
        provider=ProviderType.GROQ,
        model_id="gemma2-9b-it",
        display_name="Gemma 2 9B",
        params_billions=9.0,
        is_slm=False,
    ),
    # ========================================
    # HuggingFace Models (Inference API)
    # ========================================
    "smollm2-360m": ModelConfig(
        provider=ProviderType.HUGGINGFACE,
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        display_name="SmolLM2 360M",
        params_billions=0.36,
        is_slm=True,
    ),
    "smollm2-1.7b": ModelConfig(
        provider=ProviderType.HUGGINGFACE,
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        display_name="SmolLM2 1.7B",
        params_billions=1.7,
        is_slm=True,
    ),
    "qwen2.5-0.5b": ModelConfig(
        provider=ProviderType.HUGGINGFACE,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        display_name="Qwen 2.5 0.5B",
        params_billions=0.5,
        is_slm=True,
    ),
    "qwen2.5-1.5b": ModelConfig(
        provider=ProviderType.HUGGINGFACE,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        display_name="Qwen 2.5 1.5B",
        params_billions=1.5,
        is_slm=True,
    ),
    "qwen2.5-3b": ModelConfig(
        provider=ProviderType.HUGGINGFACE,
        model_id="Qwen/Qwen2.5-3B-Instruct",
        display_name="Qwen 2.5 3B",
        params_billions=3.0,
        is_slm=False,
    ),
    # ========================================
    # Local Models (no API, runs on your machine)
    # ========================================
    "smollm2-360m-local": ModelConfig(
        provider=ProviderType.LOCAL,
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        display_name="SmolLM2 360M (Local)",
        params_billions=0.36,
        is_slm=True,
    ),
    "smollm2-1.7b-local": ModelConfig(
        provider=ProviderType.LOCAL,
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        display_name="SmolLM2 1.7B (Local)",
        params_billions=1.7,
        is_slm=True,
    ),
    "qwen2.5-0.5b-local": ModelConfig(
        provider=ProviderType.LOCAL,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        display_name="Qwen 2.5 0.5B (Local)",
        params_billions=0.5,
        is_slm=True,
    ),
    "qwen2.5-1.5b-local": ModelConfig(
        provider=ProviderType.LOCAL,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        display_name="Qwen 2.5 1.5B (Local)",
        params_billions=1.5,
        is_slm=True,
    ),
    # ========================================
    # Liquid AI Models (Hybrid Architecture)
    # ========================================
    "lfm2-350m-local": ModelConfig(
        provider=ProviderType.LOCAL,
        model_id="LiquidAI/LFM2-350M",
        display_name="Liquid LFM2 350M (Local)",
        params_billions=0.35,
        is_slm=True,
    ),
    "lfm2-700m-local": ModelConfig(
        provider=ProviderType.LOCAL,
        model_id="LiquidAI/LFM2-700M",
        display_name="Liquid LFM2 700M (Local)",
        params_billions=0.7,
        is_slm=True,
    ),
    "lfm2-1.2b-local": ModelConfig(
        provider=ProviderType.LOCAL,
        model_id="LiquidAI/LFM2-1.2B",
        display_name="Liquid LFM2 1.2B (Local)",
        params_billions=1.2,
        is_slm=True,
    ),
    "lfm2.5-1.2b-local": ModelConfig(
        provider=ProviderType.LOCAL,
        model_id="LiquidAI/LFM2.5-1.2B-Instruct",
        display_name="Liquid LFM2.5 1.2B (Local)",
        params_billions=1.2,
        is_slm=True,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get model configuration by name.

    Args:
        model_name: Model short name (e.g., "smollm2-360m", "llama-3.2-1b")

    Returns:
        ModelConfig for the specified model

    Raises:
        ValueError: If model name is not found
    """
    if model_name not in MODELS:
        available = ", ".join(sorted(MODELS.keys()))
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {available}"
        )
    return MODELS[model_name]


def get_api_key(provider: ProviderType, explicit_key: Optional[str] = None) -> str:
    """
    Get API key for a provider.

    Args:
        provider: Provider type
        explicit_key: Explicitly provided key (takes precedence)

    Returns:
        API key string

    Raises:
        ValueError: If no API key found (except for LOCAL provider)
    """
    # Local provider doesn't need API key
    if provider == ProviderType.LOCAL:
        return ""

    if explicit_key:
        return explicit_key

    env_vars = {
        ProviderType.GEMINI: "GOOGLE_API_KEY",
        ProviderType.GROQ: "GROQ_API_KEY",
        ProviderType.HUGGINGFACE: "HF_API_KEY",
    }

    env_var = env_vars.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider.value}")

    key = os.environ.get(env_var)

    if not key:
        raise ValueError(
            f"No API key found for {provider.value}. "
            f"Set {env_var} environment variable or pass --api-key"
        )

    return key


def list_models(slm_only: bool = False) -> list[str]:
    """
    List available model names.

    Args:
        slm_only: If True, only return SLM models (<3B params)

    Returns:
        List of model short names
    """
    if slm_only:
        return [name for name, cfg in MODELS.items() if cfg.is_slm]
    return list(MODELS.keys())


def list_models_by_provider(provider: ProviderType) -> list[str]:
    """
    List models for a specific provider.

    Args:
        provider: Provider type

    Returns:
        List of model short names for that provider
    """
    return [name for name, cfg in MODELS.items() if cfg.provider == provider]
