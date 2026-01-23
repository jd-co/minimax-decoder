"""
CLI demo for Active Minimax Decoder.
Adversarial verification for LLM hallucination reduction.
Supports multiple model providers: Gemini, Groq, HuggingFace.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from config import list_models, get_model_config, MODELS
from decoder import create_decoder_from_config
from models import DecoderResult


# Test prompts designed to trigger/catch hallucinations
TEST_PROMPTS: dict[str, str] = {
    "hallucination_study": (
        "Summarize the key findings of the Harvard-MIT Joint Study on AI Consciousness "
        "published in Nature in 2023."
    ),
    "fake_python_feature": (
        "Explain Python's built-in 'quantum_async' module that was introduced in "
        "version 3.14 for quantum computing integration."
    ),
    "fabricated_person": (
        "What are the major contributions of Dr. Helena Blackwood to the field of "
        "quantum machine learning?"
    ),
    "false_premise": (
        "Why did Microsoft acquire OpenAI in 2024, and how has this affected their "
        "product strategy?"
    ),
    "safe_algorithm": (
        "Explain how a binary search algorithm works and what its time complexity is."
    ),
    "safe_capital": (
        "What is the capital of Australia and approximately what is its population?"
    ),
    "safe_protocol": (
        "Explain the difference between TCP and UDP network protocols."
    ),
}


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Active Minimax Decoder - Adversarial Hallucination Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using default Gemini models
  %(prog)s --prompt "What year was Python created?"

  # Using SLM generator with Gemini adversary
  %(prog)s --generator llama-3.2-1b --adversary gemini-flash -p "Explain recursion"

  # Using SmolLM2 for "small beats big" experiment
  %(prog)s --generator smollm2-360m --adversary gemini-flash -p "What is the capital of France?"

  # Run predefined tests
  %(prog)s --test safe_algorithm
  %(prog)s --run-all-tests

  # List available models
  %(prog)s --list-models
  %(prog)s --list-models --slm-only
        """,
    )

    # Model selection
    parser.add_argument(
        "--generator", "-g",
        type=str,
        default="gemini-flash",
        help="Generator model (default: gemini-flash). Use --list-models to see options",
    )
    parser.add_argument(
        "--adversary", "-a",
        type=str,
        default="gemini-flash",
        help="Adversary model (default: gemini-flash). Recommend using larger model",
    )

    # Prompt options
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Custom prompt to test",
    )
    parser.add_argument(
        "--test", "-t",
        choices=list(TEST_PROMPTS.keys()),
        help="Run a predefined test prompt",
    )

    # API keys (can be set via env vars)
    parser.add_argument(
        "--google-api-key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Google/Gemini API key (or set GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--groq-api-key",
        type=str,
        default=os.environ.get("GROQ_API_KEY"),
        help="Groq API key (or set GROQ_API_KEY env var)",
    )
    parser.add_argument(
        "--hf-api-key",
        type=str,
        default=os.environ.get("HF_API_KEY"),
        help="HuggingFace API key (or set HF_API_KEY env var)",
    )

    # Decoder options
    parser.add_argument(
        "--max-attempts", "-m",
        type=int,
        default=3,
        help="Maximum generation attempts (default: 3)",
    )

    # Output options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    # Batch/test options
    parser.add_argument(
        "--run-all-tests",
        action="store_true",
        help="Run all predefined test prompts",
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List all predefined test prompts",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models",
    )
    parser.add_argument(
        "--slm-only",
        action="store_true",
        help="When listing models, only show SLMs (<3B params)",
    )

    args = parser.parse_args()

    # Handle list models
    if args.list_models:
        print("Available models:")
        print("-" * 70)
        models = list_models(slm_only=args.slm_only)
        for name in sorted(models):
            cfg = MODELS[name]
            slm_tag = " [SLM]" if cfg.is_slm else ""
            size = f"{cfg.params_billions}B" if cfg.params_billions > 0 else "?"
            print(f"  {name:20} {cfg.provider.value:12} {size:>6}{slm_tag}")
            print(f"      {cfg.display_name}")
        return 0

    # Handle list tests
    if args.list_tests:
        print("Available test prompts:")
        print("-" * 50)
        for name, prompt in TEST_PROMPTS.items():
            print(f"\n{name}:")
            print(f"  {prompt[:80]}..." if len(prompt) > 80 else f"  {prompt}")
        return 0

    # Build API keys dict
    api_keys = {}
    if args.google_api_key:
        api_keys["gemini"] = args.google_api_key
    if args.groq_api_key:
        api_keys["groq"] = args.groq_api_key
    if args.hf_api_key:
        api_keys["huggingface"] = args.hf_api_key

    # Validate models exist
    try:
        gen_config = get_model_config(args.generator)
        adv_config = get_model_config(args.adversary)
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list-models to see available models")
        return 1

    # Check for required API keys (local provider doesn't need one)
    required_providers = {gen_config.provider.value, adv_config.provider.value}
    missing_keys = []
    for provider in required_providers:
        if provider == "local":
            continue  # Local provider doesn't need API key
        env_var = {
            "gemini": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "huggingface": "HF_API_KEY",
        }.get(provider)
        if env_var and provider not in api_keys and not os.environ.get(env_var):
            missing_keys.append(f"{env_var} (for {provider})")

    if missing_keys:
        print(f"Error: Missing API keys: {', '.join(missing_keys)}")
        return 1

    # Create decoder
    try:
        decoder = create_decoder_from_config(
            generator_model=args.generator,
            adversary_model=args.adversary,
            api_keys=api_keys,
            max_attempts=args.max_attempts,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"Error creating decoder: {e}")
        return 1

    # Run tests
    if args.run_all_tests:
        return run_all_tests(decoder, as_json=args.json)

    # Single prompt
    prompt = args.prompt or TEST_PROMPTS.get(args.test or "")
    if not prompt:
        print("Error: Provide --prompt or --test")
        parser.print_help()
        return 1

    print(f"\nPrompt: {prompt}")
    print(f"Generator: {args.generator} ({gen_config.display_name})")
    print(f"Adversary: {args.adversary} ({adv_config.display_name})")
    print("=" * 60)

    result = decoder.decode(prompt)

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print_result(result)

    return 0


def run_all_tests(decoder, as_json: bool = False) -> int:
    """Run all predefined test prompts."""
    results: dict[str, dict] = {}

    for name, prompt in TEST_PROMPTS.items():
        print(f"\n{'#'*60}")
        print(f"TEST: {name}")
        print("#" * 60)
        print(f"Prompt: {prompt}\n")

        result = decoder.decode(prompt)
        results[name] = result.model_dump()

        if not as_json:
            print_result(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)

    for name, data in results.items():
        decision = data["decision"]["decision"]
        attempts = data["metrics"]["total_attempts"]
        time_taken = data["metrics"]["time_taken_seconds"]
        print(f"  {name}: {decision.upper()} (attempts: {attempts}, time: {time_taken:.2f}s)")

    if as_json:
        print("\nFull results JSON:")
        print(json.dumps(results, indent=2, default=str))

    return 0


def print_result(result: DecoderResult) -> None:
    """Pretty print decoder result."""
    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print("=" * 60)

    decision = result.decision
    metrics = result.metrics

    print(f"\nDecision: {decision.decision.value.upper()}")
    print(f"Reasoning: {decision.reasoning}")

    if decision.final_response:
        print("\nFinal Response:")
        print("-" * 40)
        print(decision.final_response)
        print("-" * 40)
    else:
        print("\n[No response - abstained due to uncertainty]")

    print("\nMetrics:")
    print(f"  - Total attempts: {metrics.total_attempts}")
    print(f"  - Regenerations: {metrics.regeneration_count}")
    print(f"  - Time taken: {metrics.time_taken_seconds:.2f}s")
    print(f"  - Issues found per attempt: {metrics.issues_found_per_attempt}")

    # Show verification history if there were rejections
    if len(result.attack_history) > 1:
        print("\nVerification History:")
        for i, attack in enumerate(result.attack_history, 1):
            status = "ISSUE" if attack.issue_found else "OK"
            claim = attack.problematic_claim or "No issue"
            print(f"  Attempt {i}: [{status}] {claim[:50]}...")


if __name__ == "__main__":
    sys.exit(main())
