"""
CLI demo for Active Minimax Decoder.
Adversarial verification for LLM hallucination reduction.
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from decoder import MinimaxDecoder
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
  %(prog)s --prompt "What year was Python created?"
  %(prog)s --test safe_algorithm
  %(prog)s --run-all-tests
  %(prog)s --list-tests
        """,
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="Custom prompt to test",
    )
    parser.add_argument(
        "--test",
        "-t",
        choices=list(TEST_PROMPTS.keys()),
        help="Run a predefined test prompt",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Gemini API key (or set GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--max-attempts",
        "-m",
        type=int,
        default=3,
        help="Maximum generation attempts (default: 3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Attack confidence threshold (default: 0.7)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
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

    args = parser.parse_args()

    # Handle list tests
    if args.list_tests:
        print("Available test prompts:")
        print("-" * 50)
        for name, prompt in TEST_PROMPTS.items():
            print(f"\n{name}:")
            print(f"  {prompt[:80]}..." if len(prompt) > 80 else f"  {prompt}")
        return 0

    # Validate API key
    if not args.api_key:
        print("Error: No API key provided.")
        print("Use --api-key or set GOOGLE_API_KEY environment variable")
        return 1

    # Initialize decoder
    decoder = MinimaxDecoder(
        api_key=args.api_key,
        max_attempts=args.max_attempts,
        attack_threshold=args.threshold,
        verbose=not args.quiet,
    )

    # Run tests
    if args.run_all_tests:
        return run_all_tests(decoder, as_json=args.json)

    # Single prompt
    prompt = args.prompt or TEST_PROMPTS.get(args.test or "")
    if not prompt:
        print("Error: Provide --prompt or --test")
        parser.print_help()
        return 1

    print(f"\nPrompt: {prompt}\n")
    print("=" * 60)

    result = decoder.decode(prompt)

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print_result(result)

    return 0


def run_all_tests(decoder: MinimaxDecoder, as_json: bool = False) -> int:
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
    print(f"  - Attack confidences: {metrics.attack_confidences}")

    # Show attack history if there were rejections
    if len(result.attack_history) > 1:
        print("\nAttack History:")
        for i, attack in enumerate(result.attack_history, 1):
            print(f"  Attempt {i}: {attack.weakest_claim[:50]}... (conf: {attack.confidence_in_attack:.2f})")


if __name__ == "__main__":
    sys.exit(main())
