#!/usr/bin/env python3
"""
Run TruthfulQA benchmarks across multiple SLMs.
Compares: vanilla SLM vs SLM+Minimax vs larger model vanilla

This script tests the hypothesis:
"SmolLM2-360M + Minimax can outperform Qwen2.5-1.5B vanilla"

Usage:
    # Run all experiments (default: 50 questions)
    python experiments/run_slm_benchmark.py

    # Quick test (10 questions)
    python experiments/run_slm_benchmark.py --questions 10

    # Run specific experiment
    python experiments/run_slm_benchmark.py --experiment smollm2-minimax
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from benchmark import run_benchmark, load_truthfulqa, print_metrics, save_results
from config import MODELS, get_model_config


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    generator: str
    adversary: str | None  # None = vanilla (no minimax)
    description: str


# Experiment configurations for the "small beats big" hypothesis
EXPERIMENTS: list[ExperimentConfig] = [
    # ========================================
    # Minimax configurations (SLM + strong adversary)
    # ========================================
    ExperimentConfig(
        name="smollm2-360m-minimax",
        generator="smollm2-360m",
        adversary="gemini-flash",
        description="SmolLM2 360M with Gemini adversary (target: beat Qwen-1.5B vanilla)",
    ),
    ExperimentConfig(
        name="qwen-0.5b-minimax",
        generator="qwen2.5-0.5b",
        adversary="gemini-flash",
        description="Qwen 0.5B with Gemini adversary",
    ),
    ExperimentConfig(
        name="llama-1b-minimax",
        generator="llama-3.2-1b",
        adversary="gemini-flash",
        description="Llama 3.2 1B with Gemini adversary",
    ),
    # ========================================
    # Vanilla baselines (no adversarial verification)
    # ========================================
    ExperimentConfig(
        name="smollm2-360m-vanilla",
        generator="smollm2-360m",
        adversary=None,
        description="SmolLM2 360M vanilla baseline",
    ),
    ExperimentConfig(
        name="qwen-1.5b-vanilla",
        generator="qwen2.5-1.5b",
        adversary=None,
        description="Qwen 1.5B vanilla (target to beat)",
    ),
    ExperimentConfig(
        name="llama-3b-vanilla",
        generator="llama-3.2-3b",
        adversary=None,
        description="Llama 3.2 3B vanilla",
    ),
    ExperimentConfig(
        name="gemini-flash-vanilla",
        generator="gemini-flash",
        adversary=None,
        description="Gemini Flash vanilla (upper bound)",
    ),
]


def run_single_experiment(
    exp: ExperimentConfig,
    questions: list[dict],
    api_keys: dict[str, str],
    output_dir: Path,
    delay: float = 1.0,
) -> dict:
    """Run a single experiment and save results."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp.name}")
    print(f"Description: {exp.description}")
    print(f"Generator: {exp.generator}")
    print(f"Adversary: {exp.adversary or 'None (vanilla)'}")
    print("=" * 70)

    gen_config = get_model_config(exp.generator)

    if exp.adversary is None:
        # Vanilla run - use benchmark with no-vanilla flag (ironic but correct)
        # We need to run just the generator without minimax
        from providers import GeminiProvider, GroqProvider, HuggingFaceProvider
        from config import get_api_key, ProviderType

        key = get_api_key(gen_config.provider, api_keys.get(gen_config.provider.value))

        if gen_config.provider == ProviderType.GEMINI:
            provider = GeminiProvider(api_key=key, model_id=gen_config.model_id)
        elif gen_config.provider == ProviderType.GROQ:
            provider = GroqProvider(api_key=key, model_id=gen_config.model_id)
        elif gen_config.provider == ProviderType.HUGGINGFACE:
            provider = HuggingFaceProvider(api_key=key, model_id=gen_config.model_id)

        # Run vanilla evaluation
        from benchmark import LLMJudge
        import time

        judge_key = get_api_key(ProviderType.GEMINI, api_keys.get("gemini"))
        judge_provider = GeminiProvider(api_key=judge_key, model_id="gemini-2.0-flash")
        judge = LLMJudge(provider=judge_provider)

        results = {
            "truthful": 0,
            "hallucination": 0,
            "refusal": 0,
            "mixed": 0,
            "error": 0,
            "total_time": 0.0,
            "responses": [],
        }

        for i, q_data in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q_data['question'][:50]}...", end=" ")

            start = time.time()
            try:
                response = provider.generate(prompt=q_data["question"])
                elapsed = time.time() - start
                results["total_time"] += elapsed

                verdict = judge.evaluate(
                    question=q_data["question"],
                    response=response,
                    correct_answers=q_data["correct_answers"],
                    incorrect_answers=q_data["incorrect_answers"],
                )

                results[verdict["verdict"]] += 1
                results["responses"].append(
                    {
                        "question": q_data["question"],
                        "response": response,
                        "verdict": verdict["verdict"],
                        "time": elapsed,
                    }
                )
                print(f"-> {verdict['verdict'].upper()}")

            except Exception as e:
                results["error"] += 1
                print(f"-> ERROR: {e}")

            if delay > 0 and i < len(questions):
                time.sleep(delay)

        # Calculate rates
        valid = len(questions) - results["error"]
        metrics = {
            "experiment": exp.name,
            "generator": exp.generator,
            "adversary": None,
            "total_questions": len(questions),
            "truthful": results["truthful"],
            "hallucination": results["hallucination"],
            "refusal": results["refusal"],
            "mixed": results["mixed"],
            "error": results["error"],
            "truthful_rate": results["truthful"] / valid if valid > 0 else 0,
            "hallucination_rate": results["hallucination"] / valid if valid > 0 else 0,
            "total_time": results["total_time"],
        }

    else:
        # Minimax run
        results_list, bench_metrics = run_benchmark(
            generator_model=exp.generator,
            adversary_model=exp.adversary,
            questions=questions,
            api_keys=api_keys,
            run_vanilla=False,  # We handle vanilla separately
            verbose=True,
            delay=delay,
        )

        metrics = {
            "experiment": exp.name,
            "generator": exp.generator,
            "adversary": exp.adversary,
            "total_questions": bench_metrics.total_questions,
            "truthful": bench_metrics.minimax_truthful,
            "hallucination": bench_metrics.minimax_hallucinated,
            "refusal": bench_metrics.minimax_refusal,
            "mixed": bench_metrics.minimax_mixed,
            "error": bench_metrics.minimax_error,
            "truthful_rate": bench_metrics.minimax_truthful_rate(),
            "hallucination_rate": bench_metrics.minimax_hallucination_rate(),
            "abstention_rate": bench_metrics.minimax_abstention_rate(),
            "avg_attempts": bench_metrics.minimax_avg_attempts(),
            "total_time": bench_metrics.minimax_total_time,
        }

        # Save detailed results
        output_file = output_dir / f"{exp.name}_detailed.json"
        save_results(results_list, bench_metrics, str(output_file))

    # Save summary
    output_file = output_dir / f"{exp.name}_summary.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults for {exp.name}:")
    print(f"  Truthful:      {metrics['truthful']} ({metrics['truthful_rate']*100:.1f}%)")
    print(
        f"  Hallucination: {metrics['hallucination']} ({metrics['hallucination_rate']*100:.1f}%)"
    )
    print(f"  Total Time:    {metrics['total_time']:.1f}s")

    return metrics


def compare_results(all_metrics: list[dict]) -> None:
    """Compare and summarize all experiment results."""
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    # Sort by truthful rate
    sorted_metrics = sorted(all_metrics, key=lambda x: x["truthful_rate"], reverse=True)

    print(f"\n{'Experiment':<30} {'Truthful':>10} {'Halluc':>10} {'Params':>10}")
    print("-" * 70)

    for m in sorted_metrics:
        gen_config = get_model_config(m["generator"])
        params = f"{gen_config.params_billions}B" if gen_config.params_billions > 0 else "?"
        minimax_tag = "+MM" if m["adversary"] else ""
        name = f"{m['generator']}{minimax_tag}"

        print(
            f"{name:<30} {m['truthful_rate']*100:>9.1f}% {m['hallucination_rate']*100:>9.1f}% {params:>10}"
        )

    # Check hypothesis
    print("\n" + "=" * 80)
    print("HYPOTHESIS CHECK: Does SmolLM2-360M + Minimax beat Qwen-1.5B vanilla?")
    print("=" * 80)

    smollm2_minimax = next(
        (m for m in all_metrics if m["experiment"] == "smollm2-360m-minimax"), None
    )
    qwen_vanilla = next(
        (m for m in all_metrics if m["experiment"] == "qwen-1.5b-vanilla"), None
    )

    if smollm2_minimax and qwen_vanilla:
        diff = smollm2_minimax["truthful_rate"] - qwen_vanilla["truthful_rate"]
        halluc_diff = (
            qwen_vanilla["hallucination_rate"] - smollm2_minimax["hallucination_rate"]
        )

        print(f"\nSmolLM2-360M + Minimax: {smollm2_minimax['truthful_rate']*100:.1f}% truthful")
        print(f"Qwen 1.5B Vanilla:      {qwen_vanilla['truthful_rate']*100:.1f}% truthful")
        print(f"Difference:             {diff*100:+.1f}%")

        if diff > 0:
            print("\n*** HYPOTHESIS CONFIRMED: 360M model + Minimax beats 1.5B model! ***")
            print(
                f"    Hallucination reduction: {halluc_diff*100:.1f}% fewer hallucinations"
            )
        else:
            print("\n    Hypothesis not confirmed in this run.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SLM benchmark experiments for Minimax Decoder"
    )
    parser.add_argument(
        "--questions",
        "-n",
        type=int,
        default=50,
        help="Number of questions to test (default: 50)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/TruthfulQA.csv",
        help="Path to TruthfulQA CSV",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: results/YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default=None,
        help="Run specific experiment by name",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiments",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between questions (default: 1.0s)",
    )

    args = parser.parse_args()

    if args.list_experiments:
        print("Available experiments:")
        print("-" * 70)
        for exp in EXPERIMENTS:
            adv = exp.adversary or "None (vanilla)"
            print(f"\n  {exp.name}")
            print(f"    Generator: {exp.generator}")
            print(f"    Adversary: {adv}")
            print(f"    {exp.description}")
        return 0

    # Build API keys
    api_keys = {}
    if os.environ.get("GOOGLE_API_KEY"):
        api_keys["gemini"] = os.environ["GOOGLE_API_KEY"]
    if os.environ.get("GROQ_API_KEY"):
        api_keys["groq"] = os.environ["GROQ_API_KEY"]
    if os.environ.get("HF_API_KEY"):
        api_keys["huggingface"] = os.environ["HF_API_KEY"]

    # Check for required keys
    if "gemini" not in api_keys:
        print("Error: GOOGLE_API_KEY required for judge and adversary")
        return 1

    # Load questions
    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        return 1

    questions = load_truthfulqa(args.data, limit=args.questions)
    print(f"Loaded {len(questions)} questions from {args.data}")

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Select experiments
    if args.experiment:
        experiments = [e for e in EXPERIMENTS if e.name == args.experiment]
        if not experiments:
            print(f"Error: Unknown experiment '{args.experiment}'")
            print("Use --list-experiments to see available options")
            return 1
    else:
        experiments = EXPERIMENTS

    # Run experiments
    all_metrics = []
    for exp in experiments:
        try:
            metrics = run_single_experiment(
                exp=exp,
                questions=questions,
                api_keys=api_keys,
                output_dir=output_dir,
                delay=args.delay,
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"ERROR in {exp.name}: {e}")
            continue

    # Compare results
    if len(all_metrics) > 1:
        compare_results(all_metrics)

    # Save combined results
    combined_file = output_dir / "all_experiments.json"
    with open(combined_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "num_questions": args.questions,
                "experiments": all_metrics,
            },
            f,
            indent=2,
        )
    print(f"\nCombined results saved to: {combined_file}")

    return 0


if __name__ == "__main__":
    exit(main())
