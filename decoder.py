"""
MinimaxDecoder: Main orchestration of the adversarial verification game loop.
Refactored to use provider abstraction for multi-model support.
"""

import time
from typing import Optional

from models import (
    DraftResponse,
    AttackResult,
    MinimaxDecision,
    DecoderMetrics,
    DecoderResult,
    DecisionType,
)
from agents import GeneratorAgent, AdversaryAgent, Arbiter
from providers.base import ModelProvider


class MinimaxDecoder:
    """
    Active Minimax Decoder for hallucination reduction.

    Game Loop:
    1. Generator creates draft response
    2. Adversary attacks to find weakest claim
    3. Arbiter decides: ACCEPT, REJECT (regenerate), or ABSTAIN
    4. If REJECT: regenerate with critique, max 2 retries
    5. If still failing: ABSTAIN

    Now supports multiple model providers (Gemini, Groq, HuggingFace).
    """

    def __init__(
        self,
        generator_provider: ModelProvider,
        adversary_provider: ModelProvider,
        claim_extractor_provider: Optional[ModelProvider] = None,
        max_attempts: int = 3,
        verbose: bool = True,
    ):
        """
        Initialize the decoder with provider-based agents.

        Args:
            generator_provider: Provider for response generation (can be SLM)
            adversary_provider: Provider for adversarial checking (recommend larger model)
            claim_extractor_provider: Optional provider for claim extraction
                                     (defaults to adversary_provider for reliability)
            max_attempts: Maximum generation attempts before abstaining
            verbose: Print progress during decoding
        """
        # Use adversary provider for claim extraction by default (more reliable)
        extractor = claim_extractor_provider or adversary_provider

        self.generator = GeneratorAgent(generator_provider, claim_extractor=extractor)
        self.adversary = AdversaryAgent(adversary_provider)
        self.arbiter = Arbiter()  # Binary decisions, no thresholds

        self.generator_provider = generator_provider
        self.adversary_provider = adversary_provider

        self.max_attempts = max_attempts
        self.verbose = verbose

    def decode(self, prompt: str) -> DecoderResult:
        """
        Run the full minimax decoding game loop.

        Args:
            prompt: User's question/prompt

        Returns:
            DecoderResult with decision, metrics, and history
        """
        start_time = time.time()

        draft_history: list[DraftResponse] = []
        attack_history: list[AttackResult] = []
        issues_found: list[bool] = []

        previous_critique: Optional[str] = None

        for attempt in range(1, self.max_attempts + 1):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Attempt {attempt}/{self.max_attempts}")
                print(f"Generator: {self.generator_provider.name}")
                print(f"Adversary: {self.adversary_provider.name}")
                print("=" * 50)

            # Step 1: Generate draft
            if self.verbose:
                print("\n[Generator] Creating draft response...")

            draft = self.generator.generate(prompt, previous_critique)
            draft_history.append(draft)

            if self.verbose:
                print(f"[Generator] Draft created with {len(draft.key_claims)} claims")
                preview = (
                    draft.response_text[:200] + "..."
                    if len(draft.response_text) > 200
                    else draft.response_text
                )
                print(f"[Generator] Response preview: {preview}")

            # Step 2: Adversary verifies
            if self.verbose:
                print("\n[Verifier] Checking for factual errors...")

            attack = self.adversary.attack(prompt, draft)
            attack_history.append(attack)
            issues_found.append(attack.issue_found)

            if self.verbose:
                if attack.issue_found:
                    print(f"[Verifier] ISSUE FOUND: {attack.problematic_claim}")
                    print(f"[Verifier] Reason: {attack.reasoning}")
                else:
                    print(f"[Verifier] NO ISSUE - {attack.reasoning}")

            # Step 3: Arbiter decides
            if self.verbose:
                print("\n[Arbiter] Making decision...")

            decision, reasoning = self.arbiter.decide(
                draft=draft,
                attack=attack,
                attempt_number=attempt,
                max_attempts=self.max_attempts,
            )

            if self.verbose:
                print(f"[Arbiter] Decision: {decision.value}")
                print(f"[Arbiter] Reasoning: {reasoning}")

            # Handle decision
            if decision == DecisionType.ACCEPT:
                elapsed = time.time() - start_time
                return DecoderResult(
                    decision=MinimaxDecision(
                        decision=decision,
                        reasoning=reasoning,
                        final_response=draft.response_text,
                    ),
                    metrics=DecoderMetrics(
                        total_attempts=attempt,
                        final_decision=decision,
                        issues_found_per_attempt=issues_found,
                        time_taken_seconds=elapsed,
                        regeneration_count=attempt - 1,
                    ),
                    draft_history=draft_history,
                    attack_history=attack_history,
                )

            elif decision == DecisionType.REJECT:
                # Prepare critique for regeneration
                previous_critique = self._format_critique(attack)
                if self.verbose:
                    print("\n[System] Regenerating with critique...")

            elif decision == DecisionType.ABSTAIN:
                elapsed = time.time() - start_time
                return DecoderResult(
                    decision=MinimaxDecision(
                        decision=decision,
                        reasoning=reasoning,
                        final_response=None,
                    ),
                    metrics=DecoderMetrics(
                        total_attempts=attempt,
                        final_decision=decision,
                        issues_found_per_attempt=issues_found,
                        time_taken_seconds=elapsed,
                        regeneration_count=attempt - 1,
                    ),
                    draft_history=draft_history,
                    attack_history=attack_history,
                )

        # Should not reach here, but handle edge case
        elapsed = time.time() - start_time
        return DecoderResult(
            decision=MinimaxDecision(
                decision=DecisionType.ABSTAIN,
                reasoning="Exhausted all attempts without confident answer",
                final_response=None,
            ),
            metrics=DecoderMetrics(
                total_attempts=self.max_attempts,
                final_decision=DecisionType.ABSTAIN,
                issues_found_per_attempt=issues_found,
                time_taken_seconds=elapsed,
                regeneration_count=self.max_attempts - 1,
            ),
            draft_history=draft_history,
            attack_history=attack_history,
        )

    def _format_critique(self, attack: AttackResult) -> str:
        """Format verification result as critique for regeneration."""
        critique = f"""Your previous answer had an error:
- Problem: "{attack.problematic_claim}" - {attack.reasoning}"""

        if attack.correction:
            critique += f"\n- Correction: {attack.correction}"

        critique += """

IMPORTANT: Write a new answer to the original question. Do NOT repeat this critique.
Just answer the question directly and accurately. If unsure, say "I'm not certain"."""

        return critique


def create_decoder_from_config(
    generator_model: str,
    adversary_model: str,
    api_keys: Optional[dict[str, str]] = None,
    max_attempts: int = 3,
    verbose: bool = True,
) -> MinimaxDecoder:
    """
    Factory function to create decoder from model names.

    Args:
        generator_model: Model short name (e.g., "smollm2-360m", "llama-3.2-1b")
        adversary_model: Model short name (e.g., "gemini-flash")
        api_keys: Optional dict mapping provider names to API keys
        max_attempts: Maximum generation attempts
        verbose: Print progress

    Returns:
        Configured MinimaxDecoder instance
    """
    from config import get_model_config, get_api_key, ProviderType
    from providers import GeminiProvider, GroqProvider, HuggingFaceProvider, LocalProvider

    api_keys = api_keys or {}

    # Cache for local providers (expensive to load)
    _local_provider_cache: dict[str, LocalProvider] = {}

    def create_provider(model_name: str) -> ModelProvider:
        config = get_model_config(model_name)

        # Local provider doesn't need API key
        if config.provider == ProviderType.LOCAL:
            # Cache local providers since they're expensive to load
            if model_name not in _local_provider_cache:
                _local_provider_cache[model_name] = LocalProvider(model_id=config.model_id)
            return _local_provider_cache[model_name]

        # Get API key for API-based providers
        explicit_key = api_keys.get(config.provider.value)
        key = get_api_key(config.provider, explicit_key)

        # Create appropriate provider
        if config.provider == ProviderType.GEMINI:
            return GeminiProvider(api_key=key, model_id=config.model_id)
        elif config.provider == ProviderType.GROQ:
            return GroqProvider(api_key=key, model_id=config.model_id)
        elif config.provider == ProviderType.HUGGINGFACE:
            return HuggingFaceProvider(api_key=key, model_id=config.model_id)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

    generator_provider = create_provider(generator_model)
    adversary_provider = create_provider(adversary_model)

    return MinimaxDecoder(
        generator_provider=generator_provider,
        adversary_provider=adversary_provider,
        max_attempts=max_attempts,
        verbose=verbose,
    )
