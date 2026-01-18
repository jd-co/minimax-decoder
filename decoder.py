"""
MinimaxDecoder: Main orchestration of the adversarial verification game loop.
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


class MinimaxDecoder:
    """
    Active Minimax Decoder for hallucination reduction.

    Game Loop:
    1. Generator creates draft response
    2. Adversary attacks to find weakest claim
    3. Arbiter decides: ACCEPT, REJECT (regenerate), or ABSTAIN
    4. If REJECT: regenerate with critique, max 2 retries
    5. If still failing: ABSTAIN
    """

    def __init__(
        self,
        api_key: str,
        max_attempts: int = 3,
        attack_threshold: float = 0.7,
        verbose: bool = True,
    ):
        """
        Initialize the decoder with all agents.

        Args:
            api_key: Google Gemini API key
            max_attempts: Maximum generation attempts before abstaining
            attack_threshold: Confidence threshold for successful attacks
            verbose: Print progress during decoding
        """
        self.generator = GeneratorAgent(api_key)
        self.adversary = AdversaryAgent(api_key)
        self.arbiter = Arbiter(attack_confidence_threshold=attack_threshold)
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
        attack_confidences: list[float] = []

        previous_critique: Optional[str] = None

        for attempt in range(1, self.max_attempts + 1):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Attempt {attempt}/{self.max_attempts}")
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

            # Step 2: Adversary attacks
            if self.verbose:
                print("\n[Adversary] Analyzing for weaknesses...")

            attack = self.adversary.attack(prompt, draft)
            attack_history.append(attack)
            attack_confidences.append(attack.confidence_in_attack)

            if self.verbose:
                print(f"[Adversary] Weakest claim: {attack.weakest_claim}")
                print(f"[Adversary] Attack confidence: {attack.confidence_in_attack:.2f}")
                print(f"[Adversary] Severity: {attack.severity.value}")

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
                        attack_confidences=attack_confidences,
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
                        attack_confidences=attack_confidences,
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
                attack_confidences=attack_confidences,
                time_taken_seconds=elapsed,
                regeneration_count=self.max_attempts - 1,
            ),
            draft_history=draft_history,
            attack_history=attack_history,
        )

    def _format_critique(self, attack: AttackResult) -> str:
        """Format attack result as critique for regeneration."""
        critique = f"""The following claim was identified as problematic:

CLAIM: {attack.weakest_claim}

ISSUE: {attack.attack_reasoning}

SEVERITY: {attack.severity.value}"""

        if attack.alternative_truth:
            critique += f"\n\nCORRECTION: {attack.alternative_truth}"

        critique += "\n\nPlease regenerate with this issue addressed. If you are uncertain about something, explicitly say so."

        return critique
