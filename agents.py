"""
Agent implementations for Generator, Adversary, and Arbiter.
Refactored to use provider abstraction for multi-model support.
"""

import json
from typing import Optional

from models import (
    DraftResponse,
    AttackResult,
    ClaimAnalysis,
    DecisionType,
    ClaimType,
    ConfidenceLevel,
)
from providers.base import ModelProvider


class GeneratorAgent:
    """
    Player A: Generates draft responses with claim extraction.
    Uses any configured model provider.
    """

    SYSTEM_PROMPT = """You are a helpful AI assistant. Your task is to answer the user's question
accurately and comprehensively. As you respond, identify and list the key factual claims you make.

Be honest about uncertainty - if you're not sure about something, say so.
Focus on accuracy over comprehensiveness."""

    CLAIM_EXTRACTION_PROMPT = """Analyze this response and extract the key factual claims (max 5).

Response: {response_text}

Return a JSON array of claims. Each claim should have:
- claim_text: The specific claim
- claim_type: One of "factual", "opinion", "inference", "procedural"
- confidence_assessment: One of "high", "medium", "low"

Example format:
[{{"claim_text": "Python was created in 1991", "claim_type": "factual", "confidence_assessment": "high"}}]

Return ONLY the JSON array, no other text."""

    def __init__(self, provider: ModelProvider, claim_extractor: Optional[ModelProvider] = None):
        """
        Initialize generator agent.

        Args:
            provider: Model provider for generation
            claim_extractor: Optional separate provider for claim extraction
                            (defaults to same provider)
        """
        self.provider = provider
        self.claim_extractor = claim_extractor or provider

    def generate(
        self, prompt: str, previous_critique: Optional[str] = None
    ) -> DraftResponse:
        """
        Generate a draft response, optionally incorporating critique feedback.

        Args:
            prompt: User's original question
            previous_critique: Feedback from failed attempt (if regenerating)

        Returns:
            DraftResponse with text and extracted claims
        """
        generation_prompt = self._build_prompt(prompt, previous_critique)

        response_text = self.provider.generate(
            prompt=generation_prompt,
            system_prompt=self.SYSTEM_PROMPT,
        )

        # Extract claims in a separate call for reliability
        claims = self._extract_claims(response_text)

        return DraftResponse(response_text=response_text, key_claims=claims)

    def _build_prompt(
        self, prompt: str, previous_critique: Optional[str] = None
    ) -> str:
        """Build the generation prompt with optional critique."""
        if previous_critique:
            return f"""Question: {prompt}

{previous_critique}

Now answer the question above:"""
        return prompt

    def _extract_claims(self, response_text: str) -> list[ClaimAnalysis]:
        """Extract key claims from response text."""
        claim_prompt = self.CLAIM_EXTRACTION_PROMPT.format(response_text=response_text)

        try:
            claims_response = self.claim_extractor.generate(
                prompt=claim_prompt,
                max_tokens=512,
                temperature=0.3,  # Lower temperature for structured output
            )
            claims_text = claims_response.strip()

            # Clean up response - remove markdown code blocks if present
            if claims_text.startswith("```"):
                claims_text = claims_text.split("```")[1]
                if claims_text.startswith("json"):
                    claims_text = claims_text[4:]
            claims_text = claims_text.strip()

            claims_data = json.loads(claims_text)
            return [
                ClaimAnalysis(
                    claim_text=c.get("claim_text", ""),
                    claim_type=ClaimType(c.get("claim_type", "factual")),
                    confidence_assessment=ConfidenceLevel(
                        c.get("confidence_assessment", "medium")
                    ),
                )
                for c in claims_data
                if c.get("claim_text")
            ]
        except Exception:
            return []


class AdversaryAgent:
    """
    Player B: Fact-checker that verifies claims with BINARY decisions.
    No arbitrary confidence scores - just YES/NO issue detection.
    """

    SYSTEM_PROMPT = """You are a strict fact-checker. Your job is to determine if a response contains factual errors.

CHECK FOR:
1. FABRICATED FACTS: Made-up names, dates, numbers, statistics
2. FACTUAL ERRORS: Statements that contradict established facts
3. COMMON MISCONCEPTIONS: Popular but incorrect beliefs
4. HALLUCINATED SOURCES: Fake studies, papers, quotes

IMPORTANT RULES:
- Only flag CLEAR factual errors, not style or minor issues
- If the response is factually correct (even if incomplete), say NO ISSUE
- If the response says "I don't know" or refuses to answer, say NO ISSUE
- Be SPECIFIC about what is wrong and what is correct

You must give a binary YES/NO decision. No scores, no maybes."""

    ATTACK_PROMPT = """Question: {original_prompt}

Response to verify:
{response_text}

Claims identified:
{claims_text}

Is there a FACTUAL ERROR in this response?

Return JSON with this EXACT format:
{{
    "issue_found": true or false,
    "problematic_claim": "The specific wrong claim" or null if no issue,
    "reasoning": "Why this is wrong / why response is correct",
    "correction": "What is actually true" or null
}}

Return ONLY the JSON."""

    def __init__(self, provider: ModelProvider):
        """
        Initialize adversary agent.

        Args:
            provider: Model provider for adversarial analysis
        """
        self.provider = provider

    def attack(self, original_prompt: str, draft: DraftResponse) -> AttackResult:
        """
        Verify the draft response for factual errors.

        Args:
            original_prompt: The original user question
            draft: The generator's draft response

        Returns:
            AttackResult with binary issue_found decision
        """
        claims_text = self._format_claims(draft.key_claims)

        attack_prompt = self.ATTACK_PROMPT.format(
            original_prompt=original_prompt,
            response_text=draft.response_text,
            claims_text=claims_text,
        )

        try:
            response = self.provider.generate(
                prompt=attack_prompt,
                system_prompt=self.SYSTEM_PROMPT,
            )
            return self._parse_attack_response(response)
        except Exception as e:
            # On parsing error, assume no issue (conservative)
            return AttackResult(
                issue_found=False,
                problematic_claim=None,
                reasoning=f"Verification error: {str(e)}",
                correction=None,
            )

    def _format_claims(self, claims: list[ClaimAnalysis]) -> str:
        """Format claims for adversary analysis."""
        if not claims:
            return "No explicit claims extracted."
        return "\n".join(
            [
                f"- {c.claim_text} (type: {c.claim_type.value}, confidence: {c.confidence_assessment.value})"
                for c in claims
            ]
        )

    def _parse_attack_response(self, response_text: str) -> AttackResult:
        """Parse the adversary's binary verification response."""
        text = response_text.strip()

        # Clean up markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)

        return AttackResult(
            issue_found=bool(data.get("issue_found", False)),
            problematic_claim=data.get("problematic_claim"),
            reasoning=data.get("reasoning", "No reasoning provided"),
            correction=data.get("correction"),
        )


class Arbiter:
    """
    Binary decision-making logic gate.
    Simple logic: Issue found? → REJECT/ABSTAIN. No issue? → ACCEPT.
    No arbitrary confidence scores.
    """

    def __init__(self):
        """Initialize arbiter. No thresholds needed - binary decisions only."""
        pass

    def decide(
        self,
        draft: DraftResponse,
        attack: AttackResult,
        attempt_number: int,
        max_attempts: int = 3,
    ) -> tuple[DecisionType, str]:
        """
        Make binary decision based on whether issue was found.

        Logic:
        - No issue found → ACCEPT
        - Issue found + attempts remaining → REJECT (regenerate)
        - Issue found + max attempts reached → ABSTAIN (refuse to answer)

        Args:
            draft: The current draft response
            attack: The adversary's verification result
            attempt_number: Current attempt (1-indexed)
            max_attempts: Maximum regeneration attempts

        Returns:
            Tuple of (decision, reasoning)
        """
        if not attack.issue_found:
            # No factual error detected - accept the response
            return (
                DecisionType.ACCEPT,
                f"Verification passed. {attack.reasoning}",
            )

        # Issue was found
        if attempt_number < max_attempts:
            # Still have attempts - regenerate
            return (
                DecisionType.REJECT,
                f"Issue found: {attack.problematic_claim}. Regenerating.",
            )
        else:
            # Max attempts reached - refuse to answer
            return (
                DecisionType.ABSTAIN,
                f"Cannot provide accurate answer after {max_attempts} attempts. "
                f"Issue: {attack.problematic_claim}. {attack.reasoning}",
            )
