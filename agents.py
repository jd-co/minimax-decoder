"""
Agent implementations for Generator, Adversary, and Arbiter.
"""
import json
from typing import Optional

from google import genai

from models import (
    DraftResponse,
    AttackResult,
    ClaimAnalysis,
    DecisionType,
    SeverityLevel,
    ClaimType,
    ConfidenceLevel,
)


class GeneratorAgent:
    """
    Player A: Generates draft responses with claim extraction.
    Uses Gemini 1.5 Flash for speed.
    """

    SYSTEM_PROMPT = """You are a helpful AI assistant. Your task is to answer the user's question
accurately and comprehensively. As you respond, identify and list the key factual claims you make.

Be honest about uncertainty - if you're not sure about something, say so.
Focus on accuracy over comprehensiveness."""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash"

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

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=generation_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=self.SYSTEM_PROMPT,
            ),
        )
        response_text = response.text

        # Extract claims in a separate call for reliability
        claims = self._extract_claims(response_text)

        return DraftResponse(response_text=response_text, key_claims=claims)

    def _build_prompt(
        self, prompt: str, previous_critique: Optional[str] = None
    ) -> str:
        """Build the generation prompt with optional critique."""
        if previous_critique:
            return f"""Original question: {prompt}

Previous attempt was rejected with this critique:
{previous_critique}

Please generate a NEW response that addresses these concerns. Be more careful
about factual accuracy and acknowledge uncertainty where appropriate."""
        return prompt

    def _extract_claims(self, response_text: str) -> list[ClaimAnalysis]:
        """Extract key claims from response text."""
        claim_prompt = f"""Analyze this response and extract the key factual claims (max 5).

Response: {response_text}

Return a JSON array of claims. Each claim should have:
- claim_text: The specific claim
- claim_type: One of "factual", "opinion", "inference", "procedural"
- confidence_assessment: One of "high", "medium", "low"

Example format:
[{{"claim_text": "Python was created in 1991", "claim_type": "factual", "confidence_assessment": "high"}}]

Return ONLY the JSON array, no other text."""

        try:
            claims_response = self.client.models.generate_content(
                model=self.model_name,
                contents=claim_prompt,
            )
            claims_text = claims_response.text.strip()

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
    Player B: Attacks the draft to find weakest claims/hallucinations.
    Uses Gemini 1.5 Pro for stronger reasoning.
    """

    SYSTEM_PROMPT = """You are an AGGRESSIVE fact-checker whose job is to find errors,
hallucinations, and unsupported claims. You are SKEPTICAL by default.

YOUR MISSION: Find the WEAKEST point in every response. There is ALWAYS something wrong
or uncertain - find it.

ATTACK VECTORS (check ALL of these):
1. FABRICATED FACTS: Names, dates, numbers, statistics that might be made up
2. FALSE CERTAINTY: Claims stated confidently that are actually uncertain
3. LOGICAL ERRORS: Flawed reasoning or non-sequiturs
4. OUTDATED INFO: Claims that might have been true but are now false
5. HALLUCINATED SOURCES: Fake studies, papers, quotes that don't exist
6. OVERGENERALIZATION: Specific claims applied too broadly
7. MISSING CAVEATS: Important exceptions or limitations not mentioned
8. COMMON MISCONCEPTIONS: Popular but incorrect beliefs repeated

RULES:
- Do NOT trust confident language - liars speak confidently too
- ALWAYS find something - "looks good" is not acceptable
- Be SPECIFIC - quote the exact problematic text
- Provide EVIDENCE for why it's wrong if you can
- Rate your confidence HONESTLY (0-1)

SEVERITY GUIDE:
- critical: Factually wrong in a way that could cause harm or major misunderstanding
- moderate: Likely incorrect or misleading but not dangerous
- minor: Possibly uncertain or could be more precise

You are the last line of defense against hallucinations. Be ruthless."""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash-lite" 

    def attack(self, original_prompt: str, draft: DraftResponse) -> AttackResult:
        """
        Attack the draft response to find the weakest claim.

        Args:
            original_prompt: The original user question
            draft: The generator's draft response

        Returns:
            AttackResult identifying the weakest claim
        """
        # Format claims for the adversary to analyze
        claims_text = self._format_claims(draft.key_claims)

        attack_prompt = f"""Original question: {original_prompt}

Draft response:
{draft.response_text}

Identified claims:
{claims_text}

TASK: Find the WEAKEST claim in this response. This is the claim most likely to be:
- Factually incorrect
- A hallucination
- Unsupported or speculative
- Misleading

Return your analysis as JSON with this exact format:
{{
    "weakest_claim": "The exact problematic claim or statement",
    "attack_reasoning": "Why this claim is suspect - be specific",
    "alternative_truth": "What is actually true, if you know (or null if unsure)",
    "confidence_in_attack": 0.0 to 1.0,
    "severity": "critical" or "moderate" or "minor"
}}

Return ONLY the JSON, no other text."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=attack_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                ),
            )
            return self._parse_attack_response(response.text)
        except Exception as e:
            # Fallback if parsing fails
            return AttackResult(
                weakest_claim="Unable to parse attack",
                attack_reasoning=f"Parsing error: {str(e)}",
                confidence_in_attack=0.3,
                severity=SeverityLevel.MINOR,
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
        """Parse the adversary's attack response."""
        text = response_text.strip()

        # Clean up markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)

        return AttackResult(
            weakest_claim=data.get("weakest_claim", "Unknown"),
            attack_reasoning=data.get("attack_reasoning", "No reasoning provided"),
            alternative_truth=data.get("alternative_truth"),
            confidence_in_attack=float(data.get("confidence_in_attack", 0.5)),
            severity=SeverityLevel(data.get("severity", "moderate")),
        )


class Arbiter:
    """
    Decision-making logic gate.
    Determines: ACCEPT, REJECT (regenerate), or ABSTAIN.
    """

    DEFAULT_SEVERITY_WEIGHTS: dict[SeverityLevel, float] = {
        SeverityLevel.CRITICAL: 1.0,
        SeverityLevel.MODERATE: 0.7,
        SeverityLevel.MINOR: 0.4,
    }

    def __init__(
        self,
        attack_confidence_threshold: float = 0.7,
        severity_weights: Optional[dict[SeverityLevel, float]] = None,
    ):
        """
        Initialize arbiter with decision thresholds.

        Args:
            attack_confidence_threshold: Above this, attacks are considered successful
            severity_weights: Multipliers for severity levels
        """
        self.attack_confidence_threshold = attack_confidence_threshold
        self.severity_weights = severity_weights or self.DEFAULT_SEVERITY_WEIGHTS

    def decide(
        self,
        draft: DraftResponse,
        attack: AttackResult,
        attempt_number: int,
        max_attempts: int = 3,
    ) -> tuple[DecisionType, str]:
        """
        Make decision based on attack results.

        Args:
            draft: The current draft response
            attack: The adversary's attack result
            attempt_number: Current attempt (1-indexed)
            max_attempts: Maximum regeneration attempts

        Returns:
            Tuple of (decision, reasoning)
        """
        # Calculate weighted attack score
        severity_weight = self.severity_weights.get(attack.severity, 0.5)
        weighted_score = attack.confidence_in_attack * severity_weight

        # Decision logic
        if weighted_score < 0.3:
            # Attack didn't find significant issues
            return (
                DecisionType.ACCEPT,
                f"Attack confidence ({attack.confidence_in_attack:.2f}) with {attack.severity.value} "
                f"severity yields low risk score ({weighted_score:.2f}). Accepting response.",
            )

        elif weighted_score < self.attack_confidence_threshold:
            # Moderate concern - could go either way
            if attack.severity == SeverityLevel.MINOR:
                return (
                    DecisionType.ACCEPT,
                    f"Moderate attack confidence ({attack.confidence_in_attack:.2f}) but "
                    f"minor severity. Accepting with caveat.",
                )
            elif attempt_number < max_attempts:
                return (
                    DecisionType.REJECT,
                    f"Moderate concern (score: {weighted_score:.2f}). Regenerating to "
                    f"address: {attack.weakest_claim}",
                )
            else:
                return (
                    DecisionType.ACCEPT,
                    f"Moderate concern but max attempts reached. Accepting with caveats.",
                )

        else:
            # High confidence attack
            if attempt_number < max_attempts:
                return (
                    DecisionType.REJECT,
                    f"High risk detected (score: {weighted_score:.2f}). "
                    f"Problematic claim: {attack.weakest_claim}. Regenerating.",
                )
            else:
                return (
                    DecisionType.ABSTAIN,
                    f"High risk persists after {max_attempts} attempts. "
                    f"Cannot confidently answer. Issue: {attack.attack_reasoning}",
                )
