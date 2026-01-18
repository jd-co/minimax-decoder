"""
Pydantic models for Active Minimax Decoder.
All data structures with proper type hints and validation.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """Arbiter decision outcomes."""
    ACCEPT = "accept"
    REJECT = "reject"
    ABSTAIN = "abstain"


class SeverityLevel(str, Enum):
    """Severity levels for identified issues."""
    CRITICAL = "critical"
    MODERATE = "moderate"
    MINOR = "minor"


class ClaimType(str, Enum):
    """Types of claims that can be made in a response."""
    FACTUAL = "factual"
    OPINION = "opinion"
    INFERENCE = "inference"
    PROCEDURAL = "procedural"


class ConfidenceLevel(str, Enum):
    """Confidence assessment levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClaimAnalysis(BaseModel):
    """Single claim extracted from draft response."""
    claim_text: str = Field(description="The specific factual claim extracted")
    claim_type: ClaimType = Field(description="Type of claim")
    confidence_assessment: ConfidenceLevel = Field(
        description="How verifiable/certain this claim is"
    )


class DraftResponse(BaseModel):
    """Generator's draft output."""
    response_text: str = Field(description="The full generated response text")
    key_claims: list[ClaimAnalysis] = Field(
        description="List of key claims made in the response",
        default_factory=list
    )


class AttackResult(BaseModel):
    """Adversary's attack on the draft."""
    weakest_claim: str = Field(
        description="The claim most likely to be false or unsupported"
    )
    attack_reasoning: str = Field(
        description="Why this claim is suspect - cite specific issues"
    )
    alternative_truth: Optional[str] = Field(
        description="What might actually be true instead, if known",
        default=None
    )
    confidence_in_attack: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that this claim is indeed problematic (0-1)"
    )
    severity: SeverityLevel = Field(
        description="Severity if wrong: critical, moderate, minor"
    )


class MinimaxDecision(BaseModel):
    """Arbiter's final decision."""
    decision: DecisionType
    reasoning: str = Field(description="Explanation for the decision")
    final_response: Optional[str] = Field(
        description="Final response text (None if abstained)",
        default=None
    )


class DecoderMetrics(BaseModel):
    """Metrics tracked during decoding process."""
    total_attempts: int = Field(default=1)
    final_decision: DecisionType
    attack_confidences: list[float] = Field(default_factory=list)
    time_taken_seconds: float
    regeneration_count: int = Field(default=0)


class DecoderResult(BaseModel):
    """Complete result from the decoder."""
    decision: MinimaxDecision
    metrics: DecoderMetrics
    draft_history: list[DraftResponse] = Field(default_factory=list)
    attack_history: list[AttackResult] = Field(default_factory=list)
