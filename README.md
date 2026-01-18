# Active Minimax Decoder

An adversarial verification system for reducing LLM hallucinations using game-theoretic principles.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  GENERATOR   │───>│  ADVERSARY   │───>│   ARBITER    │
│   (Draft)    │    │   (Attack)   │    │   (Decide)   │
└──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                    GAME LOOP
```

## Overview

Standard LLMs generate responses without verification, leading to hallucinations. This system adds an **adversarial verification layer**:

1. **Generator** creates a draft response
2. **Adversary** attacks the draft to find weakest claims
3. **Arbiter** decides: ACCEPT, REJECT (regenerate), or ABSTAIN
4. If rejected, regenerate with critique feedback (max 3 attempts)

This implements the theory from ["Decoding Game: On Minimax Optimality of Heuristic Text Generation Strategies"](https://arxiv.org/abs/2410.03968) (ICLR 2025) as a working prototype.

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/yourusername/minimax-decoder.git
cd minimax-decoder
uv sync
```

## Configuration

Create a `.env` file with your Gemini API key:

```bash
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).

## Usage

### Basic Usage

```bash
# Custom prompt
uv run python main.py --prompt "What year was the Eiffel Tower built?"

# Run a predefined test
uv run python main.py --test safe_algorithm

# List all test prompts
uv run python main.py --list-tests
```

### Test Prompts

The system includes test prompts to demonstrate hallucination detection:

| Test | Type | Expected Behavior |
|------|------|-------------------|
| `hallucination_study` | Fabricated study | Should reject or say "I don't know" |
| `fake_python_feature` | Non-existent feature | Should reject false premise |
| `fabricated_person` | Made-up researcher | Should express uncertainty |
| `false_premise` | Incorrect assumption | Should correct the premise |
| `safe_algorithm` | Valid question | Should accept (possibly after refinement) |
| `safe_capital` | Simple fact | Should accept quickly |
| `safe_protocol` | Technical explanation | Should accept |

### Options

```bash
uv run python main.py --help

Options:
  --prompt, -p       Custom prompt to test
  --test, -t         Run a predefined test prompt
  --api-key, -k      Gemini API key (or use GOOGLE_API_KEY env var)
  --max-attempts, -m Maximum generation attempts (default: 3)
  --threshold        Attack confidence threshold (default: 0.7)
  --quiet, -q        Suppress verbose output
  --json             Output result as JSON
  --run-all-tests    Run all predefined test prompts
  --list-tests       List all predefined test prompts
```

### Example Output

```
Prompt: Explain how binary search works

==================================================
Attempt 1/3
==================================================

[Generator] Creating draft response...
[Generator] Draft created with 4 claims

[Adversary] Analyzing for weaknesses...
[Adversary] Weakest claim: Integer division truncation issue
[Adversary] Attack confidence: 0.90
[Adversary] Severity: moderate

[Arbiter] Decision: reject
[Arbiter] Reasoning: Moderate concern. Regenerating...

==================================================
Attempt 2/3
==================================================
...

============================================================
FINAL RESULT
============================================================

Decision: ACCEPT
Reasoning: Attack confidence (0.80) but minor severity. Accepting.

Final Response:
----------------------------------------
[Improved response addressing edge cases]
----------------------------------------

Metrics:
  - Total attempts: 3
  - Regenerations: 2
  - Time taken: 31.88s
  - Attack confidences: [0.9, 0.9, 0.8]
```

## Architecture

```
minimax_decoder/
├── models.py      # Pydantic data structures
├── agents.py      # Generator, Adversary, Arbiter implementations
├── decoder.py     # Main game loop orchestration
├── main.py        # CLI entry point
└── .env           # API key (not committed)
```

### Components

**Generator** (`GeneratorAgent`)
- Uses Gemini 2.0 Flash for fast drafting
- Extracts key claims from responses
- Incorporates critique feedback on regeneration

**Adversary** (`AdversaryAgent`)
- Aggressive fact-checker role
- Identifies weakest claim in each response
- Returns confidence score (0-1) and severity (critical/moderate/minor)

**Arbiter** (`Arbiter`)
- Simple Python logic gate
- Weighted score = confidence × severity_weight
- Thresholds determine ACCEPT/REJECT/ABSTAIN

### Decision Logic

```
weighted_score = attack_confidence × severity_weight

if weighted_score < 0.3:
    → ACCEPT (low risk)
elif weighted_score < 0.7:
    → REJECT if attempts remain, else ACCEPT with caveats
else:
    → REJECT if attempts remain, else ABSTAIN
```

## Research Background

This prototype implements concepts from:

- **Decoding Game** (ICLR 2025) - Game-theoretic analysis of text generation
- **Consensus Game** (ICLR 2024) - Equilibrium-based decoding
- **DoLa** - Contrastive decoding for factuality
- **Test-time compute scaling** - System 2 thinking approaches

### Current Limitations (vs. Full Research Implementation)

| Aspect | This Prototype | Full Research |
|--------|----------------|---------------|
| Confidence | LLM self-reported | Logit-based extraction |
| Verifier | Adversary prompt | Process Reward Model (PRM) |
| Evaluation | Manual test prompts | TruthfulQA, HaluEval benchmarks |
| Baselines | None | DoLa, self-verification, etc. |

## Programmatic Usage

```python
from decoder import MinimaxDecoder

decoder = MinimaxDecoder(
    api_key="your-api-key",
    max_attempts=3,
    attack_threshold=0.7,
    verbose=True
)

result = decoder.decode("What causes rainbows?")

if result.decision.decision.value == "accept":
    print(result.decision.final_response)
elif result.decision.decision.value == "abstain":
    print("System uncertain:", result.decision.reasoning)

# Access metrics
print(f"Attempts: {result.metrics.total_attempts}")
print(f"Time: {result.metrics.time_taken_seconds:.2f}s")
```

## License

MIT

## Citation

If you use this work, please cite:

```bibtex
@misc{minimax-decoder,
  title={Active Minimax Decoder: Adversarial Verification for LLM Hallucination Reduction},
  year={2025},
  url={https://github.com/yourusername/minimax-decoder}
}
```

## Contributing

Contributions welcome! Areas of interest:

- [ ] Benchmark evaluation (TruthfulQA, HaluEval)
- [ ] Local model support (Llama, Mistral) with logit extraction
- [ ] Process Reward Model integration
- [ ] Async/parallel execution
- [ ] Web UI demo
