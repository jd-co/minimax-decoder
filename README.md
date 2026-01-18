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

## Research Contribution

This project implements a **novel active adversarial decoder** for hallucination reduction. While prior work provides theoretical analysis, this is the first implementation of an active adversarial game loop during generation.

| Prior Work | What They Did | Our Contribution |
|------------|---------------|------------------|
| [Decoding Game (ICLR 2025)](https://arxiv.org/abs/2410.03968) | Theoretical analysis only | **Working implementation** |
| [Consensus Game (ICLR 2024)](https://arxiv.org/abs/2310.09139) | Seeks equilibrium/consensus | **Adversarial attack** approach |
| Contrastive methods (DoLa) | Layer contrast | **Active agent** attacking |

**Key novelty**: No published work implements an active adversary that attacks during generation with regeneration based on critique.

## Benchmark Results

Evaluated on [TruthfulQA](https://github.com/sylinrl/TruthfulQA) (100 questions) using LLM-as-Judge methodology:

| Metric | Minimax Decoder | Vanilla Gemini | Improvement |
|--------|-----------------|----------------|-------------|
| **Truthful Rate** | **78%** | 66% | **+12%** |
| **Hallucination Rate** | **4%** | 13% | **-9%** |
| Mixed (hedged) | 18% | 21% | -3% |

**Key findings**:
- **69% relative reduction** in hallucination rate (13% → 4%)
- **+12% absolute improvement** in truthful responses
- Minimax decoder catches and corrects potential hallucinations through adversarial verification

## Overview

Standard LLMs generate responses without verification, leading to hallucinations. This system adds an **adversarial verification layer**:

1. **Generator** creates a draft response
2. **Adversary** attacks the draft to find weakest claims
3. **Arbiter** decides: ACCEPT, REJECT (regenerate), or ABSTAIN
4. If rejected, regenerate with critique feedback (max 3 attempts)

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/jd-co/minimax-decoder.git
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

### Benchmarking

Run evaluation on TruthfulQA dataset:

```bash
# Quick test (10 questions)
uv run python benchmark.py --sample 10

# Full benchmark (50 questions)
uv run python benchmark.py --sample 50

# Without vanilla baseline (faster)
uv run python benchmark.py --sample 20 --no-vanilla

# See all options
uv run python benchmark.py --help
```

**Benchmark output:**
```
============================================================
BENCHMARK RESULTS
============================================================

Total Questions: 100

--- MINIMAX DECODER ---
Decoder Decisions:
  Accepted:   98
  Abstained:  2
  Avg Attempts: 1.85

LLM Judge Verdicts:
  Truthful:      78 (78.0%)
  Hallucination: 4 (4.0%)
  Refusal:       0
  Mixed:         18

--- VANILLA GEMINI ---
LLM Judge Verdicts:
  Truthful:      66 (66.0%)
  Hallucination: 13 (13.0%)
  Refusal:       0
  Mixed:         21

--- COMPARISON ---
Truthful Rate:      +12.0% (BETTER)
Hallucination Rate: -9.0% (BETTER)
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

### CLI Options

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

## Architecture

```
minimax_decoder/
├── models.py      # Pydantic data structures
├── agents.py      # Generator, Adversary, Arbiter implementations
├── decoder.py     # Main game loop orchestration
├── main.py        # CLI entry point
├── benchmark.py   # TruthfulQA evaluation with LLM-as-Judge
├── data/          # TruthfulQA dataset
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

**LLM Judge** (`LLMJudge`)
- Evaluates responses against TruthfulQA reference answers
- Returns verdict: truthful / hallucination / refusal / mixed
- Based on [TruthfulQA's GPT-judge methodology](https://github.com/sylinrl/TruthfulQA)

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

## Evaluation Methodology

We use **LLM-as-Judge** based on TruthfulQA's evaluation protocol:

1. **Generate response** with both Minimax decoder and vanilla baseline
2. **Judge evaluates** each response against reference answers
3. **Verdict assigned**: truthful / hallucination / refusal / mixed
4. **Metrics computed**: truthful rate, hallucination rate, etc.

This approach achieves 90-95% agreement with human evaluators ([source](https://arxiv.org/abs/2109.07958)).

## Research Background

This prototype implements concepts from:

- **Decoding Game** (ICLR 2025) - Game-theoretic analysis of text generation
- **Consensus Game** (ICLR 2024) - Equilibrium-based decoding
- **DoLa** - Contrastive decoding for factuality
- **Test-time compute scaling** - System 2 thinking approaches

### Comparison with Related Work

| Approach | Game-Theoretic | Active Adversary | Inference-Only | Abstention |
|----------|---------------|------------------|----------------|------------|
| **This Work** | Yes | **Yes** | Yes | Yes |
| Decoding Game | Yes | No (theory) | N/A | No |
| Consensus Game | Yes | No | Yes | No |
| DoLa | No | No | Yes | No |
| Self-Verification | No | No | Yes | Weak |

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

## Limitations

- **Time overhead**: ~4-8x slower than vanilla (multiple LLM calls)
- **API costs**: Requires 2-3 API calls per question
- **Modern LLMs**: Large models (GPT-4, Gemini) already have low hallucination rates on common misconceptions
- **Best suited for**: Smaller models, obscure facts, or high-stakes applications

## License

MIT

## Citation

If you use this work, please cite:

```bibtex
@misc{minimax-decoder,
  title={Active Minimax Decoder: Adversarial Verification for LLM Hallucination Reduction},
  author={jd-co},
  year={2025},
  url={https://github.com/jd-co/minimax-decoder}
}
```

## References

- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958)
- [Decoding Game: On Minimax Optimality of Heuristic Text Generation Strategies](https://arxiv.org/abs/2410.03968)
- [The Consensus Game: Language Model Generation via Equilibrium Search](https://arxiv.org/abs/2310.09139)

## Contributing

Contributions welcome! Areas of interest:

- [x] TruthfulQA benchmark evaluation
- [x] LLM-as-Judge evaluation methodology
- [ ] More datasets (HaluEval, FEVER)
- [ ] Local model support (Llama, Mistral) with logit extraction
- [ ] Process Reward Model integration
- [ ] Async/parallel execution for speed
- [ ] Web UI demo
