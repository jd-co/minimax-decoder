# Minimax Decoder

**Adversarial verification for reducing hallucinations in Small Language Models (SLMs)**

> A 360M parameter model with verification achieves **3% hallucination rate** vs **28%** for a 1.5B model without verification.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  GENERATOR   │───▶│   VERIFIER   │───▶│   ARBITER    │
│   (SLM)      │    │  (Gemini)    │    │   (Binary)   │
└──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                    VERIFICATION LOOP
```

## Key Results

Evaluated on [TruthfulQA](https://github.com/sylinrl/TruthfulQA) (100 questions):

| Model | Parameters | Truthful | Hallucination |
|-------|-----------|----------|---------------|
| SmolLM2-360M vanilla | 360M | 33% | 48% |
| **SmolLM2-360M + Minimax** | 360M | 48% | **3%** |
| Qwen-1.5B vanilla | 1.5B | 51% | 28% |

**Key finding**: A 360M model with adversarial verification has **9x fewer hallucinations** than a 4x larger 1.5B model without verification.

## How It Works

1. **Generator (SLM)** creates a draft response
2. **Verifier (Gemini)** checks for factual errors (binary: YES/NO)
3. **Arbiter** decides:
   - No issue → **ACCEPT**
   - Issue found + attempts left → **REJECT** (regenerate with feedback)
   - Issue found + max attempts → **ABSTAIN** (refuse to answer)

No arbitrary confidence scores. Binary decisions only.

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/yourusername/minimax-decoder.git
cd minimax-decoder
uv sync
```

### Install PyTorch for local models

```bash
uv pip install torch
```

## Configuration

Create a `.env` file:

```bash
GOOGLE_API_KEY=your-gemini-api-key
```

Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).

## Usage

### Run with Local SLM

```bash
# SmolLM2-360M as generator, Gemini Flash as verifier
uv run python main.py -g smollm2-360m-local -a gemini-flash \
    --prompt "What happens if you eat watermelon seeds?"
```

### Benchmarking

```bash
# Run SmolLM2 + Minimax (100 questions)
uv run python benchmark.py -g smollm2-360m-local -a gemini-flash --limit 100

# Run Qwen-1.5B vanilla baseline (for comparison)
uv run python benchmark.py -g qwen2.5-1.5b-local --vanilla-only --limit 100

# Quick test (10 questions)
uv run python benchmark.py -g smollm2-360m-local -a gemini-flash --limit 10
```

### List Available Models

```bash
uv run python benchmark.py --list-models
```

**Supported models:**
| Model | Params | Architecture | Provider |
|-------|--------|--------------|----------|
| `smollm2-360m-local` | 360M | Transformer | Local |
| `smollm2-1.7b-local` | 1.7B | Transformer | Local |
| `qwen2.5-0.5b-local` | 0.5B | Transformer | Local |
| `qwen2.5-1.5b-local` | 1.5B | Transformer | Local |
| `lfm2-350m-local` | 350M | **Hybrid (Liquid)** | Local |
| `lfm2-700m-local` | 700M | **Hybrid (Liquid)** | Local |
| `lfm2-1.2b-local` | 1.2B | **Hybrid (Liquid)** | Local |
| `gemini-flash` | - | - | Google API |
| `llama-3.2-1b` | 1B | Transformer | Groq API |

## Project Structure

```
minimax_decoder/
├── models.py          # Pydantic data structures
├── agents.py          # Generator, Adversary, Arbiter
├── decoder.py         # Main verification loop
├── benchmark.py       # TruthfulQA evaluation
├── config.py          # Model configurations
├── main.py            # CLI entry point
├── providers/         # Model providers
│   ├── local.py       # HuggingFace Transformers
│   ├── gemini.py      # Google Gemini API
│   ├── groq.py        # Groq API
│   └── huggingface.py # HuggingFace Inference API
├── data/              # TruthfulQA dataset
└── results/           # Benchmark results
```

## Programmatic Usage

```python
from decoder import create_decoder_from_config

decoder = create_decoder_from_config(
    generator_model="smollm2-360m-local",
    adversary_model="gemini-flash",
    max_attempts=3,
)

result = decoder.decode("What causes rainbows?")

if result.decision.decision.value == "accept":
    print(result.decision.final_response)
elif result.decision.decision.value == "abstain":
    print("Uncertain - refusing to answer")
    print("Reason:", result.decision.reasoning)
```

## Design Decisions

### Why Binary Decisions?

Previous versions used LLM-generated confidence scores (0-1). Problem: **LLMs hallucinate scores too**.

Current approach:
- Verifier outputs `issue_found: true/false`
- No arbitrary thresholds to tune
- More defensible for research

### Why Abstain?

When the system can't verify an answer after 3 attempts, it refuses rather than risk hallucination.

**Trade-off:**
- Higher abstention rate (42%)
- But only 3% hallucination on accepted answers

For high-stakes applications (medical, legal, financial), refusing when uncertain is better than hallucinating.

## Limitations

- **Latency**: 3-10x slower than vanilla (verification rounds)
- **API costs**: Requires Gemini API calls for verification
- **Abstention**: May refuse valid questions if verifier is too strict

## Citation

```bibtex
@misc{minimax-decoder-2025,
  title={Minimax Decoder: Adversarial Verification for SLM Hallucination Reduction},
  author={Sheikh Javeed},
  year={2025},
  url={https://github.com/yourusername/minimax-decoder}
}
```

## License

MIT

## References

- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958)
- [Decoding Game: On Minimax Optimality of Heuristic Text Generation Strategies](https://arxiv.org/abs/2410.03968)
- [SmolLM2: A Family of Small Language Models](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
