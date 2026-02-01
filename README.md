# Active Minimax Decoder (AMD)

**A training-free framework for reducing hallucinations in Small Language Models via binary adversarial verification**

> SmolLM2-360M with AMD achieves **0.61% hallucination rate** — **17× lower than Gemini-2.0-Flash** (10.65%) and 96% lower than its own vanilla baseline (15.67%).

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  GENERATOR   │───▶│  ADVERSARY   │───▶│   ARBITER    │
│    (SLM)     │    │   (Gemini)   │    │   (Binary)   │
└──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
   Generate           Verify for          Accept or
   Response        Factual Errors         Abstain
```

## Key Results

Evaluated on [TruthfulQA](https://github.com/sylinrl/TruthfulQA) (817 questions, multiple choice):

| Model | Method | Hal. Rate | Abstention | Accuracy (when answering) |
|-------|--------|-----------|------------|---------------------------|
| Gemini-2.0-Flash | Vanilla | 10.65% | 0% | 89.4% |
| SmolLM2-360M | Vanilla | 15.67% | 0% | 84.3% |
| **SmolLM2-360M** | **+ AMD** | **0.61%** | 75.4% | **97.5%** |
| LFM2-350M | Vanilla | 66.46% | 0% | 33.5% |
| **LFM2-350M** | **+ AMD** | **7.34%** | 59.6% | **81.8%** |

**Key findings:**
- A 360M model with AMD has **17× fewer hallucinations** than Gemini-2.0-Flash (~100B+ params)
- When AMD answers, it achieves **97.5% accuracy** ("know what you don't know")
- Adversary **capability matters more than size**: LFM2-350M outperforms Qwen-1.5B as adversary

## How It Works

1. **Generator (SLM)** produces a candidate response
2. **Adversary (external model)** checks for factual errors — outputs binary TRUE/FALSE
3. **Arbiter** decides:
   - No issue found → **ACCEPT** response
   - Issue found + attempts left → **RETRY** with feedback
   - Issue found + max attempts → **ABSTAIN** (refuse to answer)

No confidence scores. Binary decisions only.

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/sheikjaveed/minimax-decoder.git
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
# SmolLM2-360M as generator, Gemini Flash as adversary
uv run python main.py -g smollm2-360m-local -a gemini-flash \
    --prompt "What happens if you eat watermelon seeds?"
```

### Benchmarking

```bash
# Run SmolLM2 + AMD on TruthfulQA
uv run python benchmark.py -g smollm2-360m-local -a gemini-flash --limit 100

# Run vanilla baseline (for comparison)
uv run python benchmark.py -g smollm2-360m-local --vanilla-only --limit 100

# Quick test (10 questions)
uv run python benchmark.py -g smollm2-360m-local -a gemini-flash --limit 10
```

### Run Experiment Notebooks

The `notebooks/` folder contains standalone experiments that can be run independently on Colab/Kaggle:

```
notebooks/
├── run_smollm2_minimax.ipynb      # SmolLM2-360M + AMD
├── run_smollm2_vanilla.ipynb      # SmolLM2-360M baseline
├── run_lfm2_minimax.ipynb         # LFM2-350M + AMD
├── run_lfm2_vanilla.ipynb         # LFM2-350M baseline
├── run_qwen_minimax.ipynb         # Qwen-1.5B + AMD
├── run_qwen_vanilla.ipynb         # Qwen-1.5B baseline
├── run_smollm2_qwen_adversary.ipynb    # SmolLM2 with Qwen as adversary
├── run_smollm2_lfm2_adversary.ipynb    # SmolLM2 with LFM2 as adversary
└── run_gemini_vanilla.ipynb       # Gemini-2.0-Flash baseline
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
| `lfm2-350m-local` | 350M | Hybrid (Liquid) | Local |
| `lfm2-700m-local` | 700M | Hybrid (Liquid) | Local |
| `lfm2-1.2b-local` | 1.2B | Hybrid (Liquid) | Local |
| `gemini-flash` | ~100B+ | - | Google API |

## Project Structure

```
minimax_decoder/
├── agents.py          # Generator, Adversary, Arbiter agents
├── decoder.py         # Main AMD verification loop
├── models.py          # Pydantic data structures
├── benchmark.py       # TruthfulQA evaluation
├── config.py          # Model configurations
├── main.py            # CLI entry point
├── providers/         # Model providers
│   ├── local.py       # HuggingFace Transformers
│   ├── gemini.py      # Google Gemini API
│   ├── groq.py        # Groq API
│   └── huggingface.py # HuggingFace Inference API
├── data/              # TruthfulQA, HaluEval datasets
├── results/           # Benchmark results (JSON)
└── notebooks/         # Experiment notebooks
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

LLMs are poorly calibrated in their confidence estimates ([Yang et al., 2024](https://arxiv.org/abs/2412.14737)). If LLMs hallucinate, why trust their self-assessed confidence?

AMD uses binary accept/abstain decisions:
- No arbitrary confidence thresholds to tune
- Forces decisive commitment
- More reliable than continuous scores

### Why Abstain?

When AMD can't verify an answer after 3 attempts, it refuses rather than risk hallucination.

**Trade-off:**
- Higher abstention rate (75.4%)
- But only **0.61% hallucination** on accepted answers
- **97.5% accuracy** when answering

For high-stakes applications (medical, legal, financial), refusing when uncertain is better than hallucinating.

### Why External Adversary?

Self-verification inherits the model's own biases. AMD uses an external adversary with:
- Different parametric knowledge
- Different failure modes
- Ability to catch errors the generator cannot see

## Limitations

- **Latency**: ~2.7× slower than vanilla (adversarial verification rounds)
- **API dependency**: Best results require Gemini API for adversary
- **Coverage**: 75% abstention rate may be too high for some applications

## Citation

```bibtex
@misc{javeed2025amd,
  title={Active Minimax Decoder: Reducing Hallucinations in Small Language Models via Binary Adversarial Verification},
  author={Javeed, Sheik},
  year={2026},
  howpublished={\url{https://github.com/sheikjaveed/minimax-decoder}},
  note={RVAI Global}
}
```

## License

MIT

## References

- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958)
- [HaluEval: A Hallucination Evaluation Benchmark](https://arxiv.org/abs/2305.11747)
- [Verbalized Confidence in LLMs](https://arxiv.org/abs/2412.14737)
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
