# Finance Tool-Call Scaling: Does Fine-Tuning Compound or Substitute for Inference-Time Scaling?

A prototype that uses all three of Red Hat's open-source ML libraries —
**sdg_hub**, **training_hub**, and **its_hub** — to answer a real research question:

> **Does LoRA fine-tuning for structured tool-calling and inference-time scaling
> compound with each other, or does one substitute for the other?**

---

## Problem Setup

Given a user question in plain English (e.g. *"What was Apple's revenue in FY2022?"*),
a model must produce a correctly parameterised `search_tool` call:

```json
{
  "name": "search_tool",
  "arguments": {
    "query": "Apple total revenue fiscal year 2022",
    "company": "Apple Inc.",
    "report_type": "10K",
    "year": 2022
  }
}
```

Ground-truth labels come from the [FinanceBench](https://github.com/patronusai/financebench)
dataset, which provides company name, SEC filing type (10K/10Q/8K), and fiscal period
for each question.

**Evaluation metric**: strict exact match on `company`, `year`, `report_type`, `quarter`.

---

## Pipeline

```
FinanceBench (local)
        │
        ▼
┌──────────────────┐
│  step1            │  sdg_hub: paraphrase questions 4× and LLM-label
│  generate_data    │  query + section fields from evidence text
└──────────┬───────┘
           │ data/train.jsonl  data/eval.jsonl
           ▼
┌──────────────────┐
│  step2            │  training_hub: LoRA fine-tune Qwen2.5-1.5B-Instruct
│  train_lora       │  on tool-call formatted training data
└──────────┬───────┘
           │ models/lora-finance-toolcall/
           ▼
┌──────────────────┐
│  step3            │  its_hub: evaluate base + fine-tuned at
│  evaluate_scaling │  N ∈ {1,3,5,10} with SelfConsistency + BestOfN
└──────────┬───────┘
           │ results/scaling_results.json
           ▼
┌──────────────────┐
│  step4            │  Accuracy tables, ITS-delta plots, narrative
│  analyze_results  │
└──────────────────┘
```

---

## Quick Start

### 1. Install

```bash
cd finance-toolcall-scaling
pip install -r requirements.txt
```

### 2. Provide FinanceBench data

Download FinanceBench (parquet/jsonl/CSV) and set the path in `config.yaml`:

```yaml
dataset:
  local_path: "/path/to/financebench"
```

Or pass a HuggingFace dataset name like `"PatronusAI/financebench"` — the code
will call `datasets.load_dataset()` automatically.

### 3. Set API key for sdg_hub augmentation

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
# then update config.yaml: generation.model / generation.api_key_env
```

### 4. Run the pipeline

```bash
# Step 1: Augment FinanceBench with sdg_hub (needs API key)
python src/step1_generate_data.py

# Step 2: Fine-tune with training_hub (~30-60 min CPU / ~5 min GPU)
python src/step2_train_lora.py

# Step 3: Evaluate inference-time scaling with its_hub
python src/step3_evaluate_scaling.py

# Step 4: Analyze and plot results
python src/step4_analyze_results.py
```

Results land in `results/`:
- `scaling_results.json` — per-question scores for every condition
- `summary.txt` — narrative analysis
- `plots/accuracy_vs_budget.png` — accuracy vs N line chart
- `plots/its_delta.png` — ITS improvement delta (compound vs substitute)
- `plots/param_heatmap.png` — per-parameter accuracy heatmap

---

## Configuration

All knobs live in `config.yaml`:

| Key | Default | Notes |
|-----|---------|-------|
| `dataset.max_examples` | 200 | Increase for more statistical power |
| `dataset.eval_fraction` | 0.25 | Held-out fraction |
| `generation.model` | `anthropic/claude-haiku-4-5-20251001` | Any LiteLLM model |
| `generation.paraphrase_multiplier` | 4 | Training data multiplier |
| `training.base_model` | `Qwen/Qwen2.5-1.5B-Instruct` | Any causal LM |
| `training.lora_r` | 16 | LoRA rank |
| `training.num_epochs` | 3 | |
| `training.bf16` | false | Set `true` on GPU |
| `scaling.budgets` | [1, 3, 5, 10] | Inference samples per question |
| `scaling.mode` | `local_hf` | Switch to `api` for vLLM/Ollama |

---

## Library Usage Notes

### sdg_hub
- Flow defined in `flows/tool_call_augmentation.yaml`
- Stage A: `RowMultiplierBlock → PromptBuilderBlock → LLMChatBlock` — paraphrases each question 4×
- Stage B: `PromptBuilderBlock → LLMChatBlock → JSONParserBlock` — LLM-labels `query` and `section` from evidence text
- This is the genuine SDG contribution: FinanceBench provides metadata labels but not the free-text `query` or document `section` needed to train a retrieval-ready tool call

### training_hub
- `lora_sft()` with `dataset_type="chat_template"` applies the Qwen2.5 tokenizer template
- The Qwen2.5 template natively handles `tool_calls` in assistant messages, so training data in OpenAI format is consumed directly
- CPU-compatible: `bf16: false`, `fp16: false`, `torch.float32`

### its_hub
- Custom `LocalHFLanguageModel(AbstractLanguageModel)` in `src/local_lm.py` wraps a
  HuggingFace model without requiring a vLLM/Ollama server
- `asyncio.to_thread` bridges the synchronous transformers pipeline into the async interface
- `SelfConsistency(tool_vote="tool_hierarchical")` — uses its_hub's built-in structured
  output voting: first votes on tool name, then on each argument independently
- `BestOfN(orm=LLMJudge(...))` — reuses the same loaded model as judge to avoid
  loading two models in memory on CPU

---

## Research Question: Compound or Substitute?

The key metric is the **ITS delta**: Δ(N) = acc(N) − acc(N=1)

- **Substitution**: Δ shrinks after fine-tuning — the model already knows the right
  format, so sampling multiple times adds less marginal value
- **Compounding**: Δ grows after fine-tuning — better base output quality gives the
  voting/ranking algorithm higher-quality candidates to choose from
- **Mixed**: different algorithms (SC vs BestOfN) show opposite effects

Typical hypothesis: *structured output tasks show substitution* because the bottleneck
shifts from "knowing what format to use" (solved by training) to "knowing which company/year
to cite" (harder to fix with more samples). But ITS may still help BestOfN selectively
by surfacing rare correct calls that SC would outvote.

---

## Design Decisions

**Why FinanceBench?** Objective evaluation (exact-match on structured metadata), real
production use case (financial RAG), and well-known benchmark.

**Why no vLLM server?** The `LocalHFLanguageModel` pattern shows how to implement the
`AbstractLanguageModel` interface for any model — more architecturally instructive than
just pointing at an OpenAI endpoint. The `scaling.mode: api` config option preserves
the ability to run against a server with zero code changes.

**Why Qwen2.5-1.5B?** Small enough for CPU demo; tool-call capable out of the box;
the Qwen2.5 chat template handles OpenAI `tool_calls` format natively.

**Train/eval split at question level**: Paraphrases of the same question are always
in the same split — prevents the fine-tuned model from seeing paraphrases of eval
questions during training.

---

## AI-Assisted Development

This project was built with Claude Code (Anthropic). The AI contributed:
- Translating the research question into a concrete experiment design
- sdg_hub flow YAML syntax (two-stage: RowMultiplier + paraphrase + label)
- `LocalHFLanguageModel` interface implementation bridging sync HF pipeline → async its_hub
- `_try_parse_tool_call()` multi-format parser for robust extraction from base model output
- FinanceBench column normalization and period parsing
- Narrative analysis heuristics in `_add_narrative()` (compound/substitute thresholds)

Key decisions made by human: research question framing, tool parameter selection,
choice of FinanceBench as evaluation dataset, evaluation metric (parameter-level exact match).
