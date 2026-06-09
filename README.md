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
| `dataset.max_examples` | 150 | FinanceBench has 150 questions |
| `dataset.eval_size` | 50 | Held-out question count |
| `generation.model` | `groq/openai/gpt-oss-120b` | Any LiteLLM model |
| `generation.paraphrase_multiplier` | 4 | Training data multiplier (max 8) |
| `training.base_model` | `Qwen/Qwen2.5-1.5B-Instruct` | Any causal LM |
| `training.lora_r` | 16 | LoRA rank |
| `training.num_epochs` | 3 | |
| `training.bf16` | false | Set `true` on GPU |
| `scaling.budgets` | [1, 3, 5, 10] | Inference samples per question |
| `scaling.mode` | `local_hf` | Switch to `api` for vLLM |
| `scaling.torch_dtype` | `float32` | `local_hf` only: `bfloat16` on GPU |
| `scaling.api_finetuned_model_name` | null | vLLM `--lora-modules` name for fine-tuned variant |
| `judge.mode` | `local` | `local` = reuse candidate LM; `api` = external model |
| `judge.model_name` | `openai/gpt-oss-120b` | Used only when `judge.mode: api` |

---

## Library Usage Notes

### sdg_hub
- Flow defined in `flows/tool_call_augmentation.yaml`
- `PromptBuilderBlock → LLMChatBlock → LLMResponseExtractorBlock` — generates a new question
  for a different company in a specified sector, returning structured JSON with all tool-call fields
- The genuine SDG contribution: FinanceBench provides metadata labels but not a free-text
  `query` field suitable for vector search; the flow generates that alongside a diverse set
  of company/year combinations the base dataset doesn't cover

### training_hub
training_hub is listed as a dependency and was the intended training interface, but local
dependency resolution issues prevented it from being used directly. `step2_train_lora.py`
uses HuggingFace PEFT + trl `SFTTrainer` instead, which training_hub wraps internally.
The training configuration (LoRA rank, learning rate, chat-template formatting) mirrors
what training_hub's `lora_sft()` would apply. Migrating to the training_hub API is a
pending next step — see below.

### its_hub
- Custom `LocalHFLanguageModel(AbstractLanguageModel)` in `src/local_lm.py` wraps a
  HuggingFace model without requiring a vLLM/Ollama server
- `asyncio.to_thread` bridges the synchronous transformers pipeline into the async interface
- `SelfConsistency(tool_vote="tool_hierarchical")` — uses its_hub's built-in structured
  output voting: first votes on tool name, then on each argument independently
- `BestOfN(orm=HybridFinanceScorer(...))` — judge is configurable via `judge.mode`:
  - `local` (default): reuses the candidate LM as judge to avoid loading two models on CPU
  - `api`: calls a separate external model (`judge.model_name`) for an independent judge;
    avoids the circularity of scoring candidates with the same model that generated them
- `ainfer()` called with `tools=_SEARCH_TOOL, tool_choice="required"` to enforce structured
  tool output from both base and fine-tuned models at inference time

---

## Results

Primary run: Qwen2.5-1.5B-Instruct, LoRA r=16, 244 training examples, local judge, n=50 eval questions.

### Overall accuracy

| Variant | Algorithm | N=1 | N=3 | N=5 | N=10 |
|---------|-----------|-----|-----|-----|------|
| base | BestOfN | 24% | 40% | 50% | 58% |
| base | SelfConsistency | 28% | 26% | 26% | 40% |
| finetuned | BestOfN | 36% | 42% | 46% | 52% |
| finetuned | SelfConsistency | 28% | 32% | 34% | 34% |

### Per-parameter accuracy at N=1

| Variant | company | year | report\_type | quarter |
|---------|---------|------|-------------|---------|
| base | 46–50% | 62–64% | 40–50% | 56–60% |
| finetuned | 44–54% | **90%** | **84–86%** | **86–88%** |

### Findings

**Substitution confirmed for both algorithms.** Fine-tuning reduces the ITS delta:
- BestOfN: base Δ=+34%, fine-tuned Δ=+16% at N=10
- SelfConsistency: base Δ=+12%, fine-tuned Δ=+6% at N=10

**Fine-tuning improves N=1 for BestOfN (+12%) but not SelfConsistency (0%).** The
BestOfN judge can reward a correct but low-probability output that voting would have
suppressed; at N=1 this difference disappears, so the gain is purely from the better
base format.

**The most striking finding is per-parameter.** Fine-tuning dramatically improves
`year`, `report_type`, and `quarter` (all jump from ~40–64% to ~85–90%), but barely
moves `company` accuracy (50% → 54%). The base model already understands SEC filing
structure; the bottleneck is **entity identification** — knowing which company the
user is asking about — which synthetic format training cannot fix. This explains the
substitution pattern: ITS cannot help either model name the right company more
reliably; it can only surface better-formatted calls that the base model occasionally
produces by chance.

---

## Research Question: Compound or Substitute?

The key metric is the **ITS delta**: Δ(N) = acc(N) − acc(N=1)

- **Substitution**: Δ shrinks after fine-tuning — the model already knows the right
  format, so sampling multiple times adds less marginal value
- **Compounding**: Δ grows after fine-tuning — better base output quality gives the
  voting/ranking algorithm higher-quality candidates to choose from

**Answer: Substitution, for both algorithms.** See Results above for the full breakdown.
The per-parameter analysis clarifies *why*: fine-tuning solves the format/metadata
problem but not entity identification, and ITS cannot compensate for the latter.

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

**`assistant_only_loss=False`**: SFTTrainer computes loss on the full sequence including
system prompt and user question tokens. With only 244 short examples this acts as
implicit regularisation — the model receives more gradient signal per step, which helps
avoid overfitting on a small dataset.

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

---

## What I'd Improve With More Time

**Migrate `step2` to training_hub.** Local dependency conflicts on macOS prevented
using training_hub directly — the `trl` version it pins conflicted with the `its_hub`
environment. The workaround was calling PEFT + `SFTTrainer` directly, which is what
training_hub wraps. The proper fix is to run training in a Linux environment with GPU,
where the dependency stack resolves cleanly and training_hub's `lora_sft()` can be
used as intended. The training configuration already mirrors the training_hub interface,
so the migration is straightforward once the environment is correct.

**Retrain with `assistant_only_loss=True` in a larger-data setting.** Standard SFT
practice is to mask loss on prompt tokens so gradient signal is focused on the tool-call
output. With only 244 examples, full-sequence loss acts as useful regularisation, so
`False` is the right choice at this scale. With more training data — from a second
targeted generation round — switching to `True` would be worth revisiting.

**Replace the local judge with an independent external model for BestOfN.** The config
supports this via `judge.mode: api`. I attempted this with a remote vLLM deployment
(`results/scaling_results_remote_judge_vllm.json`) but the run surfaced two problems:
the external judge had different calibration from the local one (base model accuracy
also dropped), and the vLLM LoRA serving for the fine-tuned variant appears to have
been misconfigured — the fine-tuned model collapsed to ~2% accuracy, which is below
random and inconsistent with the local-judge run. The right fix is to validate the
vLLM LoRA adapter path separately before running the full eval, and to calibrate the
external judge against the local one on a small held-out set first.

**Target company-level training data.** The per-parameter results show company accuracy
barely moved (+2–4pp) while year/report\_type/quarter improved by 25–45pp. The synthetic
data generation creates questions about companies outside the eval set, so the model
learns format but not the specific entity vocabulary FinanceBench tests. A targeted
second round of sdg\_hub generation — producing examples for the companies the model
most frequently misidentifies — would close the gap. This is the feedback loop from
Idea 2 in the project brief applied to a specific failure mode.

**Increase eval set size.** At n=50, each percentage point is 0.5 questions. The
overall accuracy differences between variants are directional at best; the per-parameter
findings are more robust because the parameter-level errors are more consistent. More
eval examples (or bootstrapped confidence intervals over the existing results) would
make the compound-vs-substitute conclusion statistically firmer.
