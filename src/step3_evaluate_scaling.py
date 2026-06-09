"""
Step 3: Core experiment — evaluate inference-time scaling on base and fine-tuned models.

Experiment matrix:
  model_variant  ∈ {base, finetuned}
  algorithm      ∈ {self_consistency, best_of_n}
  budget         ∈ config.scaling.budgets  (e.g. [1, 3, 5, 10])

For each cell we run all eval questions and score each response.

Scoring (strict exact match on key parameters):
  - company:      normalised string equality
  - year:         integer equality
  - report_type:  exact string equality (10K / 10Q / 8K)
  - quarter:      exact string equality (Q1-Q4) or both null

Query field is not scored by exact match — it's evaluated separately by
LLMJudge (see its_hub BestOfN) when running the judge-based algorithm.

Design note on SelfConsistency:
  its_hub provides SelfConsistency(tool_vote="tool_hierarchical") which votes
  on tool name first, then on each argument. This is exactly what we want for
  tool-call consensus across N samples.

Design note on BestOfN + HybridFinanceScorer:
  Structured fields (company, year, report_type, quarter) are scored with exact
  match (0-4 pts). The free-text query is scored by an LLM judge (0-1 pt).
  BestOfN selects the highest-scoring candidate among N. One HybridFinanceScorer
  instance is created per example so it can hold the gold labels.
"""

import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from its_hub import AbstractLanguageModel, AbstractOutcomeRewardModel
from local_lm import LocalHFLanguageModel, extract_tool_args  # noqa: E402


# ── Scoring ────────────────────────────────────────────────────────────────

def _normalise_quarter(q) -> str:
    """Normalise quarter value; treats None / empty / 'nan' as no-quarter."""
    if q is None:
        return ""
    s = str(q).strip().lower()
    return "" if s in ("nan", "none", "") else s.upper()


def _normalise_company(name: Optional[str]) -> str:
    """Lower-case, strip common suffixes and punctuation for fuzzy company match."""
    if not name:
        return ""
    s = name.lower().strip()
    for suffix in [" inc.", " inc", " corp.", " corp", " co.", " co",
                   " ltd.", " ltd", " llc", " plc", " lp", ","]:
        s = s.replace(suffix, "")
    return s.strip()


def score_tool_call(predicted: Optional[dict], gold: dict) -> dict[str, bool]:
    """
    Score a predicted tool call against gold labels.

    Returns a dict of per-parameter booleans, plus an 'overall' key
    that is True only when all scored parameters are correct.
    """
    if predicted is None:
        return {k: False for k in ["company", "year", "report_type", "quarter", "overall"]}

    company_ok = _normalise_company(predicted.get("company")) == _normalise_company(gold.get("company"))

    pred_year = predicted.get("year")
    gold_year = gold.get("year")
    year_ok = (
        str(pred_year).strip() == str(gold_year).strip()
        if pred_year is not None and gold_year is not None
        else pred_year is None and gold_year is None
    )

    report_ok = str(predicted.get("report_type", "")).upper() == str(gold.get("report_type", "")).upper()

    quarter_ok = _normalise_quarter(predicted.get("quarter")) == _normalise_quarter(gold.get("quarter"))

    overall = company_ok and year_ok and report_ok and quarter_ok
    return {
        "company": company_ok,
        "year": year_ok,
        "report_type": report_ok,
        "quarter": quarter_ok,
        "overall": overall,
    }


# ── Tool schema ────────────────────────────────────────────────────────────

_SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search_tool",
        "description": "Search SEC filings for financial data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query":       {"type": "string"},
                "company":     {"type": "string"},
                "report_type": {"type": "string", "enum": ["10K", "10Q", "8K"]},
                "year":        {"type": "integer"},
                "quarter":     {"type": "string", "enum": ["Q1", "Q2", "Q3", "Q4"]},
            },
            "required": ["query", "company", "report_type", "year"],
        },
    },
}]


# ── LLM construction helpers ────────────────────────────────────────────────

_QUERY_JUDGE_PROMPT = (
    "You are evaluating a financial document search query.\n"
    "Score 0-1: does the query use specific financial terminology that would retrieve the right document?\n"
    "  1 = specific and on-target  |  0 = vague, empty, or off-topic\n"
    'Return ONLY JSON: {"score": <0-1 float>, "reasoning": "<brief explanation>"}'
)


def _parse_judge_score(content: str, fallback: float = 0.5) -> float:
    """Parse 0-1 score from a JSON judge response, with regex fallback."""
    try:
        return min(1.0, max(0.0, float(json.loads(content).get("score", fallback))))
    except (json.JSONDecodeError, ValueError, AttributeError):
        m = re.search(r'"score"\s*:\s*([\d.]+)', content)
        if m:
            try:
                return min(1.0, max(0.0, float(m.group(1))))
            except ValueError:
                pass
    return fallback


class HybridFinanceScorer(AbstractOutcomeRewardModel):
    """
    ORM for BestOfN that uses exact match for structured fields and the LLM
    only for the free-text query field.

      company / year / report_type / quarter — exact match, 1 pt each (0-4)
      query                                  — LLM judge, 0-1 pt (normalised)

    One instance per eval example (gold labels are baked in at construction).
    """

    def __init__(self, gold: dict, judge_lm: AbstractLanguageModel):
        self.gold = gold
        self.judge_lm = judge_lm

    def score(self, *_) -> float:
        raise NotImplementedError("Use ascore")

    async def ascore(self, messages, orchestrator=None, **_):
        from its_hub import LMOrchestrator
        from its_hub.api.types import ChatMessage  # type: ignore[import-untyped]

        is_batch = messages and isinstance(messages[0], list)
        conversations = messages if is_batch else [messages]

        # User question is the same across all candidates
        question = next(
            (m.content or "" for m in conversations[0] if m.role == "user"), ""
        )

        # Extract tool args from the last (assistant) message of each candidate
        candidates = [extract_tool_args(conv[-1].to_dict()) for conv in conversations]

        # Exact-match scores (0-4 pts)
        exact_scores = [
            sum(score_tool_call(pred, self.gold)[k]
                for k in ("company", "year", "report_type", "quarter"))
            for pred in candidates
        ]

        # LLM query scores (0-1 pt) — judge only the query field
        if orchestrator is None:
            orchestrator = LMOrchestrator()

        query_prompts = [
            [ChatMessage(
                role="user",
                content=(
                    _QUERY_JUDGE_PROMPT
                    + f"\n\nQuestion: {question}"
                    + f"\nQuery: {pred.get('query', '') if pred else ''}"
                ),
            )]
            for pred in candidates
        ]
        judge_responses = await orchestrator.agenerate(self.judge_lm, query_prompts)
        query_scores = [_parse_judge_score(r.get("content") or "") for r in judge_responses]

        scores = [e + q for e, q in zip(exact_scores, query_scores)]
        return scores if is_batch else scores[0]


def build_lm(
    cfg: dict,
    adapter_path: Optional[str] = None,
    variant: str = "base",
) -> AbstractLanguageModel:
    scale_cfg = cfg["scaling"]
    train_cfg = cfg["training"]

    if scale_cfg["mode"] == "api":
        from its_hub import OpenAICompatibleLanguageModel
        import os
        if variant == "finetuned":
            model_name = scale_cfg.get("api_finetuned_model_name") or scale_cfg.get("api_model_name") or train_cfg["base_model"]
        else:
            model_name = scale_cfg.get("api_model_name") or train_cfg["base_model"]
        return OpenAICompatibleLanguageModel(
            endpoint=scale_cfg["api_endpoint"],
            api_key=os.environ.get(scale_cfg["api_key_env"], "NO_KEY"),
            model_name=model_name,
        )

    return LocalHFLanguageModel(
        model_path=train_cfg["base_model"],
        adapter_path=adapter_path,
        max_new_tokens=scale_cfg["max_tokens"],
        temperature=scale_cfg["temperature"],
        torch_dtype=scale_cfg.get("torch_dtype", "float32"),
        system_prompt=(
            "You are a financial research assistant with access to a knowledge base "
            "of SEC filings. When a user asks about financial data, call the "
            "search_tool to retrieve the relevant information. Always include "
            "company, report_type, and year. Include quarter only for 10Q reports."
        ),
        tools=_SEARCH_TOOL,
    )


def build_judge_lm(cfg: dict) -> Optional[AbstractLanguageModel]:
    """Build the LLM judge for best_of_n query scoring.

    Returns None when mode='local', signalling run_all_conditions to reuse
    the candidate LM as the judge.
    """
    judge_cfg = cfg["judge"]
    if judge_cfg.get("mode", "api") == "local":
        return None
    import os
    from its_hub import OpenAICompatibleLanguageModel
    return OpenAICompatibleLanguageModel(
        endpoint=judge_cfg["api_endpoint"],
        api_key=os.environ.get(judge_cfg["api_key_env"], "NO_KEY"),
        model_name=judge_cfg["model_name"],
    )


# ── Core eval loop ─────────────────────────────────────────────────────────

async def run_single_condition(
    lm: AbstractLanguageModel,
    eval_examples: list[dict],
    algorithm_name: str,
    budget: int,
    orchestrator,
    judge_lm: Optional[AbstractLanguageModel] = None,
) -> list[dict]:
    """Run one (algorithm, budget) condition on all eval examples."""
    from its_hub import SelfConsistency, BestOfN

    # SelfConsistency is stateless per call — create once and reuse.
    # BestOfN needs gold labels in the ORM, so it is created per example below.
    sc_alg = (
        SelfConsistency(tool_vote="tool_hierarchical", orchestrator=orchestrator)
        if algorithm_name == "self_consistency"
        else None
    )
    if algorithm_name not in ("self_consistency", "best_of_n"):
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    results = []
    for i, ex in enumerate(eval_examples):
        question = ex["question"]
        gold = ex["gold"]

        logger.info(f"  [{i+1}/{len(eval_examples)}] budget={budget} alg={algorithm_name}")

        if algorithm_name == "best_of_n":
            if judge_lm is None:
                raise ValueError("judge_lm required for best_of_n")
            alg = BestOfN(
                orm=HybridFinanceScorer(gold=gold, judge_lm=judge_lm),
                orchestrator=orchestrator,
            )
        else:
            alg = sc_alg

        try:
            response = await alg.ainfer(
                lm, question, budget=budget, return_response_only=True,
                tools=_SEARCH_TOOL, tool_choice="required",
            )
            predicted = extract_tool_args(response)
        except Exception as e:
            logger.warning(f"  Inference error on example {i}: {e}")
            predicted = None

        scores = score_tool_call(predicted, gold)
        results.append({
            "question": question,
            "gold": gold,
            "predicted": predicted,
            "scores": scores,
        })

    return results


async def run_all_conditions(
    cfg: dict,
    eval_examples: list[dict],
    smoke: bool = False,
    algorithms: list[str] | None = None,
) -> dict:
    """Run the full experiment matrix and return all results."""
    scale_cfg = cfg["scaling"]
    train_cfg = cfg["training"]
    budgets = scale_cfg["budgets"]
    ckpt_dir = ROOT / train_cfg["output_dir"]
    adapter_path = str(ckpt_dir) if ckpt_dir.exists() else None

    all_results = {}

    # Judge is shared across variants — build once to avoid redundant connections.
    judge_lm = build_judge_lm(cfg)

    variants = [("base", None)] if smoke else [("base", None), ("finetuned", adapter_path)]
    for variant_name, adapter in variants:
        if variant_name == "finetuned" and adapter is None:
            logger.warning(
                "Fine-tuned model checkpoint not found — skipping finetuned variant. "
                f"Expected: {ckpt_dir}"
            )
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Model variant: {variant_name}")
        logger.info(f"{'='*60}")

        lm = build_lm(cfg, adapter_path=adapter, variant=variant_name)
        effective_judge_lm = judge_lm or lm

        from its_hub import LMOrchestrator
        orchestrator = LMOrchestrator(max_concurrency=scale_cfg["max_concurrency"])

        all_results[variant_name] = {}

        for alg_name in (algorithms or ["self_consistency", "best_of_n"]):
            all_results[variant_name][alg_name] = {}

            for budget in budgets:
                logger.info(f"\n  algorithm={alg_name}  budget={budget}")
                condition_results = await run_single_condition(
                    lm=lm,
                    eval_examples=eval_examples,
                    algorithm_name=alg_name,
                    budget=budget,
                    orchestrator=orchestrator,
                    judge_lm=effective_judge_lm if alg_name == "best_of_n" else None,
                )
                all_results[variant_name][alg_name][budget] = condition_results

                # Print running accuracy
                n_correct = sum(r["scores"]["overall"] for r in condition_results)
                acc = n_correct / len(condition_results) if condition_results else 0
                logger.info(f"  → overall accuracy: {acc:.1%}  ({n_correct}/{len(condition_results)})")

    return all_results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick sanity run: 3 examples, budgets [1, 3], base variant only.",
    )
    parser.add_argument(
        "--alg", choices=["self_consistency", "best_of_n", "all"], default="all",
        help="Algorithm(s) to run (default: all).",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge new results into the existing results file instead of overwriting.",
    )
    args = parser.parse_args()

    cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    eval_path = ROOT / cfg["paths"]["eval_data"]
    if not eval_path.exists():
        print(f"ERROR: Eval data not found at {eval_path}")
        print("Run step1_generate_data.py first.")
        sys.exit(1)

    with open(eval_path) as f:
        eval_examples = [json.loads(line) for line in f if line.strip()]

    if args.smoke:
        eval_examples = eval_examples[:3]
        cfg["scaling"]["budgets"] = [1, 3]
        logger.info("Smoke run: 3 examples, budgets [1, 3], base variant only")

    logger.info(f"Loaded {len(eval_examples)} eval examples from {eval_path}")

    algorithms = None if args.alg == "all" else [args.alg]
    results = asyncio.run(run_all_conditions(cfg, eval_examples, smoke=args.smoke, algorithms=algorithms))

    # Serialise new results (int budget keys → str for JSON)
    serialisable = {}
    for variant, alg_data in results.items():
        serialisable[variant] = {}
        for alg, budget_data in alg_data.items():
            serialisable[variant][alg] = {str(k): v for k, v in budget_data.items()}

    results_path = ROOT / cfg["paths"]["results_file"]
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if args.merge and results_path.exists():
        with open(results_path) as f:
            merged = json.load(f)
        for variant, alg_data in serialisable.items():
            merged.setdefault(variant, {})
            for alg, budget_data in alg_data.items():
                if alg in merged[variant]:
                    logger.info(f"  Overwriting existing {variant}/{alg} results")
                else:
                    logger.info(f"  Adding new {variant}/{alg} results")
                merged[variant][alg] = budget_data
        serialisable = merged
        logger.info(f"Merged results saved to {results_path}")
    else:
        if args.merge:
            logger.info("No existing results file found — writing fresh.")
        logger.info(f"Results saved to {results_path}")

    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    # Quick summary table
    print("\n── Results Summary ────────────────────────────────────────────")
    print(f"{'Variant':<12} {'Algorithm':<20} {'Budget':>6}  {'Overall Acc':>11}")
    print("-" * 55)
    for variant, alg_data in serialisable.items():
        for alg, budget_data in alg_data.items():
            for budget_str, cond_results in budget_data.items():
                n = len(cond_results)
                correct = sum(r["scores"]["overall"] for r in cond_results)
                acc = correct / n if n else 0
                print(f"{variant:<12} {alg:<20} {budget_str:>6}  {acc:>10.1%}")
    print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
