"""
Step 1: Load FinanceBench, split into train/eval, and augment the train split.

Produces:
  data/train.jsonl  — tool-call training examples in OpenAI messages format
  data/eval.jsonl   — held-out evaluation examples with gold labels

Pipeline:
  1. Load FinanceBench from local path (parquet / jsonl / CSV / HuggingFace hub)
  2. Normalise column names → standard schema
  3. Stratified split: eval_size held out, rest used as train seed
  4. Expand train seed × paraphrase_multiplier with distinct sector variation hints
  5. Run sdg_hub `tool_call_augmentation` flow to paraphrase each copy
  6. Write train.jsonl from paraphrased rows (user msg + tool call)
  7. Write eval.jsonl from eval split (original questions + gold labels)
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Search tool schema (for system prompt in training data) ────────────────
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_tool",
        "description": (
            "Search a financial knowledge base for information from SEC filings "
            "(10K annual reports, 10Q quarterly reports, 8K current reports)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing the information to retrieve",
                },
                "company": {
                    "type": "string",
                    "description": "Full legal company name (e.g. 'Apple Inc.')",
                },
                "sector": {
                    "type": "string",
                    "description": "GICS sector (e.g. 'Information Technology', 'Industrials', 'Energy')",
                },
                "report_type": {
                    "type": "string",
                    "enum": ["10K", "10Q", "8K"],
                    "description": "SEC filing type",
                },
                "year": {
                    "type": "integer",
                    "description": "Calendar year of the filing",
                },
                "quarter": {
                    "type": "string",
                    "enum": ["Q1", "Q2", "Q3", "Q4"],
                    "description": "Fiscal quarter — required for 10Q filings, omit for 10K/8K",
                },
                "section": {
                    "type": "string",
                    "description": (
                        "Document section: MD&A | Income Statement | Balance Sheet | "
                        "Cash Flow Statement | Notes to Financial Statements | "
                        "Risk Factors | Business Overview | Earnings Per Share"
                    ),
                },
                "fiscal_year_end": {
                    "type": "string",
                    "description": "Month when fiscal year ends if non-calendar (e.g. 'September')",
                },
            },
            "required": ["query", "company", "report_type", "year"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are a financial research assistant with access to a knowledge base of "
    "SEC filings. When a user asks about financial data, call the search_tool to "
    "retrieve the relevant information. Always include company, report_type, and "
    "year. Include quarter only for 10Q reports."
)

# Sector variation hints — one per augmented copy of each seed question.
# The prompt asks the LLM to generate a new question for a company in that sector,
# guaranteeing diverse companies across copies without passing original metadata.
VARIATION_HINTS = [
    "a technology or software company",
    "a healthcare or pharmaceutical company",
    "a financial services or banking company",
    "an energy or utilities company",
    "a consumer staples or retail company",
    "an industrials or manufacturing company",
    "a materials or mining company",
    "a real estate or REIT company",
]


# ── FinanceBench loading ───────────────────────────────────────────────────

_DOC_TYPE_MAP = {
    "10k": "10K",
    "10k_annualreport": "10K",
    "10q": "10Q",
    "8k": "8K",
    "earnings": "8K",
}


def _quarter_from_doc_name(doc_name: str) -> str | None:
    m = re.search(r"\d{4}(Q[1-4])", doc_name, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _load_two_file_financebench(qs_file: Path, docs_file: Path) -> pd.DataFrame:
    with open(qs_file) as f:
        questions = [json.loads(l) for l in f if l.strip()]
    with open(docs_file) as f:
        doc_info = [json.loads(l) for l in f if l.strip()]

    q_df = pd.DataFrame(questions)
    doc_df = pd.DataFrame(doc_info)[["doc_name", "doc_type", "doc_period", "gics_sector"]]

    df = q_df.merge(doc_df, on="doc_name", how="left")

    def _flatten(ev) -> str:
        if not isinstance(ev, list) or not ev:
            return ""
        return "\n\n".join(e.get("evidence_text", "") for e in ev if isinstance(e, dict))

    df["evidence"] = df["evidence"].apply(_flatten)

    df["report_type"] = (
        df["doc_type"].str.lower()
        .map(_DOC_TYPE_MAP)
        .fillna(df["doc_type"].str.upper())
    )
    df["year"] = df["doc_period"].apply(lambda x: int(x) if pd.notna(x) else None)
    df["quarter"] = df["doc_name"].apply(_quarter_from_doc_name)
    df = df.rename(columns={"gics_sector": "sector"})
    return df


def load_financebench(path: str, max_examples: int, seed: int) -> pd.DataFrame:
    p = Path(path)
    qs_file = p / "financebench_open_source.jsonl"
    docs_file = p / "financebench_document_information.jsonl"

    if p.is_dir() and qs_file.exists() and docs_file.exists():
        logger.info("Loading FinanceBench from two-file local format.")
        df = _load_two_file_financebench(qs_file, docs_file)
    elif not p.exists():
        logger.info(f"Path not found locally; trying HuggingFace: {path}")
        from datasets import load_dataset
        ds = load_dataset(path, split="train")
        df = ds.to_pandas()
    elif p.is_dir():
        files = (
            list(p.glob("*.parquet"))
            or list(p.glob("*.jsonl"))
            or list(p.glob("*.json"))
            or list(p.glob("*.csv"))
        )
        if not files:
            raise FileNotFoundError(f"No recognised data files in {path}")
        if files[0].suffix == ".parquet":
            df = pd.read_parquet(files[0])
        elif files[0].suffix in (".jsonl", ".json"):
            df = pd.read_json(files[0], lines=True)
        else:
            df = pd.read_csv(files[0])
    elif p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".jsonl", ".json"):
        df = pd.read_json(p, lines=True)
    else:
        df = pd.read_csv(p)

    logger.info(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    col_map = {
        "company_name": "company",
        "filing_type": "report_type",
        "doc_type": "report_type",
        "evidence_text": "evidence",
        "context": "evidence",
        "passage": "evidence",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns and v not in df.columns})

    if "year" not in df.columns:
        period_col = next((c for c in ("period_of_report", "period", "fiscal_period") if c in df.columns), None)
        if period_col:
            def _parse_period(s):
                s = str(s).strip()
                m_q = re.search(r"Q([1-4])", s, re.IGNORECASE)
                m_y = re.search(r"(20\d{2}|19\d{2})", s)
                return int(m_y.group(1)) if m_y else None, f"Q{m_q.group(1)}" if m_q else None
            parsed = df[period_col].apply(lambda x: pd.Series(_parse_period(x)))
            parsed.columns = ["year", "quarter"]
            df["year"] = parsed["year"]
            if "quarter" not in df.columns:
                df["quarter"] = parsed["quarter"]

    required = ["question", "company", "report_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"FinanceBench data missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    for col, default in [("year", None), ("quarter", None), ("evidence", ""), ("sector", None)]:
        if col not in df.columns:
            df[col] = default

    df = df.dropna(subset=["question", "company", "report_type"])
    df = df[df["question"].str.strip().astype(bool)]

    if len(df) > max_examples:
        df = df.sample(n=max_examples, random_state=seed).reset_index(drop=True)

    logger.info(f"Using {len(df)} FinanceBench examples after filtering/sampling.")
    return df[["question", "company", "report_type", "year", "quarter", "evidence", "sector"]].copy()


# ── sdg_hub augmentation ────────────────────────────────────────────────────

def _expand_with_hints(train_df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    """
    Expand train_df by `multiplier` with a distinct sector variation_hint per copy.
    Each copy's prompt asks the LLM to generate a new question for a company in a
    different sector, guaranteeing diverse output without passing original metadata.
    """
    if multiplier > len(VARIATION_HINTS):
        raise ValueError(
            f"paraphrase_multiplier={multiplier} exceeds available hints "
            f"({len(VARIATION_HINTS)}). Add more to VARIATION_HINTS."
        )
    copies = []
    for i in range(multiplier):
        copy = train_df[["question"]].copy()   # only question column; no metadata passed
        copy["variation_hint"] = VARIATION_HINTS[i]
        copies.append(copy)
    expanded = pd.concat(copies, ignore_index=True)
    logger.info(
        f"Expanded {len(train_df)} seed questions × {multiplier} → {len(expanded)} rows "
        f"(sector hints: {VARIATION_HINTS[:multiplier]})."
    )
    return expanded


def _extract_json_from_text(text: str) -> dict:
    """
    Try to parse a JSON object from `text`, handling markdown code fences,
    preamble text, and minor formatting quirks.
    """
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first {...} block (handles preamble / trailing text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _parse_flow_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse JSON from the LLM output column into structured columns.
    Rows missing required fields are dropped with a warning.
    """
    # The LLMResponseExtractorBlock names its output {block_name}_content.
    # Our block is named 'extract_example' → column 'extract_example_content'.
    content_col = "extract_example_content"
    if content_col not in df.columns:
        available = [c for c in df.columns if "content" in c.lower() or "extract" in c.lower()]
        logger.error(
            f"Column '{content_col}' not found. Available columns: {list(df.columns)}. "
            f"Likely candidates: {available}"
        )
        return pd.DataFrame()

    required = {"question", "company", "report_type", "year"}
    rows = []
    n_failed = 0
    for _, row in df.iterrows():
        text = str(row.get(content_col, ""))
        data = _extract_json_from_text(text)
        if required.issubset(data):
            rows.append(data)
        else:
            n_failed += 1
            if n_failed <= 3:   # log first few failures for diagnosis
                logger.warning(
                    f"Row parse failed (missing {required - data.keys()}). "
                    f"Raw content (first 200 chars): {text[:200]!r}"
                )
    if n_failed > 3:
        logger.warning(f"... and {n_failed - 3} more rows dropped.")
    elif n_failed:
        logger.warning(f"{n_failed}/{len(df)} rows dropped: missing required JSON fields.")

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    if "quarter" in result.columns:
        bad = result["quarter"].astype(str).isin({"null", "None", "nan", ""})
        result.loc[bad, "quarter"] = None
    for col in ("query", "quarter", "sector"):
        if col not in result.columns:
            result[col] = None
    return result


def run_augmentation_flow(train_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Expand `train_df` with sector hints, run the augmentation flow, and parse
    the JSON output into structured rows ready for training example assembly.
    """
    from sdg_hub import Flow

    multiplier = cfg["generation"]["paraphrase_multiplier"]
    expanded_df = _expand_with_hints(train_df, multiplier)

    flow_path = ROOT / "flows" / "tool_call_augmentation.yaml"
    flow = Flow.from_yaml(str(flow_path))

    api_key = os.environ.get(cfg["generation"]["api_key_env"], "")
    extra_kwargs = cfg["generation"].get("extra_generation_kwargs") or {}
    # reasoning_effort causes reasoning models (e.g. gpt-oss-120b) to consume all
    # tokens in the thinking phase and return empty content during generation.
    # It belongs in the inference config, not here.
    sdg_kwargs = {k: v for k, v in extra_kwargs.items() if k != "reasoning_effort"}
    flow.set_model_config(
        model=cfg["generation"]["model"],
        api_key=api_key or None,
        api_base=cfg["generation"].get("api_base") or None,
        **sdg_kwargs,
    )

    logger.info(f"Running augmentation flow on {len(expanded_df)} rows ...")
    raw_result = flow.generate(
        expanded_df,
        max_concurrency=cfg["generation"]["max_concurrency"],
    )
    logger.info(f"Flow produced {len(raw_result)} raw rows. Parsing JSON ...")
    logger.info(f"Flow output columns: {list(raw_result.columns)}")

    parsed = _parse_flow_output(raw_result)
    logger.info(f"Parsed {len(parsed)} valid augmented examples.")

    if parsed.empty:
        debug_path = ROOT / "data" / "augment_debug.jsonl"
        raw_result.to_json(debug_path, orient="records", lines=True)
        logger.error(
            f"All rows failed parsing. Raw flow output saved to {debug_path} for inspection."
        )
    return parsed


# ── Training & eval JSONL assembly ─────────────────────────────────────────

def _build_tool_call_args(row: pd.Series) -> dict:
    # For augmented rows: query comes from the LLM-generated query field.
    # For eval rows: fall back to the original question text.
    query = str(row.get("query") or row["question"])
    args: dict = {
        "query": query,
        "company": str(row["company"]),
        "report_type": str(row["report_type"]),
    }
    year = row.get("year")
    if year is not None and str(year) not in ("", "nan", "None", "<NA>"):
        args["year"] = int(year)
    quarter = row.get("quarter")
    if quarter and str(quarter) not in ("", "nan", "None", "null"):
        args["quarter"] = str(quarter)
    if row.get("sector") and str(row["sector"]) not in ("", "nan", "None"):
        args["sector"] = str(row["sector"])
    return args


def _build_training_example(row: pd.Series) -> dict:
    tool_args = _build_tool_call_args(row)
    user_msg = str(row["question"])
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "search_tool",
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            },
        ]
    }


def _build_eval_example(row: pd.Series) -> dict:
    gold = {
        "company": str(row["company"]),
        "report_type": str(row["report_type"]),
    }
    if row.get("year"):
        gold["year"] = int(row["year"])
    if row.get("quarter"):
        gold["quarter"] = str(row["quarter"])
    return {
        "question": str(row["question"]),
        "gold": gold,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Quick validation: 10 total FinanceBench rows, 2 paraphrase each.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Override dataset.max_examples.")
    parser.add_argument("--eval-size", type=int, default=None,
                        help="Override dataset.eval_size.")
    parser.add_argument("--paraphrase-multiplier", type=int, default=None,
                        help="Override generation.paraphrase_multiplier.")
    args = parser.parse_args()

    cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if args.smoke:
        cfg["dataset"]["max_examples"] = 10
        cfg["dataset"]["eval_size"] = 5
        cfg["generation"]["paraphrase_multiplier"] = 2
    if args.max_examples is not None:
        cfg["dataset"]["max_examples"] = args.max_examples
    if args.eval_size is not None:
        cfg["dataset"]["eval_size"] = args.eval_size
    if args.paraphrase_multiplier is not None:
        cfg["generation"]["paraphrase_multiplier"] = args.paraphrase_multiplier

    seed = cfg["dataset"]["seed"]
    eval_size = cfg["dataset"]["eval_size"]
    multiplier = cfg["generation"]["paraphrase_multiplier"]

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)

    # 1. Load all FinanceBench up to max_examples
    raw_df = load_financebench(
        path=cfg["dataset"]["local_path"],
        max_examples=cfg["dataset"]["max_examples"],
        seed=seed,
    )

    if len(raw_df) < eval_size + 1:
        raise ValueError(
            f"Only {len(raw_df)} examples loaded but eval_size={eval_size}. "
            "Reduce eval_size or increase max_examples."
        )

    # 2. Split: stratified by report_type if possible
    eval_df = raw_df.sample(n=eval_size, random_state=seed)
    train_seed_df = raw_df.drop(eval_df.index).reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    logger.info(f"Split: {len(train_seed_df)} train seed, {len(eval_df)} eval.")

    # 3. Run augmentation: expand with sector hints + generate new questions via LLM
    augmented_df = run_augmentation_flow(train_seed_df, cfg)

    if augmented_df.empty:
        raise RuntimeError(
            "Augmentation produced no valid rows. "
            "Check LLM credentials and inspect the flow output JSON."
        )

    # 5. Write train.jsonl — generated questions with varied metadata labels
    train_path = out_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for _, row in augmented_df.iterrows():
            f.write(json.dumps(_build_training_example(row)) + "\n")
    logger.info(f"Wrote {len(augmented_df)} training examples to {train_path}")

    # 6. Write eval.jsonl — original questions + gold labels (no paraphrasing)
    eval_path = out_dir / "eval.jsonl"
    with open(eval_path, "w") as f:
        for _, row in eval_df.iterrows():
            f.write(json.dumps(_build_eval_example(row)) + "\n")
    logger.info(f"Wrote {len(eval_df)} eval examples to {eval_path}")

    print("\n── Data Generation Complete ─────────────────────────────────")
    print(f"  FinanceBench loaded   : {len(raw_df)}")
    print(f"  Eval split            : {len(eval_df)} (original questions)")
    print(f"  Train seed            : {len(train_seed_df)}")
    print(f"  Paraphrase multiplier : {multiplier}×")
    print(f"  Training examples     : {len(augmented_df)}")
    print(f"  Output directory      : {out_dir.resolve()}")
    print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
