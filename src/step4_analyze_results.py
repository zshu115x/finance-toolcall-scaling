"""
Step 4: Analyze and visualize the scaling experiment results.

Produces:
  results/plots/accuracy_vs_budget.png   — line chart: accuracy vs N by variant & algorithm
  results/plots/its_delta.png            — bar chart: ITS improvement delta Δ(N) = acc(N) - acc(1)
  results/plots/param_heatmap.png        — per-parameter accuracy heatmap
  results/summary.txt                    — narrative analysis printed to stdout and saved

Research questions answered:
  1. Does fine-tuning improve N=1 accuracy? (baseline shift)
  2. Does the ITS delta shrink after fine-tuning? (substitution)
  3. Does the ITS delta grow after fine-tuning? (compounding)
  4. Which parameter (company/year/report_type/quarter) benefits most from each intervention?
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _budgets(table: dict) -> list[int]:
    """Return sorted budget list from the first populated cell of the table."""
    return sorted(next(iter(next(iter(table.values())).values())).keys())


def compute_accuracy_table(results: dict) -> dict:
    """
    Returns nested dict: variant → algorithm → budget → {param: accuracy, overall: accuracy}
    """
    table = {}
    params = ["company", "year", "report_type", "quarter", "overall"]
    for variant, alg_data in results.items():
        table[variant] = {}
        for alg, budget_data in alg_data.items():
            table[variant][alg] = {}
            for budget_str, examples in budget_data.items():
                n = len(examples)
                if n == 0:
                    table[variant][alg][int(budget_str)] = {p: 0.0 for p in params}
                    continue
                accs = {}
                for p in params:
                    accs[p] = sum(ex["scores"].get(p, False) for ex in examples) / n
                table[variant][alg][int(budget_str)] = accs
    return table


def compute_its_delta(table: dict) -> dict:
    """
    ITS delta: Δ(N) = acc_overall(N) - acc_overall(N=1)
    Returns: variant → algorithm → {budget: delta}
    """
    delta = {}
    for variant, alg_data in table.items():
        delta[variant] = {}
        for alg, budget_data in alg_data.items():
            budgets = sorted(budget_data.keys())
            baseline = budget_data.get(1, budget_data.get(budgets[0], {})).get("overall", 0.0)
            delta[variant][alg] = {b: budget_data[b]["overall"] - baseline for b in budgets}
    return delta


def print_summary(table: dict, delta: dict) -> str:
    lines = []

    def h(title):
        lines.append(f"\n{'='*60}")
        lines.append(title)
        lines.append("=" * 60)

    h("ACCURACY TABLE  (overall exact-match accuracy)")
    header = f"{'Variant':<12} {'Algorithm':<22} " + "  ".join(f"N={b:>2}" for b in _budgets(table))
    lines.append(header)
    lines.append("-" * len(header))
    for variant in sorted(table):
        for alg in sorted(table[variant]):
            budgets = sorted(table[variant][alg].keys())
            row = f"{variant:<12} {alg:<22} " + "  ".join(
                f"{table[variant][alg][b]['overall']:>5.1%}" for b in budgets
            )
            lines.append(row)

    h("ITS DELTA  Δ(N) = acc(N) - acc(N=1)")
    for variant in sorted(delta):
        for alg in sorted(delta[variant]):
            budgets = sorted(delta[variant][alg].keys())
            row = f"{variant:<12} {alg:<22} " + "  ".join(
                f"{delta[variant][alg][b]:>+6.1%}" for b in budgets
            )
            lines.append(row)

    h("PER-PARAMETER ACCURACY  (N=1 baseline)")
    params = ["company", "year", "report_type", "quarter"]
    lines.append(f"{'Variant':<12} {'Algorithm':<22} " + "  ".join(f"{p:<12}" for p in params))
    lines.append("-" * 80)
    for variant in sorted(table):
        for alg in sorted(table[variant]):
            budgets = sorted(table[variant][alg].keys())
            b1 = budgets[0]  # N=1
            row = f"{variant:<12} {alg:<22} " + "  ".join(
                f"{table[variant][alg][b1].get(p, 0):>12.1%}" for p in params
            )
            lines.append(row)

    # Narrative analysis
    h("ANALYSIS")
    try:
        _add_narrative(lines, table)
    except Exception as e:
        lines.append(f"(narrative generation failed: {e})")

    return "\n".join(lines)


def _add_narrative(lines, table):
    """Print a narrative interpretation of the compound vs substitute question."""
    variants = list(table.keys())
    algs = list(next(iter(table.values())).keys())

    if "base" not in variants or "finetuned" not in variants:
        lines.append("(Need both base and finetuned variants for full analysis)")
        return

    for alg in algs:
        budgets = sorted(table["base"][alg].keys())
        b1 = budgets[0]
        b_max = budgets[-1]

        base_b1 = table["base"][alg][b1]["overall"]
        base_bmax = table["base"][alg][b_max]["overall"]
        ft_b1 = table["finetuned"][alg][b1]["overall"]
        ft_bmax = table["finetuned"][alg][b_max]["overall"]

        base_delta = base_bmax - base_b1
        ft_delta = ft_bmax - ft_b1

        lines.append(f"\n[{alg.upper()}]")
        lines.append(
            f"  Base model:       N=1 → {base_b1:.1%},  N={b_max} → {base_bmax:.1%}  (Δ={base_delta:+.1%})"
        )
        lines.append(
            f"  Fine-tuned model: N=1 → {ft_b1:.1%},  N={b_max} → {ft_bmax:.1%}  (Δ={ft_delta:+.1%})"
        )

        ft_lift = ft_b1 - base_b1
        if ft_lift > 0.05:
            lines.append(f"  → Fine-tuning improves N=1 accuracy by {ft_lift:+.1%}")
        elif ft_lift < -0.02:
            lines.append(f"  → Unexpected: fine-tuned N=1 accuracy is {ft_lift:+.1%} vs base")
        else:
            lines.append(f"  → Fine-tuning has minimal N=1 effect ({ft_lift:+.1%})")

        if abs(ft_delta - base_delta) < 0.03:
            lines.append(
                f"  → NEUTRAL: ITS delta similar for both models "
                f"(base {base_delta:+.1%}, FT {ft_delta:+.1%})"
            )
        elif ft_delta < base_delta - 0.03:
            lines.append(
                f"  → SUBSTITUTION: Fine-tuning reduces ITS benefit "
                f"(base Δ={base_delta:+.1%}, FT Δ={ft_delta:+.1%}). "
                "The model already learned to call the tool correctly; extra samples add less."
            )
        else:
            lines.append(
                f"  → COMPOUNDING: Fine-tuning amplifies ITS benefit "
                f"(base Δ={base_delta:+.1%}, FT Δ={ft_delta:+.1%}). "
                "Better base format lets voting surface higher-quality candidates."
            )


def plot_accuracy_vs_budget(table: dict, plots_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    alg_titles = {"self_consistency": "Self-Consistency", "best_of_n": "Best-of-N"}
    colors = {"base": "#4c72b0", "finetuned": "#dd8452"}
    linestyles = {"base": "-o", "finetuned": "--s"}

    for ax, alg in zip(axes, ["self_consistency", "best_of_n"]):
        for variant in sorted(table):
            if alg not in table[variant]:
                continue
            budgets = sorted(table[variant][alg].keys())
            accs = [table[variant][alg][b]["overall"] for b in budgets]
            ax.plot(
                budgets, [a * 100 for a in accs],
                linestyles[variant],
                color=colors.get(variant, "gray"),
                label=variant,
                linewidth=2,
                markersize=7,
            )
        ax.set_title(alg_titles.get(alg, alg), fontsize=13)
        ax.set_xlabel("Budget (N)", fontsize=11)
        ax.set_ylabel("Overall Accuracy (%)", fontsize=11)
        ax.set_xticks(_budgets(table))
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)

    fig.suptitle(
        "Financial Tool-Call Accuracy vs Inference Budget\n(finance-toolcall-scaling experiment)",
        fontsize=14,
    )
    plt.tight_layout()
    out = plots_dir / "accuracy_vs_budget.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_its_delta(delta: dict, plots_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    algs = list(next(iter(delta.values())).keys())
    variants = sorted(delta.keys())
    budgets = sorted(next(iter(next(iter(delta.values())).values())).keys())
    non_one_budgets = [b for b in budgets if b != 1]

    fig, axes = plt.subplots(1, len(algs), figsize=(6 * len(algs), 5))
    if len(algs) == 1:
        axes = [axes]

    colors = {"base": "#4c72b0", "finetuned": "#dd8452"}
    x = np.arange(len(non_one_budgets))
    width = 0.35

    for ax, alg in zip(axes, algs):
        for i, variant in enumerate(variants):
            if alg not in delta.get(variant, {}):
                continue
            vals = [delta[variant][alg].get(b, 0) * 100 for b in non_one_budgets]
            offset = (i - len(variants) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=variant, color=colors.get(variant, "gray"))
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.3 if v >= 0 else -0.3),
                    f"{v:+.1f}%",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8,
                )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(alg.replace("_", " ").title(), fontsize=13)
        ax.set_xlabel("Budget N", fontsize=11)
        ax.set_ylabel("ITS Delta Δ(N) = acc(N) - acc(1)  (%)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={b}" for b in non_one_budgets])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "ITS Improvement Delta: Does Fine-Tuning Compound or Substitute?\n"
        "Positive = ITS helps more after fine-tuning (compounding); "
        "Negative = fine-tuning reduces ITS benefit (substitution)",
        fontsize=11,
    )
    plt.tight_layout()
    out = plots_dir / "its_delta.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_param_heatmap(table: dict, plots_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    params = ["company", "year", "report_type", "quarter"]
    variants = sorted(table.keys())
    algs = sorted(next(iter(table.values())).keys())
    # Use N=1 for parameter breakdown (isolates fine-tuning effect without ITS noise)
    budgets_available = _budgets(table)
    b1 = budgets_available[0]

    rows = [f"{v}\n{a}" for v in variants for a in algs if a in table.get(v, {})]
    data = []
    for v in variants:
        for a in algs:
            if a not in table.get(v, {}):
                continue
            data.append([table[v][a][b1].get(p, 0) * 100 for p in params])

    if not data:
        return

    data_arr = np.array(data)
    _, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.8 + 1.5)))
    im = ax.imshow(data_arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(params, fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    for i in range(len(rows)):
        for j in range(len(params)):
            ax.text(j, i, f"{data_arr[i, j]:.0f}%", ha="center", va="center",
                    fontsize=9, color="white" if data_arr[i, j] < 35 else "black")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set_title(f"Per-Parameter Accuracy at N={b1}\n(red=low, green=high)", fontsize=12)
    plt.tight_layout()
    out = plots_dir / "param_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    results_path = ROOT / cfg["paths"]["results_file"]
    if not results_path.exists():
        print(f"ERROR: Results not found at {results_path}")
        print("Run step3_evaluate_scaling.py first.")
        sys.exit(1)

    plots_dir = ROOT / cfg["paths"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw = load_results(results_path)
    table = compute_accuracy_table(raw)
    delta = compute_its_delta(table)

    summary = print_summary(table, delta)
    print(summary)

    summary_path = ROOT / "results" / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary saved to {summary_path}")

    print("\nGenerating plots...")
    plot_accuracy_vs_budget(table, plots_dir)
    plot_its_delta(delta, plots_dir)
    plot_param_heatmap(table, plots_dir)
    print("Done.")


if __name__ == "__main__":
    main()
