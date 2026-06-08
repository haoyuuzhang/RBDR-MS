"""Compute and print the Two-Stage Perturbation Resilience table.

Stage 1 : J9+J10 insertion at t=2
Stage 2 : M3 breakdown at t=6

Output saved to  output/table_resilience.txt
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OUTPUT_DIR = os.path.join(_HERE, "output")
BASELINE_CACHE = os.path.join(OUTPUT_DIR, "baseline_results.json")
RULE_CACHE = os.path.join(OUTPUT_DIR, "pure_rule_results.json")
BI_LEVEL_CACHE = os.path.join(OUTPUT_DIR, "bi_level_results.json")


def fmt_pct(v: float) -> str:
    """Format a percentage with explicit sign."""
    if v > 0:
        return f"+{v:.2f}%"
    elif v < 0:
        return f"-{abs(v):.2f}%"
    else:
        return " 0.00%"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(BASELINE_CACHE, "r", encoding="utf-8") as f:
        bl_raw = json.load(f)
    with open(RULE_CACHE, "r", encoding="utf-8") as f:
        rule_raw = json.load(f)

    # -- Baseline metrics -------------------------------------------------
    cmax_A = bl_raw["baselines"]["A"]["cmax"]   # J1-J8 optimal
    cmax_B = bl_raw["baselines"]["B"]["cmax"]   # J1-J10 clairvoyant
    cmax_C = bl_raw["baselines"]["C"]["cmax"]   # J1-J10 + M3 clairvoyant

    bl_deg1 = (cmax_B - cmax_A) / cmax_A * 100
    bl_deg2 = (cmax_C - cmax_B) / cmax_B * 100

    # -- Rule-based metrics -----------------------------------------------
    rules_data = {}
    for rule_name in ["SPT", "FIFO", "WINQ"]:
        r = rule_raw["rules"][rule_name]
        cmax_t0 = r["0.0"]["cmax"]
        cmax_t2 = r["2.0"]["cmax"]
        cmax_t6 = r["6.0"]["cmax"]

        deg1 = (cmax_t2 - cmax_t0) / cmax_t0 * 100
        deg2 = (cmax_t6 - cmax_t2) / cmax_t2 * 100
        excess_deg1 = deg1 - bl_deg1
        excess_deg2 = deg2 - bl_deg2

        # Stage 1: J9+J10 insertion
        # J9 has 2 ops, J10 has 3 ops -> 5 new operations, 2 new jobs
        affected_ops_s1 = 5
        affected_jobs_s1 = 2

        # Stage 2: M3 breakdown -> interrupted ops from partial_entries
        partial = r["6.0"].get("partial_entries", [])
        interrupted_ops = len(partial)
        interrupted_jobs = len(set(p["job_id"] for p in partial))

        rules_data[rule_name] = {
            "deg1": deg1,
            "deg2": deg2,
            "excess_deg1": excess_deg1,
            "excess_deg2": excess_deg2,
            "affected_ops_s1": affected_ops_s1,
            "affected_jobs_s1": affected_jobs_s1,
            "interrupted_ops": interrupted_ops,
            "interrupted_jobs": interrupted_jobs,
        }

    # -- Bi-level metrics ---------------------------------------------------
    bi_level_data = {}
    if os.path.exists(BI_LEVEL_CACHE):
        with open(BI_LEVEL_CACHE, "r", encoding="utf-8") as f:
            bl_raw2 = json.load(f)
        for rule_name in ["SPT", "FIFO", "WINQ"]:
            if rule_name not in bl_raw2.get("rules", {}):
                continue
            r = bl_raw2["rules"][rule_name]
            cmax_t0 = r["snapshot_cmax"]["0.0"]
            cmax_t2 = r["snapshot_cmax"]["2.0"]
            cmax_t6 = r["snapshot_cmax"]["6.0"]

            deg1 = (cmax_t2 - cmax_t0) / cmax_t0 * 100
            deg2 = (cmax_t6 - cmax_t2) / cmax_t2 * 100
            excess_deg1 = deg1 - bl_deg1
            excess_deg2 = deg2 - bl_deg2

            affected_ops_s1 = 5   # J9: 2 ops, J10: 3 ops
            affected_jobs_s1 = 2

            partial = r.get("partial_entries", [])
            interrupted_ops = len(partial)
            interrupted_jobs = len(set(p["job_id"] for p in partial))

            bi_level_data[rule_name] = {
                "deg1": deg1,
                "deg2": deg2,
                "excess_deg1": excess_deg1,
                "excess_deg2": excess_deg2,
                "affected_ops_s1": affected_ops_s1,
                "affected_jobs_s1": affected_jobs_s1,
                "interrupted_ops": interrupted_ops,
                "interrupted_jobs": interrupted_jobs,
            }

    # =====================================================================
    #  Build the table
    # =====================================================================

    SEP = "  "
    HL = "-" * 120

    lines = [
        "",
        "Table X: Two-Stage Perturbation Resilience",
        "=" * 80,
        "",
        "  Deg1 = (C_max after J9+J10 insertion - C_max initial) / C_max initial  [Stage 1]",
        "  Deg2 = (C_max after M3 breakdown - C_max after insertion) / C_max after insertion  [Stage 2]",
        "  Excess Deg = Method's Deg - Optimal Deg  (positive -> more fragile)",
        "",
        HL,
    ]

    # Column headers
    h1 = (f"{'Method':<10s}{SEP}"
          f"{'--- Stage 1 (J9+J10 Insertion) ---':^52s}{SEP}"
          f"{'--- Stage 2 (M3 Breakdown) ---':^52s}")
    h2 = (f"{'':10s}{SEP}"
          f"{'Deg1(%)':>8s}  {'Aff.Ops':>8s}  {'Aff.Jobs':>8s}  {'Exc.Deg1':>8s}{SEP}"
          f"{'Deg2(%)':>8s}  {'Int.Ops':>8s}  {'Aff.Jobs':>8s}  {'Exc.Deg2':>8s}")

    lines.append(h1)
    lines.append(h2)
    lines.append(HL)

    # -- Baseline row -----------------------------------------------------
    bl_line = (
        f"{'Baseline':10s}{SEP}"
        f"{fmt_pct(bl_deg1):>8s}  {'--':>8s}  {'--':>8s}  {'0.00%':>8s}{SEP}"
        f"{fmt_pct(bl_deg2):>8s}  {'--':>8s}  {'--':>8s}  {'0.00%':>8s}"
    )
    lines.append(bl_line)

    # -- Rule rows --------------------------------------------------------
    for rule_name in ["SPT", "FIFO", "WINQ"]:
        d = rules_data[rule_name]
        rl = (
            f"{rule_name:10s}{SEP}"
            f"{fmt_pct(d['deg1']):>8s}  {d['affected_ops_s1']:>8d}  "
            f"{d['affected_jobs_s1']:>8d}  {fmt_pct(d['excess_deg1']):>8s}{SEP}"
            f"{fmt_pct(d['deg2']):>8s}  {d['interrupted_ops']:>8d}  "
            f"{d['interrupted_jobs']:>8d}  {fmt_pct(d['excess_deg2']):>8s}"
        )
        lines.append(rl)

    # -- Bi-level rows -------------------------------------------------------
    if bi_level_data:
        lines.append("")  # blank spacer before bi-level section
        for rule_name in ["SPT", "FIFO", "WINQ"]:
            if rule_name not in bi_level_data:
                continue
            d = bi_level_data[rule_name]
            rl = (
                f"{rule_name + '-MILP':10s}{SEP}"
                f"{fmt_pct(d['deg1']):>8s}  {d['affected_ops_s1']:>8d}  "
                f"{d['affected_jobs_s1']:>8d}  {fmt_pct(d['excess_deg1']):>8s}{SEP}"
                f"{fmt_pct(d['deg2']):>8s}  {d['interrupted_ops']:>8d}  "
                f"{d['interrupted_jobs']:>8d}  {fmt_pct(d['excess_deg2']):>8s}"
            )
            lines.append(rl)

    lines.append(HL)
    lines.append("")

    # -- Interpretation notes ---------------------------------------------
    lines.append("Notes:")
    lines.append(
        f"  Optimal Stage-1 degradation: {fmt_pct(bl_deg1)}  "
        f"(Baseline: cmax {cmax_A:.0f}h -> {cmax_B:.0f}h with J9/J10)"
    )
    lines.append(
        f"  Optimal Stage-2 degradation: {fmt_pct(bl_deg2)}  "
        f"(Baseline: cmax {cmax_B:.0f}h -> {cmax_C:.0f}h with M3 breakdown)"
    )

    # Best/worst per stage (single-level rules only)
    for label, dkey in [("Stage 1 (J9+J10)", "excess_deg1"),
                         ("Stage 2 (M3 breakdown)", "excess_deg2")]:
        vals = [(rules_data[r][dkey], r) for r in ["SPT", "FIFO", "WINQ"]]
        best = min(vals, key=lambda x: x[0])
        worst = max(vals, key=lambda x: x[0])
        lines.append(
            f"  {label} (single-level):  most resilient = {best[1]} ({fmt_pct(best[0])}),  "
            f"most fragile = {worst[1]} ({fmt_pct(worst[0])})"
        )
        if bi_level_data:
            bl_vals = [(bi_level_data[r][dkey], r + '-MILP')
                       for r in ["SPT", "FIFO", "WINQ"] if r in bi_level_data]
            if bl_vals:
                bl_best = min(bl_vals, key=lambda x: x[0])
                bl_worst = max(bl_vals, key=lambda x: x[0])
                lines.append(
                    f"  {label} (bi-level):      most resilient = {bl_best[1]} ({fmt_pct(bl_best[0])}),  "
                    f"most fragile = {bl_worst[1]} ({fmt_pct(bl_worst[0])})"
                )

    lines.append("")

    # =====================================================================
    #  Print & save
    # =====================================================================

    output = "\n".join(lines)
    print(output)

    out_path = os.path.join(OUTPUT_DIR, "table_resilience.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
