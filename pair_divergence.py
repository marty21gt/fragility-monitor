#!/usr/bin/env python3
"""Paired divergence analyzer for the v3.0 vs v3.1 forward tracks.

Both tracks hold identical net exposure every day, so differencing their
realized returns removes the market entirely and leaves only the
implementation edge (fees minus decay/tracking). This script reads the two
ledgers' OUTCOME records, matches them by market_date, and reports:

  * the cumulative wealth ratio v3.1 / v3.0 (the "am I ahead" line);
  * the daily log-divergence series and its annualized drift;
  * a concentration test -- what share of the total edge comes from the
    largest-magnitude handful of days (lumpy = artifact, even = structural);
  * the correlation of daily divergence with |QQQ return| (a positive number
    means the edge rides big-move days, i.e. it is path/decay-driven rather
    than the boring fee drift you want to see); and
  * the fraction of divergence occurring on leveraged days (should be ~100%,
    since risk-off days are identical across tracks by construction).

Usage:
    python pair_divergence.py \
        --v30 forward/v3.0/ledger.jsonl \
        --v31 forward/v3.1/ledger.jsonl \
        [--top-k 5] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def read_outcomes(path: Path) -> dict[str, dict[str, Any]]:
    """Return {market_date: details} for every outcome record in a ledger."""
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("record_type") != "outcome":
                continue
            payload = rec.get("payload", {})
            date = payload.get("market_date")
            details = payload.get("details", {})
            if date is not None:
                out[date] = details
    return out


def _annualize(total_log: float, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    return math.exp(total_log * (252.0 / n_days)) - 1.0


def analyze(v30: Path, v31: Path, top_k: int) -> dict[str, Any]:
    a = read_outcomes(v30)
    b = read_outcomes(v31)
    dates = sorted(set(a) & set(b))
    if not dates:
        raise SystemExit("No overlapping outcome dates between the two ledgers.")

    rows = []
    cum_log = 0.0
    for d in dates:
        r30 = float(a[d]["realized_return"])
        r31 = float(b[d]["realized_return"])
        qqq = float(a[d].get("benchmark_qqq_return", b[d].get("benchmark_qqq_return", 0.0)))
        # log wealth-ratio increment: ln((1+r31)/(1+r30)); additive over time.
        log_div = math.log1p(r31) - math.log1p(r30)
        cum_log += log_div
        levered = abs(r31 - r30) > 1e-12
        rows.append(
            {
                "date": d,
                "r_v30": r30,
                "r_v31": r31,
                "qqq": qqq,
                "log_div": log_div,
                "cum_ratio": math.exp(cum_log),
                "levered_day": levered,
            }
        )

    n = len(rows)
    total_log = cum_log
    logs = [row["log_div"] for row in rows]
    total_abs = sum(abs(x) for x in logs) or 1.0

    # Concentration: share of NET edge from the top-k |days|.
    by_mag = sorted(rows, key=lambda r: abs(r["log_div"]), reverse=True)
    topk_net = sum(r["log_div"] for r in by_mag[:top_k])
    topk_share = topk_net / total_log if abs(total_log) > 1e-12 else float("nan")

    # Correlation of daily divergence with |QQQ| move (path/decay indicator).
    absqqq = [abs(row["qqq"]) for row in rows]
    mean_ld = sum(logs) / n
    mean_aq = sum(absqqq) / n
    cov = sum((logs[i] - mean_ld) * (absqqq[i] - mean_aq) for i in range(n)) / n
    var_ld = sum((x - mean_ld) ** 2 for x in logs) / n
    var_aq = sum((x - mean_aq) ** 2 for x in absqqq) / n
    corr = cov / math.sqrt(var_ld * var_aq) if var_ld > 0 and var_aq > 0 else float("nan")

    levered_days = sum(1 for row in rows if row["levered_day"])
    levered_log = sum(row["log_div"] for row in rows if row["levered_day"])

    return {
        "rows": rows,
        "summary": {
            "matched_days": n,
            "date_range": [rows[0]["date"], rows[-1]["date"]],
            "cumulative_wealth_ratio_v31_over_v30": math.exp(total_log),
            "annualized_divergence_drift": _annualize(total_log, n),
            "levered_days": levered_days,
            "levered_day_fraction": levered_days / n,
            "share_of_edge_on_levered_days": (
                levered_log / total_log if abs(total_log) > 1e-12 else float("nan")
            ),
            f"top_{top_k}_day_share_of_net_edge": topk_share,
            "abs_concentration_top_k": sum(abs(r["log_div"]) for r in by_mag[:top_k]) / total_abs,
            "corr_divergence_vs_abs_qqq": corr,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v30", type=Path, required=True)
    p.add_argument("--v31", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    result = analyze(args.v30, args.v31, args.top_k)
    print(json.dumps(result["summary"], indent=2))

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["date", "r_v30", "r_v31", "qqq", "log_div", "cum_ratio", "levered_day"],
            )
            writer.writeheader()
            for row in result["rows"]:
                writer.writerow(row)
        print(f"\nWrote per-day series to {args.csv}")


if __name__ == "__main__":
    main()
