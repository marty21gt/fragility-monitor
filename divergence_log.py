#!/usr/bin/env python3
"""
QLD-vs-TQQQ divergence tracker  (stripped-down, no ledger, no hash chain)
=========================================================================
v3.0 (QQQ/QLD) and v3.1 (QQQ/TQQQ) hold the IDENTICAL target exposure every
day, so the difference in their realized returns is pure implementation P&L --
fees minus decay -- with the market cancelled out by construction. On any
non-levered day the two are identical and the divergence is exactly zero.

This appends one dated row per trading day to divergence.csv and, on request,
reports whether the v3.1 edge is accruing as a slow boring drift (good) or in
lumps (a regime artifact). The git commit of divergence.csv is your proof the
number existed before the next day's data did -- that's all the "it came first"
guarantee you actually need.

Reuses reference_backtest.py so the strategy logic has ONE source of truth.

USAGE
  python divergence_log.py update    # append any new days to divergence.csv
  python divergence_log.py report    # print the drift / lumpiness summary
"""
import csv, math, sys
from pathlib import Path
import numpy as np, pandas as pd
import reference_backtest as R

CSV = Path("divergence.csv")
TQQQ_EXPENSE = 0.0082          # 0.82% annual, ProShares TQQQ
COST_BPS = 5.0                 # charged on |Δexposure|, identical for both tracks


def both_tracks():
    """Recompute v3.0 and v3.1 realized daily returns from the published data.
    Both use the exact same exposure path; only the leveraged sleeve differs."""
    df   = R.load("data.json", "daily_vt.json")
    cash = R.cash_rate(df)
    sig  = R.monitor_position(df)
    held = R.target_exposure(df, sig).shift(1).fillna(0)     # executed exposure

    # v3.0 : QQQ / QLD  -- straight from the engine
    r_v30 = R.strategy_returns(df, held, cash, cost_bps=COST_BPS)

    # v3.1 : QQQ / TQQQ -- matched net exposure, 3x financing-aware sleeve
    qqq  = df["bhqqq"].fillna(0)
    tqqq = 3 * qqq - 2 * cash - TQQQ_EXPENSE / 252
    e = held
    w_tqqq = ((e - 1) / 2).clip(lower=0)
    w_qqq  = pd.Series(np.where(e > 1, (3 - e) / 2, e), df.index).clip(lower=0)
    w_cash = (1 - w_qqq - w_tqqq).clip(lower=0)
    r_v31 = (w_qqq * qqq + w_tqqq * tqqq + w_cash * cash).fillna(0)
    r_v31 = r_v31 - e.diff().abs().fillna(0) * (COST_BPS / 10000.0)

    return df.index, e, r_v30, r_v31, qqq


def update():
    idx, e, r0, r1, qqq = both_tracks()
    seen = set()
    if CSV.exists():
        seen = {row["date"] for row in csv.DictReader(CSV.open())}
    added = 0
    with CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if not seen:
            w.writerow(["date", "exposure", "r_v30", "r_v31", "qqq_ret"])
        for i in range(len(idx)):
            d = idx[i].date().isoformat()
            if d in seen:
                continue
            w.writerow([d, f"{e.iloc[i]:.4f}", f"{r0.iloc[i]:.8f}",
                        f"{r1.iloc[i]:.8f}", f"{qqq.iloc[i]:.8f}"])
            added += 1
    print(f"appended {added} new day(s); divergence.csv now holds {len(seen)+added} days")


def report(top_k=5):
    if not CSV.exists():
        sys.exit("no divergence.csv yet -- run `update` first")
    rows = list(csv.DictReader(CSV.open()))
    logs, absq = [], []
    cum, lev_log, lev_days = 0.0, 0.0, 0
    for row in rows:
        r0, r1, q = float(row["r_v30"]), float(row["r_v31"]), float(row["qqq_ret"])
        ld = math.log1p(r1) - math.log1p(r0)      # daily log wealth-ratio step
        cum += ld
        logs.append(ld); absq.append(abs(q))
        if abs(r1 - r0) > 1e-12:                   # a day where the vehicle mattered
            lev_log += ld; lev_days += 1

    n = len(logs)
    ann = math.exp(cum * 252 / n) - 1 if n else 0.0
    by_mag = sorted(logs, key=abs, reverse=True)
    topk = sum(by_mag[:top_k]) / cum if abs(cum) > 1e-12 else float("nan")
    ml, ma = sum(logs) / n, sum(absq) / n
    cov = sum((logs[i] - ml) * (absq[i] - ma) for i in range(n)) / n
    vl = sum((x - ml) ** 2 for x in logs) / n
    va = sum((x - ma) ** 2 for x in absq) / n
    corr = cov / math.sqrt(vl * va) if vl > 0 and va > 0 else float("nan")

    print(f"days logged             : {n}   ({rows[0]['date']} .. {rows[-1]['date']})")
    print(f"cumulative v3.1 / v3.0  : {math.exp(cum):.4f}    (>1 means TQQQ ahead)")
    print(f"annualized drift        : {ann*100:+.3f}%/yr")
    print(f"days the vehicle mattered: {lev_days} of {n}  ({lev_days/n*100:.0f}% levered)")
    print(f"top {top_k}-day share of edge : {topk*100:5.0f}%    <- high = lumpy = artifact")
    print(f"corr with |QQQ| move    : {corr:+.2f}     <- ~0 = fee-driven (good); "
          f"+ = decay/path-driven")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "update"
    if cmd == "update":
        update()
    elif cmd == "report":
        report()
    else:
        sys.exit("usage: python divergence_log.py [update|report]")
