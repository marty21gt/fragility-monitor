#!/usr/bin/env python3
# =====================================================================
#  build_backtest_v33.py -- regenerate the Timeline Explorer series
#
#  WHY
#  ---
#  The v3.3 page ships a hard-coded `const EXP={...}` blob holding the
#  1986-2026 backtest that drives the Timeline Explorer (growth curve,
#  V/T path, leverage shading, the stat cards). It is a static snapshot:
#  it only advances when someone regenerates and re-pastes it, so the
#  chart drifts months behind the live Daily Action panel above it.
#
#  This script rebuilds that exact structure from the published feed on
#  every pipeline run and writes backtest_v33.json, which the page fetches.
#
#  IMPORTANT -- this is the FULL v3.3 strategy, monitor + emergency overlay.
#  Do NOT rebuild these curves from data.json's `stret`/`pos`: those carry
#  the monitor only. Verified: through COVID, data.json has pos==1 every day
#  and stret == bhqqq exactly (-28% drawdown, no exit), whereas true v3.3
#  exits 2020-02-26 on the -8%/3-session rule, sits out the 35-day lockout,
#  and takes -14%. Using the feed's own series would silently delete the
#  fast-crash behaviour from the chart.
#
#  OUTPUT SCHEMA (matches the embedded EXP exactly)
#    n    int            number of rows
#    t    [float]        v3.3 tactical equity, rebased to 1.0 at the start
#    q    [float]        QQQ buy & hold equity, rebased to 1.0
#    s    [float]        S&P 500 buy & hold equity, rebased to 1.0
#    off  [0|1]          1 = out of the market (SGOV)
#    e    [int]          effective exposure x100 (108 = 1.08x)
#    V    [int]          DAILY nowcast V x100  (daily_vt.json Vd)
#    T    [int]          DAILY nowcast T x100  (daily_vt.json Td)
#    yr   {year: idx}    first row index of each calendar year
#    y0,y1 int           first / last year
#  Floats are emitted at 4 significant figures, as in the original blob,
#  which keeps the file about a third the size of full precision.
#
#  Run:  python build_backtest_v33.py            (writes backtest_v33.json)
# =====================================================================
import json
import sys

import numpy as np
import pandas as pd

import reference_backtest_v33 as R

DATA = "data.json"
VT = "daily_vt.json"
OUT = "backtest_v33.json"


def log(m):
    print(m, flush=True)


def sig4(x):
    """4 significant figures, matching the precision of the original export."""
    if x is None or not np.isfinite(x):
        return None
    return float(f"{float(x):.4g}")


def main():
    df = R.load(DATA, VT)

    # ---- the full v3.3 stack: monitor gate -> vol-scaled target -> emergency overlay ----
    pos = R.monitor_position(df)                       # monthly V/T gate + 200-day state machine
    base = R.target_exposure(df, pos)                  # vol scaling + leverage gate + band
    held = R.emergency_overlay(df["px"], base)         # -8%/3-session fast-crash exit
    cash = R.cash_rate(df)
    exec_held = held.shift(1).fillna(0)                # what you actually HOLD that day
    rets = R.strategy_returns(df, exec_held, cash)

    # ---- equity curves, all rebased to 1.0 on the first row ----
    raw_tq = json.load(open(DATA))["timeline_qqq"]          # bhret (S&P) isn't in R.load()
    bhret = pd.Series(raw_tq["bhret"], df.index, dtype=float).fillna(0)
    t = (1 + rets).cumprod()
    q = (1 + df["bhqqq"].fillna(0)).cumprod()
    s = (1 + bhret).cumprod()
    t, q, s = (x / x.iloc[0] for x in (t, q, s))

    # ---- daily nowcast V/T (what the V/T path plots) ----
    dv = json.load(open(VT))
    vd = pd.Series(dv["Vd"], pd.to_datetime(dv["dates"]), dtype=float).reindex(df.index).ffill().bfill()
    td = pd.Series(dv["Td"], pd.to_datetime(dv["dates"]), dtype=float).reindex(df.index).ffill().bfill()

    # ---- year -> first row index ----
    years = df.index.year
    yr = {}
    for i, y in enumerate(years):
        if str(y) not in yr:
            yr[str(y)] = i

    exp = {
        "n": int(len(df)),
        "t": [sig4(x) for x in t.values],
        "q": [sig4(x) for x in q.values],
        "s": [sig4(x) for x in s.values],
        # `off`/`e` describe the EXECUTED book (signal lagged one session), which is
        # what the chart shading should show -- and matches the original export exactly.
        "off": [int(x == 0) for x in exec_held.values],
        "e": [int(round(float(x) * 100)) for x in exec_held.values],
        "V": [int(round(float(x) * 100)) for x in vd.values],
        "T": [int(round(float(x) * 100)) for x in td.values],
        "yr": yr,
        "y0": int(years[0]),
        "y1": int(years[-1]),
        # provenance -- ignored by the page, useful when auditing a commit
        "asof": df.index[-1].strftime("%Y-%m-%d"),
        "engine": "reference_backtest_v33 (monitor + emergency overlay)",
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(exp, f, separators=(",", ":"))

    dd = (t / t.cummax() - 1).min()
    yrs = len(t) / 252
    log(f"  wrote {OUT}: {exp['n']} rows, {df.index[0].date()} -> {df.index[-1].date()}")
    log(f"    tactical x{t.iloc[-1]:,.0f}  ({t.iloc[-1] ** (1/yrs) - 1:.1%} CAGR)  maxDD {dd:.0%}  "
        f"| buy&hold x{q.iloc[-1]:,.0f}")
    log(f"    time out of market: {np.mean(exec_held.values == 0):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
