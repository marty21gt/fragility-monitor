#!/usr/bin/env python3
# =====================================================================
#  preflight_check.py  --  verify the live leverage target before a trade
#
#  Answers one question: "Is the target the pipeline is showing me
#  trustworthy, or is it built on bad data?"  It checks the two things
#  that actually went wrong in the 2026-08 incident:
#     1. COMPLETENESS -- is the Nasdaq-100 price series gap-free in the
#        recent window?  (Missing sessions corrupt realized vol and flip
#        the target -- the 1.31x/1.05x whipsaw.)
#     2. CLEAN STEP    -- on a rebalance day, does the raw target equal
#        the banded target?  (They match on a genuine step; a mismatch
#        means you're looking at mid-band drift, not a real trade.)
#
#  Reads the committed data.json + daily_vt.json in the repo (the same
#  files the stage job uses) and computes the target with the same engine.
#  Exits 1 (red run) if the data has a gap, so a bad target can never
#  pass silently.  Optional: set CURRENT_POS to your live exposure and it
#  will tell you the trade direction.
# =====================================================================
import os, sys, json
import numpy as np, pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday,
    USMartinLutherKingJr, USPresidentsDay, GoodFriday,
    USMemorialDay, USLaborDay, USThanksgivingDay)

DATA = os.environ.get("DATA_JSON", "data.json")
VT   = os.environ.get("VT_JSON", "daily_vt.json")
BAND = 0.10          # REBALANCE_BAND (must match the engine)
WINDOW_DAYS = 120    # completeness look-back


def rule(c="-"): print(c * 68)


# ---------- 1) completeness: no missing NYSE session in the recent window ----------
class _NYSECal(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2022-06-19"),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday)]


def completeness(dates):
    qd = pd.to_datetime(dates).sort_values()
    end = qd[-1]
    start = max(qd[0], end - pd.Timedelta(days=WINDOW_DAYS))
    wk = pd.bdate_range(start, end)
    hol = set(_NYSECal().holidays(wk[0], wk[-1]))
    expected = [x for x in wk if x not in hol]
    have = set(qd)
    missing = [x.strftime("%Y-%m-%d") for x in expected if x not in have]
    return qd, len(expected), missing


def main():
    if not (os.path.exists(DATA) and os.path.exists(VT)):
        print(f"ERROR: could not find {DATA} / {VT} in the working directory.")
        sys.exit(2)

    try:
        import reference_backtest_v33 as R
    except Exception as e:
        print(f"ERROR: could not import the engine (reference_backtest_v33.py): {e}")
        print("Make sure that file sits next to this one in the repo.")
        sys.exit(2)

    tq = json.load(open(DATA))["timeline_qqq"]
    qd, n_expected, missing = completeness(tq["dates"])

    rule("=")
    print("  MODERN EDGE TACTICAL v3.3 -- PREFLIGHT TARGET CHECK")
    rule("=")
    import datetime as _dt
    last = qd[-1].date()
    run_date = _dt.datetime.now(_dt.timezone.utc).date()
    stale_days = int(np.busday_count(last, run_date))
    print(f"  data.json through : {last}   (target applies to the NEXT session)")
    print(f"  run date (UTC)    : {run_date}"
          + (f"   [WARNING: {stale_days} business days stale]" if stale_days > 1 else ""))
    print()

    # ---- check 1: completeness ----
    print("  [1] DATA COMPLETENESS (last %d days)" % WINDOW_DAYS)
    jul29 = pd.Timestamp("2026-07-29") in set(qd)
    aug07 = pd.Timestamp("2026-08-07") in set(qd)
    if missing:
        print(f"      FAIL -- {len(missing)} expected session(s) MISSING: {missing}")
        print("      A gap here corrupts realized vol. DO NOT trust the target.")
    else:
        print(f"      PASS -- {n_expected} expected sessions, all present (no gaps)")

    # ---- target math (same engine as the stage job) ----
    df = R.load(DATA, VT)
    pos = R.monitor_position(df)
    rv = R.realized_vol(df["px"])
    size = (R.TARGET_VOL / rv).clip(lower=R.LEVERAGE_FLOOR, upper=R.LEVERAGE_CAP)
    can = (R.leverage_gate(df["Td"]) & (df["px"] > df["ma"])
           & (R.ma_slope(df["ma"]) > R.SLOPE_DEADBAND))
    raw = pd.Series(np.where(can, size, 1.0), df.index).where(pos == 1, 0.0)
    banded = R.apply_band(raw)

    raw_now, band_now = float(raw.iloc[-1]), float(banded.iloc[-1])
    stepped = abs(band_now - float(banded.iloc[-2])) > 1e-9
    clean_step = abs(raw_now - band_now) < 1e-6

    print()
    print("  [2] TARGET (banded held vs raw model)")
    print(f"      target to hold : {band_now:.3f}x")
    print(f"      model wants    : {raw_now:.3f}x   (vol {rv.iloc[-1]*100:.1f}%)")
    if stepped:
        if clean_step:
            print("      -> REBALANCE DAY, and raw == banded: clean, genuine step. GOOD.")
        else:
            print(f"      -> stepped but raw != banded (gap {abs(raw_now-band_now):.3f}) -- unusual, inspect.")
    else:
        print("      -> no step today (holding). raw sitting inside the band is normal drift.")

    # ---- check 3: vol stability (a flip-flopping vol = a data problem) ----
    vlast = [round(float(rv.iloc[-k]) * 100, 1) for k in range(5, 0, -1)]
    swing = max(vlast) - min(vlast)
    print()
    print("  [3] VOL STABILITY (last 5 sessions)")
    print(f"      realized vol   : {vlast}   swing {swing:.1f} pts")
    print("      " + ("smooth -- trustworthy" if swing < 4 else
                      "LARGE swing -- if unexplained, suspect a data revision/gap"))

    # ---- optional: trade direction vs current position ----
    cur = os.environ.get("CURRENT_POS", "").strip()
    if cur:
        try:
            cur = float(cur)
            delta = band_now - cur
            direction = ("no change" if abs(delta) <= BAND / 2 else
                         ("INCREASE" if delta > 0 else "DECREASE"))
            print()
            print("  [4] YOUR TRADE")
            print(f"      current {cur:.3f}x  ->  target {band_now:.3f}x   ({direction}, delta {delta:+.3f}x)")
        except ValueError:
            pass

    # ---- verdict ----
    print()
    rule("=")
    if missing:
        print("  VERDICT: CAUTION -- data has a gap. Do NOT trade off this target.")
        print("           Re-run the pipeline; the completeness gate should block it.")
        rule("=")
        sys.exit(1)
    print("  VERDICT: GO -- data complete, target computed on clean, gap-free prices.")
    rule("=")
    sys.exit(0)


if __name__ == "__main__":
    main()
