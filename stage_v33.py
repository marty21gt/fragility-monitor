#!/usr/bin/env python3
"""
STAGE v3.3 — daily post-close staging job  (runs ~5:00pm ET via daily-v33.yml)
==============================================================================
Semi-automated by design: this job only STAGES. It computes the v3.3 target
exposure (v3.2 vol-scaled base + fast-crash emergency overlay), advances the
emergency state machine one day, writes a rebalance PROPOSAL for review, raises a
RED_FLAG marker when a trade is needed, and publishes a STATE-ONLY emergency.json
for the dashboard card. Nothing is submitted here — the existing Approve & Submit
step consumes proposal.json after a human clicks approve.

OPACITY: the public emergency.json carries only {state, exit, days, released} —
never the -8% rule, the 3-session window, or the 35-day threshold. Those live in
reference_backtest_v33.py (imported) and never reach the feed or the page.

Reads local data.json + daily_vt.json (checked out / fetched by the workflow).
Writes: proposal.json, emergency.json, and RED_FLAG (only if a trade is needed).
"""
import json, os, sys, datetime
import numpy as np, pandas as pd
import reference_backtest_v33 as R

ACCOUNT_EQUITY_FALLBACK = float(os.environ.get("ACCOUNT_EQUITY", "0") or 0)

# --------------------------------------------------------------------------
# emergency state machine, with the CURRENT-day status pulled out for publishing
# (re-runs deterministically from history — the state is implicit in the price
#  path, so there is no separate state file to drift out of sync)
# --------------------------------------------------------------------------
def emergency_status(px, base):
    trig = (px / px.shift(R.EMERGENCY_SESSIONS) - 1 <= -R.EMERGENCY_DROP).fillna(False).values
    e = np.asarray(base, dtype=float); n = len(e); out = np.empty(n)
    in_em = False; entry = None; released_today = False
    for i in range(n):
        released_today = False
        if in_em:
            if (i - entry) >= R.EMERGENCY_REENTRY:
                in_em = False; released_today = True      # today is the re-entry day
        elif trig[i]:
            in_em = True; entry = i
        out[i] = 0.0 if in_em else e[i]
    idx = px.index
    status = dict(
        state    = "active" if in_em else ("released" if released_today else "normal"),
        exit     = (idx[entry].date().isoformat() if entry is not None and in_em else None),
        days     = int(n - 1 - entry) if (in_em and entry is not None) else 0,
        released = bool(released_today),
        asof     = idx[-1].date().isoformat(),
    )
    return pd.Series(out, idx), status

# --------------------------------------------------------------------------
# exposure E -> QQQ / TQQQ / SGOV weights (no-margin blend that sums to 1.0)
# --------------------------------------------------------------------------
def alloc(E, mult=R.MULT):
    w_lev  = max((E - 1) / (mult - 1), 0.0)
    w_qqq  = max(1 - (E - 1) / (mult - 1), 0.0) if E > 1 else max(E, 0.0)
    w_sgov = max(1 - w_qqq - w_lev, 0.0)
    return {"QQQ": round(w_qqq, 4), "TQQQ": round(w_lev, 4), "SGOV": round(w_sgov, 4)}

def current_weights():
    """Pull current paper positions from Alpaca; empty dict if unavailable."""
    key, sec = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    base = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    if not (key and sec): return {}, None
    import urllib.request
    def get(path):
        req = urllib.request.Request(base + path,
              headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
        return json.load(urllib.request.urlopen(req, timeout=30))
    try:
        acct = get("/v2/account"); eq = float(acct["equity"])
        pos = {p["symbol"]: float(p["market_value"]) for p in get("/v2/positions")}
        w = {s: round(pos.get(s, 0.0) / eq, 4) for s in ("QQQ", "TQQQ")}
        w["SGOV"] = round(1 - w["QQQ"] - w["TQQQ"], 4)
        return w, eq
    except Exception as ex:
        print(f"  (Alpaca read failed: {ex}; staging without live position)"); return {}, None

def main():
    df   = R.load("data.json", "daily_vt.json")
    pos  = R.monitor_position(df)
    base = R.target_exposure(df, pos)                 # v3.2 vol-scaled base
    tgt, status = emergency_status(df["px"], base)    # v3.3 = base + overlay
    E = float(tgt.iloc[-1])                            # today's target exposure
    want = alloc(E)
    have, eq = current_weights()

    band = R.REBALANCE_BAND
    drift = max(abs(want[s] - have.get(s, 0.0)) for s in want) if have else 1.0
    trade_needed = (not have) or (drift > band)

    proposal = dict(
        version   = "v3.3",
        asof      = status["asof"],
        target_exposure = round(E, 3),
        target_weights  = want,
        current_weights = have or None,
        drift     = round(drift, 4),
        band      = band,
        trade_needed = bool(trade_needed),
        emergency = status["state"],                  # active / released / normal
        note      = ("EMERGENCY EXIT — move fully to SGOV" if status["state"] == "active" and E == 0
                     else "re-entry from emergency" if status["state"] == "released"
                     else "routine vol-scaled rebalance"),
    )
    json.dump(proposal, open("proposal.json", "w"), indent=2)
    print("staged proposal.json:", json.dumps(proposal, separators=(",", ":")))

    # STATE-ONLY public feed for the dashboard card (no rule parameters ever)
    pub = {k: status[k] for k in ("state", "exit", "days", "released", "asof")}
    json.dump(pub, open("emergency.json", "w"), indent=2)
    print("published emergency.json:", json.dumps(pub, separators=(",", ":")))

    if trade_needed:
        open("RED_FLAG", "w").write(proposal["note"] + f"  (target {want})\n")
        print("RED_FLAG raised — approval required before submit.")
    else:
        print("no trade needed (within band); no red flag.")

if __name__ == "__main__":
    main()
