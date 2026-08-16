#!/usr/bin/env python3
"""
INTRADAY CRASH CHECK v3.3 — runs ~3:45pm ET  (via intraday-crash-check.yml)
==========================================================================
The same-close half of the emergency exit. Fires ~15 minutes before the bell,
pulls the live QQQ price, and evaluates the fast-crash trigger against the last
completed closes. If the trigger is breached AND we're not already out, it stages
an emergency exit to SGOV and raises the RED_FLAG so you can Approve & Submit
before the close — capturing the same-day fill the backtest assumes.

It only STAGES. It never submits. And it publishes only STATE to emergency.json.

Manual backstop reminder: you always know the two prior sessions, so on a red day
you can eyeball whether the third session trips the trigger even if this job or the
cron miss. This is the primary path; your arithmetic is the fallback.

Env: APCA_API_KEY_ID / APCA_API_SECRET_KEY (any matched paper pair — data only).
Reads local data.json + daily_vt.json. Writes proposal.json, emergency.json, RED_FLAG.
"""
import json, os, datetime, urllib.request
import numpy as np, pandas as pd
import reference_backtest_v33 as R
from stage_v33 import emergency_status, alloc

def live_price(sym="QQQ"):
    key, sec = os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"]
    url = f"https://data.alpaca.markets/v2/stocks/{sym}/trades/latest?feed=iex"
    req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
    return float(json.load(urllib.request.urlopen(req, timeout=30))["trade"]["p"])

def main():
    df   = R.load("data.json", "daily_vt.json")
    pos  = R.monitor_position(df)
    base = R.target_exposure(df, pos)
    _, status = emergency_status(df["px"], base)
    if status["state"] == "active":
        print("already in emergency exit — no intraday action needed."); return

    px = df["px"]; today = datetime.date.today().isoformat()
    # close from EMERGENCY_SESSIONS sessions ago, skipping today's bar if it's already committed
    k = R.EMERGENCY_SESSIONS + (1 if px.index[-1].date().isoformat() == today else 0)
    ref_close = float(px.iloc[-k]); ref_date = px.index[-k].date().isoformat()

    p = live_price("QQQ")
    move = p / ref_close - 1
    print(f"3:45 check {today}: QQQ {p:.2f}  vs  close {ref_date} {ref_close:.2f}  ->  "
          f"{R.EMERGENCY_SESSIONS}-session {move*100:+.2f}%  (threshold {-R.EMERGENCY_DROP*100:.0f}%)")

    if move <= -R.EMERGENCY_DROP:
        pub = {"state": "active", "exit": today, "days": 0, "released": False, "asof": today}
        json.dump(pub, open("emergency.json", "w"), indent=2)
        proposal = dict(version="v3.3", asof=today, target_exposure=0.0,
                        target_weights=alloc(0.0), trade_needed=True, emergency="active",
                        note=f"INTRADAY EMERGENCY EXIT — {R.EMERGENCY_SESSIONS}-session move "
                             f"{move*100:.1f}% breached; move fully to SGOV, same-close.")
        json.dump(proposal, open("proposal.json", "w"), indent=2)
        open("RED_FLAG", "w").write(proposal["note"] + "\n")
        print("RED_FLAG raised — emergency exit staged. Approve & Submit before the close.")
    else:
        print("trigger not breached; no action.")

if __name__ == "__main__":
    main()
