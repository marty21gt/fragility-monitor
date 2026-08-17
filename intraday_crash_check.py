#!/usr/bin/env python3
"""
INTRADAY CRASH CHECK v3.3  (runs ~3:45pm ET via intraday-crash-check-v33.yml)
============================================================================
Same-close half of the emergency exit. Pulls the live QQQ price ~15 min before
the bell, tests the fast-crash trigger against the close SESSIONS sessions back,
and if it fires (and we're not already out) PERSISTS the emergency state via
alpaca_exec.set_emergency(). It stages nothing itself -- the stage_order.py step
that runs next sees exposure 0 through current_target() and produces the exit
proposal through the existing (safety-checked) machinery, for Approve & Submit
before the close.

Manual backstop: you always know the two prior sessions, so on a red day you can
eyeball whether the third trips the trigger even if this job or the cron miss.

Env: APCA_API_KEY_ID / APCA_API_SECRET_KEY (V33 paper pair -- data only here).
"""
import os, json, datetime, urllib.request
import alpaca_exec as X
import reference_backtest as R


def live_price(sym="QQQ"):
    key, sec = os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"]
    url = f"https://data.alpaca.markets/v2/stocks/{sym}/trades/latest?feed=iex"
    req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
    return float(json.load(urllib.request.urlopen(req, timeout=30))["trade"]["p"])


def main():
    active, _ = X._read_emergency_state()
    if active:
        print("already in emergency exit -- no intraday action needed."); return

    df = R.load("data.json", "daily_vt.json")
    dates = [str(d.date()) for d in df.index]; px = df["px"]
    today = datetime.date.today().isoformat()
    # close SESSIONS sessions ago; skip today's bar if the pipeline already committed it
    k = X.EMERGENCY_SESSIONS + (1 if dates[-1] == today else 0)
    ref, ref_date = float(px.iloc[-k]), dates[-k]

    p = live_price("QQQ")
    move = p / ref - 1
    print(f"3:45 check {today}: QQQ {p:.2f}  vs close {ref_date} {ref:.2f}  ->  "
          f"{X.EMERGENCY_SESSIONS}-session {move*100:+.2f}%  (threshold {-X.EMERGENCY_DROP*100:.0f}%)")

    if move <= -X.EMERGENCY_DROP:
        X.set_emergency(today)
        open("EMERGENCY_FIRED", "w").write(today + "\n")   # marker: workflow stages ONLY when this exists
        print(f"EMERGENCY: state set active (exit {today}). The staging step will propose "
              f"the exit to SGOV -- approve & submit before the close.")
    else:
        print("trigger not breached; no action.")


if __name__ == "__main__":
    main()
