#!/usr/bin/env python3
"""
INTRADAY CRASH CHECK v3.3  (runs ~3:45pm ET via intraday-crash-check-v33.yml)
============================================================================
Same-close half of the emergency exit. Both the live price AND the 3-session-ago
reference come from the QQQ ETF via Alpaca, so they are on the SAME scale (the
earlier bug compared the live QQQ ETF price to data.json's NDX INDEX level, ~51x
apart, and false-fired at ~-98%). A -8%/3-session QQQ move tracks the NDX trigger
the backtest uses to within tracking error, which is immaterial at the -8% bar.

If it fires (and we're not already out) it PERSISTS the emergency state via
alpaca_exec.set_emergency() and drops the EMERGENCY_FIRED marker; the stage step
that follows produces the SGOV exit proposal through the existing machinery.

Sanity guard: a real 3-session move is never below ~-40%, so anything worse is a
data/scale glitch, not a crash -- the check refuses to fire on it.

Env: APCA_API_KEY_ID / APCA_API_SECRET_KEY (V33 paper pair; market-data only here).
"""
import os, json, datetime, urllib.request
import alpaca_exec as X

_HDR = lambda: {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
                "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}


def live_price(sym="QQQ"):
    url = f"https://data.alpaca.markets/v2/stocks/{sym}/trades/latest?feed=iex"
    req = urllib.request.Request(url, headers=_HDR())
    return float(json.load(urllib.request.urlopen(req, timeout=30))["trade"]["p"])


def recent_closes(sym="QQQ"):
    """Completed daily closes from Alpaca, oldest-first: [(YYYY-MM-DD, close), ...].
    Excludes today's in-progress bar so the reference is a settled session."""
    end = datetime.date.today()
    start = end - datetime.timedelta(days=15)              # ~10 sessions of cushion
    url = (f"https://data.alpaca.markets/v2/stocks/{sym}/bars"
           f"?timeframe=1Day&start={start.isoformat()}&end={end.isoformat()}"
           f"&limit=50&feed=iex&adjustment=split")
    req = urllib.request.Request(url, headers=_HDR())
    bars = json.load(urllib.request.urlopen(req, timeout=30)).get("bars") or []
    today = end.isoformat()
    return [(b["t"][:10], float(b["c"])) for b in bars if b["t"][:10] < today]


def main():
    active, _ = X._read_emergency_state()
    if active:
        print("already in emergency exit -- no intraday action needed."); return

    today  = datetime.date.today().isoformat()
    closes = recent_closes("QQQ")
    if len(closes) < X.EMERGENCY_SESSIONS:
        print(f"WARN: only {len(closes)} completed QQQ closes (need {X.EMERGENCY_SESSIONS}); skipping."); return

    ref_date, ref = closes[-X.EMERGENCY_SESSIONS]         # QQQ close SESSIONS sessions ago
    p = live_price("QQQ")                                  # live QQQ ETF -- same instrument, same scale
    move = p / ref - 1
    print(f"3:45 check {today}: QQQ {p:.2f}  vs QQQ close {ref_date} {ref:.2f}  ->  "
          f"{X.EMERGENCY_SESSIONS}-session {move*100:+.2f}%  (threshold {-X.EMERGENCY_DROP*100:.0f}%)")

    if move < -0.40:
        print(f"ABORT: {move*100:.0f}% is implausible for {X.EMERGENCY_SESSIONS} sessions "
              f"-- treating as a data/scale error, NOT firing."); return

    if move <= -X.EMERGENCY_DROP:
        X.set_emergency(today)
        open("EMERGENCY_FIRED", "w").write(today + "\n")   # marker: workflow stages ONLY when this exists
        print(f"EMERGENCY: state set active (exit {today}). Approve & submit before the close.")
    else:
        print("trigger not breached; no action.")


if __name__ == "__main__":
    main()
