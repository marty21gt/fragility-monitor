#!/usr/bin/env python3
"""
update_ledger_v33.py -- nightly update of ledger/v33_paper_trading.xlsx (v3.3 paper track)

Runs at 8pm ET, after the evening pipeline has committed data.json / daily_vt.json /
emergency.json. Each run:

  1. ACTUALS. Rebuilds daily share counts and cash from the account's complete Alpaca
     order and cash-activity history and marks them at official (SIP) closes. Fills
     every completed session row whose QQQ close is still blank, so a missed night
     simply catches up on the next run.
  2. MODEL COLUMNS (L-U). Writes the NEXT session's row exactly as the v3.3 dashboard's
     Daily Action panel computes it (a line-by-line port of build() in
     modern-edge-tactical-v33/index.html), from the as-of close. Only fills blanks.
  3. TRADE LOG. One row per filled order. The reference price is the ARRIVAL price:
     the NBBO midpoint when the order was sent; orders sent before the open use the
     official open. Slippage therefore measures execution only, not the day's move.
  4. RECONCILIATION. Share counts and cash must match Alpaca exactly, and Alpaca's
     last_equity must match a reconstructed close to the cent. Any mismatch is written
     to ledger_alert.txt, which turns the workflow run red so GitHub emails you.

It never overwrites anything you typed (notes, flags, model columns): it fills blanks.
The Trade Log reference price is the one exception -- it is a computed field.

Safety: GET requests only. Account calls are locked to the PAPER endpoint.
"""
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from copy import copy
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import openpyxl
from openpyxl.formula.translate import Translator
from openpyxl.workbook.properties import CalcProperties

LEDGER = "ledger/v33_paper_trading.xlsx"
DATA, VT, EMRG = "data.json", "daily_vt.json", "emergency.json"
ALERT = "ledger_alert.txt"

PAPER = "https://paper-api.alpaca.markets"     # PAPER endpoint only
MKT = "https://data.alpaca.markets"
ET = ZoneInfo("America/New_York")
HISTORY_FROM = "2026-07-20"                    # before the account was funded (7/27)
FUNDED = "2026-07-27"
RECORD_START = "2026-08-03"                    # v3.3 record begins at this close
SYMS = ["QQQ", "TQQQ", "SGOV"]
RUN_HOUR_ET = 20                               # scheduled runs do nothing before 8pm ET

# Sheet1 columns (1-based)
C_DATE, C_QSH, C_QPX, C_TSH, C_TPX = 1, 3, 4, 7, 8
C_L, C_M, C_N, C_O, C_P, C_Q, C_R, C_S, C_T, C_U = range(12, 22)
C_SGSH, C_SGPX, C_CASH, C_TRADED = 29, 30, 33, 34
HOLD_TEXT = "no change - hold current position"   # the sheet's own wording
TL_REF_HEADER = "Arrival price (NBBO mid at order)"

KEY = os.environ.get("APCA_API_KEY_ID")
SECRET = os.environ.get("APCA_API_SECRET_KEY")
_summary = []


def log(s=""):
    print(s, flush=True)
    _summary.append(s)


# ----------------------------------------------------------------------------
# Alpaca (GET only)
# ----------------------------------------------------------------------------
def get(base, path, soft=False, **params):
    """soft=True returns None on an HTTP error instead of stopping the run."""
    params = {k: v for k, v in params.items() if v is not None}
    url = base + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, method="GET", headers={
        "APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        msg = f"Alpaca returned {e.code} for {path}: {e.read().decode()[:400]}"
        if soft:
            log(f"  note: {msg}")
            return None
        sys.exit(msg)


def to_et(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ET)


def pull_orders():
    out, after = [], f"{HISTORY_FROM}T00:00:00Z"
    while True:
        page = get(PAPER, "/v2/orders", status="closed", after=after, direction="asc", limit=500)
        out += page
        if len(page) < 500:
            break
        after = page[-1]["submitted_at"]
    return [o for o in out if float(o.get("filled_qty") or 0) > 0]


def pull_cash_activities():
    acts, token = [], None
    while True:
        page = get(PAPER, "/v2/account/activities", after=f"{HISTORY_FROM}T00:00:00Z",
                   direction="asc", page_size=100, page_token=token)
        acts += page
        if len(page) < 100:
            break
        token = page[-1]["id"]
    return [a for a in acts if a.get("activity_type") != "FILL"]


def pull_daily_bars(end_iso):
    bars, token = defaultdict(dict), None
    while True:
        page = get(MKT, "/v2/stocks/bars", symbols=",".join(SYMS), timeframe="1Day",
                   start=f"{HISTORY_FROM}T00:00:00Z", end=end_iso, feed="sip",
                   adjustment="raw", limit=10000, page_token=token)
        for sym, lst in (page.get("bars") or {}).items():
            for b in lst:
                bars[to_et(b["t"]).date().isoformat()][sym] = {"o": b["o"], "c": b["c"]}
        token = page.get("next_page_token")
        if not token:
            break
    return bars


def arrival_price(order, bars):
    """NBBO midpoint at submission; the official open for orders sent before the open."""
    sub = to_et(order["submitted_at"])
    d, sym = sub.date().isoformat(), order["symbol"]
    if sub.hour * 60 + sub.minute < 9 * 60 + 30:
        return bars.get(d, {}).get(sym, {}).get("o")
    q = get(MKT, "/v2/stocks/quotes", soft=True, symbols=sym, start=order["submitted_at"],
            limit=1, feed="sip") or {}
    lst = (q.get("quotes") or {}).get(sym) or []
    if lst and lst[0].get("ap") and lst[0].get("bp"):
        return round((lst[0]["ap"] + lst[0]["bp"]) / 2, 6)
    m = get(MKT, "/v2/stocks/bars", soft=True, symbols=sym, timeframe="1Min",
            start=order["submitted_at"], limit=1, feed="sip") or {}
    lst = (m.get("bars") or {}).get(sym) or []
    return lst[0]["o"] if lst else None


# ----------------------------------------------------------------------------
# Daily holdings + cash, rebuilt from the full order / activity history
# ----------------------------------------------------------------------------
def rebuild_state(sessions, orders, cash_acts, bars):
    by_day = defaultdict(list)
    for o in orders:
        by_day[to_et(o["filled_at"]).date().isoformat()].append(o)
    flows = defaultdict(float)
    for a in cash_acts:
        d = a.get("date") or (to_et(a["transaction_time"]).date().isoformat()
                              if a.get("transaction_time") else None)
        if d and a.get("net_amount") not in (None, ""):
            flows[d[:10]] += float(a["net_amount"])
    sh, cash, state = defaultdict(float), 0.0, {}
    for d in sessions:
        cash += flows.get(d, 0.0)
        for o in by_day.get(d, []):
            s = 1 if o["side"] == "buy" else -1
            qty, px = float(o["filled_qty"]), float(o["filled_avg_price"])
            sh[o["symbol"]] += s * qty
            cash -= s * round(qty * px, 2)
        px = bars.get(d, {})
        eq = cash + sum(q * px[s]["c"] for s, q in sh.items() if abs(q) > 1e-12 and s in px)
        state[d] = {"sh": dict(sh), "cash": round(cash, 2), "eq": round(eq, 2),
                    "px": {s: px[s]["c"] for s in px}, "traded": bool(by_day.get(d))}
    return state


# ----------------------------------------------------------------------------
# Model columns: port of build() in the v3.3 dashboard (modern-edge-tactical-v33/index.html)
# ----------------------------------------------------------------------------
CFG = dict(tv=0.25, cap=3.0, floor=0.10, tCut=0.60, tRestore=0.50, band=0.10, reentry=15)
WINDOW, WARMUP = 260, 44


def _rvol(px, i, w=20, fl=0.10):
    s = sum(math.log(px[k] / px[k - 1]) ** 2 for k in range(i - w + 1, i + 1))
    return max(math.sqrt(s / w * 252), fl)


def _slope(ma, i, w=21):
    xm = (w - 1) / 2
    ys = ma[i - w + 1:i + 1]
    ym = sum(ys) / w
    n = sum((k - xm) * (ys[k] - ym) for k in range(w))
    dn = sum((k - xm) ** 2 for k in range(w))
    return (n / dn) * 252 / ma[i]


def _debounce(Td):
    cur, out = True, []
    for t in Td:
        if cur and t >= CFG["tCut"]:
            cur = False
        elif (not cur) and t < CFG["tRestore"]:
            cur = True
        out.append(cur)
    return out


def _target(pos, lev_ok, px, ma, vol, slope):
    if pos == 0:
        return 0.0
    if not lev_ok:
        return 1.0
    if px > ma and slope > 0.03:
        return max(1.0, min(CFG["tv"] / vol, CFG["cap"]))
    return 1.0


def _banded(ts, b):
    cur, out = 0.0, []
    for t in ts:
        if abs(t - cur) > b:
            cur = t
        out.append(cur)
    return out


def _alloc(e):
    if e == 0:
        return "100% SGOV"
    if e <= 1:
        return "100% QQQ"
    w = (e - 1) / 2
    return f"{(1 - w) * 100:.2f}% QQQ / {w * 100:.2f}% TQQQ"


def model_columns(tq, dv, gate, emergency, asof):
    """Columns L..U for the session AFTER `asof`, as the dashboard shows them at that close."""
    L = max(k for k, d in enumerate(tq["dates"]) if d <= asof) + 1
    a = max(0, L - WINDOW)
    dts, px, ma = tq["dates"][a:L], tq["px"][a:L], tq["ma"][a:L]
    pos = [int(x) for x in tq["pos"][a:L]]
    tdmap = dict(zip(dv["dates"], dv["Td"]))
    last, Td = 0.5, []
    for d in dts:
        if tdmap.get(d) is not None:
            last = tdmap[d]
        Td.append(last)
    lev_ok = _debounce(Td)
    n = len(dts)
    targets = [0.0 if i < WARMUP else
               _target(pos[i], lev_ok[i], px[i], ma[i], _rvol(px, i, 20, CFG["floor"]), _slope(ma, i, 21))
               for i in range(n)]
    held = _banded(targets, CFG["band"])
    i = n - 1
    vol, slope = _rvol(px, i, 20, CFG["floor"]), _slope(ma, i, 21)
    if pos[i] == 1:                                         # nextMonitorState()
        next_pos = 0 if (gate.get("armed") and px[i] < ma[i]) else 1
    else:
        run = 0
        for k in range(i, -1, -1):
            if px[k] >= ma[k]:
                run += 1
            else:
                break
        next_pos = 1 if run >= CFG["reentry"] else 0
    raw_next = _target(next_pos, lev_ok[i], px[i], ma[i], vol, slope)
    last_held, prior_held = held[i], (held[i - 1] if i > 0 else held[i])
    cur_held = raw_next if abs(raw_next - last_held) > CFG["band"] else last_held
    changed = abs(cur_held - prior_held) > 1e-9
    wants = min(CFG["tv"] / vol, CFG["cap"])
    above = px[-1] > ma[-1]
    run = 1
    for k in range(n - 2, -1, -1):
        if (px[k] > ma[k]) == above:
            run += 1
        else:
            break
    vk = max(k for k, d in enumerate(dv["dates"]) if d <= asof)
    em_on = bool(emergency) and emergency.get("state") == "active"
    return {
        C_L: f"{cur_held:.2f}x",
        C_M: (f"REBALANCE — {_alloc(prior_held)} → {_alloc(cur_held)}" if changed else HOLD_TEXT),
        C_N: f"{wants:.2f}x",
        C_O: round(dv["Vd"][vk], 2), C_P: round(dv["Td"][vk], 2),
        C_Q: round(gate["V"], 2), C_R: round(gate["T"], 2),
        C_S: run, C_T: "above" if above else "below",
        C_U: "Yes" if em_on else "No",
    }


# ----------------------------------------------------------------------------
# Workbook helpers
# ----------------------------------------------------------------------------
def last_formula_row(ws):
    r = ws.max_row
    while r > 3 and not str(ws.cell(r, 2).value or "").startswith("="):
        r -= 1
    return r


def append_session_row(ws, d):
    """Add a row for session date d by copying the last formula row (formulas translated)."""
    src = last_formula_row(ws)
    dst = src + 1
    for c in range(1, ws.max_column + 1):
        s, t = ws.cell(src, c), ws.cell(dst, c)
        t._style = copy(s._style)
        if isinstance(s.value, str) and s.value.startswith("="):
            t.value = Translator(s.value, origin=s.coordinate).translate_formula(t.coordinate)
    ws.cell(dst, C_DATE).value = datetime(d.year, d.month, d.day)
    return dst


def sheet_rows(ws):
    rows = {}
    for r in range(3, ws.max_row + 1):
        v = ws.cell(r, C_DATE).value
        if isinstance(v, datetime):
            rows[v.date().isoformat()] = r
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(now=None):
    now = now or datetime.now(ET)
    if os.environ.get("GITHUB_EVENT_NAME") == "schedule" and now.hour < RUN_HOUR_ET:
        print(f"{now:%H:%M} ET is before {RUN_HOUR_ET}:00 (the standard-time cron slot); skipping.")
        return 0
    if not KEY or not SECRET:
        sys.exit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (the v3.3 PAPER keys).")

    today = now.date().isoformat()
    cal = get(PAPER, "/v2/calendar", start=HISTORY_FROM, end=(now.date() + timedelta(days=21)).isoformat())
    sessions_all = [c["date"] for c in cal]

    def closed(c):
        if c["date"] < today:
            return True
        if c["date"] > today:
            return False
        close_at = datetime.fromisoformat(f"{c['date']}T{c['close']}").replace(tzinfo=ET)
        return now >= close_at + timedelta(minutes=20)

    done = [c["date"] for c in cal if closed(c)]
    if not done:
        sys.exit("No completed session found.")
    asof = done[-1]
    nxt = next((d for d in sessions_all if d > asof), None)
    log(f"v3.3 ledger update | as-of close {asof} | next session {nxt} | run {now:%Y-%m-%d %H:%M} ET")

    orders = pull_orders()
    cash_acts = pull_cash_activities()
    end = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bars = pull_daily_bars(end)
    sessions = [d for d in done if d >= FUNDED]
    missing_bars = [d for d in sessions if not all(s in bars.get(d, {}) for s in ("QQQ", "TQQQ"))]
    if missing_bars:
        sys.exit(f"Closing prices not available yet for {missing_bars}; run again later.")
    state = rebuild_state(sessions, orders, cash_acts, bars)

    wb = openpyxl.load_workbook(LEDGER)
    ws, tl = wb["Sheet1"], wb["Trade Log"]
    rows = sheet_rows(ws)
    changed_any = False

    # ---- 1. actuals for every completed session row still blank -------------------
    filled = []
    for d in [x for x in sessions if x >= RECORD_START]:
        if d not in rows:
            rows[d] = append_session_row(ws, date.fromisoformat(d))
        r = rows[d]
        if ws.cell(r, C_QPX).value not in (None, ""):
            continue
        s = state[d]
        ws.cell(r, C_QSH).value = round(s["sh"].get("QQQ", 0.0), 9)
        ws.cell(r, C_QPX).value = s["px"]["QQQ"]
        ws.cell(r, C_TSH).value = round(s["sh"].get("TQQQ", 0.0), 9)
        ws.cell(r, C_TPX).value = s["px"]["TQQQ"]
        if abs(s["sh"].get("SGOV", 0.0)) > 1e-9:
            ws.cell(r, C_SGSH).value = round(s["sh"]["SGOV"], 9)
            ws.cell(r, C_SGPX).value = s["px"].get("SGOV")
        ws.cell(r, C_CASH).value = s["cash"]
        if s["traded"] and ws.cell(r, C_TRADED).value in (None, ""):
            ws.cell(r, C_TRADED).value = "Y"
        filled.append(d)
    if filled:
        changed_any = True
        s = state[asof]
        log(f"Actuals filled for: {', '.join(filled)}")
        log(f"  {asof}: QQQ {s['sh'].get('QQQ', 0):.6f} @ {s['px']['QQQ']} | "
            f"TQQQ {s['sh'].get('TQQQ', 0):.6f} @ {s['px']['TQQQ']} | "
            f"cash {s['cash']:,.2f} | account value {s['eq']:,.2f}")
    else:
        log("Actuals: nothing new to fill.")

    # ---- 2. next session's model columns ------------------------------------------
    data = json.load(open(DATA))
    tq, gate = data["timeline_qqq"], data.get("acting_gate") or {}
    dv = json.load(open(VT))
    em = json.load(open(EMRG)) if os.path.exists(EMRG) else None
    feed_date, vt_date = tq["dates"][-1], dv["dates"][-1]
    if nxt and feed_date == asof and vt_date == asof:
        if nxt not in rows:
            rows[nxt] = append_session_row(ws, date.fromisoformat(nxt))
        r = rows[nxt]
        if ws.cell(r, C_L).value in (None, ""):
            cols = model_columns(tq, dv, gate, em, asof)
            for c, v in cols.items():
                if ws.cell(r, c).value in (None, ""):
                    ws.cell(r, c).value = v
            changed_any = True
            log(f"Model columns for {nxt}: target {cols[C_L]} | {cols[C_M]} | wants {cols[C_N]} | "
                f"daily V/T {cols[C_O]}/{cols[C_P]} | gate V/T {cols[C_Q]}/{cols[C_R]} | "
                f"{cols[C_S]} days {cols[C_T]} the 200-day | emergency {cols[C_U]}")
        else:
            log(f"Model columns for {nxt}: already filled, left as is.")
    else:
        log(f"Model columns NOT written: feeds are dated {feed_date} / {vt_date}, expected {asof}. "
            f"They fill on the next run after the pipeline catches up.")

    # ---- 3. Trade Log --------------------------------------------------------------
    if tl.cell(3, 6).value != TL_REF_HEADER:
        tl.cell(3, 6).value = TL_REF_HEADER
        tl.cell(2, 1).value = ("Slippage auto-computes in bps vs the ARRIVAL price (NBBO midpoint when the order "
                               "was sent; the official open for orders sent before the open), signed so POSITIVE = "
                               "adverse. It measures execution only, not the day's market move.")
        changed_any = True
    tl_rows, last_tl = {}, 3
    for r in range(4, tl.max_row + 1):
        a = tl.cell(r, 1).value
        if isinstance(a, datetime):
            last_tl = r
            key = (a.date().isoformat(), tl.cell(r, 2).value, str(tl.cell(r, 3).value).lower(),
                   round(float(tl.cell(r, 4).value or 0), 6))
            tl_rows[key] = r
    added = 0
    for o in orders:
        d = to_et(o["filled_at"]).date().isoformat()
        key = (d, o["symbol"], o["side"], round(float(o["filled_qty"]), 6))
        r = tl_rows.get(key)
        if r is None:
            last_tl += 1
            r = last_tl
            for c in range(1, 10):
                tl.cell(r, c)._style = copy(tl.cell(5, c)._style)
            tl.cell(r, 1).value = datetime.fromisoformat(d)
            tl.cell(r, 2).value, tl.cell(r, 3).value = o["symbol"], o["side"]
            tl.cell(r, 4).value = float(o["filled_qty"])
            tl.cell(r, 5).value = f'=IF(OR(D{r}="",G{r}=""),"",D{r}*G{r})'
            tl.cell(r, 7).value = float(o["filled_avg_price"])
            tl.cell(r, 8).value = (f'=IF(OR(F{r}="",G{r}="",C{r}=""),"",(G{r}-F{r})/F{r}*10000'
                                   f'*IF(LOWER(C{r})="buy",1,-1))')
            tl_rows[key] = r
            added += 1
        if d >= RECORD_START:
            ap = arrival_price(o, bars)
            if ap is not None and tl.cell(r, 6).value != ap:
                tl.cell(r, 6).value = ap
                changed_any = True
    if added:
        changed_any = True
        log(f"Trade Log: added {added} order(s).")

    # ---- 4. reconciliation ---------------------------------------------------------
    acct = get(PAPER, "/v2/account")
    pos = {p["symbol"]: float(p["qty"]) for p in get(PAPER, "/v2/positions")}
    s = state[asof]
    problems = []
    for sym in sorted(set(pos) | {k for k, v in s["sh"].items() if abs(v) > 1e-9}):
        if abs(pos.get(sym, 0.0) - s["sh"].get(sym, 0.0)) > 1e-6:
            problems.append(f"{sym} shares: Alpaca {pos.get(sym, 0.0):.9f} vs ledger {s['sh'].get(sym, 0.0):.9f}")
    if abs(float(acct["cash"]) - s["cash"]) > 0.011:
        problems.append(f"cash: Alpaca {float(acct['cash']):,.2f} vs ledger {s['cash']:,.2f}")
    le = float(acct["last_equity"])
    prev = [d for d in sessions if d < asof]
    candidates = [s["eq"]] + ([state[prev[-1]]["eq"]] if prev else [])
    if not any(abs(le - c) <= 0.02 for c in candidates):
        problems.append(f"last_equity {le:,.2f} matches neither reconstructed close "
                        f"({', '.join(f'{c:,.2f}' for c in candidates)})")
    if problems:
        log("RECONCILIATION FAILED:")
        for p in problems:
            log(f"  - {p}")
        with open(ALERT, "w") as f:
            f.write("\n".join(problems) + "\n")
    else:
        log(f"Reconciliation OK: shares and cash match Alpaca; last_equity {le:,.2f} matches a reconstructed close.")

    if changed_any:
        wb.calculation = CalcProperties(fullCalcOnLoad=True)   # Excel recalculates on open
        wb.save(LEDGER)
        log(f"Saved {LEDGER}.")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("```\n" + "\n".join(_summary) + "\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
