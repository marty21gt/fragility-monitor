#!/usr/bin/env python3
"""
backfill_v33.py -- one-time, READ-ONLY pull of the v3.3 paper account's history.

Rebuilds the record back to the original $100,000 funding from Alpaca's own data.
Writes to backfill/:
  account.json             account creation date, current equity and cash
  portfolio_history.csv    Alpaca's daily end-of-day equity (covers the unrecorded days)
  orders.csv               every filled order: qty, average fill price, timestamps
  fills.csv                every individual execution (partial fills included)
  cash_activities.csv      non-trade activity: deposits, fees, dividends, journals
  positions.csv            current share counts, to reconcile the sheet
  bars.csv                 daily OHLC for QQQ/TQQQ from the consolidated (SIP) feed
  summary.txt              the highlights, also printed to the run log

Safety: GET requests only -- this script cannot place, change, or cancel orders.
Account calls are locked to the PAPER endpoint.
"""
import csv
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

PAPER = "https://paper-api.alpaca.markets"   # PAPER endpoint only
DATA = "https://data.alpaca.markets"
ET = ZoneInfo("America/New_York")
SYMBOLS = ["QQQ", "TQQQ"]
OUT = "backfill"

KEY = os.environ.get("APCA_API_KEY_ID")
SECRET = os.environ.get("APCA_API_SECRET_KEY")
START = (os.environ.get("START_DATE") or "").strip() or "2026-07-20"

_summary = []


def log(line=""):
    print(line)
    _summary.append(line)


def get(base, path, **params):
    """GET only. Returns parsed JSON; on failure prints Alpaca's error and stops."""
    params = {k: v for k, v in params.items() if v is not None}
    url = base + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, method="GET", headers={
        "APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"Alpaca returned {e.code} for {path}: {e.read().decode()[:500]}")


def to_et(ts):
    """ISO-8601 string or epoch seconds -> 'YYYY-MM-DD HH:MM:SS' New York time."""
    if ts in (None, ""):
        return ""
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    else:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt.astimezone(ET).strftime("%Y-%m-%d %H:%M:%S")


def to_utc(ts):
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return ts or ""


def write_csv(name, rows, fields):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {name}: {len(rows)} rows")


def main():
    if not KEY or not SECRET:
        sys.exit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (the v3.3 PAPER keys).")
    os.makedirs(OUT, exist_ok=True)
    start_iso = f"{START}T00:00:00Z"
    log(f"v3.3 paper account history from {START} (read-only pull, "
        f"{datetime.now(ET):%Y-%m-%d %H:%M} ET)")

    # 1. Account snapshot
    acct = get(PAPER, "/v2/account")
    keep = ["status", "currency", "created_at", "cash", "equity", "last_equity",
            "portfolio_value"]
    with open(os.path.join(OUT, "account.json"), "w") as f:
        json.dump({k: acct.get(k) for k in keep}, f, indent=2)
    log(f"Account created {to_et(acct.get('created_at'))} ET | "
        f"equity ${float(acct['equity']):,.2f} | cash ${float(acct['cash']):,.2f}")

    # 2. Daily end-of-day equity, including the days the sheet never recorded
    ph = get(PAPER, "/v2/account/portfolio/history", period="6M", timeframe="1D")
    with open(os.path.join(OUT, "portfolio_history_raw.json"), "w") as f:
        json.dump(ph, f, indent=2)
    hist = []
    for i, ts in enumerate(ph.get("timestamp") or []):
        hist.append({"date_et": to_et(ts)[:10], "timestamp_utc": to_utc(ts), "epoch": ts,
                     "equity": ph["equity"][i], "profit_loss": ph["profit_loss"][i],
                     "profit_loss_pct": ph["profit_loss_pct"][i]})
    write_csv("portfolio_history.csv", hist,
              ["date_et", "timestamp_utc", "epoch", "equity", "profit_loss", "profit_loss_pct"])
    funded = [h for h in hist if h["equity"] not in (None, 0, 0.0)]
    if funded:
        log("First days with equity:")
        for h in funded[:6]:
            log(f"  {h['date_et']}  ${h['equity']:,.2f}")

    # 3. Filled orders (one row per order -> Trade Log rows)
    orders, after = [], start_iso
    while True:
        page = get(PAPER, "/v2/orders", status="closed", after=after,
                   direction="asc", limit=500)
        orders += page
        if len(page) < 500:
            break
        after = page[-1]["submitted_at"]
    filled = []
    for o in orders:
        if float(o.get("filled_qty") or 0) <= 0:
            continue
        qty, px = float(o["filled_qty"]), float(o["filled_avg_price"])
        filled.append({"submitted_et": to_et(o.get("submitted_at")),
                       "filled_et": to_et(o.get("filled_at")),
                       "symbol": o["symbol"], "side": o["side"], "type": o.get("type"),
                       "time_in_force": o.get("time_in_force"),
                       "notional_requested": o.get("notional"), "qty_requested": o.get("qty"),
                       "filled_qty": o["filled_qty"], "filled_avg_price": o["filled_avg_price"],
                       "filled_value": round(qty * px, 2), "status": o["status"],
                       "order_id": o["id"]})
    write_csv("orders.csv", filled, list(filled[0].keys()) if filled else ["order_id"])
    if filled:
        log("First filled orders:")
        for o in filled[:4]:
            log(f"  {o['filled_et']}  {o['side']:<4} {o['symbol']:<5} "
                f"{float(o['filled_qty']):.9f} @ {float(o['filled_avg_price']):.6f}"
                f"  = ${o['filled_value']:,.2f}")

    # 4. Account activities: executions, plus any non-trade cash movement
    acts, token = [], None
    while True:
        page = get(PAPER, "/v2/account/activities", after=start_iso,
                   direction="asc", page_size=100, page_token=token)
        acts += page
        if len(page) < 100:
            break
        token = page[-1]["id"]
    fills = [{"time_et": to_et(a.get("transaction_time")), "symbol": a.get("symbol"),
              "side": a.get("side"), "qty": a.get("qty"), "price": a.get("price"),
              "fill_type": a.get("type"), "cum_qty": a.get("cum_qty"),
              "leaves_qty": a.get("leaves_qty"), "order_id": a.get("order_id")}
             for a in acts if a.get("activity_type") == "FILL"]
    write_csv("fills.csv", fills, ["time_et", "symbol", "side", "qty", "price", "fill_type",
                                   "cum_qty", "leaves_qty", "order_id"])
    cash = [{"date": a.get("date") or to_et(a.get("transaction_time"))[:10],
             "activity_type": a.get("activity_type"), "net_amount": a.get("net_amount"),
             "symbol": a.get("symbol"), "qty": a.get("qty"),
             "per_share_amount": a.get("per_share_amount"),
             "description": a.get("description"), "status": a.get("status")}
            for a in acts if a.get("activity_type") != "FILL"]
    write_csv("cash_activities.csv", cash, ["date", "activity_type", "net_amount", "symbol",
                                            "qty", "per_share_amount", "description", "status"])
    log(f"Non-trade cash activities since {START}: {len(cash)}")
    for c in cash:
        log(f"  {c['date']}  {c['activity_type']}  {c['net_amount']}  {c['description'] or ''}")

    # 5. Current positions
    pos = get(PAPER, "/v2/positions")
    write_csv("positions.csv", pos, ["symbol", "qty", "avg_entry_price", "cost_basis",
                                     "market_value", "current_price"])
    for p in pos:
        log(f"Position now: {p['symbol']:<5} {float(p['qty']):.9f} shares")

    # 6. Daily bars (SIP = consolidated tape; free plan allows it once 15+ min old)
    end = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bars, token = [], None
    while True:
        page = get(DATA, "/v2/stocks/bars", symbols=",".join(SYMBOLS), timeframe="1Day",
                   start=start_iso, end=end, feed="sip", adjustment="raw",
                   limit=10000, page_token=token)
        for sym, lst in (page.get("bars") or {}).items():
            for b in lst:
                bars.append({"date_et": to_et(b["t"])[:10], "symbol": sym, "open": b["o"],
                             "high": b["h"], "low": b["l"], "close": b["c"],
                             "volume": b["v"], "vwap": b.get("vw")})
        token = page.get("next_page_token")
        if not token:
            break
    bars.sort(key=lambda r: (r["date_et"], r["symbol"]))
    write_csv("bars.csv", bars, ["date_et", "symbol", "open", "high", "low", "close",
                                 "volume", "vwap"])

    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("\n".join(_summary) + "\n")
    print("Done. Download the 'v33-backfill' artifact from this run.")


if __name__ == "__main__":
    main()
