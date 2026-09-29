#!/usr/bin/env python3
"""
AUTO-SUBMIT -- unattended version of submit_order.py, for the v3.2 CANARY only.

    python auto_submit_order.py            (DRY_RUN=1 to compute and print, submitting nothing)

WHY A SEPARATE FILE
  v3.3 still runs stage -> human review -> submit_order.py. Nothing here changes
  alpaca_exec.py, stage_order.py or submit_order.py; this script only imports them.
  It hard-refuses every track except those in AUTO_TRACKS, so it can't be pointed at
  v3.3 by a copy-pasted workflow.

HOW IT DIFFERS FROM submit_order.py (and why)
  * The staged pending file is the AUTHORIZATION, not the order ticket. Orders are
    RECOMPUTED at submit time from live equity and positions. The staged notionals were
    priced at last night's close; by noon they no longer land on target.
  * The model target is RE-VALIDATED: current_target() must reproduce the staged
    target. If data.json was revised between staging and submission (the 2026-08
    missing-session bug flipped a target 1.05x -> 1.31x), this aborts instead of trading.
  * If live exposure is already within the band of target -- e.g. you traded by hand --
    it archives the proposal as skipped and does nothing.
  * Orders go one at a time, waiting for each fill, so a buy never races an unfilled sell.
  * An idempotency marker is written BEFORE the first order. If a run dies mid-way, the
    next run refuses to trade until you reconcile by hand: a partial trade must never be
    re-sent automatically.
  * Every fill is logged with TWO separate measures (see the 09-2026 slippage discussion):
      slippage_bps -- fill vs the NBBO-ish mid at the moment of submission (execution cost)
      timing_bps   -- fill vs the prior session's close (the backtest's reference price)
    Both are signed so POSITIVE = adverse. They answer different questions; don't sum them.

EXIT CODES (read by the workflow)
   0  nothing to do / kill switch / market closed / skipped / dry run   -> green
  10  trade executed                                                    -> green, or red-as-FYI
   1  aborted -- needs a human                                          -> red
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import alpaca_exec as X

# ---- configuration ----------------------------------------------------------
AUTO_TRACKS = {"v32"}              # hard allowlist -- this script refuses every other track
KILL_SWITCH = Path("AUTOTRADE_OFF")  # create this file in the repo (web UI) to stop auto-trading
DRY_RUN = os.environ.get("DRY_RUN", "0").strip().lower() in ("1", "true", "yes")
MIN_MINUTES_TO_CLOSE = 15          # refuse if the session is about to end (early-close days)
TARGET_TOL = 0.02                  # re-validated target must match the staged one within this
FILL_TIMEOUT_S = 45                # market orders in QQQ/TQQQ fill in well under a second
DATA_URL = "https://data.alpaca.markets"

LOG = Path(f"autotrade_log_{X.TRACK}.csv")
SUBMITTING = X.PENDING.with_suffix(".submitting.json")   # idempotency marker
SUBMITTED = X.PENDING.with_suffix(".submitted.json")     # same name submit_order.py uses
SKIPPED = X.PENDING.with_suffix(".skipped.json")

RC_NOTHING, RC_EXECUTED, RC_ABORT = 0, 10, 1
TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day"}

LOG_FIELDS = ["run_utc", "data_date", "target_exposure", "symbol", "side", "action",
              "notional_requested", "order_id", "status", "filled_qty", "filled_avg_price",
              "filled_notional", "arrival_bid", "arrival_ask", "arrival_mid", "slippage_bps",
              "prior_close", "timing_bps", "minutes_to_close"]


# ---- client: adds read-only endpoints; order submission is inherited unchanged ----
class AutoClient(X.AlpacaClient):
    def get_order(self, oid: str):
        return self._req("GET", f"/v2/orders/{oid}")

    def calendar(self, start: str, end: str):
        return self._req("GET", f"/v2/calendar?start={start}&end={end}")

    def _data(self, path: str):
        r = urllib.request.Request(DATA_URL + path, method="GET")
        r.add_header("APCA-API-KEY-ID", self.key)
        r.add_header("APCA-API-SECRET-KEY", self.sec)
        with urllib.request.urlopen(r, timeout=20) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}

    def latest_quote(self, sym: str):
        """(bid, ask) from the free IEX feed. IEX is one venue, so treat the mid as an
        approximation of the consolidated NBBO -- fine for liquid ETFs, not gospel."""
        q = self._data(f"/v2/stocks/{sym}/quotes/latest?feed=iex").get("quote") or {}
        return float(q.get("bp") or 0.0), float(q.get("ap") or 0.0)

    def prior_close(self, sym: str, today: str):
        start = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")
        bars = self._data(f"/v2/stocks/{sym}/bars?timeframe=1Day&start={start}"
                          f"&limit=20&feed=iex").get("bars") or []
        prev = [b for b in bars if str(b.get("t", ""))[:10] < today]
        return float(prev[-1]["c"]) if prev else None


# ---- helpers ----------------------------------------------------------------
def _parse_ts(s: str) -> datetime:
    """Alpaca can return nanosecond fractions, which fromisoformat rejects."""
    return datetime.fromisoformat(re.sub(r"\.\d+", "", s).replace("Z", "+00:00"))


def prev_session(client, today: str):
    """Most recent completed trading session before `today`, from Alpaca's own
    calendar -- so holidays and early closes are handled, not guessed."""
    start = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")
    days = [c["date"] for c in client.calendar(start, today) if c["date"] < today]
    return days[-1] if days else None


def _bps(fill, ref, side):
    if not fill or not ref:
        return None
    sign = 1.0 if side == "buy" else -1.0          # positive = adverse either way
    return round(sign * (fill - ref) / ref * 1e4, 2)


def _safe(fn, *a, default=None):
    """Measurement must never block a trade."""
    try:
        return fn(*a)
    except Exception as e:                            # noqa: BLE001
        print(f"    (measurement skipped: {type(e).__name__}: {str(e)[:80]})")
        return default


def wait_fill(client, oid: str):
    deadline, o = time.time() + FILL_TIMEOUT_S, {}
    while time.time() < deadline:
        o = client.get_order(oid)
        if o.get("status") in TERMINAL:
            return o
        time.sleep(1)
    return o


def append_log(row: dict):
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def _archive(dest: Path, extra: dict):
    rec = json.loads(X.PENDING.read_text())
    rec.update(extra)
    dest.write_text(json.dumps(rec, indent=2))
    X.PENDING.unlink()


# ---- main -------------------------------------------------------------------
def main(client=None, target_fn=None) -> int:
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    print(f"auto-submit  track={X.TRACK}  vehicle={X.VEHICLE}  dry_run={DRY_RUN}  {now_utc}")

    # 0. hard gates that need no network
    if X.TRACK not in AUTO_TRACKS:
        print(f"REFUSE: track {X.TRACK!r} is not approved for auto-submit {sorted(AUTO_TRACKS)}.")
        return RC_ABORT
    if KILL_SWITCH.exists():
        print(f"Kill switch present ({KILL_SWITCH}). Auto-trading is OFF. Nothing sent.")
        return RC_NOTHING
    if SUBMITTING.exists():
        print(f"ABORT: {SUBMITTING.name} exists -- a previous run started submitting and did not "
              f"finish. Orders may be PARTIALLY executed. Reconcile the account by hand, then "
              f"delete {SUBMITTING.name}. Refusing to trade until then.")
        return RC_ABORT
    if not X.PENDING.exists():
        print("No pending proposal. Nothing to do.")
        return RC_NOTHING

    pending = json.loads(X.PENDING.read_text())
    if pending.get("vehicle") != X.VEHICLE:
        print(f"ABORT: proposal vehicle {pending.get('vehicle')!r} != configured {X.VEHICLE!r}.")
        return RC_ABORT

    client = client or AutoClient()

    # 1. market must be open, with room before the close
    clock = client.clock()
    if not clock.get("is_open", False):
        print("Market closed (holiday?). Leaving the proposal for the next session.")
        return RC_NOTHING
    now, close = _parse_ts(clock["timestamp"]), _parse_ts(clock["next_close"])
    mins_left = (close - now).total_seconds() / 60
    if mins_left < MIN_MINUTES_TO_CLOSE:
        print(f"ABORT: only {mins_left:.0f} min to the close (< {MIN_MINUTES_TO_CLOSE}). "
              f"Not submitting into the bell.")
        return RC_ABORT
    today = clock["timestamp"][:10]

    # 2. the proposal must be built on the most recent completed session
    ps = prev_session(client, today)
    if pending["data_date"] != ps:
        print(f"ABORT: proposal is on data {pending['data_date']}, but the last completed "
              f"session was {ps}. Stale -- let tonight's stage job re-propose.")
        return RC_ABORT

    # 3. re-validate: the model must still want what was staged
    data_date, e, weights = (target_fn or X.current_target)()
    if data_date != pending["data_date"]:
        print(f"ABORT: model data is now {data_date}, staged on {pending['data_date']}.")
        return RC_ABORT
    if abs(e - float(pending["target_exposure"])) > TARGET_TOL:
        print(f"ABORT: target moved since staging ({pending['target_exposure']:.3f}x -> {e:.3f}x). "
              f"The feed was likely revised. Not trading on a changed target.")
        return RC_ABORT

    # 4. live account: is a trade still needed?
    equity = float(client.account()["equity"])
    positions = client.positions_by_value()
    e_live = X.net_exposure(positions, equity)
    print(f"target {e:.3f}x   live {e_live:.3f}x   equity ${equity:,.0f}   {mins_left:.0f} min to close")
    if abs(e_live - e) <= X.REBALANCE_BAND:
        print(f"Live exposure already within {X.REBALANCE_BAND:.2f} of target (traded by hand?). "
              f"Archiving the proposal as skipped.")
        if not DRY_RUN:
            _archive(SKIPPED, {"skipped_at_utc": now_utc, "live_exposure": round(e_live, 4)})
        return RC_NOTHING

    # 5. recompute orders at today's prices, then the existing circuit breaker
    orders = X.compute_orders(weights, equity, positions)
    if not orders:
        print("No orders above the dust threshold. Archiving as skipped.")
        if not DRY_RUN:
            _archive(SKIPPED, {"skipped_at_utc": now_utc, "live_exposure": round(e_live, 4)})
        return RC_NOTHING
    problems = X.safety_check(orders, equity, weights, positions)
    if problems:
        for p in problems:
            print("  BLOCKED:", p)
        print("ABORT: safety check failed on the recomputed orders.")
        return RC_ABORT

    for o in orders:
        print(f"  {o['action'].upper():6} {o['symbol']:5} ${o['notional']:>12,.2f}")
    if DRY_RUN:
        print("DRY RUN -- nothing submitted, proposal left in place.")
        return RC_NOTHING

    # 6. idempotency marker FIRST, then one order at a time
    X.PENDING.rename(SUBMITTING)
    fills = []
    for o in orders:
        sym, side = o["symbol"], o["side"]
        bid, ask = _safe(client.latest_quote, sym, default=(0.0, 0.0))
        mid = (bid + ask) / 2 if bid and ask else None
        prior = _safe(client.prior_close, sym, today)

        resp = client.close_position(sym) if o["action"] == "close" else \
            client.submit(sym, side, o["notional"])
        oid = resp.get("id") if isinstance(resp, dict) else None
        fin = wait_fill(client, oid) if oid else {}
        status = fin.get("status", "unknown")
        qty = float(fin.get("filled_qty") or 0.0)
        px = float(fin.get("filled_avg_price") or 0.0)

        row = {"run_utc": now_utc, "data_date": data_date, "target_exposure": round(e, 4),
               "symbol": sym, "side": side, "action": o["action"],
               "notional_requested": o["notional"], "order_id": oid, "status": status,
               "filled_qty": qty, "filled_avg_price": px, "filled_notional": round(qty * px, 2),
               "arrival_bid": bid, "arrival_ask": ask, "arrival_mid": mid,
               "slippage_bps": _bps(px, mid, side), "prior_close": prior,
               "timing_bps": _bps(px, prior, side), "minutes_to_close": round(mins_left)}
        append_log(row)
        fills.append(row)
        print(f"  -> {sym:5} {status:9} {qty:.4f} @ {px:.4f}   "
              f"slippage {row['slippage_bps']} bps   timing {row['timing_bps']} bps")

        if status != "filled":
            print(f"ABORT: {sym} order ended {status!r}. Remaining orders NOT sent. "
                  f"{SUBMITTING.name} left in place -- reconcile by hand.")
            return RC_ABORT

    # 7. verify where we actually landed, then archive
    eq2 = float(client.account()["equity"])
    pos2 = client.positions_by_value()
    e_after = X.net_exposure(pos2, eq2)
    rec = json.loads(SUBMITTING.read_text())
    rec.update({"submitted_at_utc": now_utc, "recomputed_orders": orders, "fills": fills,
                "exposure_after": round(e_after, 4)})
    SUBMITTED.write_text(json.dumps(rec, indent=2))
    SUBMITTING.unlink()
    print(f"DONE. Exposure after: {e_after:.3f}x (target {e:.3f}x). Archived to {SUBMITTED.name}.")
    if abs(e_after - e) > X.REBALANCE_BAND / 2:
        print(f"  WARNING: landed {abs(e_after - e):.3f}x from target -- check fills.")
    return RC_EXECUTED


if __name__ == "__main__":
    sys.exit(main())
