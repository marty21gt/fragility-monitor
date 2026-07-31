#!/usr/bin/env python3
"""
Shared helpers for the semi-automated Alpaca rebalancer (PAPER endpoint).

Design: the pure-logic functions (target from the model, order math, safety
checks) are kept separate from network I/O (AlpacaClient) so the logic can be
tested offline with no keys and no live calls. stage_order.py and
submit_order.py both import this.

Nothing here submits an order on import. AlpacaClient is only constructed when a
script actually needs the account, and it reads keys from the environment:
    APCA_API_KEY_ID / APCA_API_SECRET_KEY   (use your PAPER keys)
"""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path
import urllib.request, urllib.error
import numpy as np

# ---- configuration ---------------------------------------------------------
VEHICLE          = os.environ.get("VEHICLE", "QLD")   # "QLD" = v3.0; "TQQQ" = v3.1 (set per workflow)
CASH_TICKER      = "SGOV"     # risk-off / de-levered sleeve (per the v3.1 ETF spec)
if VEHICLE not in ("QLD", "TQQQ"):
    raise SystemExit(f"VEHICLE must be QLD or TQQQ, got {VEHICLE!r}")
WEIGHT_TOLERANCE = 0.03       # circuit breaker: post-trade weights must match target within this
MIN_TRADE_USD    = 25.0       # skip trades smaller than this (avoid churn / dust)
MAX_DATA_AGE_BD  = 4          # freshness: refuse if model data older than this (business days)
BASE_URL         = "https://paper-api.alpaca.markets"   # PAPER endpoint (not live)
TRACK            = os.environ.get("TRACK", VEHICLE)   # names the pending file; defaults to VEHICLE
PENDING          = Path(f"pending_order_{TRACK}.json")   # separate file per track


# ---- target from the model -------------------------------------------------
def exposure_to_weights(e: float) -> dict[str, float]:
    """Map net exposure e -> fund weights for the configured vehicle.
    v3.0 (QLD, 2x):  e>=1 -> QQQ=2-e, QLD=e-1 ;  e<1 -> QQQ=e, BIL=1-e
    v3.1 (TQQQ, 3x): e>=1 -> QQQ=(3-e)/2, TQQQ=(e-1)/2 ; e<1 as above
    """
    mult = 3.0 if VEHICLE == "TQQQ" else 2.0
    if e >= 1.0:
        lev = (e - 1.0) / (mult - 1.0)
        return {"QQQ": 1.0 - lev, VEHICLE: lev, CASH_TICKER: 0.0}
    return {"QQQ": e, VEHICLE: 0.0, CASH_TICKER: 1.0 - e}


def current_target():
    """Return (data_date, exposure, weights) the model wants for the NEXT session.
    Reuses the locked engine so the strategy logic has one source of truth."""
    import reference_backtest as R
    # Per-track sizing overrides. Default to v3.0's locked values; the v3.1
    # workflow sets these hotter (e.g. TARGET_VOL=0.25, LEVERAGE_CAP=3.0).
    # Python resolves these globals at call time, so reassigning them here
    # changes what target_exposure()/realized_vol() use -- engine stays untouched.
    R.TARGET_VOL   = float(os.environ.get("TARGET_VOL",   R.TARGET_VOL))
    R.VOL_FLOOR    = float(os.environ.get("VOL_FLOOR",    R.VOL_FLOOR))
    R.LEVERAGE_CAP = float(os.environ.get("LEVERAGE_CAP", R.LEVERAGE_CAP))
    df = R.load("data.json", "daily_vt.json")
    sig = R.monitor_position(df)
    e = float(R.target_exposure(df, sig).iloc[-1])   # composite target at the latest close
    data_date = str(df.index[-1].date())
    return data_date, e, exposure_to_weights(e)


def business_days_old(data_date: str) -> int:
    last = datetime.strptime(data_date, "%Y-%m-%d").date()
    today = datetime.now(timezone.utc).date()
    return int(np.busday_count(last, today))


def is_fresh(data_date: str) -> bool:
    return business_days_old(data_date) <= MAX_DATA_AGE_BD


# ---- order math ------------------------------------------------------------
def compute_orders(weights: dict[str, float], equity: float,
                   positions: dict[str, float]) -> list[dict]:
    """positions: {symbol: market_value_usd}. Returns orders to reach target.
    Sells/closes are ordered first so cash is freed before buys."""
    orders = []
    for sym in sorted(set(weights) | set(positions)):
        tgt = weights.get(sym, 0.0) * equity
        cur = positions.get(sym, 0.0)
        delta = tgt - cur
        if abs(delta) < MIN_TRADE_USD:
            continue
        if weights.get(sym, 0.0) == 0.0 and cur > 0:
            orders.append({"symbol": sym, "side": "sell", "action": "close",
                           "notional": round(cur, 2)})
        elif delta > 0:
            orders.append({"symbol": sym, "side": "buy", "action": "buy",
                           "notional": round(delta, 2)})
        else:
            orders.append({"symbol": sym, "side": "sell", "action": "sell",
                           "notional": round(-delta, 2)})
    orders.sort(key=lambda o: 0 if o["side"] == "sell" else 1)
    return orders


def safety_check(orders: list[dict], equity: float,
                 weights: dict[str, float], positions: dict[str, float]) -> list[str]:
    """Circuit breaker. Returns a list of problems; empty list means OK.

    The guard is *consistency*, not size: it simulates the portfolio AFTER the
    orders and confirms it matches the target weights. This ALLOWS a legitimate
    full flip to cash (which correctly reaches 100% BIL) while catching a bug
    whose orders would land somewhere other than the stated target. A second
    absurdity cap flags any single order larger than the whole account, which
    can't happen for a real target (weights sum to 1)."""
    problems = []
    post = dict(positions)
    for o in orders:
        d = o["notional"] if o["side"] == "buy" else -o["notional"]
        post[o["symbol"]] = post.get(o["symbol"], 0.0) + d
    total = sum(v for v in post.values() if v > 0) or equity
    for sym in sorted(set(weights) | set(post)):
        target_w = weights.get(sym, 0.0)
        actual_w = max(post.get(sym, 0.0), 0.0) / total
        if abs(target_w - actual_w) > WEIGHT_TOLERANCE:
            problems.append(
                f"{sym}: orders land at {actual_w:.0%} but target is {target_w:.0%} "
                f"(> {WEIGHT_TOLERANCE:.0%} off) -- orders do not match target")
    for o in orders:
        if o["notional"] > 1.05 * equity:
            problems.append(
                f"{o['symbol']} order ${o['notional']:,.0f} exceeds account equity "
                f"(${equity:,.0f}) -- impossible for a valid target")
    return problems


# ---- thin Alpaca REST client (stdlib only) ---------------------------------
class AlpacaClient:
    def __init__(self, base: str = BASE_URL):
        self.base = base
        self.key = os.environ.get("APCA_API_KEY_ID")
        self.sec = os.environ.get("APCA_API_SECRET_KEY")
        if not self.key or not self.sec:
            raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (paper keys).")

    def _req(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.base + path, data=data, method=method)
        r.add_header("APCA-API-KEY-ID", self.key)
        r.add_header("APCA-API-SECRET-KEY", self.sec)
        r.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raise SystemExit(f"Alpaca API error {e.code}: {e.read().decode()[:300]}")

    def account(self):    return self._req("GET", "/v2/account")
    def positions(self):  return self._req("GET", "/v2/positions")
    def clock(self):      return self._req("GET", "/v2/clock")

    def submit(self, symbol: str, side: str, notional: float):
        return self._req("POST", "/v2/orders", {
            "symbol": symbol, "notional": str(round(notional, 2)),
            "side": side, "type": "market", "time_in_force": "day"})

    def close_position(self, symbol: str):
        return self._req("DELETE", f"/v2/positions/{symbol}")

    def positions_by_value(self) -> dict[str, float]:
        return {p["symbol"]: float(p["market_value"]) for p in self.positions()}
