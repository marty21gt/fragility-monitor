#!/usr/bin/env python3
"""
APPROVE + SUBMIT step -- re-checks every safety condition, then sends the staged
orders to your Alpaca PAPER account. Run this only after reviewing the proposal
stage_order.py wrote.

    python submit_order.py

Aborts (submits nothing) if: no proposal exists, the proposal is stale, the
model data is stale, the market is closed, your positions changed since staging,
or any order trips the circuit breaker.
"""
import json, sys
from datetime import datetime, timezone
import alpaca_exec as X

PROPOSAL_MAX_AGE_H = 18   # refuse to submit a proposal older than this


def main():
    if not X.PENDING.exists():
        print("No pending_order.json. Nothing to submit.")
        return
    pending = json.loads(X.PENDING.read_text())

    # 1. proposal freshness -- don't fire a stale plan
    staged = datetime.fromisoformat(pending["staged_at_utc"].replace("Z", "+00:00"))
    age_h = (datetime.now(timezone.utc) - staged).total_seconds() / 3600
    if age_h > PROPOSAL_MAX_AGE_H:
        sys.exit(f"ABORT: proposal is {age_h:.0f}h old (> {PROPOSAL_MAX_AGE_H}h). Re-stage first.")

    # 2. model-data freshness -- re-checked NOW, not just when staged
    if not X.is_fresh(pending["data_date"]):
        sys.exit(f"ABORT: model data {pending['data_date']} is "
                 f"{X.business_days_old(pending['data_date'])} business days old.")

    client = X.AlpacaClient()

    # 3. market must be open (DAY orders). Near the close approximates MOC.
    if not client.clock().get("is_open", False):
        sys.exit("ABORT: market is closed. Run while open (ideally near the close).")

    # 4. position sanity -- holdings must still match what we staged against
    equity = float(client.account()["equity"])
    now_pos = client.positions_by_value()
    staged_pos = pending["positions_at_stage"]
    for sym in set(now_pos) | set(staged_pos):
        a, b = now_pos.get(sym, 0.0), staged_pos.get(sym, 0.0)
        if abs(a - b) > max(50.0, 0.05 * max(a, b, 1.0)):
            sys.exit(f"ABORT: {sym} position changed since staging "
                     f"(${b:,.0f} -> ${a:,.0f}). Re-stage.")

    # 5. circuit breaker
    problems = X.safety_check(pending["orders"], equity, pending["target_weights"], now_pos)
    if problems:
        for p in problems:
            print("BLOCKED:", p)
        sys.exit("ABORT: safety check failed.")

    # 6. submit -- sells/closes first (already ordered by compute_orders)
    print(f"Submitting {len(pending['orders'])} order(s) to PAPER account...")
    for o in pending["orders"]:
        if o["action"] == "close":
            resp = client.close_position(o["symbol"])
        else:
            resp = client.submit(o["symbol"], o["side"], o["notional"])
        oid = resp.get("id", "?") if isinstance(resp, dict) else "?"
        print(f"  {o['side'].upper():4} {o['symbol']:5} ${o['notional']:,.0f}  -> order {oid}")

    # 7. archive so it can't be double-submitted
    done = X.PENDING.with_suffix(".submitted.json")
    X.PENDING.rename(done)
    print(f"\nDone. Archived proposal to {done.name}.")


if __name__ == "__main__":
    main()
