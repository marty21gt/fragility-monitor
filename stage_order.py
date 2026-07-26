#!/usr/bin/env python3
"""
STAGE step -- computes the rebalance the model wants and writes it for your
review. Submits NOTHING. Run after the daily signal; it only pings you when a
trade is actually needed (most days: nothing).

    python stage_order.py

Writes pending_order.json when a trade is needed; prints a plain-English
proposal. Then you review, and run submit_order.py to execute.
"""
import json, sys
from datetime import datetime, timezone
import alpaca_exec as X


def main():
    data_date, e, weights = X.current_target()

    if not X.is_fresh(data_date):
        print(f"STALE: model data is {data_date} "
              f"({X.business_days_old(data_date)} business days old). Not staging.")
        sys.exit(1)

    client = X.AlpacaClient()
    equity = float(client.account()["equity"])
    positions = client.positions_by_value()

    orders = X.compute_orders(weights, equity, positions)
    _warn = X.safety_check(orders, equity, weights, positions)
    if _warn:
        print("WARNING at stage (review carefully):")
        for _w in _warn: print("  -", _w)
    if not orders:
        print(f"No action. Current holdings already match the "
              f"{e:.2f}x target within ${X.MIN_TRADE_USD:.0f}.")
        if X.PENDING.exists():
            X.PENDING.unlink()   # clear any stale proposal
        return

    pending = {
        "staged_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "data_date": data_date,
        "vehicle": X.VEHICLE,
        "target_exposure": round(e, 4),
        "target_weights": {k: round(v, 4) for k, v in weights.items()},
        "equity_at_stage": round(equity, 2),
        "positions_at_stage": {k: round(v, 2) for k, v in positions.items()},
        "orders": orders,
    }
    X.PENDING.write_text(json.dumps(pending, indent=2))

    print(f"PROPOSED REBALANCE   (model data {data_date}, target {e:.2f}x, vehicle {X.VEHICLE})")
    print(f"  account equity: ${equity:,.0f}")
    print(f"  target weights: " +
          ", ".join(f"{k} {v:.0%}" for k, v in weights.items() if v > 0))
    for o in orders:
        verb = {"close": "SELL ALL", "buy": "BUY", "sell": "SELL"}[o["action"]]
        print(f"    {verb:9} {o['symbol']:5} ${o['notional']:,.0f}")
    print(f"\nWrote {X.PENDING.name}. Review it, then run submit_order.py to execute.")


if __name__ == "__main__":
    main()
