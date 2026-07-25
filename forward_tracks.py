#!/usr/bin/env python3
"""Single source of truth for the two forward tracks.

v3.0 and v3.1 share one signal engine and emit the IDENTICAL target net
exposure every day. They differ only in how that exposure is mapped to funds:

    v3.0  ->  QQQ / QLD  / BIL   (2x sleeve)
    v3.1  ->  QQQ / TQQQ / BIL   (3x sleeve, less of it)

Because both mappings are computed here from the same exposure `e`, the two
tracks can never disagree on the decision. Any difference in their realized
returns is therefore pure implementation P&L -- fees minus decay/tracking --
with the market direction cancelled out by construction.

Note a useful property: for e <= 1 (risk-off or de-levered) both mappings are
identical (QQQ + cash), so the divergence between tracks is exactly zero on
those days. Divergence only accrues while leverage is on.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

MAX_EXPOSURE = 2.0
TRACKS = ("v3.0", "v3.1")


def _check_exposure(e: float) -> None:
    if not math.isfinite(e) or e < 0.0 or e > MAX_EXPOSURE + 1e-9:
        raise ValueError(f"exposure {e} outside [0, {MAX_EXPOSURE}]")


def v30_weights(e: float) -> dict[str, float]:
    """QQQ/QLD/BIL weights delivering net exposure e."""
    _check_exposure(e)
    if e >= 1.0:
        return {"QQQ": 2.0 - e, "QLD": e - 1.0, "BIL": 0.0}
    return {"QQQ": e, "QLD": 0.0, "BIL": 1.0 - e}


def v31_weights(e: float) -> dict[str, float]:
    """QQQ/TQQQ/BIL weights delivering the same net exposure e.

    At exposure e >= 1: TQQQ = (e-1)/2, QQQ = (3-e)/2. This holds HALF the
    leveraged notional of the QLD construction because the sleeve is 3x, not
    2x, so the single-day P&L is identical at matched e.
    """
    _check_exposure(e)
    if e >= 1.0:
        return {"QQQ": (3.0 - e) / 2.0, "TQQQ": (e - 1.0) / 2.0, "BIL": 0.0}
    return {"QQQ": e, "TQQQ": 0.0, "BIL": 1.0 - e}


def weights_for(track: str, e: float) -> dict[str, float]:
    if track == "v3.0":
        return v30_weights(e)
    if track == "v3.1":
        return v31_weights(e)
    raise ValueError(f"unknown track {track!r}; expected one of {TRACKS}")


def implied_exposure(weights: dict[str, float]) -> float:
    """Recover net exposure from a weight vector (the ledger's own check)."""
    mult = {"QLD": 2.0, "TQQQ": 3.0}
    lev = next((t for t in mult if t in weights), None)
    lever = mult.get(lev, 0.0) * weights.get(lev, 0.0) if lev else 0.0
    return weights.get("QQQ", 0.0) + lever


def track_return(weights: dict[str, float], fund_returns: dict[str, float]) -> float:
    """Published-convention daily return: sum of target weight * fund return."""
    return sum(weights[t] * float(fund_returns[t]) for t in weights)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_signal_payload(
    track: str,
    market_date: str,
    target_exposure: float,
    state: dict[str, Any],
    data_as_of: dict[str, str],
    intended_execution_session: str,
    input_hashes: dict[str, str],
    source_hash: str,
    status: str = "tradeable",
    generated_at_utc: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build one signal payload matching the ledger schema.

    Call this TWICE from the same post-close run -- once per track, with the
    same target_exposure/state/data_as_of -- so the two payloads are guaranteed
    to share the decision. The weights are the only field that differs.
    """
    weights = weights_for(track, target_exposure)
    # Cross-check: the mapping must reproduce the shared exposure exactly.
    if not math.isclose(implied_exposure(weights), target_exposure, abs_tol=1e-8):
        raise AssertionError(
            f"{track} mapping desynced from shared exposure {target_exposure}"
        )
    return {
        "market_date": market_date,
        "generated_at_utc": generated_at_utc or _utc_now(),
        "model_version": track,
        "data_as_of": dict(data_as_of),
        "status": status,
        "state": dict(state),
        "target_exposure": target_exposure,
        "weights": weights,
        "intended_execution_session": intended_execution_session,
        "input_hashes": dict(input_hashes),
        "source_hash": source_hash,
        "notes": notes,
    }


def build_outcome_payload(
    linked_record_hash: str,
    market_date: str,
    weights: dict[str, float],
    fund_returns: dict[str, float],
    benchmark_qqq_return: float,
    execution: str = "paper_close",
    slippage: float = 0.0,
    trading_cost: float = 0.0,
    exception: str | None = None,
    observed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the outcome record linked to a prior signal.

    Stores the fund returns and the realized track return so the divergence
    analyzer can difference the two tracks without re-deriving anything.
    """
    realized = track_return(weights, fund_returns) - trading_cost
    return {
        "market_date": market_date,
        "observed_at_utc": observed_at_utc or _utc_now(),
        "linked_record_hash": linked_record_hash,
        "details": {
            "execution": execution,
            "weights": dict(weights),
            "fund_returns": dict(fund_returns),
            "realized_return": realized,
            "benchmark_qqq_return": float(benchmark_qqq_return),
            "slippage": slippage,
            "trading_cost": trading_cost,
            "exception": exception,
        },
    }
