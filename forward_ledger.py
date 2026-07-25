#!/usr/bin/env python3
"""Append and verify a small SHA-256-chained prospective signal ledger.

Patched from the original v3.0-only version to accept either implementation
vehicle: QQQ/QLD/BIL (2x sleeve) or QQQ/TQQQ/BIL (3x sleeve). The only change
is inside validate_signal(): the fund set and the weights-implied exposure
check are now derived from whichever leveraged fund is present, instead of
being hardcoded to QLD at 2x. Everything else -- the hash chain, sequence
enforcement, duplicate detection, and signal/outcome linkage -- is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GENESIS_HASH = "0" * 64
RECORD_TYPES = ("signal", "outcome", "correction")

# The base fund is always QQQ and cash is always BIL. Exactly one leveraged
# sleeve is held, and its multiple fixes the weights-implied exposure check.
LEVERAGED_FUNDS = {"QLD": 2.0, "TQQQ": 3.0}


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def calculate_hash(record_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(record_without_hash)).hexdigest()


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on ledger line {line_number}: {error}"
                ) from error
    return records


def verify_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    previous_hash = GENESIS_HASH
    seen_hashes: set[str] = set()
    seen_signals: set[tuple[str, str]] = set()
    for sequence, record in enumerate(records, start=1):
        if record.get("sequence") != sequence:
            raise ValueError(f"Invalid sequence at record {sequence}")
        if record.get("previous_hash") != previous_hash:
            raise ValueError(f"Broken previous hash at record {sequence}")
        record_hash = record.get("record_hash")
        if not isinstance(record_hash, str) or len(record_hash) != 64:
            raise ValueError(f"Invalid record hash at record {sequence}")
        body = dict(record)
        body.pop("record_hash", None)
        if calculate_hash(body) != record_hash:
            raise ValueError(f"Hash mismatch at record {sequence}")
        if record_hash in seen_hashes:
            raise ValueError(f"Duplicate record hash at record {sequence}")
        seen_hashes.add(record_hash)

        payload = record.get("payload", {})
        record_type = record.get("record_type")
        if record_type == "signal":
            key = (
                str(payload.get("model_version")),
                str(payload.get("market_date")),
            )
            if key in seen_signals:
                raise ValueError(
                    "Duplicate signal for model/date; append a correction "
                    f"instead: {key}"
                )
            seen_signals.add(key)
        elif record_type in {"outcome", "correction"}:
            linked = payload.get("linked_record_hash")
            if linked not in seen_hashes:
                raise ValueError(
                    f"Record {sequence} links to an unknown earlier hash"
                )
        previous_hash = record_hash
    return {
        "valid": True,
        "records": len(records),
        "head_hash": previous_hash,
    }


def validate_iso_timestamp(value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")


def validate_signal(payload: dict[str, Any]) -> None:
    required = (
        "market_date",
        "generated_at_utc",
        "model_version",
        "data_as_of",
        "status",
        "state",
        "target_exposure",
        "weights",
        "input_hashes",
        "source_hash",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Signal payload missing fields: {missing}")
    validate_iso_timestamp(payload["generated_at_utc"], "generated_at_utc")
    datetime.fromisoformat(str(payload["market_date"]))

    weights = payload["weights"]
    present = [t for t in LEVERAGED_FUNDS if t in weights]
    if len(present) != 1:
        raise ValueError(
            "weights must contain exactly one leveraged fund "
            f"({' or '.join(LEVERAGED_FUNDS)})"
        )
    lev = present[0]
    multiple = LEVERAGED_FUNDS[lev]
    if set(weights) != {"QQQ", lev, "BIL"}:
        raise ValueError(f"weights must contain exactly QQQ, {lev}, and BIL")

    values = {ticker: float(weights[ticker]) for ticker in ("QQQ", lev, "BIL")}
    if any(not math.isfinite(v) or v < 0.0 for v in values.values()):
        raise ValueError("weights must be finite and non-negative")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-8):
        raise ValueError("weights must sum to 1")

    exposure = float(payload["target_exposure"])
    implied = values["QQQ"] + multiple * values[lev]
    if not math.isclose(exposure, implied, abs_tol=1e-8):
        raise ValueError(
            f"target exposure {exposure} != weights-implied {implied} "
            f"(base QQQ + {multiple}x {lev})"
        )


def validate_linked(payload: dict[str, Any], record_type: str) -> None:
    required = (
        "market_date",
        "observed_at_utc",
        "linked_record_hash",
        "details",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"{record_type} payload missing fields: {missing}")
    validate_iso_timestamp(payload["observed_at_utc"], "observed_at_utc")
    linked = payload["linked_record_hash"]
    if not isinstance(linked, str) or len(linked) != 64:
        raise ValueError("linked_record_hash must be a SHA-256 hex digest")


def append_record(
    ledger: Path,
    record_type: str,
    payload_path: Path,
) -> dict[str, Any]:
    records = read_records(ledger)
    state = verify_records(records)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")
    if record_type == "signal":
        validate_signal(payload)
    else:
        validate_linked(payload, record_type)

    body = {
        "sequence": len(records) + 1,
        "record_type": record_type,
        "recorded_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "previous_hash": state["head_hash"],
        "payload": payload,
    }
    record = {**body, "record_hash": calculate_hash(body)}
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(canonical_bytes(record).decode("utf-8") + "\n")
        stream.flush()
    verify_records(read_records(ledger))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    append_parser = subparsers.add_parser("append")
    append_parser.add_argument("--ledger", type=Path, required=True)
    append_parser.add_argument(
        "--record-type", choices=RECORD_TYPES, required=True
    )
    append_parser.add_argument("--payload", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "append":
        record = append_record(args.ledger, args.record_type, args.payload)
        print(
            json.dumps(
                {
                    "appended": True,
                    "sequence": record["sequence"],
                    "record_hash": record["record_hash"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(verify_records(read_records(args.ledger)), indent=2))


if __name__ == "__main__":
    main()
