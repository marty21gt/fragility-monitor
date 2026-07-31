#!/usr/bin/env python3
"""
ALFRED REVISION DIAGNOSTIC  (run this BEFORE any V-rebuild)
==========================================================
Your V already percentile-ranks every component on an expanding, point-in-time
window, so there is no look-ahead in the *ranking*. The only revision leak left is
that the input LEVELS are today's revised values rather than what was published at
the time. This script measures how large that leak actually is for the three
ALFRED-addressable V inputs -- so you know whether a full vintage V-rebuild is worth
the complexity before committing the pipeline to it.

For each series it pulls two versions from the same FRED/ALFRED endpoint:
    latest = current revised values     (output_type=1, default real-time period)
    first  = value as first published   (output_type=4, "Initial Release Only")
and reports the revision |latest - first|: its size, its size specifically over
2010-2026, and its size relative to the series' own dispersion -- a proxy for
whether the expanding-window percentile (how V consumes these) would actually shift.

NOT covered: Shiller CAPE. It comes from Shiller's ie_data.xls (+ multpl), not FRED,
so ALFRED has no vintage for it. If the three below prove material, CAPE needs a
separate Shiller-specific point-in-time treatment.

REQUIRES: FRED_API_KEY in the environment, and network access to api.stlouisfed.org
(your GitHub Actions runner or a local machine -- not a restricted sandbox).
Pure standard library + pandas; no extra dependencies.

    FRED_API_KEY=xxxx python alfred_revision_diagnostic.py
"""
import os, sys, json, time
import urllib.request, urllib.parse
import pandas as pd

FRED_KEY = os.environ.get("FRED_API_KEY", "").strip()
if not FRED_KEY:
    print("ERROR: FRED_API_KEY not set in the environment."); sys.exit(1)

def fred(series, output_type=1, retries=3):
    """FRED/ALFRED observations. output_type=1 -> latest revised (default real-time
    period); output_type=4 -> initial release only (value as first published)."""
    q = urllib.parse.urlencode({"series_id": series, "api_key": FRED_KEY,
                                "file_type": "json", "output_type": output_type})
    url = f"https://api.stlouisfed.org/fred/series/observations?{q}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                obs = json.load(r)["observations"]
            break
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(3)
    idx, val = [], []
    for o in obs:
        v = o.get("value", ".")
        if v in (".", "", None):
            continue
        idx.append(pd.Timestamp(o["date"])); val.append(float(v))
    return pd.Series(val, index=idx).sort_index()

SERIES = {
    "BOGZ1FL663067003Q": "Margin-loan leverage (Z.1)",
    "QUSPAM770A":        "BIS credit-to-GDP (lev spine)",
    "BOGZ1FL153064486Q": "Household equity alloc (Z.1)",
}

hdr = (f"{'series':<20}{'label':<32}{'n':>5}{'vint from':>10}"
       f"{'|rev| med':>11}{'|rev| max':>11}{'rev/sigma':>10}{'2010+ med':>11}")
print(hdr); print("-" * len(hdr))

for sid, label in SERIES.items():
    try:
        latest = fred(sid, output_type=1)
        first  = fred(sid, output_type=4)          # initial release only
    except Exception as e:
        print(f"{sid:<20}{label:<32}  FETCH FAILED: {e}"); continue
    both = pd.concat({"latest": latest, "first": first}, axis=1).dropna()
    if both.empty:
        print(f"{sid:<20}{label:<32}  no overlapping vintages (series not archived pre-now)")
        continue
    rev   = (both["latest"] - both["first"]).abs()
    sigma = latest.std()
    sub   = rev[both.index >= "2010-01-01"]
    print(f"{sid:<20}{label:<32}{len(both):>5}{both.index.min().strftime('%Y-%m'):>10}"
          f"{rev.median():>11.4g}{rev.max():>11.4g}"
          f"{(rev.median()/sigma if sigma else float('nan')):>10.3f}"
          f"{(sub.median() if len(sub) else float('nan')):>11.4g}")

print("\nHow to read it:")
print("  rev/sigma = median revision as a fraction of the series' own std deviation.")
print("  < ~0.05  -> the expanding-window percentile V uses barely moves; skip the rebuild.")
print("  > ~0.15  -> ranks likely shift some exit dates; the vintage V-rebuild is justified.")
print("  in between -> borderline; next step is to recompute the actual percentile rank")
print("               both ways for that series and compare, rather than eyeball rev/sigma.")
print("  (CAPE is excluded -- not on ALFRED; treat separately only if the above matter.)")
