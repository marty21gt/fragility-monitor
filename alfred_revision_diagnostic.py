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

METHOD (why not output_type=4): FRED's "initial release only" (output_type=4) 400s
unless paired with an explicit real-time window. So instead we pull EVERY vintage of
each series with the full real-time span (realtime_start=1776-07-04,
realtime_end=9999-12-31) -- the standard fredapi approach -- and reconstruct, per
observation date:
    first  = value at the earliest realtime_start  (as first published)
    latest = value at the newest  realtime_start    (today's revised number)
then report the revision |latest - first|: its size, its size over 2010-2026, and its
size relative to the series' own dispersion (a proxy for whether the expanding-window
percentile V uses would actually shift). 'vints' shows the archival depth -- if a
series has only ~1 vintage it simply isn't archived, and first==latest.

NOT covered: Shiller CAPE. It comes from Shiller's ie_data.xls (+ multpl), not FRED,
so ALFRED has no vintage for it; handle separately only if the three below matter.

REQUIRES: FRED_API_KEY in the environment, and network access to api.stlouisfed.org
(your GitHub Actions runner or a local machine -- not a restricted sandbox).
Standard library + pandas; no other dependencies.

    FRED_API_KEY=xxxx python alfred_revision_diagnostic.py
"""
import os, sys, json, time
import urllib.request, urllib.parse, urllib.error
import pandas as pd

FRED_KEY = os.environ.get("FRED_API_KEY", "").strip()
if not FRED_KEY:
    print("ERROR: FRED_API_KEY not set in the environment."); sys.exit(1)

def fred_vintages(series, obs_start="1990-01-01", retries=3):
    """All ALFRED vintages for `series` since obs_start. Returns rows of
    (observation_date, value, realtime_start)."""
    q = urllib.parse.urlencode({
        "series_id": series, "api_key": FRED_KEY, "file_type": "json",
        "realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
        "observation_start": obs_start, "limit": "100000",
    })
    url = f"https://api.stlouisfed.org/fred/series/observations?{q}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                obs = json.load(r)["observations"]
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            if attempt == retries - 1:
                raise RuntimeError(f"HTTP {e.code}: {body}")
            time.sleep(3)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3)
    rows = []
    for o in obs:
        v = o.get("value", ".")
        if v in (".", "", None):
            continue
        rows.append((pd.Timestamp(o["date"]), float(v),
                     pd.Timestamp(o["realtime_start"])))
    return pd.DataFrame(rows, columns=["date", "value", "rt_start"])

SERIES = {
    "BOGZ1FL663067003Q": "Margin-loan leverage (Z.1)",
    "QUSPAM770A":        "BIS credit-to-GDP (lev spine)",
    "BOGZ1FL153064486Q": "Household equity alloc (Z.1)",
}

hdr = (f"{'series':<20}{'label':<32}{'obs':>5}{'vints':>7}"
       f"{'|rev| med':>11}{'|rev| max':>11}{'rev/sigma':>10}{'2010+ med':>11}")
print(hdr); print("-" * len(hdr))

for sid, label in SERIES.items():
    try:
        v = fred_vintages(sid)
    except Exception as e:
        print(f"{sid:<20}{label:<32}  FETCH FAILED: {e}"); continue
    if v.empty:
        print(f"{sid:<20}{label:<32}  no observations returned"); continue
    n_vint = v["rt_start"].nunique()
    g = v.sort_values("rt_start").groupby("date")["value"]
    both = pd.concat({"first": g.first(), "latest": g.last()}, axis=1).dropna()
    rev = (both["latest"] - both["first"]).abs()
    sigma = both["latest"].std()
    sub = rev[both.index >= "2010-01-01"]
    print(f"{sid:<20}{label:<32}{len(both):>5}{n_vint:>7}"
          f"{rev.median():>11.4g}{rev.max():>11.4g}"
          f"{(rev.median()/sigma if sigma else float('nan')):>10.3f}"
          f"{(sub.median() if len(sub) else float('nan')):>11.4g}")

print("\nHow to read it:")
print("  vints     = distinct archival vintages found. ~1 means the series isn't")
print("              revised/archived, so first==latest and there's nothing to vintage.")
print("  rev/sigma = median revision as a fraction of the series' own std deviation.")
print("  < ~0.05  -> the expanding-window percentile V uses barely moves; skip the rebuild.")
print("  > ~0.15  -> ranks likely shift some exit dates; the vintage V-rebuild is justified.")
print("  in between -> borderline; next step is to recompute the actual percentile rank")
print("               both ways for that series and compare, rather than eyeball rev/sigma.")
print("  (CAPE is excluded -- not on ALFRED; treat separately only if the above matter.)")
