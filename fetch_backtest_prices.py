#!/usr/bin/env python3
"""Fetch daily OHLC for the backtest and (optionally) 3:50 PM intraday marks.

Daily OHLC (the important part) gives real next-open execution across the whole
history. ^NDX (the Nasdaq-100 index) is used for the deep history because it
reaches back to 1985 and covers Black Monday; QQQ (the ETF, 1999+) is fetched
too as a cross-check for the modern era.

Optional: if ALPACA_KEY / ALPACA_SECRET are set, it also pulls QQQ 1-minute bars
for a few crash windows and extracts the 15:50 ET price per day, so the same-day
(3:50) execution can be validated on the days that actually matter.

Run it in GitHub Actions (see fetch-prices.yml); it writes CSVs to the repo root.
"""
import sys, os
import pandas as pd

def fetch_daily():
    import yfinance as yf
    for sym, out in [("^NDX", "ndx_daily.csv"), ("QQQ", "qqq_daily.csv")]:
        d = yf.download(sym, start="1985-01-01", auto_adjust=False, progress=False)
        if d is None or len(d) == 0:
            print(f"WARNING: no data for {sym}"); continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in d.columns]
        d = d[cols].dropna(how="all")
        d.index.name = "Date"
        d.to_csv(out)
        print(f"{sym}: {len(d)} rows  {d.index[0].date()}..{d.index[-1].date()}  -> {out}")

def fetch_intraday_350():
    key, sec = os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")
    if not (key and sec):
        print("ALPACA_KEY/SECRET not set — skipping intraday 3:50 marks (daily OHLC is the important file)")
        return
    import requests
    windows = [("2020-02-14","2020-04-15"),("2018-01-15","2018-03-01"),
               ("2022-01-01","2022-07-01"),("2011-07-15","2011-09-15")]
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    rows = []
    for a, b in windows:
        try:
            url = "https://data.alpaca.markets/v2/stocks/QQQ/bars"
            params = {"timeframe":"1Min","start":f"{a}T13:00:00Z","end":f"{b}T21:00:00Z",
                      "limit":10000,"adjustment":"split","feed":"iex"}
            got, page = 0, None
            while True:
                if page: params["page_token"] = page
                r = requests.get(url, headers=hdr, params=params, timeout=30).json()
                if r.get("message"):
                    print(f"  {a}..{b}: Alpaca says '{r['message']}' -- skipping"); break
                for bar in (r.get("bars") or []):
                    t = pd.Timestamp(bar["t"]).tz_convert("America/New_York")
                    if t.hour == 15 and t.minute == 50:
                        rows.append((t.date(), bar["c"])); got += 1
                page = r.get("next_page_token")
                if not page: break
            print(f"  {a}..{b}: {got} 3:50 marks")
        except Exception as e:
            print(f"  {a}..{b}: skipped ({e})")
    if rows:
        df = pd.DataFrame(rows, columns=["Date","Price350"]).drop_duplicates("Date")
        df.to_csv("qqq_intraday_350.csv", index=False)
        print(f"3:50 marks: {len(df)} days -> qqq_intraday_350.csv")

if __name__ == "__main__":
    fetch_daily()                       # the important file(s) -- must succeed
    try:
        fetch_intraday_350()            # optional spot-check -- never allowed to fail the job
    except Exception as e:
        print(f"intraday 3:50 fetch skipped: {e}")
