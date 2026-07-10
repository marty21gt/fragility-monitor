#!/usr/bin/env python3
# =====================================================================
#  Market Fragility Monitor  --  data builder
#  Runs on a schedule (GitHub Actions). Fetches daily + monthly data,
#  computes the credit-momentum model, and writes data.json for the page.
#  Needs one secret: FRED_API_KEY (set in the GitHub repo, never in code).
# =====================================================================
import os, io, json, sys, datetime as dt
import numpy as np, pandas as pd, requests

FRED_KEY = os.environ.get("FRED_API_KEY", "").strip()
if not FRED_KEY:
    print("ERROR: FRED_API_KEY not set."); sys.exit(1)

def log(m): print(m, flush=True)

# ---------- fetch helpers ----------
def fred(series, retries=3):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={FRED_KEY}&file_type=json"
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=40); r.raise_for_status()
            obs = r.json()["observations"]
            s = pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".","") else np.nan) for o in obs})
            return s.dropna()
        except Exception as e:
            log(f"  retry {series}: {e}")
    log(f"  WARN: could not fetch {series}"); return pd.Series(dtype=float)

def _clean_series(s):
    """Force any price download into a clean, tz-naive, unique-index float Series."""
    s = pd.Series(pd.to_numeric(pd.Series(s).values.ravel(), errors="coerce"),
                  index=pd.DatetimeIndex(pd.Series(s).index))
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()

def daily_sp():
    # primary: Stooq daily S&P 500 close (no key)
    try:
        hdr = {"User-Agent": "Mozilla/5.0"}
        r = requests.get("https://stooq.com/q/d/l/?s=%5Espx&i=d", headers=hdr, timeout=40); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "Close" in df.columns and len(df) > 500:
            df["Date"] = pd.to_datetime(df["Date"])
            log(f"  daily S&P from Stooq: {len(df)} rows")
            return _clean_series(df.set_index("Date")["Close"])
    except Exception as e:
        log(f"  Stooq failed: {e}")
    # fallback: yfinance (handle both Series and DataFrame return shapes)
    try:
        import yfinance as yf
        raw = yf.download("^GSPC", start="1990-01-01", progress=False, auto_adjust=False)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        log(f"  daily S&P from Yahoo: {len(close)} rows")
        return _clean_series(close)
    except Exception as e:
        log(f"  Yahoo failed: {e}"); return pd.Series(dtype=float)

# ---------- model helpers ----------
def epct(s, m=120):
    s = s.astype(float); out = pd.Series(index=s.index, dtype=float); h=[]
    for t,v in s.items():
        if not np.isnan(v) and len(h)>=m: out[t]=(np.asarray(h)<=v).mean()
        if not np.isnan(v): h.append(v)
    return out
def blend(c):
    v=[x for x in c if x is not None and not np.isnan(x)]
    if not v: return np.nan
    srt=sorted(v,reverse=True); tail=np.mean(srt[:2]) if len(srt)>=2 else srt[0]
    return .70*np.mean(v)+.30*tail

log("Fetching data...")
sp_daily = daily_sp()
sh = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv")
sh["Date"]=pd.to_datetime(sh["Date"]); sh=sh.set_index("Date")
jst = pd.read_excel("https://github.com/bank-of-england/MachineLearningCrisisPrediction/raw/master/data/JSTdatasetR3.xlsx", sheet_name="Data")
usj = jst[jst["country"]=="USA"].set_index("year")
baa, aaa = fred("BAA"), fred("AAA")
bogz = fred("BOGZ1FL663067003Q")     # broker margin loans (live leverage)
nfci = fred("NFCI")                   # financial conditions

# ---------- monthly panel + credit-momentum model (the timeline) ----------
d = pd.DataFrame(index=sh.index)
d["px"]=sh["SP500"]; d["div"]=sh["Dividend"].replace(0,np.nan); d["cape"]=sh["PE10"].replace(0,np.nan); d["yr"]=d.index.year
cg = usj["tloans"]/usj["gdp"]; d["cg5"]=d["yr"].map(cg-cg.shift(5)); d["stir"]=d["yr"].map(usj["stir"])
for y,v in {2017:.93,2018:1.94,2019:2.11,2020:.37,2021:.04,2022:2.02,2023:5.14,2024:4.98,2025:4.30,2026:4.20}.items():
    d.loc[d.yr==y,"stir"]=v
d["stir"]=d["stir"].ffill()
spread=(baa-aaa); spread.index=spread.index.to_period("M"); d["spread"]=d.index.to_period("M").map(spread)
d["rvol"]=d["px"].pct_change().rolling(12).std()*np.sqrt(12); d["ma10"]=d["px"].rolling(10).mean()
d["tr"]=(d["px"]+d["div"]/12)/d["px"].shift(1)-1
d=d.dropna(subset=["px"])
capef=epct(d["cape"]); volsup=1-epct(d["rvol"]); levf=epct(d["cg5"]); comp=1-epct(d["spread"])
sm=epct(d["spread"]-d["spread"].shift(3))
volup=d["rvol"]/d["rvol"].rolling(12).min()-1; mom3=d["px"].pct_change(3)
d["V"]=pd.DataFrame({"a":capef,"b":volsup,"c":levf,"d":comp}).apply(lambda r:blend(list(r.values)),axis=1)
d["T"]=pd.DataFrame({"a":epct(volup),"b":epct(-mom3),"c":sm}).apply(lambda r:blend(list(r.values)),axis=1)
d=d.dropna(subset=["V","T","tr","ma10"])
pos=[]; state="in"; Ts,Vs,px,ma=d["T"].values,d["V"].values,d["px"].values,d["ma10"].values
for i in range(len(d)):
    pos.append(1 if state=="in" else 0)
    if state=="in":
        if Vs[i]>=.62 and Ts[i]>=.55 and px[i]<ma[i]: state="out"
    else:
        if i>=1 and Ts[i]<.55 and Ts[i-1]<.55 and px[i]>ma[i]: state="in"
d["pos"]=pos; d["sr"]=np.where(np.array(pos)==1, d["tr"], d["stir"]/100/12)
timeline={"dates":[t.strftime("%Y-%m") for t in d.index],
 "V":[round(float(x),3) for x in d["V"]],"T":[round(float(x),3) for x in d["T"]],
 "px":[round(float(x),1) for x in d["px"]],"ma":[round(float(x),1) for x in d["ma10"]],
 "pos":[int(x) for x in d["pos"]],"bhret":[round(float(x),5) for x in d["tr"]],"stret":[round(float(x),5) for x in d["sr"]]}
monthly_pos = pd.Series(d["pos"].values, index=pd.DatetimeIndex(d.index))

# ---------- daily price series + 200-day MA + mapped position (recent window) ----------
priceSeries={"dates":[],"px":[],"ma":[],"pos":[]}
if len(sp_daily) > 250:
    ma200 = sp_daily.rolling(200).mean()
    cutoff = sp_daily.index[-1] - pd.DateOffset(months=18)
    for dte in sp_daily.index[sp_daily.index >= cutoff]:
        pxv = float(sp_daily.loc[dte]); mav = ma200.loc[dte]
        mp = monthly_pos[monthly_pos.index <= dte]
        priceSeries["dates"].append(dte.strftime("%Y-%m-%d"))
        priceSeries["px"].append(round(pxv, 1))
        priceSeries["ma"].append(round(float(mav), 1) if pd.notna(mav) else None)
        priceSeries["pos"].append(int(mp.iloc[-1]) if len(mp) else 1)
else:
    tailN = 72
    priceSeries = {"dates":timeline["dates"][-tailN:], "px":timeline["px"][-tailN:],
                   "ma":timeline["ma"][-tailN:], "pos":timeline["pos"][-tailN:]}

# ---------- current live gauges (richer set for the snapshot) ----------
def latest_pct(series, invert=False, mom=None):
    s=pd.Series(series).dropna()
    if len(s)<24: return None
    if mom is not None: s=(s-s.shift(mom)).dropna()
    p=epct(s).dropna()
    if not len(p): return None
    v=float(p.iloc[-1])
    return round(1-v if invert else v, 2)

curV, curT = [], []
def addV(label, sub, frag):
    if frag is not None: curV.append({"label":label,"sub":sub,"frag":frag})
def addT(label, sub, frag):
    if frag is not None: curT.append({"label":label,"sub":sub,"frag":frag})

cape_now = d["cape"].dropna().iloc[-1] if d["cape"].dropna().size else None
addV("Valuation (CAPE)", (f"CAPE {cape_now:.0f}" if cape_now else "Shiller CAPE"), latest_pct(d["cape"]))
addV("Volatility suppression","low realized vol = complacency", latest_pct(d["rvol"], invert=True))
if len(bogz)>8:
    yoy=(bogz.rolling(2).mean().pct_change(4)*100).dropna()
    addV("Margin-loan leverage", (f"{yoy.iloc[-1]:+.0f}% YoY" if len(yoy) else "FRED margin loans"),
         (round(float((yoy<=yoy.iloc[-1]).mean()),2) if len(yoy)>8 else None))
sp_m=(baa-aaa).dropna()
if len(sp_m)>24:
    addV("Credit-spread compression", f"Baa\u2013Aaa {sp_m.iloc[-1]:.2f}%", latest_pct(sp_m, invert=True))
    addT("Credit spreads widening","3-month momentum", latest_pct(sp_m, mom=3))
addV("Loose financial conditions", (f"NFCI {nfci.iloc[-1]:+.2f}" if len(nfci) else "NFCI"), latest_pct(nfci))
addT("Volatility rising","vs recent calm", latest_pct(d["rvol"], mom=3))
addT("Price momentum","trailing 3-month fall", latest_pct(-d["px"].pct_change(3)))
if len(nfci): addT("Conditions tightening","NFCI momentum", latest_pct(nfci, mom=3))

# commentary: editable file in the repo (later written by the approval workflow)
commentary = "Commentary pending review."
if os.path.exists("commentary.txt"):
    commentary = open("commentary.txt", encoding="utf-8").read().strip() or commentary

data = {
  "as_of": dt.date.today().strftime("%B %Y"),
  "current": {"vulnerability":curV, "trigger":curT},
  "commentary": commentary,
  "priceSeries": priceSeries,
  "timeline": timeline
}
with open("data.json","w",encoding="utf-8") as f:
    json.dump(data, f, separators=(",",":"), ensure_ascii=False)
log(f"Wrote data.json  |  V-gauges {len(curV)}  T-gauges {len(curT)}  |  daily pts {len(priceSeries['dates'])}  |  timeline {len(timeline['dates'])} months")
