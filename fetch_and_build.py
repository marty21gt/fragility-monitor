#!/usr/bin/env python3
# =====================================================================
#  Market Fragility Monitor  --  data builder
#  Runs on a schedule (GitHub Actions). Fetches daily + monthly data,
#  computes the credit-momentum model, and writes data.json for the page.
#  Needs one secret: FRED_API_KEY (set in the GitHub repo, never in code).
# =====================================================================
import os, io, json, sys, datetime as dt
import numpy as np, pandas as pd, requests
try:
    import lxml  # noqa: F401  (needed by pandas.read_html for the recent-CAPE extension)
except Exception:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lxml"])
try:
    import xlrd  # noqa: F401  (needed by pandas.read_excel for Shiller's legacy .xls)
except Exception:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xlrd>=2.0.1"])

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
        raw = yf.download("^GSPC", start="1983-01-01", progress=False, auto_adjust=False)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        log(f"  daily S&P from Yahoo: {len(close)} rows")
        return _clean_series(close)
    except Exception as e:
        log(f"  Yahoo failed: {e}"); return pd.Series(dtype=float)

def daily_ndx():
    # Nasdaq-100 (index since 1985) for the QQQ variant
    import time as _t
    for attempt in range(3):
        try:
            import yfinance as yf
            raw = yf.download("^NDX", start="1985-01-01", progress=False, auto_adjust=False)
            close = raw["Close"]
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            s = _clean_series(close)
            if len(s) > 250:
                log(f"  daily Nasdaq-100 from Yahoo: {len(s)} rows")
                return s
            log(f"  Nasdaq-100 attempt {attempt+1}: only {len(s)} rows, retrying...")
        except Exception as e:
            log(f"  Nasdaq-100 attempt {attempt+1} failed: {e}")
        _t.sleep(3)
    log("  WARN: Nasdaq-100 unavailable -- QQQ toggles will be hidden this run")
    return pd.Series(dtype=float)

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

# ---- authoritative CAPE: Robert Shiller's own dataset (ie_data.xls) ----
# The GitHub mirror -- and the old Yale URL -- both stop updating CAPE in Sept 2023.
# The live file is hosted off shillerdata.com behind a versioned CDN link that rotates
# on each update, so we scrape the current download URL from the page rather than
# hardcoding it. Falls back to the estimator below if unreachable.
def _shiller_urls():
    # 1) the direct CDN file that shillerdata.com links to (versioned, but the path is stable)
    urls = ["https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/"
            "downloads/907c87f4-4176-4a13-9487-abddeadceb1b/ie_data.xls?ver=1783525168910",
            "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/"
            "downloads/907c87f4-4176-4a13-9487-abddeadceb1b/ie_data.xls"]
    # 2) whatever the page currently links to (in case the CDN path rotates)
    try:
        html = requests.get("https://shillerdata.com/", headers={"User-Agent":"Mozilla/5.0"},
                            timeout=45).text
        import re as _re
        for m in _re.findall(r'href="([^"]*ie_data[^"]*\.xls[^"]*)"', html, _re.I):
            u = m.replace("&amp;", "&")
            if not u.startswith("http"):
                u = "https://shillerdata.com/" + u.lstrip("/")
            if "0001" not in u and u not in urls:
                urls.append(u)
        log(f"  shillerdata.com: {len(urls)} candidate download link(s)")
    except Exception as e:
        log(f"  could not read shillerdata.com ({e})")
    urls.append("http://www.econ.yale.edu/~shiller/data/ie_data.xls")   # stale fallback
    return urls

shiller_ok = False
for _u in _shiller_urls():
    try:
        raw = requests.get(_u, timeout=90, headers={
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer":"https://shillerdata.com/",
            "Accept":"application/vnd.ms-excel,application/octet-stream,*/*"}).content
        # content sniff: a real .xls is an OLE2 compound file (magic D0 CF 11 E0).
        # Anything starting with '<' is an HTML error/interstitial page -- reject it.
        if raw[:4] != b"\xd0\xcf\x11\xe0":
            head = raw[:12].decode("ascii", "replace")
            raise ValueError(f"not an .xls file (got {head!r})")
        xl = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=None, engine="xlrd")
        # find the header row (the one whose first cell is "Date")
        hdr = next(i for i in range(15)
                   if str(xl.iloc[i,0]).strip().lower().startswith("date"))
        tbl = xl.iloc[hdr+1:].copy()
        cols = [str(c).strip() for c in xl.iloc[hdr].tolist()]
        tbl.columns = cols
        def _find(*names):
            for j,c in enumerate(cols):
                cl = c.lower().replace(" ","")
                if any(n in cl for n in names): return j
            return None
        j_date = 0
        j_p    = _find("^p$","price") if False else 1      # col B = nominal price
        j_d    = 2                                          # col C = dividend
        j_cape = _find("cape","p/e10")
        if j_cape is None: raise ValueError("CAPE column not found")
        # stage the parse first -- do NOT touch `sh` until we know the file is fresh
        stage = {}
        for _, r in tbl.iterrows():
            dv = r.iloc[j_date]
            if pd.isna(dv): continue
            # Shiller encodes dates as YYYY.MM  (note 1871.1 == October, not January)
            s = f"{float(dv):.2f}"
            yy, mm = int(s.split(".")[0]), int(s.split(".")[1])
            if mm < 1 or mm > 12: continue
            m0 = pd.Timestamp(yy, mm, 1)
            stage[m0] = (pd.to_numeric(r.iloc[j_p], errors="coerce"),
                         pd.to_numeric(r.iloc[j_d], errors="coerce"),
                         pd.to_numeric(r.iloc[j_cape], errors="coerce"))
        capes = {k:v[2] for k,v in stage.items() if pd.notna(v[2])}
        if not capes: raise ValueError("no CAPE values parsed")
        newest = max(capes)
        age_mo = (pd.Timestamp.today().to_period("M") - newest.to_period("M")).n
        # A live Shiller file should be at most a few months behind. If it is years old,
        # it is an archived copy -- reject it and try the next source.
        if age_mo > 9:
            log(f"  Shiller file is STALE (latest CAPE {newest:%Y-%m}, {age_mo} months old) "
                f"-- rejecting, trying next source")
            continue
        for m0,(pp,dd_,cp) in stage.items():
            if pd.notna(pp): sh.loc[m0,"SP500"] = float(pp)
            if pd.notna(dd_): sh.loc[m0,"Dividend"] = float(dd_)
            if pd.notna(cp): sh.loc[m0,"PE10"] = float(cp)
        sh = sh.sort_index()
        log(f"  Shiller CAPE (authoritative): {len(capes)} months, latest "
            f"{newest:%Y-%m} = {capes[newest]:.1f}  [{age_mo} mo old]")
        shiller_ok = True
        break
    except Exception as e:
        log(f"  Shiller fetch failed ({_u.split('/')[2]}): {e}")
if not shiller_ok:
    log("  WARN: no fresh Shiller file -- using ESTIMATED CAPE for recent months")


# ---- extend Shiller CAPE / price / dividend to the present (free mirror lags ~2 yrs) ----
try:
    _hdr={"User-Agent":"Mozilla/5.0"}
    def _multpl(url,col):
        html=requests.get(url,headers=_hdr,timeout=40).text
        t=pd.read_html(io.StringIO(html))[0].iloc[:,:2].copy()
        t.columns=["Date",col]; t["Date"]=pd.to_datetime(t["Date"],errors="coerce"); t=t.dropna(subset=["Date"])
        t[col]=pd.to_numeric(t[col].astype(str).str.replace(",","",regex=False).str.replace("%","",regex=False).str.strip(), errors="coerce")
        return t.dropna().set_index("Date")[col]
    cape_m=_multpl("https://www.multpl.com/shiller-pe/table/by-month","v")
    dy_m=_multpl("https://www.multpl.com/s-p-500-dividend-yield/table/by-month","v")
    mpx=(sp_daily.resample("MS").mean() if len(sp_daily)>250 else pd.Series(dtype=float))
    last=sh.index.max(); added=0
    for dte in sorted(cape_m.index):
        m0=pd.Timestamp(dte.year,dte.month,1)
        if m0<=last: continue
        price=mpx.get(m0,np.nan)
        if np.isnan(price):
            nn=sp_daily[sp_daily.index<=dte]; price=float(nn.iloc[-1]) if len(nn) else np.nan
        if np.isnan(price): continue
        dyv=dy_m.get(dte,np.nan)
        sh.loc[m0,"SP500"]=price; sh.loc[m0,"PE10"]=float(cape_m[dte])
        sh.loc[m0,"Dividend"]=price*(dyv/100) if not np.isnan(dyv) else float(sh["Dividend"].dropna().iloc[-1])
        added+=1
    sh=sh.sort_index()
    log(f"  extended Shiller to {sh.index.max().strftime('%Y-%m')} (+{added} months)")
except Exception as e:
    log(f"  recent-extension via multpl skipped: {e}")

# ---- fill any month that has a price but blank CAPE / dividend (the mirror's 2023+ gap) ----
try:
    pxc = sh["SP500"].replace(0, np.nan)
    capec = sh["PE10"].replace(0, np.nan)
    divc = sh["Dividend"].replace(0, np.nan)
    realc = capec.dropna()                       # months with a real CAPE
    dyv = (divc / pxc).dropna()                  # dividend yield history
    dy_last = float(dyv.iloc[-1]) if len(dyv) else 0.018
    fC = fD = 0
    for m0 in pxc.dropna().index:
        if pd.isna(capec.get(m0, np.nan)):
            earlier = realc.index[realc.index <= m0]
            if len(earlier):
                rc = earlier.max(); yrs = (m0 - rc).days / 365.25
                sh.loc[m0, "PE10"] = float(realc.loc[rc]) * (float(pxc[m0]) / float(pxc[rc])) / (1.07 ** yrs)
                fC += 1
        if pd.isna(divc.get(m0, np.nan)):
            sh.loc[m0, "Dividend"] = float(pxc[m0]) * dy_last
            fD += 1
    sh = sh.sort_index()
    if fC or fD:
        log(f"  gap-filled recent months: {fC} CAPE + {fD} dividend (through {sh.index.max().strftime('%Y-%m')}; recent CAPE approximate)")
except Exception as e:
    log(f"  gap-fill skipped: {e}")
jst = pd.read_excel("https://github.com/bank-of-england/MachineLearningCrisisPrediction/raw/master/data/JSTdatasetR3.xlsx", sheet_name="Data")
usj = jst[jst["country"]=="USA"].set_index("year")
baa, aaa = fred("BAA"), fred("AAA")
bogz = fred("BOGZ1FL663067003Q")     # broker margin loans (live leverage)
nfci = fred("NFCI")                   # financial conditions

# ---------- monthly panel + credit-momentum model (the timeline) ----------
d = pd.DataFrame(index=sh.index)
d["px"]=sh["SP500"]; d["div"]=sh["Dividend"].replace(0,np.nan); d["cape"]=sh["PE10"].replace(0,np.nan); d["yr"]=d.index.year
jst_cg = usj["tloans"]/usj["gdp"]        # JST BANK LOANS / GDP (annual, 1870-2016)

# ---- Leverage gauge: BIS total credit as the SPINE, JST back-spliced for the early era ----
# JST "tloans" is bank loans only. BIS QUSPAM770A is TOTAL credit to the private
# non-financial sector (loans + debt securities) -- the right concept for a modern
# leverage gauge, since a large share of corporate leverage is now bond-financed, which
# JST cannot see. BIS is also quarterly and still updating.
#
# So: use BIS wherever it exists (1947+), and extend BACKWARD with JST's proportional
# shape for the pre-1947 tail. This puts every crisis that matters (1973, 1987, 2000,
# 2008, 2020, 2022) on one consistent unspliced series, and moves the seam to 1947 --
# far from any live decision, and to the era where "bank loans ~= credit" is most nearly
# true, so the splice assumption is at its strongest.
cg = jst_cg.copy()                        # fallback: JST-only (previous behavior)
lev_note = "JST bank-loans only (BIS unavailable)"
try:
    bis = fred("QUSPAM770A")              # quarterly, % of GDP, total private credit
    if len(bis) > 20:
        bis_a = (bis/100.0).groupby(bis.index.year).mean()     # annual ratio
        b0 = int(bis_a.index.min())                            # first BIS year (~1947)
        combined = {}
        # BIS is the spine wherever it exists
        for y in bis_a.index:
            combined[int(y)] = float(bis_a[y])
        # back-splice JST proportionally for years before BIS starts
        if b0 in jst_cg.index and not pd.isna(jst_cg[b0]):
            anchor_b, anchor_j = float(bis_a[b0]), float(jst_cg[b0])
            back = 0
            for y in sorted(jst_cg.dropna().index):
                if int(y) < b0:
                    combined[int(y)] = anchor_b * (float(jst_cg[y]) / anchor_j)
                    back += 1
            cg = pd.Series(combined).sort_index()
            lev_note = (f"BIS total credit {b0}-{int(cg.index.max())} (spine) + "
                        f"JST back-splice for {back} pre-{b0} yrs")
        else:
            cg = pd.Series(combined).sort_index()
            lev_note = f"BIS total credit {b0}-{int(cg.index.max())} (no early splice)"
        log(f"  leverage gauge: {lev_note}")
    else:
        log(f"  WARN: BIS credit/GDP unavailable -- leverage gauge = {lev_note}")
except Exception as e:
    log(f"  WARN: BIS fetch failed ({e}) -- leverage gauge = {lev_note}")

d["cg5"]=d["yr"].map(cg-cg.shift(5)); d["stir"]=d["yr"].map(usj["stir"])
d["cg5_jst"]=d["yr"].map(jst_cg-jst_cg.shift(5))     # old gauge: JST bank loans, ends 2016
for y,v in {2017:.93,2018:1.94,2019:2.11,2020:.37,2021:.04,2022:2.02,2023:5.14,2024:4.98,2025:4.30,2026:4.20}.items():
    d.loc[d.yr==y,"stir"]=v
d["stir"]=d["stir"].ffill()
spread=(baa-aaa); spread.index=spread.index.to_period("M"); d["spread"]=d.index.to_period("M").map(spread)
d["rvol"]=d["px"].pct_change().rolling(12).std()*np.sqrt(12); d["ma10"]=d["px"].rolling(10).mean()
d["tr"]=(d["px"]+d["div"]/12)/d["px"].shift(1)-1
d=d.dropna(subset=["px"])
capef=epct(d["cape"]); volsup=1-epct(d["rvol"]); comp=1-epct(d["spread"])
levf     = epct(d["cg5_jst"])   # GAUGE OF RECORD: JST bank loans (preserves the 1973-74 exit)
levf_bis = epct(d["cg5"])       # comparison variant: BIS total credit
sm=epct(d["spread"]-d["spread"].shift(3))
volup=d["rvol"]/d["rvol"].rolling(12).min()-1; mom3=d["px"].pct_change(3)
# GAUGE OF RECORD: credit-to-GDP is DROPPED from vulnerability.
#   - it changed no positions since 1991
#   - JST data dies in 2016; BIS cannot rank its own early history (missed 1973-74)
#   - BIS currently sits at the 1st percentile and would SUPPRESS V during a genuine
#     market-leverage boom (margin debt +28% YoY) -- it measures economy-wide debt,
#     not market speculation, which is the wrong instrument for an equity-crash model
# Credit risk is still captured by credit-spread COMPRESSION (here) and credit-spread
# MOMENTUM (in the trigger) -- both live, official, market-priced FRED series.
d["V"]    =pd.DataFrame({"a":capef,"b":volsup,"c":comp}).apply(lambda r:blend(list(r.values)),axis=1)
d["V_alt"]=pd.DataFrame({"a":capef,"b":volsup,"c":levf,"d":comp}).apply(lambda r:blend(list(r.values)),axis=1)  # comparison: with JST credit/GDP
d["T"]=pd.DataFrame({"a":epct(volup),"b":epct(-mom3),"c":sm}).apply(lambda r:blend(list(r.values)),axis=1)
d=d.dropna(subset=["V","T","tr","ma10"])
Ts,Vs,px,ma=d["T"].values,d["V"].values,d["px"].values,d["ma10"].values
Vs_alt=d["V_alt"].values
def make_pos(mode, Vv=None):
    if Vv is None: Vv = Vs
    pos=[]; state="in"
    for i in range(len(d)):
        pos.append(1 if state=="in" else 0)
        if state=="in":
            if Vv[i]>=.62 and Ts[i]>=.55 and px[i]<ma[i]: state="out"
        else:
            if mode=="v1":
                if px[i]>ma[i]: state="in"                       # faster: re-enter once price reclaims the 200-day MA
            else:
                if i>=1 and Ts[i]<.55 and Ts[i-1]<.55 and px[i]>ma[i]: state="in"
    return pos
d["pos"]=make_pos("base"); d["pos_v1"]=make_pos("v1")
d["pos_alt"]=make_pos("base", Vs_alt)                 # comparison: BIS leverage gauge
d["sr"]=np.where(np.array(d["pos"])==1, d["tr"], d["stir"]/100/12)
d["sr_v1"]=np.where(np.array(d["pos_v1"])==1, d["tr"], d["stir"]/100/12)
d["sr_alt"]=np.where(np.array(d["pos_alt"])==1, d["tr"], d["stir"]/100/12)

# ---------- timeline: monthly deep history + DAILY from 1985 (true fast-crash depth) ----------
SPLICE = pd.Timestamp("1986-01-01")
dm = d[d.index < SPLICE]
tl_dates=[t.strftime("%Y-%m") for t in dm.index]
tl_V=[round(float(x),3) for x in dm["V"]]; tl_T=[round(float(x),3) for x in dm["T"]]
tl_px=[round(float(x),1) for x in dm["px"]]; tl_ma=[round(float(x),1) for x in dm["ma10"]]
tl_pos=[int(x) for x in dm["pos"]]; tl_bh=[round(float(x),5) for x in dm["tr"]]; tl_sr=[round(float(x),5) for x in dm["sr"]]
tl_pos_v1=[int(x) for x in dm["pos_v1"]]; tl_sr_v1=[round(float(x),5) for x in dm["sr_v1"]]
tl_V_j=[round(float(x),3) for x in dm["V_alt"]]; tl_pos_j=[int(x) for x in dm["pos_alt"]]
tl_sr_j=[round(float(x),5) for x in dm["sr_alt"]]
def _num(x, d3):  # safe scalar
    try:
        f=float(x); return None if np.isnan(f) else f
    except Exception: return None
if len(sp_daily) > 250:
    dend = d.index[-1]
    dd = sp_daily[(sp_daily.index>=SPLICE)&(sp_daily.index<=dend)]
    ma200 = sp_daily.rolling(200).mean()
    Vm=pd.Series(d["V"].values,index=d.index.to_period("M"))
    Tm=pd.Series(d["T"].values,index=d.index.to_period("M"))
    Pm=pd.Series(d["pos"].values,index=d.index.to_period("M"))
    Pm1=pd.Series(d["pos_v1"].values,index=d.index.to_period("M"))
    PmJ=pd.Series(d["pos_alt"].values,index=d.index.to_period("M"))
    VmJ=pd.Series(d["V_alt"].values,index=d.index.to_period("M"))
    Sm=pd.Series(d["stir"].values,index=d.index.to_period("M"))
    DY=pd.Series((sh["Dividend"]/sh["SP500"]).values,index=sh.index.to_period("M"))
    prev=None
    for dte,pxv in dd.items():
        per=dte.to_period("M")
        vv=_num(Vm.get(per),3); tv=_num(Tm.get(per),3); pv=_num(Pm.get(per),0); pv1=_num(Pm1.get(per),0)
        pv = tl_pos[-1] if pv is None else int(pv)
        pv1 = tl_pos_v1[-1] if pv1 is None else int(pv1)
        pvj=_num(PmJ.get(per),0); pvj = tl_pos_j[-1] if pvj is None else int(pvj)
        vvj=_num(VmJ.get(per),3)
        divy=_num(DY.get(per),6) or 0.0
        stira=_num(Sm.get(per),4); stira=4.0 if stira is None else stira
        bh = 0.0 if prev is None else (pxv/prev-1)+divy/252.0
        sr = bh if pv==1 else stira/100/252
        sr1 = bh if pv1==1 else stira/100/252
        mav=_num(ma200.get(dte),1)
        tl_dates.append(dte.strftime("%Y-%m-%d"))
        tl_V.append(round(vv,3) if vv is not None else tl_V[-1])
        tl_T.append(round(tv,3) if tv is not None else tl_T[-1])
        tl_px.append(round(float(pxv),1))
        tl_ma.append(round(mav,1) if mav is not None else None)
        tl_pos.append(pv); tl_bh.append(round(bh,5)); tl_sr.append(round(sr,5))
        tl_pos_v1.append(pv1); tl_sr_v1.append(round(sr1,5))
        srj = bh if pvj==1 else stira/100/252
        tl_V_j.append(round(vvj,3) if vvj is not None else tl_V_j[-1])
        tl_pos_j.append(pvj); tl_sr_j.append(round(srj,5))
        prev=pxv
timeline={"dates":tl_dates,"V":tl_V,"T":tl_T,"px":tl_px,"ma":tl_ma,"pos":tl_pos,"bhret":tl_bh,"stret":tl_sr}
timeline_v1={"pos":tl_pos_v1,"stret":tl_sr_v1}
timeline_alt={"V":tl_V_j,"pos":tl_pos_j,"stret":tl_sr_j}
monthly_pos = pd.Series(d["pos"].values, index=pd.DatetimeIndex(d.index))

# ---------- QQQ variant: same S&P regime signal + S&P 200-day trend, Nasdaq-100 as risk-on vehicle ----------
timeline_qqq = None; timeline_qqq_v1 = None; timeline_qqq_alt = None
try:
    ndx = daily_ndx()
    if len(ndx) <= 250:
        log(f"  QQQ variant SKIPPED: ndx has only {len(ndx)} rows")
    if len(ndx) > 250:
        Pm=pd.Series(d["pos"].values,index=d.index.to_period("M"))
        Pm1=pd.Series(d["pos_v1"].values,index=d.index.to_period("M"))
        PmJq=pd.Series(d["pos_alt"].values,index=d.index.to_period("M"))
        VmJq=pd.Series(d["V_alt"].values,index=d.index.to_period("M"))
        Sm=pd.Series(d["stir"].values,index=d.index.to_period("M"))
        Vm=pd.Series(d["V"].values,index=d.index.to_period("M"))
        Tm=pd.Series(d["T"].values,index=d.index.to_period("M"))
        S85=pd.Timestamp("1986-01-01")
        ext=ndx[ndx.index>=(S85-pd.DateOffset(months=2))]
        nret=ext.pct_change(); nma=ext.rolling(200).mean(); QDY=0.006
        # S&P benchmark aligned by calendar: accumulate all S&P daily returns that fall
        # between consecutive Nasdaq dates, so no S&P return is silently dropped.
        # NOTE: tl_dates mixes "YYYY-MM" (pre-1985) with "YYYY-MM-DD" (daily). The QQQ
        # timeline only needs 1985+, so use just the full-date entries.
        _sp = [(dstr, r) for dstr, r in zip(tl_dates, tl_bh) if len(dstr) == 10]
        sp_ret = pd.Series([r for _, r in _sp],
                           index=pd.to_datetime([dstr for dstr, _ in _sp], format="%Y-%m-%d"))
        sp_ret = sp_ret[~sp_ret.index.duplicated(keep="last")].sort_index()
        sp_cum = (1.0 + sp_ret).cumprod()          # S&P wealth index on its own dates
        def sp_bench(prev_dt, dte):
            """compounded S&P return over (prev_dt, dte]"""
            if prev_dt is None: return 0.0
            a = sp_cum[sp_cum.index <= prev_dt]
            b = sp_cum[sp_cum.index <= dte]
            if not len(a) or not len(b): return 0.0
            return float(b.iloc[-1]/a.iloc[-1] - 1.0)
        tq={"dates":[],"V":[],"T":[],"px":[],"ma":[],"pos":[],"bhret":[],"stret":[],"bhqqq":[]}
        qpos1=[]; qsr1=[]; qposj=[]; qsrj=[]; qVj=[]; prev_q=None
        for dte in ext.index[ext.index>=S85]:
            pm=dte.to_period("M")
            if pm not in Pm.index: continue
            r=nret.loc[dte]
            if pd.isna(r): continue
            pos_i=int(Pm.loc[pm]); pos1_i=int(Pm1.loc[pm]); sti=Sm.loc[pm]
            cash=(float(sti)/100/252 if not pd.isna(sti) else 0.0)
            qbh=float(r)+QDY/252
            st=qbh if pos_i==1 else cash
            st1=qbh if pos1_i==1 else cash
            bench=sp_bench(prev_q, dte); prev_q=dte
            mav=nma.loc[dte]
            tq["dates"].append(dte.strftime("%Y-%m-%d"))
            tq["V"].append(round(float(Vm.loc[pm]),3)); tq["T"].append(round(float(Tm.loc[pm]),3))
            tq["px"].append(round(float(ext.loc[dte]),1)); tq["ma"].append(round(float(mav),1) if not pd.isna(mav) else None)
            tq["pos"].append(pos_i); tq["bhret"].append(round(bench,5)); tq["stret"].append(round(st,5)); tq["bhqqq"].append(round(qbh,5))
            qpos1.append(pos1_i); qsr1.append(round(st1,5))
            pj=int(PmJq.loc[pm]); stj = qbh if pj==1 else cash
            qposj.append(pj); qsrj.append(round(stj,5)); qVj.append(round(float(VmJq.loc[pm]),3))
        if len(tq["dates"])>250:
            # --- align all views to a common, clean start (1986) so buy&hold figures
            # reconcile AND every selectable year is a complete calendar year.
            # (The Nasdaq-100 index began late in 1985, so 1985 would be a partial year.)
            START = "1986-01-01"
            qkeep = [i for i,ds in enumerate(tq["dates"]) if ds >= START]
            if qkeep and len(qkeep) < len(tq["dates"]):
                for k in ("dates","V","T","px","ma","pos","bhret","stret","bhqqq"):
                    tq[k] = [tq[k][i] for i in qkeep]
                qpos1 = [qpos1[i] for i in qkeep]; qsr1 = [qsr1[i] for i in qkeep]
                qposj = [qposj[i] for i in qkeep]; qsrj = [qsrj[i] for i in qkeep]
                qVj   = [qVj[i]   for i in qkeep]
            # first bar defines the 100% baseline: no carried-in return
            if tq["dates"]:
                tq["bhret"][0]=0.0; tq["stret"][0]=0.0; tq["bhqqq"][0]=0.0; qsr1[0]=0.0; qsrj[0]=0.0
            timeline_qqq=tq; timeline_qqq_v1={"pos":qpos1,"stret":qsr1}
            timeline_qqq_alt={"V":qVj,"pos":qposj,"stret":qsrj}
            log(f"  QQQ variant: {len(tq['dates'])} daily points from {tq['dates'][0] if tq['dates'] else 'n/a'}")

            keep = [i for i,ds in enumerate(timeline["dates"])
                    if len(ds) != 10 or ds >= START]      # monthly (pre-1985) + daily from 1986
            if len(keep) < len(timeline["dates"]):
                dropped = len(timeline["dates"]) - len(keep)
                for k in ("dates","V","T","px","ma","pos","bhret","stret"):
                    timeline[k] = [timeline[k][i] for i in keep]
                for k in ("pos","stret"):
                    timeline_v1[k] = [timeline_v1[k][i] for i in keep]
                for k in ("V","pos","stret"):
                    timeline_alt[k] = [timeline_alt[k][i] for i in keep]
                fi = next((i for i,ds in enumerate(timeline["dates"]) if len(ds)==10), None)
                if fi is not None:
                    timeline["bhret"][fi] = 0.0; timeline["stret"][fi] = 0.0
                    timeline_v1["stret"][fi] = 0.0; timeline_alt["stret"][fi] = 0.0
                log(f"  aligned all views to {START} (trimmed {dropped} partial-1985 days)")
        else:
            log(f"  QQQ variant DROPPED: only {len(tq['dates'])} points built.")
            log(f"    ndx rows={len(ndx)} ext rows={len(ext[ext.index>=S85])} "
                f"Pm months={len(Pm.index)} Pm range={Pm.index.min()}..{Pm.index.max()}")
except Exception as e:
    import traceback
    log(f"  QQQ variant FAILED: {type(e).__name__}: {e}")
    log("  " + traceback.format_exc().replace("\n", "\n  "))

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
def addV(label, sub, frag, score=True):
    if frag is not None: curV.append({"label":label,"sub":sub,"frag":frag,"score":bool(score)})
def addT(label, sub, frag, score=True):
    if frag is not None: curT.append({"label":label,"sub":sub,"frag":frag,"score":bool(score)})

cape_now = d["cape"].dropna().iloc[-1] if d["cape"].dropna().size else None
_cape_lbl = (f"CAPE {cape_now:.0f}" if shiller_ok else f"CAPE ~{cape_now:.0f} (est.)") if cape_now else "Shiller CAPE"
addV("Valuation (CAPE)", _cape_lbl, latest_pct(d["cape"]))
addV("Volatility suppression","low realized vol = complacency", latest_pct(d["rvol"], invert=True))
if len(bogz)>8:
    yoy=(bogz.rolling(2).mean().pct_change(4)*100).dropna()
    addV("Margin-loan leverage", (f"{yoy.iloc[-1]:+.0f}% YoY \u00b7 context" if len(yoy) else "FRED margin loans \u00b7 context"),
         (round(float((yoy<=yoy.iloc[-1]).mean()),2) if len(yoy)>8 else None), score=False)
sp_m=(baa-aaa).dropna()
if len(sp_m)>24:
    addV("Credit-spread compression", f"Baa\u2013Aaa {sp_m.iloc[-1]:.2f}%", latest_pct(sp_m, invert=True))
    addT("Credit spreads widening","3-month momentum", latest_pct(sp_m, mom=3))
addV("Loose financial conditions", (f"NFCI {nfci.iloc[-1]:+.2f} \u00b7 context" if len(nfci) else "NFCI \u00b7 context"), latest_pct(nfci), score=False)
addT("Volatility rising","vs recent calm", latest_pct(d["rvol"], mom=3))
addT("Price momentum","trailing 3-month fall", latest_pct(-d["px"].pct_change(3)))
if len(nfci): addT("Conditions tightening","NFCI momentum \u00b7 context", latest_pct(nfci, mom=3), score=False)

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
if timeline_qqq: data["timeline_qqq"] = timeline_qqq
data["timeline_v1"] = timeline_v1
data["timeline_alt"] = timeline_alt
if timeline_qqq_v1: data["timeline_qqq_v1"] = timeline_qqq_v1
if timeline_qqq_alt: data["timeline_qqq_alt"] = timeline_qqq_alt
with open("data.json","w",encoding="utf-8") as f:
    json.dump(data, f, separators=(",",":"), ensure_ascii=False)

# ---- analysis.json: the raw monthly gauge components, for offline experimentation ----
# data.json only carries the FINAL blended V/T. This file carries the ingredients, so any
# variant (different weights, thresholds, gauge swaps) can be tested without re-fetching.
try:
    ana = {"note":"monthly gauge components; percentiles are point-in-time (expanding window)",
           "dates":[t.strftime("%Y-%m") for t in d.index]}
    for k, ser in [("cape",d["cape"]), ("rvol",d["rvol"]), ("spread",d["spread"]),
                   ("cg5_bis",d["cg5"]), ("cg5_jst",d["cg5_jst"]),
                   ("px",d["px"]), ("ma10",d["ma10"]), ("tr",d["tr"]), ("stir",d["stir"]),
                   ("pct_cape",capef), ("pct_volsup",volsup), ("pct_lev_jst",levf),
                   ("pct_lev_bis",levf_bis), ("pct_creditcomp",comp),
                   ("pct_volup",epct(volup)), ("pct_pricemom",epct(-mom3)), ("pct_spreadmom",sm),
                   ("V",d["V"]), ("V_bis",d["V_alt"]), ("T",d["T"]),
                   ("pos",d["pos"]), ("pos_bis",d["pos_alt"])]:
        s2 = pd.Series(ser).reindex(d.index)
        ana[k] = [None if pd.isna(v) else round(float(v),5) for v in s2.values]
    # DAILY block: monthly positions applied to daily returns -> true drawdowns
    ana["daily"] = {"dates": timeline["dates"], "bhret": timeline["bhret"],
                    "pos": timeline["pos"], "px": timeline["px"]}
    if timeline_qqq:
        ana["daily_qqq"] = {"dates": timeline_qqq["dates"], "bhqqq": timeline_qqq["bhqqq"],
                            "pos": timeline_qqq["pos"]}
    with open("analysis.json","w",encoding="utf-8") as f:
        json.dump(ana, f, separators=(",",":"))
    log(f"Wrote analysis.json  |  {len(ana['dates'])} months x {len(ana)-2} series")
except Exception as e:
    log(f"  analysis.json skipped: {e}")
log(f"Wrote data.json  |  V-gauges {len(curV)}  T-gauges {len(curT)}  |  daily pts {len(priceSeries['dates'])}  |  timeline {len(timeline['dates'])} points (daily from 1985)")
