#!/usr/bin/env python3
"""
IMPLIED-VOL (VXO/VIX) LEVERAGE-GOVERNOR TEST
============================================
Tests whether a FORWARD-looking implied-vol gauge can protect fast crashes where
trailing realized vol cannot -- because implied vol can be elevated on a still-calm
day, letting it pre-empt the leverage you're about to get hurt on.

Self-contained: replicates the v3.2 engine (validated to reproduce the Rev-3.0 audit
numbers) so it needs only data.json + daily_vt.json (committed) and a FRED key.
Fetches VXOCLS (S&P100 VXO, 1986-2021) spliced to VIXCLS (VIX, 1990-present).

All governors are REDUCE-ONLY (never add leverage) and leave the tv/vol20 leverage-up
untouched. Grid covers level triggers and relative-change ("implied vol doubles")
triggers, de-levering to 1x (governor) and one to-SGOV variant (to show the exit cost).

    FRED_API_KEY=xxxx python vix_governor_test.py [data.json] [daily_vt.json]
"""
import os, sys, json, time, urllib.request, urllib.parse, urllib.error
import numpy as np, pandas as pd

FRED_KEY=os.environ.get("FRED_API_KEY","").strip()
if not FRED_KEY: print("ERROR: FRED_API_KEY not set."); sys.exit(1)
DATA=sys.argv[1] if len(sys.argv)>1 else "data.json"
VT  =sys.argv[2] if len(sys.argv)>2 else "daily_vt.json"

# ---------- v3.2 engine (mirror of reference_backtest_v32) ----------
VTH,TTH,REENTRY=0.54,0.70,15; VW,VFL,SW,DEAD=20,0.10,21,0.03
TCUT,TREST,BAND=0.60,0.50,0.10; TV,CAP,FLOOR=0.25,3.0,1.0; MULT,EXP,BPS=3,0.0082,5.0
def load(dp,vp):
    d=json.load(open(dp)); tq=d['timeline_qqq']; dv=json.load(open(vp)); idx=pd.to_datetime(tq['dates'])
    df=pd.DataFrame({k:pd.Series(tq[k],idx,dtype=float) for k in ['px','ma','V','T','pos','stret','bhqqq']})
    df['Td']=pd.Series(dv['Td'],pd.to_datetime(dv['dates']),dtype=float).reindex(idx).ffill(); return df
def cash_rate(df): return df['stret'].where(df['pos']==0).ffill().bfill().fillna(0.02/252)
def monitor(df):
    act=lambda s:s.resample('MS').first().shift(1).reindex(df.index,method='ffill').ffill().bfill()
    dg=((act(df['V'])>=VTH)&(act(df['T'])>=TTH)).values; bl=(df['px']<df['ma']).values
    n=len(df);pos=np.ones(n);off=False;c=0
    for i in range(n):
        if off:
            c=c+1 if not bl[i] else 0
            if c>=REENTRY: off=False;c=0
        elif dg[i] and bl[i]: off=True;c=0
        pos[i]=0.0 if off else 1.0
    return pd.Series(pos,df.index)
def rvol(px): return np.sqrt((np.log(px/px.shift(1))**2).rolling(VW).mean()*252).clip(lower=VFL)
def mslope(ma):
    x=np.arange(SW);xd=x-x.mean();den=(xd**2).sum()
    return ma.rolling(SW).apply(lambda w:(xd*(w-w.mean())).sum()/den,raw=True)*252/ma
def lgate(Td):
    on=True;o=np.ones(len(Td),bool)
    for i,t in enumerate(Td.values):
        if on and t>=TCUT:on=False
        elif (not on) and t<TREST:on=True
        o[i]=on
    return pd.Series(o,Td.index)
def band(t):
    h=[];c=0.0
    for x in np.asarray(t):
        if abs(x-c)>BAND:c=x
        h.append(c)
    return pd.Series(h,t.index)
def base_exposure(df,sig):
    size=(TV/rvol(df['px'])).clip(lower=FLOOR,upper=CAP)
    can=lgate(df['Td'])&(df['px']>df['ma'])&(mslope(df['ma'])>DEAD)
    raw=pd.Series(np.where(can,size,1.0),df.index).where(sig==1,0.0)
    return band(raw).shift(1).fillna(0)
def strat_ret(df,held,cash):
    qqq=df['bhqqq'].fillna(0); lev=MULT*qqq-(EXP+(MULT-1)*cash*252)/252; e=held
    wl=((e-1)/(MULT-1)).clip(lower=0); wq=pd.Series(np.where(e>1,1-(e-1)/(MULT-1),e),df.index).clip(lower=0)
    wc=(1-wq-wl).clip(lower=0); r=(wq*qqq+wl*lev+wc*cash).fillna(0)
    return r-e.diff().abs().fillna(0)*(BPS/10000.0)
def metrics(r,cash):
    r=r.dropna();n=len(r);eq=(1+r).cumprod();exc=r-cash.reindex(r.index).fillna(0)
    return dict(CAGR=eq.iloc[-1]**(252/n)-1,Sharpe=(exc.mean()*252)/(r.std()*np.sqrt(252)),
                MaxDD=(eq/eq.cummax()-1).min(),Term=eq.iloc[-1],eq=eq)

# ---------- FRED implied vol ----------
def fred(sid,retries=3):
    q=urllib.parse.urlencode({"series_id":sid,"api_key":FRED_KEY,"file_type":"json"})
    url=f"https://api.stlouisfed.org/fred/series/observations?{q}"
    for a in range(retries):
        try:
            with urllib.request.urlopen(url,timeout=60) as r: obs=json.load(r)["observations"]; break
        except urllib.error.HTTPError as e:
            if a==retries-1: raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]}")
            time.sleep(3)
    i=[];v=[]
    for o in obs:
        if o["value"] not in (".","",None): i.append(pd.Timestamp(o["date"])); v.append(float(o["value"]))
    return pd.Series(v,i).sort_index()

df=load(DATA,VT); cash=cash_rate(df); sig=monitor(df); base=base_exposure(df,sig); px=df['px']
vxo=fred("VXOCLS"); vix=fred("VIXCLS")
iv=(vxo.reindex(df.index).combine_first(vix.reindex(df.index))/100.0).ffill()   # VXO primary, VIX fills 2021+
import csv
with open("implied_vol.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["date","iv"])
    for d,v in iv.dropna().items(): w.writerow([d.date(), round(float(v),4)])
print("wrote implied_vol.csv (spliced VXO/VIX, decimal annualized) -- upload this back for the full sweep")
try:
    vxv = fred("VXVCLS")   # CBOE 3-month VIX (real 3-month ATM IV), inception 2007
    with open("vxv_3month.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["date","vxv"])
        for d,v in (vxv/100.0).dropna().items(): w.writerow([d.date(), round(float(v),4)])
    print(f"wrote vxv_3month.csv (real 3-month ATM IV, {vxv.index.min().date()}+) -- refines the put term structure")
    print("NOTE: VXV is ATM only -- it does NOT contain the 10%-OTM skew, which the sensitivity shows is decisive")
except Exception as e:
    print("VXV fetch failed (proceeding without it):", e)
print(f"implied vol: VXO {vxo.index.min().date()}..{vxo.index.max().date()}, "
      f"VIX {vix.index.min().date()}..{vix.index.max().date()}; coverage {iv.notna().mean()*100:.0f}%")

def apply_gov(active, capto=1.0):
    return pd.Series(np.where(active, np.minimum(base.values,capto), base.values), df.index)
def level(thr):  return (iv>thr).shift(1).fillna(False).values
def change(mult,hold=10):
    fire=((iv/iv.rolling(21).median())>=mult).fillna(False).values; act=np.zeros(len(df),bool)
    for i in np.where(fire)[0]: act[i+1:i+1+hold]=True
    return act
def ev(name,held):
    m=metrics(strat_ret(df,held,cash),cash); eq=m['eq']
    cdd=lambda s,e:(eq.loc[s:e]/eq.loc[s:e].cummax()-1).min()
    return (name,m['CAGR'],m['Sharpe'],m['MaxDD'],m['Term'],
            cdd('1987-08','1988-01'),cdd('2000-03','2002-12'),cdd('2007-10','2009-06'),cdd('2020-02','2020-06'))
rows=[ev("base",base),
      ev("iv>0.30 ->1x",apply_gov(level(0.30))), ev("iv>0.35 ->1x",apply_gov(level(0.35))),
      ev("iv>0.40 ->1x",apply_gov(level(0.40))),
      ev("iv x2 ->1x",apply_gov(change(2))), ev("iv x3 ->1x",apply_gov(change(3))),
      ev("iv>0.35 ->SGOV",apply_gov(level(0.35),capto=0.0))]
print(f"\n{'governor':<18}{'CAGR':>7}{'Shrp':>6}{'MaxDD':>7}{'$1->':>9} | {'1987':>6}{'00-02':>6}{'2008':>6}{'2020':>6}")
print("-"*80)
for n,c,s,md,tm,d87,d00,d08,d20 in rows:
    print(f"{n:<18}{c*100:6.1f}%{s:6.2f}{md*100:6.0f}%{'$'+format(tm,',.0f'):>9} | {d87*100:5.0f}%{d00*100:5.0f}%{d08*100:5.0f}%{d20*100:5.0f}%")
print("\nKey question: does any governor cut 1987/2020 WITHOUT gutting CAGR/terminal?")
print("Watch whether implied vol was elevated on calm pre-crash days (its only edge over realized vol).")

# ---------- HEADFAKE PAIR: spike -> exit, VIX-collapse -> fast re-enter (reduce-only) ----------
def headfake(spike, reenter, capto):
    ivv=iv.values; act=np.zeros(len(df),bool); on=False
    for i in range(len(df)):
        if i>=1:
            if (not on) and ivv[i-1]>=spike: on=True
            elif on and ivv[i-1]<=reenter: on=False
        act[i]=on
    held=pd.Series(np.where(act, np.minimum(base.values,capto), base.values), df.index)
    m=metrics(strat_ret(df,held,cash),cash); eq=m['eq']
    fires=int(np.sum(act[1:] & ~act[:-1])); daysout=int(act.sum())
    cdd=lambda s,e:(eq.loc[s:e]/eq.loc[s:e].cummax()-1).min()
    return (m['CAGR'],m['Sharpe'],m['MaxDD'],m['Term'],fires,daysout,
            cdd('1987-08','1988-01'),cdd('2000-03','2002-12'),cdd('2007-10','2009-06'),cdd('2020-02','2020-06'))
print(f"\n=== headfake pair: spike->SGOV, VIX-collapse->reinvest (reduce-only) ===")
print(f"{'spike/reenter':<16}{'CAGR':>7}{'Shrp':>6}{'MaxDD':>7}{'$1->':>8}{'fires':>6}{'d.out':>6} | {'1987':>6}{'0002':>6}{'2008':>6}{'2020':>6}")
for sp in (0.35,0.40,0.45):
    for re in (0.20,0.25):
        c,s2,md,tm,fr,do,d87,d00,d08,d20=headfake(sp,re,0.0)
        print(f"{f'>{sp}/<{re}':<16}{c*100:6.1f}%{s2:6.2f}{md*100:6.0f}%{'$'+format(tm,',.0f'):>8}{fr:>6}{do:>6} | {d87*100:5.0f}%{d00*100:5.0f}%{d08*100:5.0f}%{d20*100:5.0f}%")

print('\n(base for reference: 20.2% CAGR, 0.76 Sharpe, -42% MaxDD, ~$1,771)')
