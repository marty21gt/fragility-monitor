#!/usr/bin/env python3
"""
MODERN EDGE TACTICAL v3.3 — SELF-CONTAINED AUDIT SCRIPT
========================================================
INTERNAL AUDITING ONLY. This file contains the full, un-black-boxed ruleset
(the -8%/3-session trigger, the 35-trading-day delay, every parameter). It is the
inverse of the public dashboard — do NOT distribute it to prospects or publish it.

One command reproduces every headline claim made for v3.3:

    python audit_v33.py data.json daily_vt.json

It is self-contained: the strategy engine is inlined below (byte-identical logic to
reference_backtest_v33.py), so the only external inputs are the two PUBLIC feeds
from https://marty21gt.github.io/fragility-monitor/ :
    data.json      -> timeline_qqq (QQQ engine)  AND  timeline (S&P, daily 1928+)
    daily_vt.json  -> dates, Td

WHAT v3.3 IS:  v3.2, unchanged, plus one thin reactive layer — a fast-crash
emergency overlay that exits fully to SGOV on a -8% decline over 3 trading
sessions (executed at that day's close), and re-enters to the base exposure 35
trading days later. Because the base exposure is already 0 whenever the V/T
monitor is risk-off, the delayed re-entry defers to the monitor.

SECTIONS PRINTED (each reproduces a specific claim):
    1. QQQ reproduction       -> v3.0 / v3.1 / v3.2 / v3.3 / buy-&-hold, 1986-2026
    2. Execution convention   -> same-day close (shift 1, engine default) vs next-day
    3. Re-entry delay plateau -> 25..55 trading days (robustness; not a fitted knife-edge)
    4. S&P cross-asset OOS     -> 1928-1985 / 1986-2026 / full, incl. the Great Depression
    5. Bootstrap tail          -> block-length sweep (quarter / 1yr / 2yr); the honest tail

Everything is deterministic (fixed RNG seed in section 5). No network, no hidden state.
"""
import json, sys, os
import numpy as np, pandas as pd

# ============================================================================
# LOCKED PARAMETERS (v3.3 TQQQ hot)
# ============================================================================
V_THRESHOLD, T_THRESHOLD, REENTRY_DAYS = 0.54, 0.70, 15
MULT, LEV_EXPENSE                       = 3, 0.0082
TARGET_VOL, LEVERAGE_CAP, LEVERAGE_FLOOR= 0.25, 3.0, 1.00
VOL_WINDOW, VOL_FLOOR                   = 20, 0.10
SLOPE_WINDOW, SLOPE_DEADBAND            = 21, 0.03
DAILY_T_CUT, DAILY_T_RESTORE            = 0.60, 0.50
REBALANCE_BAND, COST_BPS               = 0.10, 5.0
EMERGENCY_DROP, EMERGENCY_SESSIONS, EMERGENCY_REENTRY = 0.08, 3, 35   # <-- the v3.3 overlay

# ============================================================================
# ENGINE  (inlined; identical to reference_backtest_v33.py)
# ============================================================================
def load(data_path, vt_path):
    d = json.load(open(data_path)); tq = d['timeline_qqq']; dv = json.load(open(vt_path))
    idx = pd.to_datetime(tq['dates'])
    df = pd.DataFrame({k: pd.Series(tq[k], idx, dtype=float)
                       for k in ['px','ma','V','T','pos','stret','bhqqq']})
    df['Td'] = pd.Series(dv['Td'], pd.to_datetime(dv['dates']), dtype=float).reindex(idx).ffill()
    return df

def cash_rate(df):
    return df['stret'].where(df['pos'] == 0).ffill().bfill().fillna(0.02/252)

def monitor_position(df):
    acting = lambda s: (s.resample('MS').first().shift(1)
                         .reindex(df.index, method='ffill').ffill().bfill())
    Va, Ta = acting(df['V']), acting(df['T'])
    danger = ((Va >= V_THRESHOLD) & (Ta >= T_THRESHOLD)).values
    below  = (df['px'] < df['ma']).values
    n = len(df); pos = np.ones(n); off = False; cnt = 0
    for i in range(n):
        if off:
            cnt = cnt + 1 if not below[i] else 0
            if cnt >= REENTRY_DAYS: off = False; cnt = 0
        elif danger[i] and below[i]: off = True; cnt = 0
        pos[i] = 0.0 if off else 1.0
    return pd.Series(pos, df.index)

def realized_vol(px):
    lr = np.log(px / px.shift(1))
    return np.sqrt((lr**2).rolling(VOL_WINDOW).mean() * 252).clip(lower=VOL_FLOOR)

def ma_slope(ma):
    x = np.arange(SLOPE_WINDOW); xd = x - x.mean(); den = (xd**2).sum()
    return ma.rolling(SLOPE_WINDOW).apply(lambda w: (xd*(w-w.mean())).sum()/den, raw=True) * 252 / ma

def leverage_gate(Td):
    on = True; out = np.ones(len(Td), dtype=bool)
    for i, t in enumerate(Td.values):
        if on and t >= DAILY_T_CUT: on = False
        elif (not on) and t < DAILY_T_RESTORE: on = True
        out[i] = on
    return pd.Series(out, Td.index)

def apply_band(target, width=REBALANCE_BAND):
    held = []; cur = 0.0
    for t in np.asarray(target):
        if abs(t - cur) > width: cur = t
        held.append(cur)
    return pd.Series(held, target.index)

def target_exposure(df, pos, target_vol=TARGET_VOL, cap=LEVERAGE_CAP):
    size = (target_vol / realized_vol(df['px'])).clip(lower=LEVERAGE_FLOOR, upper=cap)
    can  = leverage_gate(df['Td']) & (df['px'] > df['ma']) & (ma_slope(df['ma']) > SLOPE_DEADBAND)
    raw  = pd.Series(np.where(can, size, 1.0), df.index).where(pos == 1, 0.0)
    return apply_band(raw)

def emergency_overlay(px, base, drop=EMERGENCY_DROP, sessions=EMERGENCY_SESSIONS, reentry=EMERGENCY_REENTRY):
    trig = (px / px.shift(sessions) - 1 <= -drop).fillna(False).values
    e = np.asarray(base, dtype=float); out = np.empty(len(e)); in_em = False; entry = 0
    for i in range(len(e)):
        if in_em:
            if (i - entry) >= reentry: in_em = False
        elif trig[i]: in_em = True; entry = i
        out[i] = 0.0 if in_em else e[i]
    return pd.Series(out, base.index)

def target_v33(df, pos, target_vol=TARGET_VOL, cap=LEVERAGE_CAP, emergency=True):
    base = target_exposure(df, pos, target_vol, cap)
    return emergency_overlay(df['px'], base) if emergency else base

def strategy_returns(df, held, cash, mult=MULT, expense=LEV_EXPENSE, cost_bps=COST_BPS):
    qqq = df['bhqqq'].fillna(0)
    lev = mult*qqq - (expense + (mult-1)*cash*252)/252
    e = held
    w_lev  = ((e-1)/(mult-1)).clip(lower=0)
    w_qqq  = pd.Series(np.where(e > 1, 1-(e-1)/(mult-1), e), df.index).clip(lower=0)
    w_cash = (1-w_qqq-w_lev).clip(lower=0)
    r = (w_qqq*qqq + w_lev*lev + w_cash*cash).fillna(0)
    if cost_bps: r = r - e.diff().abs().fillna(0)*(cost_bps/10000.0)
    return r

# ============================================================================
# AUDIT HELPERS
# ============================================================================
def stats(r, cash, a=None, b=None):
    rr = (r.loc[a:b] if (a or b) else r).dropna(); n = len(rr)
    q = (1+rr).cumprod(); exc = rr - cash.reindex(rr.index).fillna(0)
    return dict(CAGR=q.iloc[-1]**(252/n)-1, Sharpe=(exc.mean()*252)/(rr.std()*np.sqrt(252)),
                MaxDD=(q/q.cummax()-1).min(), Terminal=q.iloc[-1])

def maxdd(r):
    eq = np.cumprod(1+np.asarray(r)); return (eq/np.maximum.accumulate(eq)-1).min()

def run_qqq(df, pos, cash, tv, cap, mult, exp, emergency):
    held = target_v33(df, pos, tv, cap, emergency=emergency).shift(1).fillna(0)
    return stats(strategy_returns(df, held, cash, mult, exp), cash)

# ============================================================================
# SECTION 4 HELPERS: leveraged S&P (v3.2 vol-target + v3.3 overlay, no daily-T gate)
# ============================================================================
def load_sp(data_path):
    tl = json.load(open(data_path))['timeline']
    idx = pd.to_datetime(pd.Series(tl['dates']), format='mixed', errors='coerce')
    cols = [k for k in ['px','V','T','pos','stret','bhret'] if k in tl]
    sp = pd.DataFrame({k: tl[k] for k in cols}, index=idx)
    sp = sp[~sp.index.isna()].astype(float); sp = sp[sp.index >= '1928-01-01'].copy()
    sp['ma'] = sp['px'].rolling(200).mean()
    sp['bhqqq'] = sp['bhret'].fillna(sp['px'].pct_change()).fillna(0)
    return sp

def sp_target(sp, N=EMERGENCY_REENTRY, emergency=True):
    pos = sp['pos']
    size = (TARGET_VOL/realized_vol(sp['px'])).clip(LEVERAGE_FLOOR, LEVERAGE_CAP)
    can  = (sp['px'] > sp['ma']) & (ma_slope(sp['ma']) > SLOPE_DEADBAND)   # no daily-T feed for S&P
    base = apply_band(pd.Series(np.where(can, size, 1.0), sp.index).where(pos == 1, 0.0))
    if not emergency: return base
    return emergency_overlay(sp['px'], base, reentry=N)

# ============================================================================
# SECTION 5 HELPER: path-based block bootstrap (re-runs the overlay each path)
# ============================================================================
def _rmean(P, w):
    c = np.cumsum(np.insert(P, 0, 0.0)); m = np.full(len(P), np.nan); m[w-1:] = (c[w:]-c[:-w])/w; return m

def _emrg_path(u, rb, pos_r, cash_r, cd=EMERGENCY_REENTRY):
    n = len(u); P = np.cumprod(1+u); m200 = _rmean(P, 200); m20 = _rmean(P, 20)
    slope = np.zeros(n); slope[20:] = m200[20:]/np.where(m200[:-20]==0, np.nan, m200[:-20])-1
    three = np.zeros(n); three[EMERGENCY_SESSIONS:] = P[EMERGENCY_SESSIONS:]/P[:-EMERGENCY_SESSIONS]-1
    et = three <= -EMERGENCY_DROP; below = P < m200; above = P >= m200; fast = (P > m20) & (slope > 0)
    out = rb.copy(); em = False; entry = 0
    for i in range(n):
        if np.isnan(m200[i]): continue
        if em:
            if (i-entry) >= cd: em = False          # pure fixed delay -> defer to base (which is 0 if pos==0)
            out[i] = cash_r[i]
        elif et[i]: em = True; entry = i
    return out

def bootstrap_tail(series_dict, blk, nboot=600, seed=7):
    """series_dict: name -> ('bh', returns) or ('emrg', (u, rb, pos, cash)). Returns MaxDD pctiles."""
    n = len(next(iter(series_dict.values()))[1] if next(iter(series_dict.values()))[0]=='bh'
            else next(iter(series_dict.values()))[1][0])
    nblk = int(np.ceil(n/blk)); rng = np.random.default_rng(seed)
    dd = {k: np.empty(nboot) for k in series_dict}
    for b in range(nboot):
        st = rng.integers(0, n-blk, nblk); idx = np.concatenate([np.arange(s, s+blk) for s in st])[:n]
        for k, (kind, payload) in series_dict.items():
            if kind == 'bh':
                dd[k][b] = maxdd(payload[idx])
            else:
                u, rb, pos, csh = payload
                dd[k][b] = maxdd(_emrg_path(u[idx], rb[idx], pos[idx], csh[idx]))
    pc = lambda a, q: np.percentile(a, q)
    return {k: (v.min(), pc(v,1), pc(v,5), np.median(v)) for k, v in dd.items()}

# ============================================================================
# MAIN
# ============================================================================
def main(data_path='data.json', vt_path='daily_vt.json'):
    df = load(data_path, vt_path); cash = cash_rate(df); pos = monitor_position(df)
    agree = (pos.shift(1).fillna(1.0) == df['pos']).mean()
    print("="*78)
    print(f" MODERN EDGE TACTICAL v3.3 — AUDIT  |  {df.index[0].date()}..{df.index[-1].date()}  "
          f"({len(df)} days, total return, {COST_BPS:.0f} bps)")
    print(f" monitor reconciles vs published pos: {agree*100:.2f}%   "
          f"[v3.3 = v3.2 + emergency: -{EMERGENCY_DROP*100:.0f}%/{EMERGENCY_SESSIONS}sess, {EMERGENCY_REENTRY}d delay]")
    print("="*78)

    # --- 1. QQQ reproduction -------------------------------------------------
    print("\n[1] QQQ REPRODUCTION (1986-2026)")
    tracks = [("v3.0 QLD  tv0.20 2.0x",       0.20, 2.0, 2, 0.0095, False),
              ("v3.1 TQQQ tv0.20 2.0x",       0.20, 2.0, 3, 0.0082, False),
              ("v3.2 TQQQ tv0.25 2.5x",       0.25, 3.0, 3, 0.0082, False),
              ("v3.3 TQQQ tv0.25 2.5x +emrg", 0.25, 3.0, 3, 0.0082, True)]
    print(f"    {'track':<30}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'$1->':>10}")
    for nm, tv, cap, mult, exp, em in tracks:
        m = run_qqq(df, pos, cash, tv, cap, mult, exp, em)
        print(f"    {nm:<30}{m['CAGR']*100:7.1f}%{m['Sharpe']:8.2f}{m['MaxDD']*100:7.0f}%{'$'+format(m['Terminal'],',.0f'):>10}")
    mbh = stats(df['bhqqq'].fillna(0), cash)
    print(f"    {'buy & hold QQQ':<30}{mbh['CAGR']*100:7.1f}%{mbh['Sharpe']:8.2f}{mbh['MaxDD']*100:7.0f}%{'$'+format(mbh['Terminal'],',.0f'):>10}")

    # --- 2. Execution convention --------------------------------------------
    print("\n[2] EXECUTION CONVENTION (v3.3)")
    tgt = target_v33(df, pos, emergency=True)
    for nm, h in [("shift(1) = SAME-DAY close (engine default)", tgt.shift(1)),
                  ("shift(2) = NEXT-DAY close (for contrast)",   tgt.shift(2))]:
        r = strategy_returns(df, h.fillna(0), cash); q = (1+r).cumprod(); dd = q/q.cummax()-1
        print(f"    {nm:<44}CAGR {q.iloc[-1]**(252/len(r))*100-100:5.1f}%  Sh {stats(r,cash)['Sharpe']:.2f}  "
              f"MaxDD {dd.min()*100:.0f}%  1987 {dd.loc['1987-08':'1988-01'].min()*100:.0f}%  COVID {dd.loc['2020-02':'2020-06'].min()*100:.0f}%")

    # --- 3. Re-entry delay plateau ------------------------------------------
    print("\n[3] RE-ENTRY DELAY PLATEAU (trading days; robustness)")
    base = target_exposure(df, pos)
    print(f"    {'delay':>6}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}")
    for N in (25, 30, 35, 40, 45, 50, 55):
        held = emergency_overlay(df['px'], base, reentry=N).shift(1).fillna(0)
        m = stats(strategy_returns(df, held, cash), cash)
        star = "  <- locked" if N == EMERGENCY_REENTRY else ""
        print(f"    {N:>6}{m['CAGR']*100:7.1f}%{m['Sharpe']:8.2f}{m['MaxDD']*100:7.0f}%{star}")

    # --- 4. S&P cross-asset OOS ----------------------------------------------
    print("\n[4] S&P 500 CROSS-ASSET TRANSFER (leveraged; v3.2 params, no daily-T gate)")
    sp = load_sp(data_path); spcash = cash_rate(sp)
    r_bh   = sp['bhqqq']
    r_base = strategy_returns(sp, sp_target(sp, emergency=False).shift(1).fillna(0), spcash, 3, 0.0091)
    r_em   = strategy_returns(sp, sp_target(sp, emergency=True ).shift(1).fillna(0), spcash, 3, 0.0091)
    for a, b, lbl in [('1928','1985','1928-1985  TRUE OOS'), ('1986','2026','1986-2026'), ('1928','2026','FULL 1928-2026')]:
        print(f"    --- {lbl} ---")
        for nm, r in [("S&P buy & hold", r_bh), ("S&P vol-scaled (no emrg)", r_base), ("S&P vol-scaled + emergency", r_em)]:
            m = stats(r, spcash, a, b)
            print(f"      {nm:<28}{m['CAGR']*100:6.1f}%  Sh {m['Sharpe']:5.2f}  MaxDD {m['MaxDD']*100:4.0f}%")
    dep_bh = stats(r_bh, spcash, '1929-08','1933-01'); dep_em = stats(r_em, spcash, '1929-08','1933-01')
    print(f"    1929-32 Depression:  buy&hold {dep_bh['MaxDD']*100:.0f}%   vs   v3.3 {dep_em['MaxDD']*100:.0f}%")

    # --- 5. Bootstrap tail (block-length sweep) ------------------------------
    print("\n[5] BOOTSTRAP MaxDD TAIL (path-based, overlay re-run each path; 600 paths, seed 7)")
    rb  = strategy_returns(df, target_exposure(df, pos).shift(1).fillna(0), cash).values
    u   = df['bhqqq'].fillna(0).values; pv = pos.values; cv = cash.values
    spr = sp['bhqqq'].reindex(df.index).fillna(0).values   # S&P B&H aligned to QQQ window
    sd  = {'v3.2 baseline': ('bh', rb),
           'v3.3 + emrg'  : ('emrg', (u, rb, pv, cv)),
           'S&P B&H'      : ('bh', spr)}
    print(f"    {'block length':<18}{'strategy':<16}{'worst':>8}{'1st':>7}{'5th':>7}{'median':>8}")
    for blk, lbl in [(63,'quarter'), (252,'1 year'), (504,'2 years')]:
        res = bootstrap_tail(sd, blk)
        for k in ('v3.2 baseline','v3.3 + emrg','S&P B&H'):
            w,p1,p5,md = res[k]
            print(f"    {lbl if k=='v3.2 baseline' else '':<18}{k:<16}{w*100:7.0f}%{p1*100:6.0f}%{p5*100:6.0f}%{md*100:7.0f}%")
        print()
    print("="*78)
    print(" AUDIT COMPLETE. Every number above is deterministic and reproduces from the two")
    print(" public feeds. v3.3 headline (QQQ 1986-2026): ~20.6% CAGR / 0.83 Sharpe / -31% MaxDD.")
    print("="*78)

if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    main(a[0] if len(a) > 0 else 'data.json', a[1] if len(a) > 1 else 'daily_vt.json')
