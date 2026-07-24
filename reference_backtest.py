#!/usr/bin/env python3
"""
QQQ TACTICAL ACTION — REFERENCE BACKTEST (Rev 3.0)
==================================================
Self-contained reproduction of the locked strategy from published data files.
No hidden state, no external calls. Every convention is stated explicitly below.

INPUTS (both public, from https://marty21gt.github.io/fragility-monitor/):
  data.json      -> timeline_qqq: dates, V, T, px, ma, pos, stret, bhqqq, bhret
  daily_vt.json  -> dates, Vd, Td   (daily V/T nowcast; only Td is used)

RUN:  python reference_backtest.py data.json daily_vt.json

EXPECTED OUTPUT (1986-01-02 .. 2026-07-23, total return, 5 bps):
  Integrated tv0.20 :  CAGR 19.2%   Sharpe 0.77   MaxDD -42%
  Monitor alone     :  CAGR 17.8%   Sharpe 0.77   MaxDD -40%
  Buy & hold QQQ    :  CAGR 14.9%   Sharpe 0.56   MaxDD -83%

REV 3.0 — two corrections found by independent audit (2026-07):
  1. EXECUTION LAG. The published `pos` is the position HELD on day t, already
     executed; it equals our raw state-machine signal shifted one day. Rev 2.x
     lagged it a SECOND time. Verified: on position-change days, same-day `pos`
     reproduces the published `stret` with MAE 9.4e-06; lagging again gives
     1.17e-02. This module now lags exactly once, at the composite target.
  2. RETURN BASIS. `px` is a PRICE series; the published `bhqqq` benchmark is
     TOTAL return (~0.60%/yr of dividends). Rev 2.x compared a price-return
     strategy against a total-return benchmark. Both are now total return.
  Together these understated the strategy by ~1.0% CAGR.
"""
import json, sys
import numpy as np, pandas as pd

# ============================================================================
# LOCKED PARAMETERS
# ============================================================================
V_THRESHOLD      = 0.54   # vulnerability: danger requires V >= this
T_THRESHOLD      = 0.70   # trigger:       danger requires T >= this
MA_WINDOW        = 200    # trend gate (the 'ma' field is precomputed at 200d)
REENTRY_DAYS     = 15     # consecutive closes above MA required to re-enter
TARGET_VOL       = 0.20   # leverage sizing numerator
LEVERAGE_CAP     = 2.00   # max effective exposure (QLD is 2x)
LEVERAGE_FLOOR   = 1.00   # never de-risk below fully invested via the overlay
VOL_WINDOW       = 20     # trading days for realized vol
VOL_FLOOR        = 0.10   # annualized; prevents divide-by-tiny
SLOPE_WINDOW     = 21     # trading days for MA slope regression
SLOPE_DEADBAND   = 0.03   # annualized fractional slope required to lever
DAILY_T_CUT      = 0.60   # daily-T at/above this -> leverage OFF
DAILY_T_RESTORE  = 0.50   # daily-T below this    -> leverage back ON (hysteresis)
REBALANCE_BAND   = 0.10   # only trade when target exposure moves > this
QLD_EXPENSE      = 0.0095 # 0.95% annual, used for the synthetic pre-2006 QLD

# ============================================================================
# 1. LOAD
# ============================================================================
def load(data_path, vt_path):
    d  = json.load(open(data_path));  tq = d['timeline_qqq']
    dv = json.load(open(vt_path))
    idx = pd.to_datetime(tq['dates'])
    df = pd.DataFrame({
        'px'   : pd.Series(tq['px'],   idx, dtype=float),  # QQQ/NDX price level
        'ma'   : pd.Series(tq['ma'],   idx, dtype=float),  # 200-day MA of px
        'V'    : pd.Series(tq['V'],    idx, dtype=float),  # monthly-frozen vulnerability
        'T'    : pd.Series(tq['T'],    idx, dtype=float),  # monthly-frozen trigger
        'pos'  : pd.Series(tq['pos'],  idx, dtype=float),  # published monitor state (0/1)
        'stret': pd.Series(tq['stret'],idx, dtype=float),  # published monitor daily return
        'bhqqq': pd.Series(tq['bhqqq'],idx, dtype=float),  # QQQ buy&hold TOTAL return
    })
    df['Td'] = pd.Series(dv['Td'], pd.to_datetime(dv['dates']), dtype=float)\
                 .reindex(idx).ffill()                     # daily trigger nowcast
    return df

# ============================================================================
# 2. CASH RATE
#    The published feed has no explicit T-bill series. On days the monitor is
#    risk-off, its strategy return IS the cash return, so we recover the rate
#    from those days and forward-fill across risk-on stretches.
#    LIMITATION: risk-on stretches inherit the last risk-off rate. Over 1986-2026
#    the model is in cash ~15% of days, so the fill is coarse. Sensitivity: using
#    a flat 2% instead changes CAGR by <0.1%.
# ============================================================================
def cash_rate(df):
    c = df['stret'].where(df['pos'] == 0)
    return c.ffill().bfill().fillna(0.02/252)

# ============================================================================
# 3. MONITOR STATE MACHINE
#    Acting gate = PRIOR completed month's V/T (month-start value, shifted 1 month).
#    Exit  : danger (V>=0.54 AND T>=0.70) AND close < 200d MA        -> 1-day rule
#    Re-enter: REENTRY_DAYS consecutive closes >= MA
#    Returns the SIGNAL generated at close t. The published `pos` equals this
#    series shifted one day (verified 100.00%), i.e. `pos` is already executed.
#    The single execution lag is applied once, to the composite target, in main().
# ============================================================================
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
            if cnt >= REENTRY_DAYS:
                off = False; cnt = 0
        else:
            if danger[i] and below[i]:
                off = True; cnt = 0
        pos[i] = 0.0 if off else 1.0
    return pd.Series(pos, df.index)    # SIGNAL at close t (pos_published[t+1])

# ============================================================================
# 4. SIGNALS FOR THE LEVERAGE OVERLAY
# ============================================================================
def realized_vol(px):
    lr = np.log(px / px.shift(1))
    return np.sqrt((lr**2).rolling(VOL_WINDOW).mean() * 252).clip(lower=VOL_FLOOR)

def ma_slope(ma):
    """OLS slope of the MA over SLOPE_WINDOW days, annualized and normalized by level."""
    x = np.arange(SLOPE_WINDOW); xd = x - x.mean(); den = (xd**2).sum()
    return ma.rolling(SLOPE_WINDOW).apply(
        lambda w: (xd * (w - w.mean())).sum() / den, raw=True) * 252 / ma

def leverage_gate(Td):
    """Debounced daily-T gate: OFF at >=0.60, back ON only below 0.50."""
    on = True; out = np.ones(len(Td), dtype=bool)
    for i, t in enumerate(Td.values):
        if on and t >= DAILY_T_CUT:        on = False
        elif (not on) and t < DAILY_T_RESTORE: on = True
        out[i] = on
    return pd.Series(out, Td.index)

# ============================================================================
# 5. TARGET EXPOSURE + REBALANCE BAND
#    exposure = 0                          if monitor risk-off
#             = clip(tv/vol, 1.0, 2.0)     if gate ON and px>ma and slope>deadband
#             = 1.0                        otherwise
# ============================================================================
def apply_band(target, width=REBALANCE_BAND):
    held = []; cur = 0.0
    for t in np.asarray(target):
        if abs(t - cur) > width: cur = t
        held.append(cur)
    return pd.Series(held, target.index)

def target_exposure(df, pos):
    size  = (TARGET_VOL / realized_vol(df['px'])).clip(lower=LEVERAGE_FLOOR, upper=LEVERAGE_CAP)
    can   = leverage_gate(df['Td']) & (df['px'] > df['ma']) & (ma_slope(df['ma']) > SLOPE_DEADBAND)
    raw   = pd.Series(np.where(can, size, 1.0), df.index).where(pos == 1, 0.0)
    return apply_band(raw)

# ============================================================================
# 6. RETURN CONSTRUCTION
#    Exposure E is delivered WITHOUT MARGIN as a QQQ/QLD/SGOV blend:
#        E == 0        -> 100% cash
#        0 <  E <= 1    -> E in QQQ, (1-E) in cash        [only E=1 occurs in practice]
#        1 <  E <= 2    -> (2-E) in QQQ, (E-1) in QLD, 0 cash
#    Weights always sum to 1.0 -> no borrowing.
#    QLD daily return is synthetic throughout for consistency:
#        qld = 2*qqq - (expense + cash_annualized)/252
#    (validated at 0.9998 return-correlation vs real QLD post-2006 inception)
#    Held exposure is shifted one more day: signal at close t -> traded at close t+1.
# ============================================================================
def strategy_returns(df, held, cash, eq_ret=None, cost_bps=0.0):
    """`held` is the position ALREADY held (post-execution-lag). eq_ret defaults
    to TOTAL return (published bhqqq) so the strategy and benchmark match."""
    qqq = df['bhqqq'].fillna(0) if eq_ret is None else eq_ret
    qld = 2*qqq - (QLD_EXPENSE + cash*252)/252
    e   = held                                        # already lagged by caller
    w_qld  = (e - 1).clip(lower=0)
    w_qqq  = pd.Series(np.where(e > 1, 2 - e, e), df.index).clip(lower=0)
    w_cash = (1 - w_qqq - w_qld).clip(lower=0)
    assert (abs(w_qqq + w_qld + w_cash - 1) < 1e-9).all(), "weights must sum to 1"
    r = (w_qqq*qqq + w_qld*qld + w_cash*cash).fillna(0)
    if cost_bps:
        r = r - e.diff().abs().fillna(0) * (cost_bps/10000.0)
    return r

# ============================================================================
# 7. METRICS
#    CAGR   : geometric, 252 trading days per year
#    Sharpe : (mean EXCESS return over the cash series * 252) / (stdev * sqrt(252))
#    MaxDD  : min of equity/cummax - 1, on the daily compounded series
# ============================================================================
def metrics(r, cash):
    r = r.dropna(); n = len(r); eq = (1+r).cumprod()
    cagr = eq.iloc[-1]**(252/n) - 1
    vol  = r.std()*np.sqrt(252)
    exc  = r - cash.reindex(r.index).fillna(0)
    return dict(CAGR=cagr, Sharpe=(exc.mean()*252)/vol, Vol=vol,
                MaxDD=(eq/eq.cummax() - 1).min(), Terminal=eq.iloc[-1])

# ============================================================================
# MAIN
# ============================================================================
def main(data_path='data.json', vt_path='daily_vt.json', cost_bps=5.0):
    df   = load(data_path, vt_path)
    cash = cash_rate(df)
    sig  = monitor_position(df)                 # signal at close t

    # --- integrity check: signal shifted once must equal the published `pos` ---
    agree = (sig.shift(1).fillna(1.0) == df['pos']).mean()
    print(f"state-machine signal, shifted once, vs published `pos`: {agree*100:.2f}%"
          f"  ({int((sig.shift(1).fillna(1.0) != df['pos']).sum())} of {len(sig)} days differ)")

    target = target_exposure(df, sig)           # composite target at close t
    held   = target.shift(1).fillna(0)          # THE single execution lag
    mon_h  = sig.shift(1).fillna(1.0)           # monitor alone, same convention

    r_int = strategy_returns(df, held,  cash, cost_bps=cost_bps)
    r_mon = strategy_returns(df, mon_h, cash, cost_bps=cost_bps)
    r_bh  = df['bhqqq'].fillna(0)

    print(f"\nsample: {df.index[0].date()} .. {df.index[-1].date()}  ({len(df)} days)"
          f"   basis: TOTAL return   costs: {cost_bps:.0f} bps/one-way")
    print(f"{'strategy':<24}{'CAGR':>8}{'Sharpe':>8}{'Vol':>8}{'MaxDD':>8}{'$1 ->':>10}")
    for name, r in [('Integrated tv0.20', r_int), ('Monitor alone', r_mon), ('Buy & hold QQQ', r_bh)]:
        m = metrics(r, cash)
        print(f"{name:<24}{m['CAGR']*100:7.1f}%{m['Sharpe']:8.2f}{m['Vol']*100:7.1f}%"
              f"{m['MaxDD']*100:7.0f}%{'$'+format(m['Terminal'],',.0f'):>10}")

    flips = int((held.diff().abs() > 1e-9).sum())
    rt    = int(((held == 0).astype(int).diff() == 1).sum())
    print(f"\ntrades: {flips} ({flips/(len(df)/252):.1f}/yr)   risk-off round-trips: {rt}"
          f"   time in cash: {(held==0).mean()*100:.0f}%   levered: {(held>1).mean()*100:.0f}%")

    def wdd(r, a, b):
        s = r.loc[a:b]; eq = (1+s).cumprod(); return (eq/eq.cummax()-1).min()
    print(f"\n{'regime':<16}{'integrated':>12}{'buy & hold':>12}")
    for lbl, a, b in [('Dot-com','2000-01-01','2002-12-31'), ('GFC','2007-10-01','2009-06-30'),
                      ('COVID','2020-02-01','2020-04-15'), ('2022','2022-01-01','2022-12-31')]:
        print(f"{lbl:<16}{wdd(r_int,a,b)*100:11.0f}%{wdd(r_bh,a,b)*100:11.0f}%")

if __name__ == '__main__':
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else ()))
