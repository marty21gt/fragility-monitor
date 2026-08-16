#!/usr/bin/env python3
"""
QQQ TACTICAL ACTION — REFERENCE BACKTEST (Rev 3.3 · TQQQ hot + fast-crash overlay)
==================================================================================
Self-contained reproduction of the strategy from the published data files, and
the single source of truth for the v3.3 dashboard's backtest chart.

Rev 3.3 == Rev 3.2, UNCHANGED, plus one thin reactive layer — a fast-crash
emergency overlay:

    * EXIT fully to SGOV on a -8% decline over 3 trading sessions, executed at
      that day's close (the live pipeline decides on the 3:45pm price).
    * RE-ENTER to the base exposure 35 trading days later. Because the base
      exposure is already 0 whenever the V/T monitor is risk-off, the delayed
      re-entry DEFERS to the monitor: if the monitor is still risk-off at day 35,
      the position simply stays in cash until the monitor itself clears.

The monitor, vol-target, leverage gate, rebalance band, vehicle math, expense
model and single execution lag are all IDENTICAL to Rev 3.2. The overlay closes
v3.2's one weakness — a fast crash the smooth vol-scaling can't get ahead of —
without touching anything else. Validated out-of-sample on the S&P 1928-1985
(incl. the Great Depression) and across a 25-55 trading-day re-entry plateau.

  track  vehicle  mult  target_vol  cap    expense   overlay
  v3.0   QLD      2x    0.20        2.0    0.95%     none         (real-money anchor)
  v3.1   TQQQ     3x    0.20        2.0    0.82%     none         (matched net exposure)
  v3.2   TQQQ     3x    0.25        3.0    0.82%     none         (hot; floor -> 2.5x)
  v3.3   TQQQ     3x    0.25        3.0    0.82%     -8%/3d, 35d  (v3.2 + fast-crash exit)

INPUTS (both public, from https://marty21gt.github.io/fragility-monitor/):
  data.json      -> timeline_qqq: dates, V, T, px, ma, pos, stret, bhqqq
  daily_vt.json  -> dates, Td            (daily trigger nowcast; only Td used)

RUN:
  python reference_backtest_v33.py data.json daily_vt.json
      -> prints the v3.0 / v3.1 / v3.2 / v3.3 / buy-&-hold comparison table
  python reference_backtest_v33.py data.json daily_vt.json --patch index.html
      -> regenerates the v3.3 chart (t, e arrays) inside index.html in place

LEVERAGED-SLEEVE MODEL (synthetic throughout, for consistency across history):
      lev = mult*qqq - (expense + (mult-1)*cash_annualized)/252
  i.e. mult x the QQQ total return, less the daily expense, less financing on
  the (mult-1)x borrowed notional at the recovered cash rate. For mult=2 this is
  exactly the Rev 3.0 QLD model (validated at 0.9998 return-corr vs real QLD).
  Delivered WITHOUT MARGIN as a QQQ / lev-ETF / SGOV blend that sums to 1.0:
      E == 0        -> 100% cash (SGOV)          <- also the emergency-exit state
      0 <  E <= 1   -> E in QQQ, (1-E) in cash
      1 <  E <= cap -> weight (E-1)/(mult-1) in the lev ETF, remainder in QQQ

NOTE ON THE 0.82% TQQQ EXPENSE: that is the *net* ratio, held below the 0.97%
gross by a 0.15% ProShares fee waiver scheduled through ~2026-09-30. Applied as
a constant across synthetic pre-2010 history it is mildly optimistic; set
LEV_EXPENSE = 0.0097 for the conservative gross-cost sensitivity.
"""
import json, sys, os, re
import numpy as np, pandas as pd

# ============================================================================
# LOCKED PARAMETERS (defaults = v3.3 TQQQ hot; env overrides match alpaca_exec)
# ============================================================================
V_THRESHOLD      = 0.54
T_THRESHOLD      = 0.70
REENTRY_DAYS     = 15
VEHICLE          = os.environ.get("VEHICLE", "TQQQ")
MULT             = 3                      # TQQQ sleeve multiple (QLD = 2)
LEV_EXPENSE      = 0.0082                  # TQQQ net expense (QLD = 0.0095)
TARGET_VOL       = float(os.environ.get("TARGET_VOL", 0.25))
LEVERAGE_CAP     = float(os.environ.get("LEVERAGE_CAP", 3.0))   # floor binds -> 2.5x eff
LEVERAGE_FLOOR   = 1.00
VOL_WINDOW       = 20
VOL_FLOOR        = 0.10
SLOPE_WINDOW     = 21
SLOPE_DEADBAND   = 0.03
DAILY_T_CUT      = 0.60
DAILY_T_RESTORE  = 0.50
REBALANCE_BAND   = 0.10
COST_BPS         = 5.0

# --- v3.3 fast-crash emergency overlay -------------------------------------
EMERGENCY_ENABLED  = True
EMERGENCY_DROP     = float(os.environ.get("EMERGENCY_DROP", 0.08))   # -8% ...
EMERGENCY_SESSIONS = int(os.environ.get("EMERGENCY_SESSIONS", 3))    # ... over 3 sessions
EMERGENCY_REENTRY  = int(os.environ.get("EMERGENCY_REENTRY", 35))    # trading days out, then defer to monitor

# ============================================================================
# 1. LOAD
# ============================================================================
def load(data_path, vt_path):
    d  = json.load(open(data_path)); tq = d['timeline_qqq']
    dv = json.load(open(vt_path))
    idx = pd.to_datetime(tq['dates'])
    df = pd.DataFrame({
        'px'   : pd.Series(tq['px'],   idx, dtype=float),
        'ma'   : pd.Series(tq['ma'],   idx, dtype=float),
        'V'    : pd.Series(tq['V'],    idx, dtype=float),
        'T'    : pd.Series(tq['T'],    idx, dtype=float),
        'pos'  : pd.Series(tq['pos'],  idx, dtype=float),
        'stret': pd.Series(tq['stret'],idx, dtype=float),
        'bhqqq': pd.Series(tq['bhqqq'],idx, dtype=float),
    })
    df['Td'] = pd.Series(dv['Td'], pd.to_datetime(dv['dates']), dtype=float)\
                 .reindex(idx).ffill()
    return df

# ============================================================================
# 2. CASH RATE  (recovered from risk-off days, forward-filled)
# ============================================================================
def cash_rate(df):
    return df['stret'].where(df['pos'] == 0).ffill().bfill().fillna(0.02/252)

# ============================================================================
# 3. MONITOR STATE MACHINE  (identical across all tracks)
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
    return pd.Series(pos, df.index)

# ============================================================================
# 4. LEVERAGE-OVERLAY SIGNALS
# ============================================================================
def realized_vol(px):
    lr = np.log(px / px.shift(1))
    return np.sqrt((lr**2).rolling(VOL_WINDOW).mean() * 252).clip(lower=VOL_FLOOR)

def ma_slope(ma):
    x = np.arange(SLOPE_WINDOW); xd = x - x.mean(); den = (xd**2).sum()
    return ma.rolling(SLOPE_WINDOW).apply(
        lambda w: (xd * (w - w.mean())).sum() / den, raw=True) * 252 / ma

def leverage_gate(Td):
    on = True; out = np.ones(len(Td), dtype=bool)
    for i, t in enumerate(Td.values):
        if on and t >= DAILY_T_CUT:            on = False
        elif (not on) and t < DAILY_T_RESTORE: on = True
        out[i] = on
    return pd.Series(out, Td.index)

# ============================================================================
# 5. TARGET EXPOSURE + REBALANCE BAND  (v3.2 base)  +  v3.3 EMERGENCY OVERLAY
# ============================================================================
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

def emergency_overlay(px, base, drop=EMERGENCY_DROP, sessions=EMERGENCY_SESSIONS,
                      reentry=EMERGENCY_REENTRY):
    """v3.3 fast-crash overlay. Exit fully to SGOV (exposure 0) the day a `drop`
    (default -8%) decline over `sessions` (default 3) trading days completes, then
    re-enter to `base` exactly `reentry` (default 35) trading days later. Since
    `base` is already 0 whenever the monitor is risk-off, the delayed re-entry
    defers to the monitor. Same-close execution: the exposure change lands on the
    trigger day itself; the caller's single .shift(1) is the one execution lag."""
    trig = (px / px.shift(sessions) - 1 <= -drop).fillna(False).values
    e    = np.asarray(base, dtype=float)
    out  = np.empty(len(e)); in_em = False; entry = 0
    for i in range(len(e)):
        if in_em:
            if (i - entry) >= reentry:            # served the delay -> hand back to base
                in_em = False
        elif trig[i]:                             # fast crash -> emergency exit
            in_em = True; entry = i
        out[i] = 0.0 if in_em else e[i]
    return pd.Series(out, base.index)

def target_v33(df, pos, target_vol=TARGET_VOL, cap=LEVERAGE_CAP, emergency=EMERGENCY_ENABLED):
    """The published v3.3 target exposure: the v3.2 vol-scaled target with the
    fast-crash overlay applied on top (set emergency=False to recover pure v3.2)."""
    base = target_exposure(df, pos, target_vol, cap)
    return emergency_overlay(df['px'], base) if emergency else base

# ============================================================================
# 6. RETURN CONSTRUCTION  (generalized leveraged sleeve)
# ============================================================================
def strategy_returns(df, held, cash, mult=MULT, expense=LEV_EXPENSE,
                     cost_bps=COST_BPS, tqqq_actual=None):
    qqq = df['bhqqq'].fillna(0)
    syn = mult*qqq - (expense + (mult - 1)*cash*252)/252   # synthetic sleeve
    if tqqq_actual is not None:
        # use the fund's REALIZED daily total return where it exists (post-inception),
        # fall back to the synthetic model before the ETF listed. actual returns already
        # embed the real expense + financing, so nothing is subtracted from them.
        real = tqqq_actual.reindex(df.index)
        lev  = real.where(real.notna(), syn)
        _n = int(real.notna().sum())
        strategy_returns._actual_days = _n
    else:
        lev = syn
    e   = held                                            # already lagged by caller
    w_lev  = ((e - 1) / (mult - 1)).clip(lower=0)
    w_qqq  = pd.Series(np.where(e > 1, 1 - (e - 1)/(mult - 1), e), df.index).clip(lower=0)
    w_cash = (1 - w_qqq - w_lev).clip(lower=0)
    assert (abs(w_qqq + w_lev + w_cash - 1) < 1e-9).all(), "weights must sum to 1"
    r = (w_qqq*qqq + w_lev*lev + w_cash*cash).fillna(0)
    if cost_bps:
        r = r - e.diff().abs().fillna(0) * (cost_bps/10000.0)
    return r

# ============================================================================
# 7. METRICS
# ============================================================================
def metrics(r, cash):
    r = r.dropna(); n = len(r); eq = (1 + r).cumprod()
    cagr = eq.iloc[-1]**(252/n) - 1
    vol  = r.std()*np.sqrt(252)
    exc  = r - cash.reindex(r.index).fillna(0)
    return dict(CAGR=cagr, Sharpe=(exc.mean()*252)/vol,
                MaxDD=(eq/eq.cummax() - 1).min(), Terminal=eq.iloc[-1])

def run_track(df, sig, cash, target_vol, cap, mult, expense, tqqq_actual=None,
              emergency=False):
    """emergency=False -> pure v3.2 base (used for the v3.0/3.1/3.2 rows).
       emergency=True  -> v3.3 (base + fast-crash overlay)."""
    target = target_v33(df, sig, target_vol, cap, emergency=emergency)
    held   = target.shift(1).fillna(0)                    # the single execution lag
    r      = strategy_returns(df, held, cash, mult, expense, tqqq_actual=tqqq_actual)
    return held, (1 + r).cumprod(), metrics(r, cash)

def load_tqqq_returns(path):
    """Daily total-return series from a TQQQ CSV. Accepts Stooq/Yahoo layout;
    prefers an 'Adj Close' column (dividend-adjusted) and falls back to 'Close'.
    TQQQ listed 2010-02-11, so returns exist only from then on."""
    t = pd.read_csv(path)
    t.columns = [c.strip().lower() for c in t.columns]
    date_col = next(c for c in t.columns if c.startswith('date'))
    price_col = ('adj close' if 'adj close' in t.columns
                 else 'adj_close' if 'adj_close' in t.columns else 'close')
    s = pd.Series(t[price_col].astype(float).values,
                  pd.to_datetime(t[date_col])).sort_index()
    s = s[~s.index.duplicated(keep='last')]
    return s.pct_change()                                 # daily total return

# ============================================================================
# 8. PAGE ARRAYS + HTML PATCH
#    Only the two strategy-dependent arrays change vs the baked page: the
#    tactical equity `t` (rebased to 1.0 at the window start, EXP convention)
#    and the exposure `e` (stored as x100 integers). The benchmarks (q, s),
#    the SGOV on/off flags, V/T and the year map are strategy-independent and
#    are reused byte-for-byte, so alignment is verified against EXP.q / EXP.off.
#    (v3.3: `off` now includes emergency-exit days, since exposure is 0 there.)
# ============================================================================
def _extract_exp(html):
    i = html.find('const EXP='); j = html.find('{', i); depth = 0; k = j
    while k < len(html):
        c = html[k]
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0: break
        k += 1
    return j, k, json.loads(html[j:k+1])

def patch_html(df, sig, cash, html_path, tqqq_actual=None, emergency=EMERGENCY_ENABLED):
    held, eq, _ = run_track(df, sig, cash, TARGET_VOL, LEVERAGE_CAP, MULT, LEV_EXPENSE,
                            tqqq_actual=tqqq_actual, emergency=emergency)
    e_full = held.tolist(); t_full = (eq / eq.iloc[0]).tolist()   # rebase day0 -> 1.0

    html = open(html_path).read()
    j, k, EXP = _extract_exp(html); N = len(EXP['q'])

    # align: reconstruct B&H QQQ equity, find the offset matching EXP.q
    bh = (1 + df['bhqqq'].fillna(0)); qeq = (bh / bh.iloc[0]).cumprod().tolist()
    best_a, best_err = 0, 1e9
    for a in range(0, 13):
        seg = qeq[a:a+N]
        if len(seg) < N: break
        err = max(abs(seg[m]/seg[0] - EXP['q'][m]) for m in range(0, N, 500))
        if err < best_err: best_a, best_err = a, err
    a = best_a
    off_seg = [1 if x == 0 else 0 for x in e_full[a:a+N]]
    off_agree = sum(1 for m in range(N) if off_seg[m] == EXP['off'][m]) / N
    print(f"align offset a={a}  q-recon err={best_err:.3g}  off-agreement={off_agree*100:.2f}% "
          f"(v3.3 off includes emergency days; agreement < 100% is expected)")

    sig4 = lambda x: float(f"{x:.4g}")
    EXP['t'] = [sig4(v/t_full[a] if a else v) for v in t_full[a:a+N]]
    EXP['e'] = [int(round(v*100)) for v in e_full[a:a+N]]
    # v3.3: refresh the SGOV shading to the actual exposure-0 days (monitor OR emergency)
    EXP['off'] = [1 if v == 0 else 0 for v in e_full[a:a+N]]

    html = html[:j] + json.dumps(EXP, separators=(',', ':')) + html[k+1:]
    html = re.sub(r'\s*<div id="btVersionWarn".*?</div>\n', '\n', html, flags=re.S)
    open(html_path, 'w').write(html)
    print(f"patched {html_path}: t[0]={EXP['t'][0]} t[-1]={EXP['t'][-1]} e_max={max(EXP['e'])} "
          f"n={len(EXP['t'])} sgov_days={sum(EXP['off'])}")

# ============================================================================
# MAIN
# ============================================================================
def main(data_path='data.json', vt_path='daily_vt.json', tqqq_actual=None):
    df   = load(data_path, vt_path)
    cash = cash_rate(df)
    sig  = monitor_position(df)
    agree = (sig.shift(1).fillna(1.0) == df['pos']).mean()
    print(f"signal shifted once vs published pos: {agree*100:.2f}%  "
          f"sample {df.index[0].date()}..{df.index[-1].date()} ({len(df)} days, total return, {COST_BPS:.0f} bps)")
    if tqqq_actual is not None:
        n_act = int(tqqq_actual.reindex(df.index).notna().sum())
        print(f"TQQQ sleeve: actual returns on {n_act} days ({tqqq_actual.dropna().index.min().date()}+), "
              f"synthetic on the {len(df)-n_act} prior")

    # v3.1/v3.2/v3.3 use the actual TQQQ sleeve where available; v3.0 is QLD (synthetic only).
    # (name, target_vol, cap, mult, expense, tqqq_actual, emergency)
    tracks = [("v3.0 QLD  tv0.20 2.0x",       0.20, 2.0, 2, 0.0095, None,        False),
              ("v3.1 TQQQ tv0.20 2.0x",       0.20, 2.0, 3, 0.0082, tqqq_actual, False),
              ("v3.2 TQQQ tv0.25 2.5x",       0.25, 3.0, 3, 0.0082, tqqq_actual, False),
              ("v3.3 TQQQ tv0.25 2.5x +emrg", 0.25, 3.0, 3, 0.0082, tqqq_actual, True)]
    print(f"\n{'track':<30}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'$1 ->':>10}")
    for name, tv, cap, mult, exp, tq, emrg in tracks:
        _, _, m = run_track(df, sig, cash, tv, cap, mult, exp, tqqq_actual=tq, emergency=emrg)
        print(f"{name:<30}{m['CAGR']*100:7.1f}%{m['Sharpe']:8.2f}{m['MaxDD']*100:7.0f}%"
              f"{'$'+format(m['Terminal'],',.0f'):>10}")
    mbh = metrics(df['bhqqq'].fillna(0), cash)
    print(f"{'buy & hold QQQ':<30}{mbh['CAGR']*100:7.1f}%{mbh['Sharpe']:8.2f}"
          f"{mbh['MaxDD']*100:7.0f}%{'$'+format(mbh['Terminal'],',.0f'):>10}")

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    data_path = args[0] if len(args) > 0 else 'data.json'
    vt_path   = args[1] if len(args) > 1 else 'daily_vt.json'
    tqqq = None
    if '--tqqq' in sys.argv:                              # optional actual-TQQQ CSV
        tqqq = load_tqqq_returns(sys.argv[sys.argv.index('--tqqq') + 1])
    if '--patch' in sys.argv:
        target = sys.argv[sys.argv.index('--patch') + 1]
        df = load(data_path, vt_path); cash = cash_rate(df); sig = monitor_position(df)
        patch_html(df, sig, cash, target, tqqq_actual=tqqq)
    else:
        main(data_path, vt_path, tqqq_actual=tqqq)
