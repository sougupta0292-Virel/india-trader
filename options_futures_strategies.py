"""
options_futures_strategies.py
==============================
Chandelier Exit + TUX EMA Scalper (20 EMA) + Supertrend
for Nifty / BankNifty Options & Futures only.

Includes:
  - calc_chandelier_exit()
  - calc_tux_ema()
  - calc_supertrend()
  - scan_all()               — scans 5m + 15m + 1h simultaneously
  - get_straddle_strangle()  — Long/Short Straddle/Strangle recommendation
  - StrategyBacktester       — bar-by-bar backtest, futures & options P&L
  - OptionsFuturesAutoTrader — background thread auto-trader
"""

import os, json, time, threading, math
from datetime import datetime, date, timedelta, timezone, time as dtime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ── Constants ─────────────────────────────────────────────────────────────────

IST = timezone(timedelta(hours=5, minutes=30))

_TICKER_MAP = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
_LOT_SIZES  = {"NIFTY": 75,       "BANKNIFTY": 35,         "SENSEX": 20}
# Strike steps per instrument
_STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100, "SENSEX": 100}

# Expiry rules (post-SEBI 2024 / NSE 2025 schedule)
# Python weekday: 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri
# Monthly futures: NIFTY & BANKNIFTY → last Tuesday; SENSEX → last Friday
_FUT_EXPIRY_DAY = {"NIFTY": 1, "BANKNIFTY": 1, "SENSEX": 4}
# Options: NIFTY weekly → Thursday; BANKNIFTY monthly → Tuesday (weekly discontinued); SENSEX weekly → Friday
_OPT_EXPIRY_DAY = {"NIFTY": 3, "BANKNIFTY": 1, "SENSEX": 4}
# True = weekly options exist; False = only monthly expiry
_OPT_IS_WEEKLY  = {"NIFTY": True, "BANKNIFTY": False, "SENSEX": True}

_TF_YF = {"5m": "5m", "15m": "15m", "1h": "1h"}
# yfinance max lookback for intraday (days)
_TF_DAYS_LIMIT = {"5m": 58, "15m": 58, "1h": 728}

STRATEGIES = ["CHANDELIER", "TUX_SUPERTREND", "CE_REGIME"]
TIMEFRAMES  = ["5m", "15m", "1h"]

# ── Data Fetch ────────────────────────────────────────────────────────────────

def fetch_index_ohlcv(instrument: str, tf: str = "5m", days: int = 5) -> Optional[pd.DataFrame]:
    """Download OHLCV from yfinance. Returns df with lowercase cols or None."""
    ticker_sym = _TICKER_MAP.get(instrument, "^NSEI")
    try:
        tk  = yf.Ticker(ticker_sym)
        df  = tk.history(period=f"{days}d", interval=tf, auto_adjust=True)
        if df is None or len(df) < 10:
            return None
        df = df.rename(columns=str.lower)
        df = df[['open','high','low','close','volume']].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"fetch_index_ohlcv({instrument},{tf}): {e}")
        return None


def fetch_index_ohlcv_range(instrument: str, tf: str, start: date, end: date) -> Optional[pd.DataFrame]:
    """Download OHLCV for a specific date range."""
    ticker_sym = _TICKER_MAP.get(instrument, "^NSEI")
    try:
        tk  = yf.Ticker(ticker_sym)
        df  = tk.history(start=str(start), end=str(end + timedelta(days=1)),
                         interval=tf, auto_adjust=True)
        if df is None or len(df) < 5:
            return None
        df = df.rename(columns=str.lower)
        df = df[['open','high','low','close','volume']].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"fetch_index_ohlcv_range({instrument},{tf}): {e}")
        return None


# ── Indicators ────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    h, l, c = df['high'], df['low'], df['close']
    prev_c  = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Adds columns: tr_adx, plus_dm, minus_dm, plus_di, minus_di, adx."""
    df = df.copy()
    h, l, c = df['high'], df['low'], df['close'].shift(1)
    df['tr_adx']   = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    raw_plus_dm    = (h - h.shift(1)).clip(lower=0)
    raw_minus_dm   = (l.shift(1) - l).clip(lower=0)
    df['plus_dm']  = raw_plus_dm.where(raw_plus_dm > raw_minus_dm, 0)
    df['minus_dm'] = raw_minus_dm.where(raw_minus_dm > raw_plus_dm, 0)
    atr14    = df['tr_adx'].rolling(period).mean()
    plus_di  = 100 * df['plus_dm'].rolling(period).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * df['minus_dm'].rolling(period).mean() / atr14.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['plus_di']  = plus_di.fillna(0)
    df['minus_di'] = minus_di.fillna(0)
    df['adx']      = dx.rolling(period).mean().fillna(0)
    return df


def calc_chandelier_exit(df: pd.DataFrame, period: int = 22, mult: float = 3.0) -> pd.DataFrame:
    """
    Chandelier Exit indicator.
    Adds columns: ce_atr, ce_long_stop, ce_short_stop, ce_dir, ce_signal
    ce_signal: 'LONG' on short→long flip, 'SHORT' on long→short flip, None otherwise.
    """
    df = df.copy()
    df['ce_atr']        = _atr(df, period)
    df['ce_long_stop']  = df['high'].rolling(period).max() - mult * df['ce_atr']
    df['ce_short_stop'] = df['low'].rolling(period).min()  + mult * df['ce_atr']

    ce_dir = [1] * len(df)  # 1=Long bias, -1=Short bias
    for i in range(1, len(df)):
        prev_dir = ce_dir[i - 1]
        price    = float(df['close'].iloc[i])
        ls       = float(df['ce_long_stop'].iloc[i])
        ss       = float(df['ce_short_stop'].iloc[i])
        prev_ls  = float(df['ce_long_stop'].iloc[i - 1])
        prev_ss  = float(df['ce_short_stop'].iloc[i - 1])

        # Update stops: only allow stop to move in direction of bias
        if prev_dir == 1:
            ls = max(ls, prev_ls)
        else:
            ss = min(ss, prev_ss)
        df.at[df.index[i], 'ce_long_stop']  = ls
        df.at[df.index[i], 'ce_short_stop'] = ss

        if price > prev_ss:
            ce_dir[i] = 1
        elif price < prev_ls:
            ce_dir[i] = -1
        else:
            ce_dir[i] = prev_dir

    df['ce_dir'] = ce_dir

    # Signal = direction flip
    prev_dir = pd.Series(ce_dir).shift(1)
    sigs = []
    for i in range(len(ce_dir)):
        if ce_dir[i] == 1 and prev_dir.iloc[i] == -1:
            sigs.append('LONG')
        elif ce_dir[i] == -1 and prev_dir.iloc[i] == 1:
            sigs.append('SHORT')
        else:
            sigs.append(None)
    df['ce_signal'] = sigs
    return df


def calc_tux_ema(df: pd.DataFrame, ema_period: int = 20) -> pd.DataFrame:
    """
    TUX EMA Scalper at 20 EMA.
    Adds columns: tux_ema, tux_signal
    LONG: candle wicks below EMA but closes above EMA, majority of prior 3 bars above EMA
    SHORT: candle wicks above EMA but closes below EMA, majority of prior 3 bars below EMA
    """
    df = df.copy()
    df['tux_ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()

    sigs = [None] * len(df)
    for i in range(3, len(df)):
        ema   = float(df['tux_ema'].iloc[i])
        lo    = float(df['low'].iloc[i])
        hi    = float(df['high'].iloc[i])
        cl    = float(df['close'].iloc[i])
        op    = float(df['open'].iloc[i])

        # Trend filter: count how many of prior 3 bars are above/below EMA
        prior_closes = [float(df['close'].iloc[i - k]) for k in range(1, 4)]
        prior_emas   = [float(df['tux_ema'].iloc[i - k]) for k in range(1, 4)]
        bars_above   = sum(1 for c, e in zip(prior_closes, prior_emas) if c > e)
        bars_below   = sum(1 for c, e in zip(prior_closes, prior_emas) if c < e)

        # LONG: wick touches EMA from above (lo <= ema) but closes above EMA
        if bars_above >= 2 and lo <= ema * 1.001 and cl > ema and cl > op:
            sigs[i] = 'LONG'
        # SHORT: wick touches EMA from below (hi >= ema) but closes below EMA
        elif bars_below >= 2 and hi >= ema * 0.999 and cl < ema and cl < op:
            sigs[i] = 'SHORT'

    df['tux_signal'] = sigs
    return df


def calc_supertrend(df: pd.DataFrame, atr_period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """
    Supertrend indicator.
    Adds columns: st_atr, st_upper, st_lower, st_dir, st_signal
    st_signal: 'LONG' on -1→+1 flip, 'SHORT' on +1→-1 flip, None otherwise.
    """
    df = df.copy()
    df['st_atr'] = _atr(df, atr_period)
    hl2 = (df['high'] + df['low']) / 2
    df['st_upper_raw'] = hl2 + mult * df['st_atr']
    df['st_lower_raw'] = hl2 - mult * df['st_atr']

    st_upper = list(df['st_upper_raw'])
    st_lower = list(df['st_lower_raw'])
    st_dir   = [1] * len(df)

    for i in range(1, len(df)):
        prev_cl = float(df['close'].iloc[i - 1])

        # Lower band (support): can only go up
        if st_lower[i] < st_lower[i - 1]:
            st_lower[i] = st_lower[i - 1]
        # Upper band (resistance): can only go down
        if st_upper[i] > st_upper[i - 1]:
            st_upper[i] = st_upper[i - 1]
        if prev_cl <= st_upper[i - 1]:
            st_lower[i] = st_lower_raw = float(df['st_lower_raw'].iloc[i])

        cl = float(df['close'].iloc[i])
        if cl > st_upper[i - 1]:
            st_dir[i] = 1
        elif cl < st_lower[i - 1]:
            st_dir[i] = -1
        else:
            st_dir[i] = st_dir[i - 1]

    df['st_upper'] = st_upper
    df['st_lower'] = st_lower
    df['st_dir']   = st_dir

    prev_dir = pd.Series(st_dir).shift(1)
    sigs = []
    for i in range(len(st_dir)):
        if st_dir[i] == 1 and prev_dir.iloc[i] == -1:
            sigs.append('LONG')
        elif st_dir[i] == -1 and prev_dir.iloc[i] == 1:
            sigs.append('SHORT')
        else:
            sigs.append(None)
    df['st_signal'] = sigs
    return df


def calc_ce_regime(df: pd.DataFrame, ce_period: int = 22, ce_mult: float = 3.0,
                   adx_period: int = 14, ema_period: int = 50) -> pd.DataFrame:
    """
    Chandelier Exit + Market Regime Filter.
    Adds all CE columns plus: adx, ema50, ema50_slope, atr_pct, regime, ce_regime_signal.
    regime ∈ ('TRENDING', 'MIXED', 'SIDEWAYS')
    ce_regime_signal: CE flip only when regime != SIDEWAYS.
    """
    df = calc_chandelier_exit(df, ce_period, ce_mult)
    df = add_adx(df, adx_period)
    df['ema50']       = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['ema50_slope'] = df['ema50'].pct_change(5).abs()
    df['atr_pct']     = df['ce_atr'] / df['close']

    regimes = []
    for i in range(len(df)):
        adx_v  = float(df['adx'].iloc[i])       if not pd.isna(df['adx'].iloc[i])       else 0.0
        atr_p  = float(df['atr_pct'].iloc[i])   if not pd.isna(df['atr_pct'].iloc[i])   else 1.0
        slope  = float(df['ema50_slope'].iloc[i])if not pd.isna(df['ema50_slope'].iloc[i])else 1.0
        score  = int(adx_v < 20) + int(atr_p < 0.004) + int(slope < 0.0005)
        if adx_v > 25:
            regimes.append('TRENDING')
        elif score >= 2:
            regimes.append('SIDEWAYS')
        else:
            regimes.append('MIXED')
    df['regime'] = regimes

    ce_regime_sigs = []
    for i in range(len(df)):
        if df['ce_signal'].iloc[i] in ('LONG', 'SHORT') and regimes[i] != 'SIDEWAYS':
            ce_regime_sigs.append(df['ce_signal'].iloc[i])
        else:
            ce_regime_sigs.append(None)
    df['ce_regime_signal'] = ce_regime_sigs
    return df


# ── Signal extraction ─────────────────────────────────────────────────────────

def _signal_from_chandelier(df: pd.DataFrame) -> dict:
    """Extract latest Chandelier Exit signal from processed df."""
    df = calc_chandelier_exit(df)
    last_sig_idx = df['ce_signal'].last_valid_index()
    recent_sig   = None
    recent_bars  = 0
    for i in range(len(df) - 1, max(len(df) - 8, -1), -1):
        s = df['ce_signal'].iloc[i]
        if s in ('LONG', 'SHORT'):
            recent_sig  = s
            recent_bars = len(df) - 1 - i
            break

    entry     = float(df['close'].iloc[-1])
    atr_val   = float(df['ce_atr'].iloc[-1])
    cur_dir   = int(df['ce_dir'].iloc[-1])
    long_stop = float(df['ce_long_stop'].iloc[-1])
    short_stop= float(df['ce_short_stop'].iloc[-1])

    if cur_dir == 1:
        direction = 'LONG'
        sl     = round(long_stop, 2)
        target = round(entry + 2 * (entry - sl), 2)
        stop   = sl
    else:
        direction = 'SHORT'
        sl     = round(short_stop, 2)
        target = round(entry - 2 * (sl - entry), 2)
        stop   = sl

    rr = round(abs(target - entry) / abs(entry - stop), 2) if entry != stop else 0

    freshness = "🔴 No recent flip" if recent_sig is None else (
        f"🟢 {recent_sig} flip {recent_bars} bar(s) ago" if recent_bars <= 3 else
        f"⚪ {recent_sig} flip {recent_bars} bars ago"
    )
    return {
        'strategy': 'Chandelier Exit',
        'direction': direction,
        'fresh_signal': recent_sig if recent_bars <= 3 else None,
        'entry': round(entry, 2),
        'sl': stop,
        'target': target,
        'rr': rr,
        'ce_dir': cur_dir,
        'reason': f'CE dir={direction} | stop={stop:.0f} | ATR={atr_val:.1f} | {freshness}',
    }


def _signal_from_tux_ema(df: pd.DataFrame) -> dict:
    """Extract latest TUX EMA Scalper signal."""
    df = calc_tux_ema(df)
    recent_sig  = None
    recent_bars = 0
    for i in range(len(df) - 1, max(len(df) - 8, -1), -1):
        s = df['tux_signal'].iloc[i]
        if s in ('LONG', 'SHORT'):
            recent_sig  = s
            recent_bars = len(df) - 1 - i
            break

    entry = float(df['close'].iloc[-1])
    ema   = float(df['tux_ema'].iloc[-1])
    # Determine current bias from EMA position
    direction = 'LONG' if entry > ema else 'SHORT'
    ema_dist  = abs(entry - ema)

    if direction == 'LONG':
        sl     = round(ema * 0.998, 2)   # just below EMA
        target = round(entry + 1.5 * (entry - sl), 2)
    else:
        sl     = round(ema * 1.002, 2)
        target = round(entry - 1.5 * (sl - entry), 2)

    rr = round(abs(target - entry) / abs(entry - sl), 2) if entry != sl else 0

    freshness = "🔴 No bounce signal" if recent_sig is None else (
        f"🟢 Bounce {recent_sig} {recent_bars} bar(s) ago" if recent_bars <= 2 else
        f"⚪ Bounce {recent_sig} {recent_bars} bars ago"
    )
    return {
        'strategy': 'TUX EMA Scalper (20)',
        'direction': direction,
        'fresh_signal': recent_sig if recent_bars <= 2 else None,
        'entry': round(entry, 2),
        'sl': sl,
        'target': target,
        'rr': rr,
        'ema': round(ema, 2),
        'reason': f'20 EMA={ema:.0f} | Price={entry:.0f} | dist={ema_dist:.0f} | {freshness}',
    }


def _signal_from_tux_supertrend(df: pd.DataFrame) -> dict:
    """
    TUX EMA Scalper + Supertrend confluence.
    Both must agree: Supertrend direction AND TUX EMA bounce in same direction.
    Stronger than either alone — trend direction from Supertrend, entry timing from TUX bounce.
    """
    df = calc_tux_ema(calc_supertrend(df))

    st_dir      = int(df['st_dir'].iloc[-1])
    st_direction = 'LONG' if st_dir == 1 else 'SHORT'
    st_low      = float(df['st_lower'].iloc[-1])
    st_high     = float(df['st_upper'].iloc[-1])
    atr_val     = float(df['st_atr'].iloc[-1])

    # Find most recent TUX bounce that agrees with Supertrend direction
    tux_match_bars = None
    for i in range(len(df) - 1, max(len(df) - 6, -1), -1):
        tux_sig = df['tux_signal'].iloc[i]
        if tux_sig == st_direction:
            tux_match_bars = len(df) - 1 - i
            break

    vol_ma20    = df['volume'].rolling(20).mean().iloc[-1]
    vol_current = float(df['volume'].iloc[-1])
    vol_ok      = (not pd.isna(vol_ma20)) and float(vol_ma20) > 0 and vol_current > float(vol_ma20)

    fresh = tux_match_bars is not None and tux_match_bars <= 3 and vol_ok
    entry = float(df['close'].iloc[-1])

    if st_dir == 1:
        sl     = round(st_low, 2)
        target = round(entry + 2 * (entry - sl), 2)
    else:
        sl     = round(st_high, 2)
        target = round(entry - 2 * (sl - entry), 2)

    rr = round(abs(target - entry) / abs(entry - sl), 2) if entry != sl else 0

    if tux_match_bars is None:
        tux_info = "no TUX bounce"
    elif fresh:
        tux_info = f"TUX bounce {tux_match_bars}bar(s) ago FRESH"
    else:
        tux_info = f"TUX bounce {tux_match_bars}bars ago"

    return {
        'strategy':     'TUX + Supertrend',
        'direction':    st_direction,
        'fresh_signal': st_direction if fresh else None,
        'entry':        round(entry, 2),
        'sl':           sl,
        'target':       target,
        'rr':           rr,
        'st_dir':       st_dir,
        'reason':       f'ST={st_direction} | sl={sl:.0f} | ATR={atr_val:.1f} | VOL={vol_current:.0f}(ma={float(vol_ma20) if not pd.isna(vol_ma20) else 0:.0f}) | {tux_info}',
    }


def _signal_from_ce_regime(df: pd.DataFrame) -> dict:
    """Chandelier Exit filtered by Market Regime. Returns signal dict."""
    df = calc_ce_regime(df)
    recent_sig  = None
    recent_bars = 0
    for i in range(len(df) - 1, max(len(df) - 8, -1), -1):
        s = df['ce_regime_signal'].iloc[i]
        if s in ('LONG', 'SHORT'):
            recent_sig  = s
            recent_bars = len(df) - 1 - i
            break

    entry      = float(df['close'].iloc[-1])
    atr_val    = float(df['ce_atr'].iloc[-1])
    cur_dir    = int(df['ce_dir'].iloc[-1])
    long_stop  = float(df['ce_long_stop'].iloc[-1])
    short_stop = float(df['ce_short_stop'].iloc[-1])
    regime     = str(df['regime'].iloc[-1])
    adx_val    = float(df['adx'].iloc[-1])

    if cur_dir == 1:
        direction = 'LONG'
        sl = round(long_stop, 2)
    else:
        direction = 'SHORT'
        sl = round(short_stop, 2)

    sl_dist = abs(entry - sl)
    target  = round(entry + 2 * sl_dist, 2) if direction == 'LONG' else round(entry - 2 * sl_dist, 2)
    rr      = round(abs(target - entry) / sl_dist, 2) if sl_dist > 0 else 0

    freshness = "🔴 No recent flip" if recent_sig is None else (
        f"🟢 {recent_sig} flip {recent_bars} bar(s) ago" if recent_bars <= 3 else
        f"⚪ {recent_sig} flip {recent_bars} bars ago"
    )
    return {
        'strategy':     'CE + Regime Filter',
        'direction':    direction,
        'fresh_signal': (recent_sig if recent_bars <= 3 and regime != 'SIDEWAYS' else None),
        'entry':        round(entry, 2),
        'sl':           sl,
        'target':       target,
        'rr':           rr,
        'ce_dir':       cur_dir,
        'regime':       regime,
        'reason':       (f'CE dir={direction} | sl={sl:.0f} | ATR={atr_val:.1f} | '
                         f'ADX={adx_val:.1f} | Regime={regime} | {freshness}'),
    }


# ── Multi-TF Scanner ──────────────────────────────────────────────────────────

def scan_all(instrument: str, timeframes: list = None) -> dict:
    """
    Scan all 3 strategies on 5m, 15m, 1h simultaneously.
    Returns dict: {tf: [signal_dict, ...]}
    Each signal_dict has keys: strategy, direction, fresh_signal, entry, sl, target, rr, reason.
    """
    if timeframes is None:
        timeframes = ["5m", "15m", "1h"]

    results = {}
    for tf in timeframes:
        days = min(10, _TF_DAYS_LIMIT[tf])
        df   = fetch_index_ohlcv(instrument, tf, days)
        tf_sigs = []
        if df is not None and len(df) >= 30:
            # Pre-compute ADX for live entry filter
            try:
                df = add_adx(df, period=14)
            except Exception:
                pass
            _adx_val = (float(df['adx'].iloc[-1])
                        if 'adx' in df.columns and not pd.isna(df['adx'].iloc[-1]) else 0.0)

            try:
                _s = _signal_from_chandelier(df)
                _s['adx_val'] = _adx_val
                tf_sigs.append(_s)
            except Exception as e:
                tf_sigs.append({'strategy': 'Chandelier Exit', 'direction': 'FLAT',
                                 'fresh_signal': None, 'entry': 0, 'sl': 0,
                                 'target': 0, 'rr': 0, 'reason': str(e), 'adx_val': 0.0})
            try:
                _s = _signal_from_tux_supertrend(df)
                _s['adx_val'] = _adx_val
                tf_sigs.append(_s)
            except Exception as e:
                tf_sigs.append({'strategy': 'TUX + Supertrend', 'direction': 'FLAT',
                                 'fresh_signal': None, 'entry': 0, 'sl': 0,
                                 'target': 0, 'rr': 0, 'reason': str(e), 'adx_val': 0.0})
            try:
                _s = _signal_from_ce_regime(df)
                _s['adx_val'] = _adx_val
                tf_sigs.append(_s)
            except Exception as e:
                tf_sigs.append({'strategy': 'CE + Regime Filter', 'direction': 'FLAT',
                                 'fresh_signal': None, 'entry': 0, 'sl': 0,
                                 'target': 0, 'rr': 0, 'reason': str(e), 'adx_val': 0.0})
        else:
            for s in ['Chandelier Exit', 'TUX + Supertrend', 'CE + Regime Filter']:
                tf_sigs.append({'strategy': s, 'direction': 'FLAT', 'fresh_signal': None,
                                 'entry': 0, 'sl': 0, 'target': 0, 'rr': 0,
                                 'reason': 'No data'})
        tf_sigs.append({'tf': tf, '_timestamp': datetime.now(IST).strftime('%H:%M:%S IST')})
        results[tf] = tf_sigs

    return results


# ── Straddle / Strangle ───────────────────────────────────────────────────────

def _est_premium(spot: float, strike: int, option_type: str,
                 vix: float, dte: int) -> float:
    """Black-Scholes proxy for option premium."""
    try:
        sigma = vix / 100.0
        T     = max(dte, 0.5) / 365.0
        d1    = (math.log(spot / strike) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        d2    = d1 - sigma * math.sqrt(T)
        from scipy.stats import norm
        if option_type == "CE":
            premium = spot * norm.cdf(d1) - strike * norm.cdf(d2)
        else:
            premium = strike * norm.cdf(-d2) - spot * norm.cdf(-d1)
        # Add intrinsic if deep ITM
        if option_type == "CE" and spot > strike:
            premium = max(premium, spot - strike)
        elif option_type == "PE" and spot < strike:
            premium = max(premium, strike - spot)
        return round(max(premium, 5.0), 2)
    except Exception:
        # Fallback: rough ATM premium ≈ 0.4 × σ × S × √T
        sigma = (vix or 15) / 100.0
        T     = max(dte, 0.5) / 365.0
        return round(max(0.4 * sigma * spot * math.sqrt(T), 5.0), 2)


def _get_vix() -> float:
    """Fetch India VIX from yfinance."""
    try:
        tk = yf.Ticker("^INDIAVIX")
        return float(tk.fast_info.last_price or 15.0)
    except Exception:
        return 15.0


def _last_weekday_of_month(yr: int, mo: int, wd: int) -> date:
    """Return last occurrence of weekday wd (0=Mon…4=Fri) in month mo/yr."""
    import calendar
    last_day = calendar.monthrange(yr, mo)[1]
    d = date(yr, mo, last_day)
    return d - timedelta(days=(d.weekday() - wd) % 7)


def _next_monthly_opt_expiry(instrument: str) -> date:
    """
    For BANKNIFTY (monthly options only): return nearest upcoming monthly expiry
    (last Tuesday). Same logic as futures expiry for BANKNIFTY.
    """
    today = date.today()
    wd = _OPT_EXPIRY_DAY.get(instrument, 1)
    exp = _last_weekday_of_month(today.year, today.month, wd)
    if exp <= today:
        mo = today.month + 1 if today.month < 12 else 1
        yr = today.year if today.month < 12 else today.year + 1
        exp = _last_weekday_of_month(yr, mo, wd)
    return exp


def _get_expiry(instrument: str) -> tuple:
    """
    Get nearest option expiry and DTE.
    NIFTY: next Thursday (weekly) | BANKNIFTY: next monthly Tuesday | SENSEX: next Friday (weekly)
    On expiry day, roll to next cycle after 2:30 PM IST.
    """
    today = date.today()
    if not _OPT_IS_WEEKLY.get(instrument, True):
        # BANKNIFTY — monthly only
        exp = _next_monthly_opt_expiry(instrument)
        dte = (exp - today).days
        return exp.strftime('%d%b%Y').upper(), dte

    target_day = _OPT_EXPIRY_DAY.get(instrument, 3)
    days_ahead = (target_day - today.weekday()) % 7
    if days_ahead == 0:
        now = datetime.now().time()
        if now >= dtime(14, 30):
            days_ahead = 7
    expiry_date = today + timedelta(days=days_ahead)
    return expiry_date.strftime('%d%b%Y').upper(), days_ahead


def get_two_expiries(instrument: str) -> list:
    """
    Return the 2 nearest option expiries.
    Weekly instruments (NIFTY/SENSEX): consecutive Thursdays/Fridays.
    Monthly instruments (BANKNIFTY): current + next monthly Tuesday.
    Returns list of (expiry_str, dte) tuples.
    """
    today = date.today()
    if not _OPT_IS_WEEKLY.get(instrument, True):
        exp1 = _next_monthly_opt_expiry(instrument)
        wd   = _OPT_EXPIRY_DAY.get(instrument, 1)
        mo2  = exp1.month + 1 if exp1.month < 12 else 1
        yr2  = exp1.year if exp1.month < 12 else exp1.year + 1
        exp2 = _last_weekday_of_month(yr2, mo2, wd)
        return [
            (exp1.strftime('%d%b%Y').upper(), (exp1 - today).days),
            (exp2.strftime('%d%b%Y').upper(), (exp2 - today).days),
        ]

    target_day = _OPT_EXPIRY_DAY.get(instrument, 3)
    days1 = (target_day - today.weekday()) % 7
    if days1 == 0:
        now = datetime.now().time()
        if now >= dtime(14, 30):
            days1 = 7
    exp1 = today + timedelta(days=days1)
    exp2 = exp1 + timedelta(days=7)
    return [
        (exp1.strftime('%d%b%Y').upper(), days1),
        (exp2.strftime('%d%b%Y').upper(), days1 + 7),
    ]


def get_three_option_expiries(instrument: str) -> list:
    """
    Return the 3 nearest option expiries.
    Weekly instruments (NIFTY/SENSEX): 3 consecutive weekly dates.
    Monthly instruments (BANKNIFTY): 3 consecutive monthly Tuesdays.
    Returns list of (expiry_str, dte) tuples.
    """
    today = date.today()
    if not _OPT_IS_WEEKLY.get(instrument, True):
        wd   = _OPT_EXPIRY_DAY.get(instrument, 1)
        exp1 = _next_monthly_opt_expiry(instrument)
        exps = [exp1]
        yr, mo = exp1.year, exp1.month
        for _ in range(2):
            mo += 1
            if mo > 12:
                mo, yr = 1, yr + 1
            exps.append(_last_weekday_of_month(yr, mo, wd))
        return [(e.strftime('%d%b%Y').upper(), (e - today).days) for e in exps]

    target_day = _OPT_EXPIRY_DAY.get(instrument, 3)
    days1 = (target_day - today.weekday()) % 7
    if days1 == 0:
        now = datetime.now().time()
        if now >= dtime(14, 30):
            days1 = 7
    exp1 = today + timedelta(days=days1)
    exp2 = exp1 + timedelta(days=7)
    exp3 = exp1 + timedelta(days=14)
    return [
        (exp1.strftime('%d%b%Y').upper(), days1),
        (exp2.strftime('%d%b%Y').upper(), days1 + 7),
        (exp3.strftime('%d%b%Y').upper(), days1 + 14),
    ]


def get_two_futures_expiries(instrument: str) -> list:
    """
    Return 2 monthly futures expiries: near month + far month.
    NIFTY & BANKNIFTY → last Tuesday of month; SENSEX → last Friday.
    Returns list of (expiry_str, dte) tuples.
    """
    today      = date.today()
    wd         = _FUT_EXPIRY_DAY.get(instrument, 1)

    exp1 = _last_weekday_of_month(today.year, today.month, wd)
    if exp1 <= today:
        mo2 = today.month + 1 if today.month < 12 else 1
        yr2 = today.year if today.month < 12 else today.year + 1
        exp1 = _last_weekday_of_month(yr2, mo2, wd)

    next_yr = exp1.year + (1 if exp1.month == 12 else 0)
    next_mo = 1 if exp1.month == 12 else exp1.month + 1
    exp2 = _last_weekday_of_month(next_yr, next_mo, wd)

    return [
        (exp1.strftime('%d%b%Y').upper(), (exp1 - today).days),
        (exp2.strftime('%d%b%Y').upper(), (exp2 - today).days),
    ]


def get_past_n_option_expiries(instrument: str, n: int = 6, offset: int = 0) -> list:
    """
    Return n past option expiry dates (newest first), starting offset cycles back.
    Weekly instruments (NIFTY/SENSEX): weekly Thursday/Friday dates.
    Monthly instruments (BANKNIFTY): monthly Tuesday dates.
    """
    today = date.today()
    if not _OPT_IS_WEEKLY.get(instrument, True):
        # Monthly: collect past last-Tuesday expiries
        wd   = _OPT_EXPIRY_DAY.get(instrument, 1)
        past = []
        mo, yr = today.month, today.year
        while len(past) < offset + n + 2:
            exp = _last_weekday_of_month(yr, mo, wd)
            if exp < today:
                past.append(exp)
            mo -= 1
            if mo == 0:
                mo, yr = 12, yr - 1
        return past[offset: offset + n]

    # Weekly
    target_day = _OPT_EXPIRY_DAY.get(instrument, 3)
    days_since = (today.weekday() - target_day) % 7
    if days_since == 0:
        days_since = 7
    latest_past = today - timedelta(days=days_since)
    return [latest_past - timedelta(weeks=offset + i) for i in range(n)]


def get_past_n_futures_expiries(instrument: str, n: int = 6, offset: int = 0) -> list:
    """
    Return n past monthly futures expiry dates (newest first), starting offset months back.
    NIFTY & BANKNIFTY → last Tuesday; SENSEX → last Friday.
    """
    today = date.today()
    wd    = _FUT_EXPIRY_DAY.get(instrument, 1)
    past  = []
    mo, yr = today.month, today.year
    while len(past) < offset + n + 2:
        exp = _last_weekday_of_month(yr, mo, wd)
        if exp < today:
            past.append(exp)
        mo -= 1
        if mo == 0:
            mo, yr = 12, yr - 1
    return past[offset: offset + n]


def get_futures_expiry(instrument: str) -> tuple:
    """
    Get nearest monthly futures expiry.
    NIFTY & BANKNIFTY → last Tuesday of month; SENSEX → last Friday.
    Returns (expiry_str, dte).
    """
    today = date.today()
    wd    = _FUT_EXPIRY_DAY.get(instrument, 1)
    exp   = _last_weekday_of_month(today.year, today.month, wd)
    if exp <= today:
        mo2 = today.month + 1 if today.month < 12 else 1
        yr2 = today.year if today.month < 12 else today.year + 1
        exp = _last_weekday_of_month(yr2, mo2, wd)
    return exp.strftime('%d%b%Y').upper(), (exp - today).days


def get_all_expiries(instrument: str) -> dict:
    """Return 3 option expiries (near/mid/far) and 2 futures expiries for an instrument."""
    opts = get_three_option_expiries(instrument)
    futs = get_two_futures_expiries(instrument)
    return {
        'opt_near': opts[0],    # (str, dte)
        'opt_mid':  opts[1],
        'opt_far':  opts[2],
        'fut_near': futs[0],
        'fut_far':  futs[1],
    }


def get_straddle_strangle(instrument: str) -> dict:
    """
    Recommend Long/Short Straddle or Strangle based on volatility regime.
    Returns dict with setup, legs, premiums, breakevens, reason.
    """
    spot_sym    = _TICKER_MAP.get(instrument, "^NSEI")
    strike_step = _STRIKE_STEP.get(instrument, 50)

    try:
        tk   = yf.Ticker(spot_sym)
        spot = float(tk.fast_info.last_price)
    except Exception:
        return {'error': 'Cannot fetch spot price'}

    vix  = _get_vix()
    exps = get_two_expiries(instrument)   # [(str, dte), (str, dte)]
    expiry_str, dte = exps[0]
    expiry2_str, dte2 = exps[1]
    atm  = round(spot / strike_step) * strike_step
    otm1c = atm + strike_step
    otm1p = atm - strike_step
    otm2c = atm + 2 * strike_step
    otm2p = atm - 2 * strike_step

    # ATR-based volatility regime (using 5m data)
    df_5m = fetch_index_ohlcv(instrument, "5m", days=5)
    atr_now, atr_avg = 0.0, 0.0
    if df_5m is not None and len(df_5m) >= 30:
        atr_series = _atr(df_5m, 14)
        atr_now    = float(atr_series.iloc[-5:].mean())
        atr_avg    = float(atr_series.iloc[-30:-5].mean()) if len(atr_series) > 30 else atr_now
    atr_ratio  = atr_now / atr_avg if atr_avg > 0 else 1.0
    vol_expand = atr_ratio > 1.1 or vix > 14.0
    vol_pinned = atr_ratio < 0.85

    # Range check (last 5 bars)
    rng_pct = 0.0
    if df_5m is not None and len(df_5m) >= 5:
        tail = df_5m.tail(5)
        rng_pct = float(tail['high'].max() - tail['low'].min()) / spot * 100

    # Premium estimates
    p_atm_ce  = _est_premium(spot, atm,  "CE", vix, dte)
    p_atm_pe  = _est_premium(spot, atm,  "PE", vix, dte)
    p_otm1c   = _est_premium(spot, otm1c,"CE", vix, dte)
    p_otm1p   = _est_premium(spot, otm1p,"PE", vix, dte)
    p_otm2c   = _est_premium(spot, otm2c,"CE", vix, dte)
    p_otm2p   = _est_premium(spot, otm2p,"PE", vix, dte)

    straddle_net = p_atm_ce + p_atm_pe
    strangle_net = p_otm1c + p_otm1p

    # Recommendation logic (relaxed so fires daily)
    if vol_pinned and rng_pct < 0.8:
        setup = "SHORT_STRADDLE"
        net   = -(straddle_net)
        reason= f"Range-bound | VIX={vix:.1f} | ATR ratio={atr_ratio:.2f} | Range={rng_pct:.2f}% | Sell premium"
        legs  = [
            {'type':'SELL','strike':atm,'option_type':'CE','premium':p_atm_ce},
            {'type':'SELL','strike':atm,'option_type':'PE','premium':p_atm_pe},
        ]
        bep_lo = atm - straddle_net
        bep_hi = atm + straddle_net
    elif vol_pinned:
        setup = "SHORT_STRANGLE"
        net   = -(p_otm2c + p_otm2p)
        reason= f"Low vol | VIX={vix:.1f} | ATR ratio={atr_ratio:.2f} | Strangle credit"
        legs  = [
            {'type':'SELL','strike':otm2c,'option_type':'CE','premium':p_otm2c},
            {'type':'SELL','strike':otm2p,'option_type':'PE','premium':p_otm2p},
        ]
        bep_lo = otm2p - (p_otm2c + p_otm2p)
        bep_hi = otm2c + (p_otm2c + p_otm2p)
    elif vol_expand:
        setup = "LONG_STRADDLE"
        net   = straddle_net
        reason= f"Volatility expanding | VIX={vix:.1f} | ATR ratio={atr_ratio:.2f} | Expect big move"
        legs  = [
            {'type':'BUY','strike':atm,'option_type':'CE','premium':p_atm_ce},
            {'type':'BUY','strike':atm,'option_type':'PE','premium':p_atm_pe},
        ]
        bep_lo = atm - straddle_net
        bep_hi = atm + straddle_net
    else:
        setup = "LONG_STRANGLE"
        net   = strangle_net
        reason= f"Moderate vol | VIX={vix:.1f} | ATR ratio={atr_ratio:.2f} | Cheaper strangle"
        legs  = [
            {'type':'BUY','strike':otm1c,'option_type':'CE','premium':p_otm1c},
            {'type':'BUY','strike':otm1p,'option_type':'PE','premium':p_otm1p},
        ]
        bep_lo = otm1p - strangle_net
        bep_hi = otm1c + strangle_net

    # Also compute week-2 net premiums (higher dte = higher premium)
    p_atm_ce2  = _est_premium(spot, atm,  "CE", vix, dte2)
    p_atm_pe2  = _est_premium(spot, atm,  "PE", vix, dte2)
    p_otm1c2   = _est_premium(spot, otm1c,"CE", vix, dte2)
    p_otm1p2   = _est_premium(spot, otm1p,"PE", vix, dte2)

    return {
        'setup': setup,
        'instrument': instrument,
        'spot': round(spot, 2),
        'atm': atm,
        'vix': round(vix, 2),
        # Expiry 1 (near week)
        'dte': dte,
        'expiry': expiry_str,
        'net_premium': round(abs(net), 2),
        'net_sign': 'DEBIT' if setup.startswith('LONG') else 'CREDIT',
        'breakeven_lo': round(bep_lo, 2),
        'breakeven_hi': round(bep_hi, 2),
        'legs': legs,
        # Expiry 2 (next week)
        'dte2': dte2,
        'expiry2': expiry2_str,
        'net_premium2': round(abs(p_atm_ce2 + p_atm_pe2), 2),
        'legs2': [
            {'type': legs[0]['type'], 'strike': atm, 'option_type': 'CE', 'premium': p_atm_ce2},
            {'type': legs[1]['type'], 'strike': atm, 'option_type': 'PE', 'premium': p_atm_pe2},
        ],
        'reason': reason,
        'atr_ratio': round(atr_ratio, 2),
        'vol_regime': 'EXPANDING' if vol_expand else ('PINNED' if vol_pinned else 'NEUTRAL'),
    }


def get_all_option_strategies(instrument: str) -> list:
    """
    Compute 10 option strategies across 3 weekly expiries = 30 setups.
    Returns list of dicts, each with:
      strategy_name, setup_type, market_view, risk_profile, expiry, dte,
      legs, net_premium, net_sign, breakeven_lo, breakeven_hi, reason
    """
    spot_sym    = _TICKER_MAP.get(instrument, "^NSEI")
    strike_step = _STRIKE_STEP.get(instrument, 50)

    try:
        tk   = yf.Ticker(spot_sym)
        spot = float(tk.fast_info.last_price)
    except Exception:
        return []

    vix  = _get_vix()
    exps = get_three_option_expiries(instrument)   # [(str,dte), (str,dte), (str,dte)]

    atm   = round(spot / strike_step) * strike_step
    otm1c = atm + strike_step
    otm1p = atm - strike_step
    otm2c = atm + 2 * strike_step
    otm2p = atm - 2 * strike_step

    results = []

    for (expiry_str, dte) in exps:
        p_atm_ce  = _est_premium(spot, atm,   "CE", vix, dte)
        p_atm_pe  = _est_premium(spot, atm,   "PE", vix, dte)
        p_otm1c   = _est_premium(spot, otm1c, "CE", vix, dte)
        p_otm1p   = _est_premium(spot, otm1p, "PE", vix, dte)
        p_otm2c   = _est_premium(spot, otm2c, "CE", vix, dte)
        p_otm2p   = _est_premium(spot, otm2p, "PE", vix, dte)

        def _add(name, view, risk, legs, net, sign, bep_lo, bep_hi, note=""):
            results.append({
                'strategy_name':  name,
                'market_view':    view,
                'risk_profile':   risk,
                'expiry':         expiry_str,
                'dte':            dte,
                'legs':           legs,
                'net_premium':    round(abs(net), 2),
                'net_sign':       sign,
                'breakeven_lo':   round(bep_lo, 2),
                'breakeven_hi':   round(bep_hi, 2),
                'spot':           round(spot, 2),
                'atm':            atm,
                'vix':            round(vix, 2),
                'reason':         note or f"VIX={vix:.1f} | DTE={dte} | ATM={atm}",
            })

        # 1. Long Call
        _add('Long Call', 'Bullish', 'Limited Risk',
             [{'type': 'BUY', 'strike': atm, 'option_type': 'CE', 'premium': p_atm_ce}],
             p_atm_ce, 'DEBIT',
             atm + p_atm_ce, atm + p_atm_ce,   # BEP = strike + premium (same lo/hi for directional)
             f"Buy ATM CE {atm} @ {p_atm_ce:.0f} | BEP={atm + p_atm_ce:.0f}")

        # 2. Long Put
        _add('Long Put', 'Bearish', 'Limited Risk',
             [{'type': 'BUY', 'strike': atm, 'option_type': 'PE', 'premium': p_atm_pe}],
             p_atm_pe, 'DEBIT',
             atm - p_atm_pe, atm - p_atm_pe,
             f"Buy ATM PE {atm} @ {p_atm_pe:.0f} | BEP={atm - p_atm_pe:.0f}")

        # 3. Long Straddle
        strad_net = p_atm_ce + p_atm_pe
        _add('Long Straddle', 'High Volatility', 'Limited Risk',
             [{'type': 'BUY', 'strike': atm, 'option_type': 'CE', 'premium': p_atm_ce},
              {'type': 'BUY', 'strike': atm, 'option_type': 'PE', 'premium': p_atm_pe}],
             strad_net, 'DEBIT',
             atm - strad_net, atm + strad_net,
             f"Buy CE+PE @ ATM {atm} | Net={strad_net:.0f} | BEP={atm - strad_net:.0f}/{atm + strad_net:.0f}")

        # 4. Long Strangle
        strang_net = p_otm1c + p_otm1p
        _add('Long Strangle', 'Extreme Move', 'Limited Risk',
             [{'type': 'BUY', 'strike': otm1c, 'option_type': 'CE', 'premium': p_otm1c},
              {'type': 'BUY', 'strike': otm1p, 'option_type': 'PE', 'premium': p_otm1p}],
             strang_net, 'DEBIT',
             otm1p - strang_net, otm1c + strang_net,
             f"Buy OTM CE {otm1c}+PE {otm1p} | Net={strang_net:.0f} | Cheaper than straddle")

        # 5. Covered Call (Long Futures simulated + Sell OTM CE)
        _add('Covered Call', 'Neutral/Mild Bull', 'Capped Upside',
             [{'type': 'LONG_FUT', 'strike': 0, 'option_type': 'FUT', 'premium': 0},
              {'type': 'SELL',     'strike': otm1c, 'option_type': 'CE', 'premium': p_otm1c}],
             p_otm1c, 'CREDIT',
             spot - p_otm1c, otm1c,
             f"Sell OTM CE {otm1c} @ {p_otm1c:.0f} | Requires long futures | Max profit at {otm1c}")

        # 6. Protective Put (Long Futures + Buy ATM PE)
        _add('Protective Put', 'Bullish + Hedged', 'Insurance Cost',
             [{'type': 'LONG_FUT', 'strike': 0, 'option_type': 'FUT', 'premium': 0},
              {'type': 'BUY',      'strike': atm, 'option_type': 'PE', 'premium': p_atm_pe}],
             p_atm_pe, 'DEBIT',
             atm - p_atm_pe, atm - p_atm_pe,
             f"Buy PE {atm} @ {p_atm_pe:.0f} to protect long futures | Floor at {atm - p_atm_pe:.0f}")

        # 7. Collar (Long Futures + Buy OTM PE + Sell OTM CE)
        collar_net = p_otm1p - p_otm2c   # buy put, sell call
        collar_sign = 'DEBIT' if collar_net > 0 else 'CREDIT'
        _add('Collar', 'Neutral/Capped', 'Full Hedge',
             [{'type': 'LONG_FUT', 'strike': 0,    'option_type': 'FUT', 'premium': 0},
              {'type': 'BUY',      'strike': otm1p, 'option_type': 'PE',  'premium': p_otm1p},
              {'type': 'SELL',     'strike': otm2c, 'option_type': 'CE',  'premium': p_otm2c}],
             collar_net, collar_sign,
             otm1p - abs(collar_net), otm2c,
             f"Buy PE {otm1p} + Sell CE {otm2c} | Net={abs(collar_net):.0f} {collar_sign} | Cap={otm2c}")

        # 8. Synthetic Long (Buy ATM CE + Sell ATM PE)
        syn_net = p_atm_ce - p_atm_pe
        syn_sign = 'DEBIT' if syn_net > 0 else 'CREDIT'
        _add('Synthetic Long', 'Bullish', 'Futures-Like Risk',
             [{'type': 'BUY',  'strike': atm, 'option_type': 'CE', 'premium': p_atm_ce},
              {'type': 'SELL', 'strike': atm, 'option_type': 'PE', 'premium': p_atm_pe}],
             syn_net, syn_sign,
             atm + syn_net, atm + syn_net,
             f"Buy CE + Sell PE @ ATM {atm} | Net={abs(syn_net):.0f} {syn_sign} | Mimics long futures")

        # 9. Call Backspread (Sell 1 ATM CE + Buy 2 OTM1 CE)
        cbs_net = 2 * p_otm1c - p_atm_ce
        cbs_sign = 'DEBIT' if cbs_net > 0 else 'CREDIT'
        cbs_bep  = otm1c + max(cbs_net, 0)
        _add('Call Backspread', 'Explosive Up', 'Unlimited Profit',
             [{'type': 'SELL', 'strike': atm,   'option_type': 'CE', 'premium': p_atm_ce},
              {'type': 'BUY',  'strike': otm1c, 'option_type': 'CE', 'premium': p_otm1c, 'qty': 2}],
             cbs_net, cbs_sign,
             cbs_bep, cbs_bep,
             f"Sell ATM CE {atm} + Buy 2× OTM CE {otm1c} | BEP={cbs_bep:.0f} | Unlimited above BEP")

        # 10. Put Backspread (Sell 1 ATM PE + Buy 2 OTM1 PE)
        pbs_net = 2 * p_otm1p - p_atm_pe
        pbs_sign = 'DEBIT' if pbs_net > 0 else 'CREDIT'
        pbs_bep  = otm1p - max(pbs_net, 0)
        _add('Put Backspread', 'Explosive Down', 'Unlimited Profit',
             [{'type': 'SELL', 'strike': atm,   'option_type': 'PE', 'premium': p_atm_pe},
              {'type': 'BUY',  'strike': otm1p, 'option_type': 'PE', 'premium': p_otm1p, 'qty': 2}],
             pbs_net, pbs_sign,
             pbs_bep, pbs_bep,
             f"Sell ATM PE {atm} + Buy 2× OTM PE {otm1p} | BEP={pbs_bep:.0f} | Unlimited below BEP")

    return results   # 30 setups (10 strategies × 3 expiries)


def backtest_option_strategies_period(instrument: str, expiry_date: date,
                                       period_days: int = 8) -> list:
    """
    Backtest all 10 structural option strategies for a single expiry period.
    Entry: period_start = expiry_date - period_days, premiums via Black-Scholes.
    Exit:  expiry_date, values = intrinsic (no theta left) + futures P&L where applicable.
    Returns list of dicts per strategy: strategy_name, entry_spot, exit_spot,
            dte, atm, net_pnl_pts, net_pnl_inr, expiry.
    """
    ticker_sym  = _TICKER_MAP.get(instrument, "^NSEI")
    lot_size    = _LOT_SIZES.get(instrument, 75)
    strike_step = _STRIKE_STEP.get(instrument, 50)
    period_start = expiry_date - timedelta(days=period_days)

    try:
        tk   = yf.Ticker(ticker_sym)
        df_d = tk.history(
            start=str(period_start - timedelta(days=7)),
            end=str(expiry_date + timedelta(days=1)),
            interval='1d', auto_adjust=True)
        if df_d is None or len(df_d) < 2:
            return []
        df_d = df_d.rename(columns=str.lower)
        df_d.index = pd.to_datetime(df_d.index).normalize().tz_localize(None)
    except Exception:
        return []

    # Spot at entry (last close on or before period_start)
    en_idx = df_d.index[df_d.index <= pd.Timestamp(period_start)]
    if len(en_idx) == 0:
        en_idx = df_d.index[:1]
    entry_spot = float(df_d.loc[en_idx[-1], 'close'])

    # Spot at exit (last close on or before expiry_date)
    ex_idx = df_d.index[df_d.index <= pd.Timestamp(expiry_date)]
    if len(ex_idx) == 0:
        return []
    exit_spot = float(df_d.loc[ex_idx[-1], 'close'])

    vix = _get_vix()
    dte = period_days

    atm   = round(entry_spot / strike_step) * strike_step
    otm1c = atm + strike_step
    otm1p = atm - strike_step
    otm2c = atm + 2 * strike_step
    otm2p = atm - 2 * strike_step

    # Entry premiums (Black-Scholes, dte days to expiry)
    pe_atm_ce  = _est_premium(entry_spot, atm,   "CE", vix, dte)
    pe_atm_pe  = _est_premium(entry_spot, atm,   "PE", vix, dte)
    pe_otm1c   = _est_premium(entry_spot, otm1c, "CE", vix, dte)
    pe_otm1p   = _est_premium(entry_spot, otm1p, "PE", vix, dte)
    pe_otm2c   = _est_premium(entry_spot, otm2c, "CE", vix, dte)

    # Exit intrinsic values (dte=0, max(0, S-K) or max(0, K-S))
    def _iv(strike, opt):
        return max(0.0, exit_spot - strike) if opt == "CE" else max(0.0, strike - exit_spot)

    xi_atm_ce  = _iv(atm,   "CE")
    xi_atm_pe  = _iv(atm,   "PE")
    xi_otm1c   = _iv(otm1c, "CE")
    xi_otm1p   = _iv(otm1p, "PE")
    xi_otm2c   = _iv(otm2c, "CE")
    fut_pnl    = exit_spot - entry_spot   # per-point P&L for futures leg

    exp_str    = expiry_date.strftime('%d%b%Y').upper()
    entry_str  = period_start.strftime('%d %b %y')
    exit_str   = expiry_date.strftime('%d %b %y')

    def _res(name, pnl_pts):
        return {
            'strategy_name': name,
            'entry_date':    entry_str,    # period start (when premium is paid)
            'exit_date':     exit_str,     # expiry date (when intrinsic is collected)
            'entry_spot':    round(entry_spot, 2),
            'exit_spot':     round(exit_spot, 2),
            'dte':           dte,
            'atm':           atm,
            'net_pnl_pts':   round(pnl_pts, 2),
            'net_pnl_inr':   round(pnl_pts * lot_size, 2),
            'expiry':        exp_str,
            'period_days':   period_days,
        }

    results = [
        _res('Long Call',      (xi_atm_ce - pe_atm_ce)),
        _res('Long Put',       (xi_atm_pe - pe_atm_pe)),
        _res('Long Straddle',  (xi_atm_ce + xi_atm_pe - pe_atm_ce - pe_atm_pe)),
        _res('Long Strangle',  (xi_otm1c + xi_otm1p - pe_otm1c - pe_otm1p)),
        # Covered Call: long futures + sell OTM1 CE
        _res('Covered Call',   fut_pnl + (pe_otm1c - xi_otm1c)),
        # Protective Put: long futures + buy ATM PE
        _res('Protective Put', fut_pnl + (xi_atm_pe - pe_atm_pe)),
        # Collar: long futures + buy OTM1 PE + sell OTM2 CE
        _res('Collar',         fut_pnl + (xi_otm1p - pe_otm1p) + (pe_otm2c - xi_otm2c)),
        # Synthetic Long: buy ATM CE + sell ATM PE
        _res('Synthetic Long', (xi_atm_ce - pe_atm_ce) - (xi_atm_pe - pe_atm_pe)),
        # Call Backspread: sell 1 ATM CE + buy 2 OTM1 CE
        _res('Call Backspread', -(xi_atm_ce - pe_atm_ce) + 2 * (xi_otm1c - pe_otm1c)),
        # Put Backspread: sell 1 ATM PE + buy 2 OTM1 PE
        _res('Put Backspread',  -(xi_atm_pe - pe_atm_pe) + 2 * (xi_otm1p - pe_otm1p)),
    ]
    return results


# ── Backtester ────────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    date:         str
    strategy:     str
    tf:           str
    direction:    str
    entry:        float
    exit_price:   float
    sl:           float
    target:       float
    hit_sl:       bool
    hit_target:   bool
    hold_bars:    int
    pnl_pts:      float
    pnl_inr:      float
    lot_size:     int
    entry_time:   str   = ""    # HH:MM of entry bar
    exit_time:    str   = ""    # HH:MM of exit bar
    rr_ratio:     float = 2.0   # always 1:2 (target = entry ± 2 × SL_distance)
    exit_reason:  str   = ""    # 'SL' | 'TARGET' | 'EOD'
    theta_entry:  float = 0.0   # BS premium at entry (options mode only)
    theta_exit:   float = 0.0   # BS premium at exit (options mode only)
    instrument:   str   = ""    # NIFTY | BANKNIFTY | SENSEX
    iv_pct:       float = 0.0   # IV % used for Black-Scholes (options only, e.g. 20.0)


class StrategyBacktester:
    """
    Bar-by-bar backtest for Chandelier Exit + TUX+Supertrend.
    Supports Futures (fixed point × lot) and Options (theta-decay premium P&L).
    """

    def __init__(self, instrument: str = "NIFTY"):
        self.instrument = instrument
        self.lot_size   = _LOT_SIZES.get(instrument, 75)
        self._tf        = "5m"   # set in run(), used by _calc_pnl_inr fallback

    def run(self, strategy: str, tf: str,
            start: date, end: date,
            mode: str = 'futures',
            expiry_date: date = None,
            sl_pct_opt: float = 40.0,
            target_pct_opt: float = 80.0,
            rr_mult: float = 2.0,
            vix_override: float = None) -> list:
        """
        Run backtest.
        strategy: 'CHANDELIER', 'TUX_SUPERTREND', 'CE_REGIME'
        tf: '5m', '15m', '1h'
        mode: 'futures' or 'options'
        expiry_date: actual option/futures expiry (enables theta-decay P&L for options)
        vix_override: if set, use this IV% instead of live _get_vix() (e.g. 20.0 = 20%)
        Returns list of TradeResult.
        """
        self._tf = tf
        df = fetch_index_ohlcv_range(self.instrument, tf, start, end)
        if df is None or len(df) < 30:
            return []

        # Calculate indicators
        if strategy == 'CHANDELIER':
            df = calc_chandelier_exit(df)
            sig_col = 'ce_signal'
            def get_sl(df, i):
                d = int(df['ce_dir'].iloc[i])
                return float(df['ce_long_stop'].iloc[i]) if d == 1 else float(df['ce_short_stop'].iloc[i])
        elif strategy == 'TUX_SUPERTREND':
            df = calc_tux_ema(calc_supertrend(df))
            # Synthesize signal: Supertrend direction + TUX bounce agree
            combo_sigs = [None] * len(df)
            for _i in range(1, len(df)):
                st_d = int(df['st_dir'].iloc[_i])
                st_prev = int(df['st_dir'].iloc[_i - 1])
                tux_s = df['tux_signal'].iloc[_i]
                st_dir_str = 'LONG' if st_d == 1 else 'SHORT'
                # Emit signal on Supertrend flip + matching TUX bounce (within 3 bars)
                if st_d != st_prev:
                    # Check TUX in recent 3 bars
                    for _j in range(_i, max(_i - 3, -1), -1):
                        if df['tux_signal'].iloc[_j] == st_dir_str:
                            combo_sigs[_i] = st_dir_str
                            break
                elif tux_s == st_dir_str:
                    # TUX bounce in existing ST direction
                    combo_sigs[_i] = st_dir_str
            df['combo_signal'] = combo_sigs
            sig_col = 'combo_signal'
            def get_sl(df, i):
                d = int(df['st_dir'].iloc[i])
                return float(df['st_lower'].iloc[i]) if d == 1 else float(df['st_upper'].iloc[i])
        elif strategy == 'CE_REGIME':
            df = calc_ce_regime(df)
            sig_col = 'ce_regime_signal'
            def get_sl(df, i):
                d = int(df['ce_dir'].iloc[i])
                return float(df['ce_long_stop'].iloc[i]) if d == 1 else float(df['ce_short_stop'].iloc[i])
        else:
            return []

        # ── Pre-computations used by mode-specific filters ────────────────
        if 'adx' not in df.columns:
            df = add_adx(df, period=14)

        df['vol_ma20'] = df['volume'].rolling(20).mean()

        _atr14_series = _atr(df, 14)   # ATR14 Series for trailing SL

        _bars_1h = {'5m': 12, '15m': 4, '1h': 1}.get(tf, 4)
        df['prev_1h_high'] = df['high'].rolling(_bars_1h).max().shift(1)
        df['prev_1h_low']  = df['low'].rolling(_bars_1h).min().shift(1)

        _MAX_OPT_BARS = {'5m': 15, '15m': 6, '1h': 2}

        results = []
        in_trade = False
        trade_dir = None
        trade_entry = 0.0
        trade_sl = 0.0
        trade_target = 0.0
        trade_entry_i = 0
        trade_date = ""
        trade_entry_date = None   # calendar date for theta decay
        trade_entry_time = ""     # HH:MM of entry bar
        _breakeven_hit   = False  # CHANDELIER options: SL moved to entry at 1:1

        vix     = vix_override if vix_override else _get_vix()
        _iv_pct = round(vix * 100, 1) if mode == 'options' else 0.0

        for i in range(1, len(df) - 1):
            row = df.iloc[i]
            ts  = df.index[i]

            # IST filtering: only 9:20–15:00
            try:
                t = ts.time()
                if not (dtime(9, 20) <= t <= dtime(15, 0)):
                    continue
                # Force exit at 15:00 (EOD)
                if in_trade and t >= dtime(15, 0):
                    ep = float(df['close'].iloc[i])
                    pnl_pts = (ep - trade_entry) * (1 if trade_dir == 'LONG' else -1)
                    exit_dt = ts.date()
                    pnl_inr, th_en, th_ex = self._calc_pnl_inr(
                        pnl_pts, trade_entry, trade_dir, vix,
                        i - trade_entry_i, mode, sl_pct_opt, target_pct_opt,
                        entry_date=trade_entry_date, exit_date=exit_dt,
                        exit_price=ep, expiry_date=expiry_date)
                    results.append(TradeResult(
                        date=str(ts.date()), strategy=strategy, tf=tf,
                        direction=trade_dir, entry=round(trade_entry, 2),
                        exit_price=round(ep, 2), sl=round(trade_sl, 2),
                        target=round(trade_target, 2), hit_sl=False,
                        hit_target=False, hold_bars=i - trade_entry_i,
                        pnl_pts=round(pnl_pts, 2), pnl_inr=round(pnl_inr, 2),
                        lot_size=self.lot_size,
                        entry_time=trade_entry_time, exit_time=ts.strftime('%H:%M'),
                        rr_ratio=rr_mult, exit_reason='EOD',
                        theta_entry=th_en, theta_exit=th_ex,
                        iv_pct=_iv_pct,
                    ))
                    in_trade = False
                    continue
            except Exception:
                pass

            # Overnight carry guard: if a trade crosses to a new calendar day, close it immediately
            if in_trade and trade_entry_date is not None and ts.date() != trade_entry_date:
                ep = float(row['close'])
                pnl_pts = (ep - trade_entry) * (1 if trade_dir == 'LONG' else -1)
                exit_dt = ts.date()
                pnl_inr, th_en, th_ex = self._calc_pnl_inr(
                    pnl_pts, trade_entry, trade_dir, vix,
                    i - trade_entry_i, mode, sl_pct_opt, target_pct_opt,
                    entry_date=trade_entry_date, exit_date=exit_dt,
                    exit_price=ep, expiry_date=expiry_date)
                results.append(TradeResult(
                    date=trade_date, strategy=strategy, tf=tf,
                    direction=trade_dir, entry=round(trade_entry, 2),
                    exit_price=round(ep, 2), sl=round(trade_sl, 2),
                    target=round(trade_target, 2), hit_sl=False,
                    hit_target=False, hold_bars=i - trade_entry_i,
                    pnl_pts=round(pnl_pts, 2), pnl_inr=round(pnl_inr, 2),
                    lot_size=self.lot_size,
                    entry_time=trade_entry_time, exit_time=ts.strftime('%H:%M'),
                    rr_ratio=rr_mult, exit_reason='EOD',
                    theta_entry=th_en, theta_exit=th_ex,
                ))
                in_trade = False

            # ── CHANDELIER futures: ATR trailing SL (ratchets in profit direction) ──
            if in_trade and strategy == 'CHANDELIER' and mode == 'futures':
                _atr_now = float(_atr14_series.iloc[i]) if not pd.isna(_atr14_series.iloc[i]) else 0.0
                if _atr_now > 0:
                    if trade_dir == 'LONG':
                        _new_sl = float(row['close']) - 2.0 * _atr_now
                        if _new_sl > trade_sl:
                            trade_sl = _new_sl
                    else:
                        _new_sl = float(row['close']) + 2.0 * _atr_now
                        if _new_sl < trade_sl:
                            trade_sl = _new_sl

            # ── CHANDELIER options: move SL to breakeven when 1:1 RR is hit ──
            if in_trade and strategy == 'CHANDELIER' and mode == 'options' and not _breakeven_hit:
                _sl_dist_orig = abs(trade_entry - trade_sl)
                _1to1_level   = (trade_entry + _sl_dist_orig if trade_dir == 'LONG'
                                 else trade_entry - _sl_dist_orig)
                _hit_1to1 = (float(df['high'].iloc[i]) >= _1to1_level if trade_dir == 'LONG'
                             else float(df['low'].iloc[i])  <= _1to1_level)
                if _hit_1to1:
                    trade_sl       = trade_entry
                    _breakeven_hit = True

            # ── Options max-hold: exit after ~75 min (15 bars × 5m) ──────────
            if in_trade and mode == 'options':
                _max_bars = _MAX_OPT_BARS.get(tf, 6)
                if (i - trade_entry_i) >= _max_bars:
                    ep = float(row['close'])
                    pnl_pts = (ep - trade_entry) * (1 if trade_dir == 'LONG' else -1)
                    exit_dt = ts.date()
                    pnl_inr, th_en, th_ex = self._calc_pnl_inr(
                        pnl_pts, trade_entry, trade_dir, vix,
                        i - trade_entry_i, mode, sl_pct_opt, target_pct_opt,
                        entry_date=trade_entry_date, exit_date=exit_dt,
                        exit_price=ep, expiry_date=expiry_date)
                    results.append(TradeResult(
                        date=trade_date, strategy=strategy, tf=tf,
                        direction=trade_dir, entry=round(trade_entry, 2),
                        exit_price=round(ep, 2), sl=round(trade_sl, 2),
                        target=round(trade_target, 2), hit_sl=False,
                        hit_target=False, hold_bars=i - trade_entry_i,
                        pnl_pts=round(pnl_pts, 2), pnl_inr=round(pnl_inr, 2),
                        lot_size=self.lot_size,
                        entry_time=trade_entry_time, exit_time=ts.strftime('%H:%M'),
                        rr_ratio=rr_mult, exit_reason='MAX_HOLD',
                        theta_entry=th_en, theta_exit=th_ex,
                        iv_pct=_iv_pct,
                    ))
                    in_trade       = False
                    _breakeven_hit = False
                    continue

            # CE_REGIME: exit early if market turns sideways mid-trade
            if in_trade and strategy == 'CE_REGIME' and 'regime' in df.columns:
                regime_now = str(df['regime'].iloc[i])
                if regime_now == 'SIDEWAYS':
                    ep = float(row['close'])
                    pnl_pts = (ep - trade_entry) * (1 if trade_dir == 'LONG' else -1)
                    exit_dt = ts.date()
                    pnl_inr, th_en, th_ex = self._calc_pnl_inr(
                        pnl_pts, trade_entry, trade_dir, vix,
                        i - trade_entry_i, mode, sl_pct_opt, target_pct_opt,
                        entry_date=trade_entry_date, exit_date=exit_dt,
                        exit_price=ep, expiry_date=expiry_date)
                    results.append(TradeResult(
                        date=trade_date, strategy=strategy, tf=tf,
                        direction=trade_dir, entry=round(trade_entry, 2),
                        exit_price=round(ep, 2), sl=round(trade_sl, 2),
                        target=round(trade_target, 2), hit_sl=False,
                        hit_target=False, hold_bars=i - trade_entry_i,
                        pnl_pts=round(pnl_pts, 2), pnl_inr=round(pnl_inr, 2),
                        lot_size=self.lot_size,
                        entry_time=trade_entry_time, exit_time=ts.strftime('%H:%M'),
                        rr_ratio=rr_mult, exit_reason='SIDEWAYS',
                        theta_entry=th_en, theta_exit=th_ex,
                        iv_pct=_iv_pct,
                    ))
                    in_trade = False
                    continue

            if in_trade:
                hi = float(row['high'])
                lo = float(row['low'])
                # Check SL / target
                sl_hit  = (trade_dir == 'LONG' and lo <= trade_sl) or \
                          (trade_dir == 'SHORT' and hi >= trade_sl)
                tgt_hit = (trade_dir == 'LONG' and hi >= trade_target) or \
                          (trade_dir == 'SHORT' and lo <= trade_target)
                if sl_hit or tgt_hit:
                    ep = trade_sl if sl_hit else trade_target
                    pnl_pts = (ep - trade_entry) * (1 if trade_dir == 'LONG' else -1)
                    exit_dt = ts.date()
                    pnl_inr, th_en, th_ex = self._calc_pnl_inr(
                        pnl_pts, trade_entry, trade_dir, vix,
                        i - trade_entry_i, mode, sl_pct_opt, target_pct_opt,
                        entry_date=trade_entry_date, exit_date=exit_dt,
                        exit_price=ep, expiry_date=expiry_date)
                    results.append(TradeResult(
                        date=trade_date, strategy=strategy, tf=tf,
                        direction=trade_dir, entry=round(trade_entry, 2),
                        exit_price=round(ep, 2), sl=round(trade_sl, 2),
                        target=round(trade_target, 2), hit_sl=sl_hit,
                        hit_target=tgt_hit, hold_bars=i - trade_entry_i,
                        pnl_pts=round(pnl_pts, 2), pnl_inr=round(pnl_inr, 2),
                        lot_size=self.lot_size,
                        entry_time=trade_entry_time, exit_time=ts.strftime('%H:%M'),
                        rr_ratio=rr_mult, exit_reason='SL' if sl_hit else 'TARGET',
                        theta_entry=th_en, theta_exit=th_ex,
                        iv_pct=_iv_pct,
                    ))
                    in_trade = False

            if not in_trade:
                sig = df[sig_col].iloc[i]
                if sig in ('LONG', 'SHORT'):
                    # ── Global ADX > 25 filter: only trade in trending markets ──
                    _adx_now = float(df['adx'].iloc[i]) if not pd.isna(df['adx'].iloc[i]) else 0.0
                    if _adx_now < 25:
                        continue

                    # ── TUX+ST futures: volume must exceed 20-bar MA ──────────
                    if strategy == 'TUX_SUPERTREND' and mode == 'futures':
                        _vol_now = float(df['volume'].iloc[i])
                        _vol_ma  = (float(df['vol_ma20'].iloc[i])
                                    if not pd.isna(df['vol_ma20'].iloc[i]) else 0.0)
                        if _vol_ma > 0 and _vol_now <= _vol_ma:
                            continue

                    # ── CE_REGIME options: require breakout of prev 1H high/low ─
                    if strategy == 'CE_REGIME' and mode == 'options':
                        _p1h_hi = df['prev_1h_high'].iloc[i]
                        _p1h_lo = df['prev_1h_low'].iloc[i]
                        if pd.isna(_p1h_hi) or pd.isna(_p1h_lo):
                            continue
                        _close_now = float(row['close'])
                        if sig == 'LONG'  and _close_now <= float(_p1h_hi):
                            continue
                        if sig == 'SHORT' and _close_now >= float(_p1h_lo):
                            continue

                    trade_entry_try = float(df['open'].iloc[i + 1]) if i + 1 < len(df) else float(row['close'])
                    sl_val_try      = get_sl(df, i)
                    # Skip trades where SL is on the wrong side of entry (inverted stop)
                    if sig == 'LONG'  and sl_val_try >= trade_entry_try:
                        pass
                    elif sig == 'SHORT' and sl_val_try <= trade_entry_try:
                        pass
                    else:
                        trade_dir        = sig
                        trade_entry      = trade_entry_try
                        sl_val           = sl_val_try
                        sl_dist          = abs(trade_entry - sl_val)
                        trade_sl         = sl_val
                        trade_target     = (trade_entry + rr_mult * sl_dist) if sig == 'LONG' else (trade_entry - rr_mult * sl_dist)
                        trade_entry_i    = i
                        trade_date       = str(ts.date())
                        trade_entry_date = ts.date()
                        trade_entry_time = ts.strftime('%H:%M')
                        in_trade         = True
                        _breakeven_hit   = False

        return results

    def _calc_pnl_inr(self, pnl_pts: float, entry: float, direction: str,
                       vix: float, hold_bars: int, mode: str,
                       sl_pct: float, target_pct: float,
                       entry_date=None, exit_date=None,
                       exit_price: float = None,
                       expiry_date=None):
        """Returns (pnl_inr, theta_entry, theta_exit). Theta values are 0 for futures."""
        if mode == 'futures':
            return pnl_pts * self.lot_size, 0.0, 0.0

        # Options: Black-Scholes at entry + exit for theta decay
        strike_step = _STRIKE_STEP.get(self.instrument, 50)
        atm      = round(entry / strike_step) * strike_step
        opt_type = "CE" if direction == 'LONG' else "PE"

        if expiry_date and entry_date and exit_date:
            dte_entry  = max(1, (expiry_date - entry_date).days)
            dte_exit   = max(0, (expiry_date - exit_date).days)
            prem_entry = _est_premium(entry, atm, opt_type, vix, dte_entry)
            ex_spot    = exit_price if exit_price else (
                entry + pnl_pts if direction == 'LONG' else entry - pnl_pts)
            prem_exit  = _est_premium(ex_spot, atm, opt_type, vix, dte_exit)
            pnl_prem   = prem_exit - prem_entry
            return round(pnl_prem * self.lot_size, 2), round(prem_entry, 2), round(prem_exit, 2)
        else:
            bars_per_day = {'5m': 75, '15m': 25, '1h': 6}.get(self._tf, 12)
            dte      = max(1, 7 - hold_bars // bars_per_day)
            prem     = _est_premium(entry, atm, opt_type, vix, dte)
            pnl_pct  = pnl_pts / entry * 100
            pnl_prem = prem * pnl_pct / 100
            return round(pnl_prem * self.lot_size, 2), round(prem, 2), 0.0

    def run_all_strategies(self, tf: str,
                            start: date, end: date,
                            mode: str = 'futures',
                            expiry_date: date = None,
                            rr_mult: float = 2.0,
                            vix_override: float = None) -> dict:
        """Run all STRATEGIES; return {strategy_name: (trades_list, summary_dict)}."""
        out = {}
        for strat in STRATEGIES:
            trades = self.run(strat, tf, start, end, mode, expiry_date,
                              rr_mult=rr_mult, vix_override=vix_override)
            out[strat] = (trades, self.summary(trades))
        return out

    def summary(self, results: list) -> dict:
        if not results:
            return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
                    'total_pnl': 0, 'profit_factor': 0, 'max_dd': 0, 'avg_rr': 0}
        wins    = [r for r in results if r.pnl_inr > 0]
        losses  = [r for r in results if r.pnl_inr <= 0]
        gross_w = sum(r.pnl_inr for r in wins)
        gross_l = abs(sum(r.pnl_inr for r in losses))

        # Max drawdown (on cumulative INR equity)
        cumul = 0.0
        peak  = 0.0
        max_dd = 0.0
        for r in results:
            cumul += r.pnl_inr
            peak   = max(peak, cumul)
            max_dd = max(max_dd, peak - cumul)

        avg_rr = sum(getattr(r, 'rr_ratio', 2.0) for r in results) / len(results) if results else 2.0

        return {
            'total':        len(results),
            'wins':         len(wins),
            'losses':       len(losses),
            'win_rate':     round(len(wins) / len(results) * 100, 1),
            'total_pnl':    round(sum(r.pnl_inr for r in results), 2),
            'profit_factor': round(gross_w / gross_l, 2) if gross_l > 0 else 99.0,
            'max_dd':       round(max_dd, 2),
            'avg_rr':       round(avg_rr, 2),
        }


# ── Auto Trader ───────────────────────────────────────────────────────────────

@dataclass
class OFTrade:
    id:           str
    date:         str
    time:         str
    instrument:   str
    tf:           str
    strategy:     str
    direction:    str
    mode:         str  # 'futures' or 'options'
    entry:        float
    sl:           float
    target:       float
    lot_size:     int
    qty:          int
    status:       str = "OPEN"
    exit_price:   float = 0.0
    exit_reason:  str = ""
    pnl_pts:      float = 0.0
    pnl_inr:      float = 0.0
    paper:        bool = True
    expiry_str:   str = ""    # contract expiry (e.g. '29MAY2026')
    exit_time:    str = ""    # HH:MM of exit
    rr_ratio:     float = 2.0 # 2.0 = 1:2, 3.0 = 1:3
    vix_at_entry: float = 0.0 # India VIX % at trade entry time

    def to_dict(self): return self.__dict__


class OptionsFuturesAutoTrader:
    """
    Background auto-trader for Chandelier Exit + TUX+Supertrend.
    Scans NIFTY, BANKNIFTY, SENSEX simultaneously across 5m/15m/1h.
    Trades options (ATM CE/PE on nearest weekly expiry) and/or futures.
    """

    _STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optfut_state.json')

    def __init__(self, instruments: list = None,
                 strategies: list = None,
                 timeframes: list = None,
                 mode: str = "both",   # 'options' | 'futures' | 'both'
                 qty: int = 1,
                 paper: bool = True,
                 angel_client=None,
                 sl_pct_opt: float = 40.0,
                 target_pct_opt: float = 80.0,
                 rr_mult: float = 2.0,
                 # legacy single-instrument compat
                 instrument: str = None):
        self.instruments    = instruments or (["NIFTY", "BANKNIFTY", "SENSEX"] if instrument is None else [instrument])
        self.instrument     = self.instruments[0]   # kept for backwards compat
        self.strategies     = strategies or STRATEGIES
        self.timeframes     = timeframes or ["5m", "15m", "1h"]
        self.mode           = mode
        self.qty            = qty
        self.paper          = paper
        self.client         = angel_client
        self.sl_pct_opt     = sl_pct_opt
        self.target_pct_opt = target_pct_opt
        self.rr_mult        = rr_mult   # 2.0 = 1:2, 3.0 = 1:3
        self._running       = False
        self._thread        = None
        self.positions: list[OFTrade] = []
        self.closed: list[OFTrade]    = []
        self.log: list[str]           = []
        self.total_pnl      = 0.0
        self._last_cycle_at = None
        self._load_state()

    def _log(self, msg: str):
        ts = datetime.now(IST).strftime('%H:%M:%S')
        self.log.append(f"[{ts}] {msg}")
        self.log = self.log[-200:]
        print(f"[OFAutoTrader] {msg}")

    def _save_state(self):
        data = {
            'date':      str(date.today()),
            'positions': [p.to_dict() for p in self.positions],
            'closed':    [t.to_dict() for t in self.closed[-100:]],
            'log':       self.log[-100:],
            'total_pnl': self.total_pnl,
        }
        try:
            with open(self._STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_state(self):
        if not os.path.exists(self._STATE_FILE):
            return
        try:
            with open(self._STATE_FILE) as f:
                data = json.load(f)
            if data.get('date') != str(date.today()):
                return  # fresh day
            self.total_pnl = float(data.get('total_pnl', 0))
            self.log       = data.get('log', [])
            for td in data.get('positions', []):
                if td.get('status') == 'OPEN':
                    self.positions.append(OFTrade(**td))
            for td in data.get('closed', []):
                self.closed.append(OFTrade(**td))
        except Exception:
            pass

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._log(f"Started | {', '.join(self.instruments)} | {self.mode} | "
                  f"TFs={self.timeframes} | Strats={self.strategies} | Paper={self.paper}")

    def stop(self):
        self._running = False
        self._log("Stopped by user")
        self._save_state()

    def _is_market_open(self) -> bool:
        now = datetime.now(IST).time()
        # 'both' and 'futures' get the later cutoff so positions can be monitored until EOD
        sq_off = dtime(15, 10) if self.mode in ('futures', 'both') else dtime(15, 0)
        return dtime(9, 20) <= now <= sq_off

    def _get_spot(self, instrument: str) -> float:
        sym = _TICKER_MAP.get(instrument, "^NSEI")
        try:
            tk = yf.Ticker(sym)
            return float(tk.fast_info.last_price)
        except Exception:
            return 0.0

    def _loop(self):
        while self._running:
            try:
                self._cycle()
            except Exception as e:
                self._log(f"Loop error: {e}")
            time.sleep(300)  # scan every 5 minutes

    def _cycle(self):
        now = datetime.now(IST)
        self._last_cycle_at = now  # heartbeat for UI

        # EOD square-off must run before market-open check so it always fires at 15:10
        if now.time() >= dtime(15, 10):
            self._square_off_all("EOD")
            return

        if not self._is_market_open():
            # Still monitor and close any open positions during wind-down window
            for pos in list(self.positions):
                spot = self._get_spot(pos.instrument)
                self._monitor_position(pos, spot)
            self._log(f"Market closed ({now.strftime('%H:%M')} IST) — monitoring only")
            return

        # Monitor all open positions (update P&L using live spot)
        for pos in list(self.positions):
            spot = self._get_spot(pos.instrument)
            self._monitor_position(pos, spot)

        # Scan for new signals — enter when no trade exists for (instrument, mode)
        # Each entry creates BOTH 1:2 and 1:3 trades, so key is (instrument, mode) only
        open_keys = {(p.instrument, p.mode) for p in self.positions if p.status == 'OPEN'}
        vix = _get_vix()
        for instr in self.instruments:
            for trade_mode in (['options', 'futures'] if self.mode == 'both' else [self.mode]):
                if (instr, trade_mode) not in open_keys:
                    self._scan_and_enter(instr, trade_mode, vix)

    # Maps internal strategy codes → display names used in signal dicts
    _STRAT_DISPLAY = {
        'CHANDELIER':     'Chandelier Exit',
        'TUX_SUPERTREND': 'TUX + Supertrend',
        'CE_REGIME':      'CE + Regime Filter',
    }

    def _scan_and_enter(self, instrument: str, trade_mode: str, vix: float):
        results = scan_all(instrument, self.timeframes)
        spot    = self._get_spot(instrument)
        if spot <= 0:
            return

        # Build allowed strategy display names from self.strategies (internal codes)
        _allowed = {self._STRAT_DISPLAY.get(s, s) for s in self.strategies}

        # Find freshest, highest-RR signal across all TFs and allowed strategies
        best_sig   = None
        best_score = 0
        for tf, tf_sigs in results.items():
            for sig in tf_sigs:
                if not isinstance(sig, dict) or '_timestamp' in sig:
                    continue
                if sig.get('strategy') not in _allowed:
                    continue   # strategy not selected by user
                fresh = sig.get('fresh_signal')
                rr    = sig.get('rr', 0)
                if fresh in ('LONG', 'SHORT') and rr >= 1.5:
                    score = rr * (2 if tf == '5m' else (1.5 if tf == '15m' else 1.0))
                    if score > best_score:
                        best_score = score
                        best_sig   = {**sig, 'tf': tf}

        if best_sig is None:
            self._log(f"No fresh signal | {instrument} | {trade_mode} | spot={spot:.0f}")
            return

        # ADX gate: skip entries in non-trending markets
        _adx_val = float(best_sig.get('adx_val', 0))
        if _adx_val < 25:
            self._log(f"Skipped {instrument}: ADX={_adx_val:.1f} < 25 (non-trending)")
            return

        direction = best_sig['fresh_signal']
        entry     = spot
        sl_val    = best_sig['sl']
        sl_dist   = abs(entry - sl_val)

        # Validate SL is in the correct direction
        if direction == 'SHORT' and sl_val <= entry:
            self._log(f"Skipped SHORT {instrument}: SL {sl_val:.0f} below entry {entry:.0f} (inverted stop)")
            return
        if direction == 'LONG' and sl_val >= entry:
            self._log(f"Skipped LONG {instrument}: SL {sl_val:.0f} above entry {entry:.0f} (inverted stop)")
            return

        target    = entry + self.rr_mult * sl_dist if direction == 'LONG' else entry - self.rr_mult * sl_dist
        lot_size  = _LOT_SIZES.get(instrument, 75)

        # Nearest expiry for this mode
        try:
            if trade_mode == 'options':
                exp_str = get_three_option_expiries(instrument)[0][0]   # nearest weekly
            else:
                exp_str = get_two_futures_expiries(instrument)[0][0]    # nearest monthly
        except Exception:
            exp_str = ""

        # Always create BOTH 1:2 and 1:3 trades simultaneously
        _ts_str = datetime.now(IST).strftime('%H%M%S')
        for _rr in [2.0, 3.0]:
            _tgt = entry + _rr * sl_dist if direction == 'LONG' else entry - _rr * sl_dist
            trade_id = f"OF_{instrument}_{trade_mode[:3].upper()}_1{int(_rr)}_{_ts_str}"
            trade = OFTrade(
                id=trade_id,
                date=str(date.today()),
                time=datetime.now(IST).strftime('%H:%M:%S'),
                instrument=instrument,
                tf=best_sig['tf'],
                strategy=best_sig['strategy'],
                direction=direction,
                mode=trade_mode,
                entry=round(entry, 2),
                sl=round(sl_val, 2),
                target=round(_tgt, 2),
                lot_size=lot_size,
                qty=self.qty,
                paper=self.paper,
                expiry_str=exp_str,
                rr_ratio=_rr,
                vix_at_entry=round(vix, 1),
            )
            self.positions.append(trade)
        self._save_state()
        self._log(f"ENTERED {direction} {instrument} | {best_sig['strategy']} [{best_sig['tf']}] | "
                  f"{trade_mode.upper()} {exp_str} | Entry={entry:.0f} SL={sl_val:.0f} "
                  f"TGT_1:2={entry + 2*sl_dist if direction=='LONG' else entry - 2*sl_dist:.0f} "
                  f"TGT_1:3={entry + 3*sl_dist if direction=='LONG' else entry - 3*sl_dist:.0f}")

    def _monitor_position(self, pos: OFTrade, spot: float):
        if spot <= 0:
            return

        # CHANDELIER options: move SL to breakeven when price hits 1:1 RR
        if pos.status == 'OPEN' and pos.strategy == 'Chandelier Exit' and pos.mode == 'options':
            _sl_dist = abs(pos.entry - pos.sl)
            _level_1to1 = (pos.entry + _sl_dist if pos.direction == 'LONG'
                           else pos.entry - _sl_dist)
            if ((pos.direction == 'LONG'  and spot >= _level_1to1 and pos.sl < pos.entry) or
                (pos.direction == 'SHORT' and spot <= _level_1to1 and pos.sl > pos.entry)):
                pos.sl = pos.entry
                self._save_state()
                self._log(f"BREAKEVEN set {pos.instrument} {pos.direction} SL→{pos.entry:.0f}")

        # Options max-hold: force exit after 75 minutes to avoid theta decay
        if pos.status == 'OPEN' and pos.mode == 'options':
            try:
                _entry_dt = datetime.strptime(
                    f"{pos.date} {pos.time[:5]}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=IST)
                _elapsed = (datetime.now(IST) - _entry_dt).total_seconds() / 60
                if _elapsed >= 75:
                    pnl_pts = (spot - pos.entry) * (1 if pos.direction == 'LONG' else -1)
                    pnl_inr = pnl_pts * pos.lot_size * pos.qty
                    pos.status     = "CLOSED"
                    pos.exit_price = round(spot, 2)
                    pos.exit_reason = "MAX_HOLD"
                    pos.exit_time  = datetime.now(IST).strftime('%H:%M')
                    pos.pnl_pts    = round(pnl_pts, 2)
                    pos.pnl_inr    = round(pnl_inr, 2)
                    self.total_pnl += pnl_inr
                    self.closed.append(pos)
                    self.positions.remove(pos)
                    self._save_state()
                    emoji = "✅" if pnl_inr > 0 else "❌"
                    self._log(f"{emoji} MAX_HOLD exit {pos.direction} {pos.instrument} | "
                              f"{pos.mode.upper()} | P&L={pnl_pts:+.0f}pts = ₹{pnl_inr:+,.0f}")
                    return
            except Exception:
                pass

        sl_hit  = (pos.direction == 'LONG'  and spot <= pos.sl) or \
                  (pos.direction == 'SHORT' and spot >= pos.sl)
        tgt_hit = (pos.direction == 'LONG'  and spot >= pos.target) or \
                  (pos.direction == 'SHORT' and spot <= pos.target)

        if sl_hit or tgt_hit:
            exit_reason = "SL" if sl_hit else "TARGET"
            exit_price  = pos.sl if sl_hit else pos.target
            pnl_pts     = (exit_price - pos.entry) * (1 if pos.direction == 'LONG' else -1)
            # Safety: correct any inversion where label contradicts P&L direction
            if exit_reason == "SL" and pnl_pts > 0:
                exit_reason = "TARGET"
            elif exit_reason == "TARGET" and pnl_pts < 0:
                exit_reason = "SL"
            pnl_inr     = pnl_pts * pos.lot_size * pos.qty

            pos.status     = "CLOSED"
            pos.exit_price = round(exit_price, 2)
            pos.exit_reason= exit_reason
            pos.exit_time  = datetime.now(IST).strftime('%H:%M')
            pos.pnl_pts    = round(pnl_pts, 2)
            pos.pnl_inr    = round(pnl_inr, 2)
            self.total_pnl += pnl_inr

            self.closed.append(pos)
            self.positions.remove(pos)
            self._save_state()
            emoji = "✅" if pnl_inr > 0 else "❌"
            self._log(f"{emoji} CLOSED {pos.direction} {pos.instrument} | {exit_reason} | "
                      f"{pos.mode.upper()} {pos.expiry_str} | P&L={pnl_pts:+.0f}pts = ₹{pnl_inr:+,.0f}")

    def _square_off_all(self, reason: str = "EOD"):
        spot = self._get_spot()
        for pos in list(self.positions):
            pnl_pts = (spot - pos.entry) * (1 if pos.direction == 'LONG' else -1)
            pnl_inr = pnl_pts * pos.lot_size * pos.qty
            pos.status     = "CLOSED"
            pos.exit_price = round(spot, 2)
            pos.exit_reason= reason
            pos.pnl_pts    = round(pnl_pts, 2)
            pos.pnl_inr    = round(pnl_inr, 2)
            self.total_pnl += pnl_inr
            self.closed.append(pos)
        self.positions.clear()
        self._save_state()
        self._log(f"Square-off all ({reason}) | Total P&L: ₹{self.total_pnl:+,.0f}")
