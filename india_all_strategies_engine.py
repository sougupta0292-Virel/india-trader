"""
india_all_strategies_engine.py
Multi-timeframe (5M · 15M · 1H · 4H) strategy scanner and backtester
for all Nifty 50 stocks, index futures, and options.

Data: yfinance (.NS suffix for NSE equities)
yfinance limits:
  5m  / 15m  → max 60 days
  1h         → max 730 days
  4h         → resampled from 1h
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import warnings, logging, os, sys
warnings.filterwarnings('ignore')

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ─── Constants ────────────────────────────────────────────────────────────────

YF_INTERVAL = {"5M": "5m", "15M": "15m", "1H": "1h", "4H": "1h"}
MAX_DAYS     = {"5M": 55,  "15M": 55,    "1H": 200,  "4H": 400}

TF_STRATEGIES = {
    "5M":  ["ORB", "5EMA_CROSS", "VWAP_BOUNCE", "INSIDE_BAR"],
    "15M": ["FABIO_FIB", "VWAP_CPR", "ORB_15", "EMA_PULLBACK"],
    "1H":  ["EMA_TREND", "BOLLINGER", "VWAP_TREND", "KEY_LEVEL"],
    "4H":  ["SWING_EMA", "FIB_SWING", "RSI_REVERSAL", "TREND_CONT"],
}

# (short_name, full description)
STRATEGY_DESCRIPTIONS: Dict[str, Tuple[str, str]] = {
    "ORB":          ("Opening Range Breakout",
                     "First 5-min candle H/L defines the session range. Enter when a later candle "
                     "closes above (LONG) or below (SHORT) that range with a 1.5× target. "
                     "Works best on gap-up/gap-down opens."),
    "5EMA_CROSS":   ("5 EMA × 20 EMA Cross",
                     "Enter when the 5-period EMA crosses the 20-period EMA on the 5-min chart. "
                     "Fast scalp momentum signal; most effective during 9:15–11:00 AM sessions."),
    "VWAP_BOUNCE":  ("VWAP Bounce",
                     "Price retests the session VWAP and shows a reversal candle (close crosses "
                     "back through VWAP). Strong institutional support/resistance. Best on "
                     "range-bound mid-day sessions."),
    "INSIDE_BAR":   ("Inside Bar Breakout",
                     "Current bar's H/L fits inside the previous bar (range contraction). Enter on "
                     "the directional breakout with a 2× range target. Tight SL gives excellent RR."),

    "FABIO_FIB":    ("Fabio Fibonacci (15M)",
                     "Enter at the 61.8% Fibonacci retracement of the previous day's H/L on the "
                     "15-min chart. Trend confirmed by day-return PCR-proxy. Core India intraday setup."),
    "VWAP_CPR":     ("VWAP + CPR Confluence",
                     "Central Pivot Range (BC / TC) combined with the VWAP forms an institutional "
                     "cluster. Enter on a rejection from this zone; target opposite CPR side."),
    "ORB_15":       ("15-Min Opening Range",
                     "First 15-min candle H/L = opening range. Break above with 20 EMA aligned → "
                     "buy; break below → sell. Cleaner signal than 5M ORB; fewer false breaks."),
    "EMA_PULLBACK": ("5 EMA Pullback",
                     "In a trending market (EMA5 > EMA20 or vice versa), enter the first pullback "
                     "that touches and bounces off the 5 EMA. Trend alignment is mandatory."),

    "EMA_TREND":    ("20/50 EMA Trend (1H)",
                     "20 EMA crossing 50 EMA on the 1H chart confirms an intraday momentum shift. "
                     "Better for positional intraday trades expected to last 2–4 hours."),
    "BOLLINGER":    ("Bollinger Squeeze (1H)",
                     "When band-width falls below its 20th-percentile (historic squeeze), a "
                     "directional expansion follows. Enter the breakout candle above/below the bands."),
    "VWAP_TREND":   ("VWAP Sustained Trend",
                     "Three consecutive 1H candles closing on the same side of VWAP signals "
                     "institutional commitment. Enter on the 3rd candle continuation."),
    "KEY_LEVEL":    ("Prev-Day H/L Break (1H)",
                     "A full 1H candle closing above the previous day's high (or below its low) "
                     "confirms the daily trend. Entry = candle close; target = 2× ATR beyond the level."),

    "SWING_EMA":    ("Swing EMA 50/200 (4H)",
                     "50 EMA relative to 200 EMA on the 4H chart establishes the multi-session swing "
                     "bias. Above 200 EMA = bullish trend; below = bearish. Enter pullbacks to 50 EMA."),
    "FIB_SWING":    ("Fibonacci Swing Entry (4H)",
                     "Measure the dominant 20-bar swing (H/L). Enter at the 61.8% retracement with "
                     "RSI confirmation. Multi-day swing setup; expects a 1–3 session hold."),
    "RSI_REVERSAL": ("RSI Extreme Reversal (4H)",
                     "RSI below 30 (oversold) or above 70 (overbought) on the 4H chart with a "
                     "long-wick reversal candle. Counter-trend entry with tight SL at the candle extreme."),
    "TREND_CONT":   ("Trend Continuation (4H)",
                     "Three or more consecutive same-direction 4H bars followed by a 1-bar pause. "
                     "Enter the breakout of the pause bar in the trend direction."),
}

INDEX_UNIVERSE = {
    "NIFTY":     {"yf": "^NSEI",    "lot": 75,  "step": 50},
    "BANKNIFTY": {"yf": "^NSEBANK", "lot": 35,  "step": 100},
    "SENSEX":    {"yf": "^BSESN",   "lot": 20,  "step": 100},
}

try:
    from nse_stocks_engine import NIFTY50_STOCKS
except ImportError:
    NIFTY50_STOCKS: Dict = {}

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class StratSignal:
    symbol:    str
    tf:        str
    strategy:  str
    direction: str    # LONG / SHORT / FLAT
    entry:     float
    target:    float
    stop:      float
    rr:        float
    score:     int    # 0–10
    reason:    str

@dataclass
class StratTrade:
    date:        str
    symbol:      str
    strategy:    str
    tf:          str
    direction:   str
    entry:       float
    target:      float
    stop:        float
    exit_price:  float
    exit_reason: str
    pnl_pts:     float
    pnl_pct:     float
    rr:          float
    won:         bool

@dataclass
class StratBtResult:
    symbol:        str
    strategy:      str
    tf:            str
    segment:       str
    total_trades:  int
    winning:       int
    losing:        int
    win_rate:      float
    profit_factor: float
    total_pnl_pts: float
    total_pnl_rs:  float
    max_drawdown:  float
    trades:        List[StratTrade] = field(default_factory=list)
    equity_curve:  List[float]      = field(default_factory=list)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _f(v) -> float:
    try:
        while hasattr(v, 'iloc'): v = v.iloc[0]
        if hasattr(v, 'item'): return float(v.item())
        return float(v)
    except:
        return 0.0


def _yf_download_silent(yf_ticker: str, **kwargs) -> Optional[pd.DataFrame]:
    """yfinance download with stderr suppressed and .BO fallback if .NS fails."""
    import yfinance as yf
    tickers = [yf_ticker]
    if yf_ticker.endswith(".NS"):
        tickers.append(yf_ticker[:-3] + ".BO")
    for t in tickers:
        try:
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            try:
                df = yf.download(t, progress=False, auto_adjust=True, **kwargs)
            finally:
                sys.stderr.close()
                sys.stderr = old_stderr
            if df is not None and len(df) >= 5:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception:
            pass
    return None


def fetch_ohlcv(yf_ticker: str, tf: str, days: int) -> Optional[pd.DataFrame]:
    """Download OHLCV from yfinance, resample to 4H if needed."""
    interval = YF_INTERVAL.get(tf, "5m")
    actual   = min(days, MAX_DAYS.get(tf, 55))
    try:
        df = _yf_download_silent(yf_ticker, period=f"{actual}d", interval=interval)
        if df is None or len(df) < 5:
            return None
        for col in ('Open', 'High', 'Low', 'Close'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        if tf == "4H":
            try:
                df = df.resample('4h').agg(
                    {'Open': 'first', 'High': 'max', 'Low': 'min',
                     'Close': 'last', 'Volume': 'sum'}).dropna()
            except Exception:
                pass
        df = df.copy()
        df['_date'] = (df.index.date if hasattr(df.index, 'date')
                       else pd.to_datetime(df.index).date)
        return df if len(df) >= 5 else None
    except Exception:
        return None


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMAs, RSI, ATR, VWAP, and Bollinger Bands."""
    df = df.copy()
    c = df['Close'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)

    df['ema5']   = c.ewm(span=5,   adjust=False).mean()
    df['ema9']   = c.ewm(span=9,   adjust=False).mean()
    df['ema20']  = c.ewm(span=20,  adjust=False).mean()
    df['ema50']  = c.ewm(span=50,  adjust=False).mean()
    df['ema200'] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, 0.0001)))

    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_mid']   = bb_mid
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid.replace(0, 1)

    # Session-reset VWAP
    if 'Volume' in df.columns:
        tp = (h + l + c) / 3
        vwap_vals: List[float] = []
        cum_tv = cum_v = 0.0
        prev_d = None
        for i, idx in enumerate(df.index):
            d = idx.date() if hasattr(idx, 'date') else None
            if d != prev_d:
                cum_tv = cum_v = 0.0
                prev_d = d
            v = max(_f(df['Volume'].iloc[i]), 1.0)
            cum_tv += _f(tp.iloc[i]) * v
            cum_v  += v
            vwap_vals.append(cum_tv / cum_v)
        df['vwap'] = vwap_vals
    else:
        df['vwap'] = c

    return df


def _prev_day_hl(df: pd.DataFrame, idx: int) -> Tuple[float, float, float]:
    """Return (prev_high, prev_low, prev_close) for the bar at position idx."""
    bar_date  = df['_date'].iloc[idx]
    prev_rows = df[df['_date'] < bar_date]
    if prev_rows.empty:
        return 0.0, 0.0, 0.0
    last_prev = prev_rows[prev_rows['_date'] == prev_rows['_date'].iloc[-1]]
    return (_f(last_prev['High'].max()),
            _f(last_prev['Low'].min()),
            _f(last_prev['Close'].iloc[-1]))


def _simulate_exit(df: pd.DataFrame, start: int, direction: str,
                   entry: float, target: float, stop: float,
                   max_bars: int = 30) -> Tuple[float, str]:
    n = min(start + max_bars, len(df))
    for j in range(start, n):
        h = _f(df.iloc[j]['High'])
        l = _f(df.iloc[j]['Low'])
        if direction == "LONG":
            if l <= stop:   return stop, "STOP_LOSS"
            if h >= target: return target, "TARGET_HIT"
        else:
            if h >= stop:   return stop, "STOP_LOSS"
            if l <= target: return target, "TARGET_HIT"
    ep = _f(df.iloc[n - 1]['Close']) if n > start else entry
    return ep, "TIME_EXIT"

# ─── Strategy signal functions ─────────────────────────────────────────────
# Each returns (direction, entry, target, stop, score, reason) or None.

def _sig_orb(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 3: return None
    today    = df['_date'].iloc[-1]
    today_df = df[df['_date'] == today]
    if len(today_df) < 2: return None
    orb_h = _f(today_df.iloc[0]['High'])
    orb_l = _f(today_df.iloc[0]['Low'])
    curr  = today_df.iloc[-1]
    c = _f(curr['Close']); atr = _f(curr['atr'])
    rng = orb_h - orb_l
    if rng < atr * 0.3: return None
    if c > orb_h:
        tgt = c + rng * 1.5; stp = orb_l
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, min(int(rr * 2), 9),
                f"ORB ↑ range={rng:.0f} RR={rr:.1f}")
    if c < orb_l:
        tgt = c - rng * 1.5; stp = orb_h
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, min(int(rr * 2), 9),
                f"ORB ↓ range={rng:.0f} RR={rr:.1f}")
    return None


def _sig_5ema_cross(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 22: return None
    curr = df.iloc[-1]; prev = df.iloc[-2]
    c = _f(curr['Close']); atr = _f(curr['atr'])
    e5 = _f(curr['ema5']); e20 = _f(curr['ema20'])
    pe5 = _f(prev['ema5']); pe20 = _f(prev['ema20'])
    if e5 > e20 and pe5 <= pe20:
        tgt = c + atr * 2.5; stp = c - atr * 1.2
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 7, f"5EMA×20EMA bull | E5={e5:.0f}")
    if e5 < e20 and pe5 >= pe20:
        tgt = c - atr * 2.5; stp = c + atr * 1.2
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 7, f"5EMA×20EMA bear | E5={e5:.0f}")
    return None


def _sig_vwap_bounce(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 5: return None
    curr = df.iloc[-1]; prev = df.iloc[-2]
    c = _f(curr['Close']); atr = _f(curr['atr'])
    vwap = _f(curr['vwap']); e5 = _f(curr['ema5'])
    pl = _f(prev['Low']); ph = _f(prev['High'])
    dist = abs(c - vwap) / max(atr, 1)
    if dist > 0.7: return None
    if pl < vwap < c and c > e5:
        tgt = c + atr * 1.5; stp = vwap - atr * 0.6
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 6, f"VWAP bounce ↑ VWAP={vwap:.0f}")
    if ph > vwap > c and c < e5:
        tgt = c - atr * 1.5; stp = vwap + atr * 0.6
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 6, f"VWAP bounce ↓ VWAP={vwap:.0f}")
    return None


def _sig_inside_bar(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 4: return None
    curr  = df.iloc[-1]; prev = df.iloc[-2]; pprev = df.iloc[-3]
    ch = _f(curr['High']); cl = _f(curr['Low'])
    ph = _f(prev['High']); pl = _f(prev['Low'])
    pph = _f(pprev['High']); ppl = _f(pprev['Low'])
    c = _f(curr['Close']); atr = _f(curr['atr'])
    if not (ph <= pph and pl >= ppl): return None   # prev was inside pprev
    rng = pph - ppl
    if rng < atr * 0.3: return None
    if c > pph:
        tgt = c + rng * 2.0; stp = ppl
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 7, f"Inside bar break ↑ range={rng:.0f}")
    if c < ppl:
        tgt = c - rng * 2.0; stp = pph
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 7, f"Inside bar break ↓ range={rng:.0f}")
    return None


def _sig_fabio_fib(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 10: return None
    ph, pl, pc = _prev_day_hl(df, len(df) - 1)
    if ph == 0: return None
    diff = ph - pl
    if diff < 50: return None
    curr = df.iloc[-1]
    c = _f(curr['Close']); atr = _f(curr['atr']); e5 = _f(curr['ema5'])
    pcr_proxy = (pc - pl) / diff if diff else 0.5
    f618l = pl + diff * 0.618; f618s = ph - diff * 0.618
    f786l = pl + diff * 0.786; f786s = ph - diff * 0.786
    if abs(c - f618l) / diff < 0.35 and pcr_proxy > 0.5 and e5 > f618l * 0.993:
        tgt = ph; stp = f786l - atr * 0.3
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 8, f"61.8% Fib LONG PH={ph:.0f} PL={pl:.0f}")
    if abs(c - f618s) / diff < 0.35 and pcr_proxy < 0.5 and e5 < f618s * 1.007:
        tgt = pl; stp = f786s + atr * 0.3
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 8, f"61.8% Fib SHORT PH={ph:.0f} PL={pl:.0f}")
    return None


def _sig_vwap_cpr(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 5: return None
    ph, pl, pc_v = _prev_day_hl(df, len(df) - 1)
    if ph == 0: return None
    pivot = (ph + pl + pc_v) / 3
    bc    = (ph + pl) / 2
    tc    = pivot * 2 - bc
    if tc < bc: tc, bc = bc, tc
    curr = df.iloc[-1]
    c = _f(curr['Close']); atr = _f(curr['atr'])
    vwap = _f(curr['vwap']); e5 = _f(curr['ema5'])
    if abs(c - bc) / max(atr, 1) < 1.2 and c > vwap and c > e5:
        tgt = tc + atr; stp = bc - atr * 0.8
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 7, f"VWAP+CPR bull BC={bc:.0f} VWAP={vwap:.0f}")
    if abs(c - tc) / max(atr, 1) < 1.2 and c < vwap and c < e5:
        tgt = bc - atr; stp = tc + atr * 0.8
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 7, f"VWAP+CPR bear TC={tc:.0f} VWAP={vwap:.0f}")
    return None


def _sig_orb15(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 3: return None
    today    = df['_date'].iloc[-1]
    today_df = df[df['_date'] == today]
    if len(today_df) < 2: return None
    orb_h = _f(today_df.iloc[0]['High'])
    orb_l = _f(today_df.iloc[0]['Low'])
    curr  = today_df.iloc[-1]
    c = _f(curr['Close']); atr = _f(curr['atr']); e20 = _f(curr['ema20'])
    rng = orb_h - orb_l
    if rng < atr * 0.5: return None
    if c > orb_h and c > e20:
        tgt = c + rng * 1.5; stp = orb_l
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 7, f"ORB15 ↑ range={rng:.0f} E20={e20:.0f}")
    if c < orb_l and c < e20:
        tgt = c - rng * 1.5; stp = orb_h
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 7, f"ORB15 ↓ range={rng:.0f} E20={e20:.0f}")
    return None


def _sig_ema_pullback(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 22: return None
    curr = df.iloc[-1]; prev = df.iloc[-2]
    c = _f(curr['Close']); atr = _f(curr['atr'])
    e5 = _f(curr['ema5']); e20 = _f(curr['ema20'])
    pl = _f(prev['Low']); ph = _f(prev['High']); pe5 = _f(prev['ema5'])
    if e5 > e20 and pl <= pe5 and c > ph:
        tgt = c + atr * 2.0; stp = e5 - atr * 0.5
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 8, f"EMA pull ↑ E5={e5:.0f} E20={e20:.0f}")
    if e5 < e20 and ph >= pe5 and c < pl:
        tgt = c - atr * 2.0; stp = e5 + atr * 0.5
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 8, f"EMA pull ↓ E5={e5:.0f} E20={e20:.0f}")
    return None


def _sig_ema_trend_1h(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 55: return None
    curr = df.iloc[-1]; prev = df.iloc[-2]
    c = _f(curr['Close']); atr = _f(curr['atr'])
    e20 = _f(curr['ema20']); e50 = _f(curr['ema50'])
    pe20 = _f(prev['ema20']); pe50 = _f(prev['ema50'])
    if e20 > e50 and pe20 <= pe50:
        tgt = c + atr * 3.0; stp = c - atr * 1.5
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 8, f"20/50 EMA bull cross E20={e20:.0f}")
    if e20 < e50 and pe20 >= pe50:
        tgt = c - atr * 3.0; stp = c + atr * 1.5
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 8, f"20/50 EMA bear cross E20={e20:.0f}")
    return None


def _sig_bollinger_1h(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 30: return None
    bw_series = df['bb_width'].dropna()
    if len(bw_series) < 20: return None
    curr = df.iloc[-1]
    c = _f(curr['Close']); atr = _f(curr['atr'])
    bw = _f(curr['bb_width'])
    bw_p20 = float(np.percentile(bw_series, 20))
    bb_u = _f(curr['bb_upper']); bb_l = _f(curr['bb_lower']); bb_m = _f(curr['bb_mid'])
    if bw > bw_p20 * 1.5: return None  # not in squeeze
    if c > bb_u:
        tgt = c + atr * 2.0; stp = bb_m
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 8, f"BB squeeze ↑ bw={bw:.4f}")
    if c < bb_l:
        tgt = c - atr * 2.0; stp = bb_m
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 8, f"BB squeeze ↓ bw={bw:.4f}")
    return None


def _sig_vwap_trend_1h(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 5: return None
    last3 = df.tail(3)
    closes = [_f(r['Close']) for _, r in last3.iterrows()]
    vwaps  = [_f(r['vwap'])  for _, r in last3.iterrows()]
    c = closes[-1]; atr = _f(df.iloc[-1]['atr'])
    if all(closes[i] > vwaps[i] for i in range(3)):
        tgt = c + atr * 2.5; stp = vwaps[-1] - atr * 0.3
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 6, f"VWAP trend ↑ 3-bar VWAP={vwaps[-1]:.0f}")
    if all(closes[i] < vwaps[i] for i in range(3)):
        tgt = c - atr * 2.5; stp = vwaps[-1] + atr * 0.3
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 6, f"VWAP trend ↓ 3-bar VWAP={vwaps[-1]:.0f}")
    return None


def _sig_key_level_1h(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 5: return None
    ph, pl, _ = _prev_day_hl(df, len(df) - 1)
    if ph == 0: return None
    curr = df.iloc[-1]
    c = _f(curr['Close']); atr = _f(curr['atr'])
    if c > ph:
        tgt = c + atr * 2.0; stp = ph - atr * 0.5
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 7, f"PDH break ↑ PDH={ph:.0f}")
    if c < pl:
        tgt = c - atr * 2.0; stp = pl + atr * 0.5
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 7, f"PDL break ↓ PDL={pl:.0f}")
    return None


def _sig_swing_ema_4h(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 60: return None
    curr = df.iloc[-1]
    c = _f(curr['Close']); atr = _f(curr['atr'])
    e50 = _f(curr['ema50']); e200 = _f(curr['ema200']); rsi = _f(curr['rsi'])
    if e50 > e200 and abs(c - e50) / max(atr, 1) < 1.5 and rsi < 60:
        tgt = c + atr * 4.0; stp = e200 - atr * 0.5
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 8, f"SwingEMA E50>E200 pullback RSI={rsi:.0f}")
    if e50 < e200 and abs(c - e50) / max(atr, 1) < 1.5 and rsi > 40:
        tgt = c - atr * 4.0; stp = e200 + atr * 0.5
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 8, f"SwingEMA E50<E200 pullback RSI={rsi:.0f}")
    return None


def _sig_fib_swing_4h(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 22: return None
    recent = df.tail(20)
    swing_h = _f(recent['High'].max())
    swing_l = _f(recent['Low'].min())
    diff    = swing_h - swing_l
    curr = df.iloc[-1]
    c = _f(curr['Close']); atr = _f(curr['atr']); rsi = _f(curr['rsi'])
    if diff < atr * 2: return None
    f618b = swing_l + diff * 0.618
    f618s = swing_h - diff * 0.618
    if abs(c - f618b) / diff < 0.12 and rsi < 55:
        tgt = swing_h; stp = swing_l - atr * 0.5
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 8, f"4H Fib 61.8% LONG swing={diff:.0f}")
    if abs(c - f618s) / diff < 0.12 and rsi > 45:
        tgt = swing_l; stp = swing_h + atr * 0.5
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 8, f"4H Fib 61.8% SHORT swing={diff:.0f}")
    return None


def _sig_rsi_reversal_4h(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 15: return None
    curr = df.iloc[-1]; prev = df.iloc[-2]
    c = _f(curr['Close']); atr = _f(curr['atr']); rsi = _f(curr['rsi'])
    pc = _f(prev['Close'])
    ch = _f(curr['High']); cl = _f(curr['Low']); co = _f(curr['Open'])
    body = abs(c - co)
    lower_wick = (min(c, co) - cl)
    upper_wick = (ch - max(c, co))
    if rsi < 30 and c > pc and lower_wick > body * 1.5:
        tgt = c + atr * 3.0; stp = cl - atr * 0.3
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 8, f"RSI oversold reversal RSI={rsi:.0f}")
    if rsi > 70 and c < pc and upper_wick > body * 1.5:
        tgt = c - atr * 3.0; stp = ch + atr * 0.3
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 8, f"RSI overbought reversal RSI={rsi:.0f}")
    return None


def _sig_trend_cont_4h(df: pd.DataFrame) -> Optional[Tuple]:
    if len(df) < 6: return None
    bars   = df.tail(5)
    opens  = [_f(r['Open'])  for _, r in bars.iterrows()]
    closes = [_f(r['Close']) for _, r in bars.iterrows()]
    highs  = [_f(r['High'])  for _, r in bars.iterrows()]
    lows   = [_f(r['Low'])   for _, r in bars.iterrows()]
    curr = bars.iloc[-1]
    c = closes[-1]; atr = _f(curr['atr'])
    # 3 bull bars, pause, breakout
    if (closes[0] > opens[0] and closes[1] > opens[1] and closes[2] > opens[2]
            and highs[4] > highs[3]):
        tgt = c + atr * 2.5; stp = lows[3] - atr * 0.3
        rr  = (tgt - c) / (c - stp) if c > stp else 0
        if rr < 1.2: return None
        return ("LONG", c, tgt, stp, 7, "4H trend cont ↑ 3-bull + break")
    if (closes[0] < opens[0] and closes[1] < opens[1] and closes[2] < opens[2]
            and lows[4] < lows[3]):
        tgt = c - atr * 2.5; stp = highs[3] + atr * 0.3
        rr  = (c - tgt) / (stp - c) if stp > c else 0
        if rr < 1.2: return None
        return ("SHORT", c, tgt, stp, 7, "4H trend cont ↓ 3-bear + break")
    return None


# ─── Dispatcher ───────────────────────────────────────────────────────────────

SIGNAL_FNS = {
    "ORB":          _sig_orb,
    "5EMA_CROSS":   _sig_5ema_cross,
    "VWAP_BOUNCE":  _sig_vwap_bounce,
    "INSIDE_BAR":   _sig_inside_bar,
    "FABIO_FIB":    _sig_fabio_fib,
    "VWAP_CPR":     _sig_vwap_cpr,
    "ORB_15":       _sig_orb15,
    "EMA_PULLBACK": _sig_ema_pullback,
    "EMA_TREND":    _sig_ema_trend_1h,
    "BOLLINGER":    _sig_bollinger_1h,
    "VWAP_TREND":   _sig_vwap_trend_1h,
    "KEY_LEVEL":    _sig_key_level_1h,
    "SWING_EMA":    _sig_swing_ema_4h,
    "FIB_SWING":    _sig_fib_swing_4h,
    "RSI_REVERSAL": _sig_rsi_reversal_4h,
    "TREND_CONT":   _sig_trend_cont_4h,
}

# ─── Scanner ──────────────────────────────────────────────────────────────────

class MultiTFScanner:
    """Scan any symbol/universe across all strategies for a given timeframe."""

    @staticmethod
    def scan_symbol(yf_ticker: str, symbol: str, tf: str,
                    days: int = 12) -> List[StratSignal]:
        strats = TF_STRATEGIES.get(tf, [])
        df = fetch_ohlcv(yf_ticker, tf, days)
        if df is None or len(df) < 5:
            return []
        df = _indicators(df)
        results: List[StratSignal] = []
        for strat in strats:
            fn = SIGNAL_FNS.get(strat)
            if fn is None:
                continue
            try:
                sig = fn(df)
            except Exception:
                sig = None
            if sig is None:
                results.append(StratSignal(symbol=symbol, tf=tf, strategy=strat,
                                           direction="FLAT", entry=0, target=0,
                                           stop=0, rr=0, score=0, reason="No setup"))
                continue
            direction, entry, target, stop, score, reason = sig
            rr = round(abs(target - entry) / abs(entry - stop), 2) if entry != stop else 0
            results.append(StratSignal(symbol=symbol, tf=tf, strategy=strat,
                                       direction=direction,
                                       entry=round(entry, 2), target=round(target, 2),
                                       stop=round(stop, 2), rr=rr, score=score,
                                       reason=reason))
        return results

    @staticmethod
    def scan_indices(tf: str, days: int = 12) -> List[StratSignal]:
        all_sigs: List[StratSignal] = []
        for name, meta in INDEX_UNIVERSE.items():
            all_sigs.extend(MultiTFScanner.scan_symbol(meta["yf"], name, tf, days))
        return all_sigs

    @staticmethod
    def scan_stocks(tf: str, symbols: Optional[List[str]] = None,
                    days: int = 12, progress_cb=None) -> List[StratSignal]:
        symbols = symbols or list(NIFTY50_STOCKS.keys())
        all_sigs: List[StratSignal] = []
        for i, sym in enumerate(symbols):
            meta = NIFTY50_STOCKS.get(sym, {})
            yf_t = meta.get("yf", f"{sym}.NS")
            all_sigs.extend(MultiTFScanner.scan_symbol(yf_t, sym, tf, days))
            if progress_cb:
                progress_cb(i + 1, len(symbols), sym)
        return all_sigs

# ─── Backtester ───────────────────────────────────────────────────────────────

class MultiTFBacktester:
    """Backtest all strategies for a given timeframe across the requested universe."""

    def __init__(self, tf: str, days: int, segment: str = "EQUITY"):
        self.tf      = tf
        self.days    = days
        self.segment = segment

    # ── single symbol ─────────────────────────────────────────────────────────
    def _bt_symbol(self, yf_ticker: str, symbol: str,
                   strategy: str, lot_size: int = 1) -> StratBtResult:
        df_raw = fetch_ohlcv(yf_ticker, self.tf, self.days)
        if df_raw is None or len(df_raw) < 25:
            return self._empty(symbol, strategy)
        df = _indicators(df_raw)
        fn = SIGNAL_FNS.get(strategy)
        if fn is None:
            return self._empty(symbol, strategy)

        trades:   List[StratTrade] = []
        equity:   List[float]      = [100_000.0]
        cur_eq    = 100_000.0
        trade_dates: set           = set()

        for i in range(20, len(df) - 1):
            bar_date = str(df['_date'].iloc[i])
            if bar_date in trade_dates:
                continue
            try:
                sig = fn(df.iloc[:i + 1])
            except Exception:
                continue
            if sig is None:
                continue
            direction, entry, target, stop, score, reason = sig
            if entry <= 0 or stop <= 0 or target <= 0:
                continue
            risk = abs(entry - stop)
            if risk == 0:
                continue
            rr = abs(target - entry) / risk
            if rr < 1.2:
                continue
            # Skip trades with inverted stops (SL on wrong side of entry)
            if direction == "LONG"  and stop >= entry:
                continue
            if direction == "SHORT" and stop <= entry:
                continue

            trade_dates.add(bar_date)
            exit_p, exit_r = _simulate_exit(df, i + 1, direction,
                                             entry, target, stop)
            pnl_pts = (exit_p - entry) if direction == "LONG" else (entry - exit_p)
            # Relabel exit reason if P&L sign contradicts the label
            if exit_r == "STOP_LOSS"  and pnl_pts > 0:
                exit_r = "TARGET_HIT"
            elif exit_r == "TARGET_HIT" and pnl_pts < 0:
                exit_r = "STOP_LOSS"
            pnl_pct = pnl_pts / entry * 100 if entry > 0 else 0

            lot_mult = (lot_size if self.segment == "FUTURES"
                        else max(1, lot_size // 2) if self.segment == "OPTIONS"
                        else 1)
            cur_eq += pnl_pts * lot_mult
            equity.append(cur_eq)

            trades.append(StratTrade(
                date=bar_date, symbol=symbol, strategy=strategy, tf=self.tf,
                direction=direction, entry=round(entry, 2),
                target=round(target, 2), stop=round(stop, 2),
                exit_price=round(exit_p, 2), exit_reason=exit_r,
                pnl_pts=round(pnl_pts, 2), pnl_pct=round(pnl_pct, 2),
                rr=round(rr, 2), won=pnl_pts > 0))

        return self._build(symbol, strategy, trades, equity, lot_size)

    def _build(self, symbol: str, strategy: str,
               trades: List[StratTrade], equity: List[float],
               lot_size: int) -> StratBtResult:
        if not trades:
            return self._empty(symbol, strategy)
        wins   = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        wr     = len(wins) / len(trades) * 100 if trades else 0
        avg_w  = float(np.mean([t.pnl_pts for t in wins]))  if wins   else 0.0
        avg_l  = abs(float(np.mean([t.pnl_pts for t in losses]))) if losses else 0.0001
        pf     = (len(wins) * avg_w) / (len(losses) * avg_l) if losses else 99.0
        total  = sum(t.pnl_pts for t in trades)
        eq_arr = np.array(equity, dtype=float)
        peak   = np.maximum.accumulate(eq_arr)
        dd     = (peak - eq_arr) / np.where(peak > 0, peak, 1) * 100
        return StratBtResult(
            symbol=symbol, strategy=strategy, tf=self.tf, segment=self.segment,
            total_trades=len(trades), winning=len(wins), losing=len(losses),
            win_rate=round(wr, 1), profit_factor=round(pf, 2),
            total_pnl_pts=round(total, 2),
            total_pnl_rs=round(total * lot_size, 2),
            max_drawdown=round(float(dd.max()), 2),
            trades=trades, equity_curve=equity)

    def _empty(self, symbol: str, strategy: str) -> StratBtResult:
        return StratBtResult(
            symbol=symbol, strategy=strategy, tf=self.tf, segment=self.segment,
            total_trades=0, winning=0, losing=0, win_rate=0, profit_factor=0,
            total_pnl_pts=0, total_pnl_rs=0, max_drawdown=0,
            trades=[], equity_curve=[100_000.0])

    # ── run all indices ────────────────────────────────────────────────────────
    def run_indices(self, strategies: Optional[List[str]] = None,
                    progress_cb=None) -> Dict[str, StratBtResult]:
        strats  = strategies or TF_STRATEGIES.get(self.tf, [])
        results: Dict[str, StratBtResult] = {}
        total   = len(INDEX_UNIVERSE) * len(strats)
        done    = 0
        for name, meta in INDEX_UNIVERSE.items():
            for strat in strats:
                results[f"{name}_{strat}"] = self._bt_symbol(
                    meta["yf"], name, strat, meta["lot"])
                done += 1
                if progress_cb: progress_cb(done, total, f"{name} {strat}")
        return results

    # ── run all Nifty 50 stocks ────────────────────────────────────────────────
    def run_stocks(self, strategies: Optional[List[str]] = None,
                   symbols: Optional[List[str]] = None,
                   progress_cb=None) -> Dict[str, StratBtResult]:
        strats  = strategies or TF_STRATEGIES.get(self.tf, [])
        symbols = symbols    or list(NIFTY50_STOCKS.keys())
        results: Dict[str, StratBtResult] = {}
        total   = len(symbols) * len(strats)
        done    = 0
        for sym in symbols:
            meta = NIFTY50_STOCKS.get(sym, {})
            yf_t = meta.get("yf", f"{sym}.NS")
            lot  = meta.get("lot", 1)
            for strat in strats:
                results[f"{sym}_{strat}"] = self._bt_symbol(yf_t, sym, strat, lot)
                done += 1
                if progress_cb: progress_cb(done, total, f"{sym} {strat}")
        return results
