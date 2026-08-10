"""
pos_auto_trader.py
==================
Automated trading engine for Power of Stocks strategies.
Uses Angel One API for live data + order placement.

Strategies:
1. 5 EMA — candle away from EMA, break = trade
2. Inside Candle — mother + inside bar breakout
3. Traffic Light — 3 consecutive candles
4. Round Number — psychological level rejection

Safety rules (hardcoded):
- Only trades 9:20 AM to 3:15 PM IST
- Max 1 trade per day
- Auto square off at 3:15 PM
- SL always placed immediately
- Paper trade by default
"""

import time
import json
import os
import threading
import pandas as pd
import numpy as np
from datetime import datetime, date, time as dtime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

_IST = timezone(timedelta(hours=5, minutes=30))

def _ist_date_str() -> str:
    """Return current date in IST as YYYY-MM-DD string."""
    return datetime.now(_IST).strftime('%Y-%m-%d')

# ─── Daily Capital Cap ────────────────────────────────────────────────────────

DAILY_CAP_RS = 1_000_000  # ₹10 lakhs max deployment per day across ALL bots

_IDX_SET = {'NIFTY', 'BANKNIFTY', 'SENSEX'}

def _get_today_deployed_rs() -> float:
    """Sum investment deployed today across all equity/futures and options state files."""
    import glob as _g
    _dir  = os.path.dirname(os.path.abspath(__file__))
    today = _ist_date_str()
    total = 0.0

    for pf in _g.glob(os.path.join(_dir, "pos_state_*.json")):
        try:
            with open(pf, encoding='utf-8') as f:
                data = json.load(f)
            for t in data.get('history', []):
                if t.get('date') != today:
                    continue
                ep  = float(t.get('entry', 0) or 0)
                qty = float(t.get('qty', 1) or 1)
                sym = str(t.get('instrument', '')).upper()
                total += ep * qty * 0.1 if sym in _IDX_SET else ep * qty
            ot = data.get('open_trade')
            if ot and ot.get('date') == today:
                ep  = float(ot.get('entry', 0) or 0)
                qty = float(ot.get('qty', 1) or 1)
                sym = str(ot.get('instrument', '')).upper()
                total += ep * qty * 0.1 if sym in _IDX_SET else ep * qty
        except Exception:
            pass

    for pf in _g.glob(os.path.join(_dir, "options_state*.json")):
        try:
            with open(pf, encoding='utf-8') as f:
                data = json.load(f)
            for t in data.get('history', []):
                if t.get('date') != today:
                    continue
                ep  = float(t.get('entry_premium', 0) or 0)
                qty = float(t.get('qty', 1) or 1)
                total += ep * qty
            ot = data.get('open_trade')
            if ot and ot.get('date') == today:
                ep  = float(ot.get('entry_premium', 0) or 0)
                qty = float(ot.get('qty', 1) or 1)
                total += ep * qty
        except Exception:
            pass

    return total

# ─── Trade Record ─────────────────────────────────────────────────────────────

@dataclass
class POSTrade:
    id:           str
    date:         str
    time:         str
    strategy:     str
    instrument:   str
    direction:    str
    entry:        float
    target:       float
    stop_loss:    float
    qty:          int
    status:       str = "OPEN"    # OPEN/CLOSED
    exit_price:   float = 0.0
    exit_reason:  str = ""
    pnl_pts:      float = 0.0
    pnl_rs:       float = 0.0
    paper:        bool = True

    def to_dict(self): return self.__dict__


# ─── Auto Trader State ────────────────────────────────────────────────────────

@dataclass
class POSState:
    status:        str = "STOPPED"
    open_trade:    Optional[POSTrade] = None
    trades_today:  int = 0
    total_trades:  int = 0
    winning:       int = 0
    total_pnl:     float = 0.0
    last_scan:     str = ""
    last_signal:   str = "—"
    last_action:   str = "—"
    log:           list = field(default_factory=list)
    history:       list = field(default_factory=list)

    STATE_FILE = os.path.join(os.path.dirname(__file__), 'pos_state.json')

    @staticmethod
    def _state_file(instrument: str = None) -> str:
        _dir = os.path.dirname(__file__)
        if instrument:
            return os.path.join(_dir, f'pos_state_{instrument.upper()}.json')
        return POSState.STATE_FILE

    def save(self, instrument: str = None):
        data = {
            'status':       self.status,
            'global_bias':  getattr(self, 'global_bias', 'NEUTRAL'),
            'trades_today': self.trades_today,
            'total_trades': self.total_trades,
            'winning':      self.winning,
            'total_pnl':    self.total_pnl,
            'last_scan':    self.last_scan,
            'last_signal':  self.last_signal,
            'last_action':  self.last_action,
            'log':          self.log[-100:],
            'history':      [t.to_dict() for t in self.history[-200:]],
            'open_trade':   self.open_trade.to_dict() if self.open_trade else None,
            'date':         _ist_date_str(),
        }
        with open(POSState._state_file(instrument), 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(instrument: str = None) -> 'POSState':
        _file = POSState._state_file(instrument)
        state = POSState()
        if not os.path.exists(_file):
            return state
        try:
            with open(_file, encoding='utf-8') as f:
                raw = f.read().strip()
            if not raw:
                return state  # empty/corrupt file — start fresh silently
            data = json.loads(raw)
            # Reset daily counters if new day (compare using IST date)
            if data.get('date') != _ist_date_str():
                data['trades_today'] = 0
                data['open_trade']   = None

            state.status       = data.get('status', 'STOPPED')
            state.trades_today = data.get('trades_today', 0)
            state.total_trades = data.get('total_trades', 0)
            state.winning      = data.get('winning', 0)
            state.total_pnl    = data.get('total_pnl', 0.0)
            state.last_scan    = data.get('last_scan', '')
            state.last_signal  = data.get('last_signal', '—')
            state.last_action  = data.get('last_action', '—')
            state.log          = data.get('log', [])
            state.history      = [POSTrade(**t) for t in data.get('history', [])]
            if data.get('open_trade'):
                state.open_trade = POSTrade(**data['open_trade'])
        except Exception:
            pass  # corrupt file — start fresh silently
        return state

    def log_msg(self, msg: str, level: str = "INFO"):
        entry = {
            'time':    datetime.now(_IST).strftime('%H:%M:%S'),
            'level':   level,
            'message': msg
        }
        self.log.append(entry)
        print(f"[{entry['time']}] {level}: {msg}")


# ─── Live Data Engine ─────────────────────────────────────────────────────────

class LiveDataEngine:
    """Fetches live 5-min candles from Angel One Smart API."""

    SYMBOLS = {
        "NIFTY":     {"token": "26000", "exchange": "NSE"},
        "BANKNIFTY": {"token": "26009", "exchange": "NSE"},
        "SENSEX":    {"token": "1",     "exchange": "BSE"},
    }

    def __init__(self, angel_client=None):
        self.client = angel_client

    def get_candles(self, symbol: str, count: int = 20,
                    interval: str = "5m") -> Optional[pd.DataFrame]:
        """Get last N candles via yfinance. interval: '5m' or '15m'."""
        return self._from_yfinance(symbol, count, interval)

    def _from_angel(self, symbol: str, count: int) -> Optional[pd.DataFrame]:
        """Fetch from Angel One historical API."""
        from datetime import datetime, timedelta
        meta    = self.SYMBOLS.get(symbol, self.SYMBOLS["NIFTY"])
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(hours=count//12 + 1)

        params = {
            "exchange":    meta["exchange"],
            "symboltoken": meta["token"],
            "interval":    "FIVE_MINUTE",
            "fromdate":    from_dt.strftime('%Y-%m-%d %H:%M'),
            "todate":      to_dt.strftime('%Y-%m-%d %H:%M'),
        }
        import requests
        url  = "https://apiconnect.angelbroking.com/rest/secure/angelbroking/historical/v1/getCandleData"
        resp = requests.post(url, json=params,
                              headers=self.client._headers(auth=True), timeout=10)
        data = resp.json()
        if data.get('status') and data.get('data'):
            df = pd.DataFrame(data['data'],
                               columns=['ts','open','high','low','close','volume'])
            df['ts']    = pd.to_datetime(df['ts'])
            df = df.set_index('ts').astype(float)
            df.columns  = ['Open','High','Low','Close','Volume']
            return df.tail(count)
        return None

    def _from_yfinance(self, symbol: str, count: int,
                       interval: str = "5m") -> Optional[pd.DataFrame]:
        """Fetch candles from yfinance with retry. interval: '5m' or '15m'."""
        import time as _time
        import yfinance as yf
        sym_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
        ticker_sym = sym_map.get(symbol, symbol + ".NS")
        _period_map = {"5m": "5d", "15m": "10d", "1h": "60d"}
        period = _period_map.get(interval, "10d")
        for attempt in range(2):
            try:
                df = yf.download(ticker_sym, period=period, interval=interval,
                                 progress=False, auto_adjust=True)
                if df is not None and len(df) > 0:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df['ema5'] = df['Close'].ewm(span=5, adjust=False).mean()
                    return df.tail(count)
            except Exception as e:
                if attempt == 0:
                    _time.sleep(2)
                else:
                    print(f"yfinance candles failed ({symbol} {interval}): {e}")
        return None

    def get_ltp(self, symbol: str) -> float:
        """Get current price with retry."""
        import time as _time
        if self.client and self.client.connected:
            try:
                quote = self.client.get_quote(symbol)
                if quote and quote.get('ltp') and float(quote['ltp']) > 0:
                    return float(quote['ltp'])
            except Exception:
                pass
        import yfinance as yf
        sym_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
        ticker_sym = sym_map.get(symbol, symbol + ".NS")
        for attempt in range(2):
            try:
                price = float(yf.Ticker(ticker_sym).fast_info.last_price)
                if price > 0:
                    return price
            except Exception as e:
                if attempt == 0:
                    _time.sleep(2)
                else:
                    print(f"yfinance LTP failed ({symbol}): {e}")
        return 0.0


# ─── Signal Detector ──────────────────────────────────────────────────────────

class POSSignalDetector:
    """Detects 5 EMA and Inside Candle signals on live candles."""

    @staticmethod
    def check_5ema(df: pd.DataFrame) -> Optional[dict]:
        """
        5 EMA Strategy:
        - Candle completely above EMA5 (no touch) + next breaks low → SHORT
        - Candle completely below EMA5 (no touch) + next breaks high → LONG
        """
        if df is None or len(df) < 6:
            return None

        df = df.copy()
        df['ema5'] = df['Close'].ewm(span=5, adjust=False).mean()

        # Check last 2 completed candles
        prev = df.iloc[-3]   # Signal candle
        curr = df.iloc[-2]   # Confirmation candle (just closed)
        ema5 = float(prev['ema5'])

        ph = float(prev['High']); pl = float(prev['Low'])
        po = float(prev['Open']); pc = float(prev['Close'])
        ch = float(curr['High']); cl = float(curr['Low'])

        # SHORT: prev candle entirely above EMA, curr breaks prev low
        if pl > ema5 and pc > ema5 and po > ema5:
            if cl < pl:   # Breaks previous low
                return {
                    'direction': 'SHORT',
                    'strategy':  '5EMA',
                    'entry':     pl,
                    'sl':        ph + 10,
                    'target':    pl - (ph - pl) * 2,
                    'reason':    f'Candle above EMA5({ema5:.0f}), break of low {pl:.0f}'
                }

        # LONG: prev candle entirely below EMA, curr breaks prev high
        if ph < ema5 and pc < ema5 and po < ema5:
            if ch > ph:   # Breaks previous high
                return {
                    'direction': 'LONG',
                    'strategy':  '5EMA',
                    'entry':     ph,
                    'sl':        pl - 10,
                    'target':    ph + (ph - pl) * 2,
                    'reason':    f'Candle below EMA5({ema5:.0f}), break of high {ph:.0f}'
                }
        return None

    @staticmethod
    def check_inside_candle(df: pd.DataFrame) -> Optional[dict]:
        """
        Inside Candle Strategy:
        - 3 candles ago = mother candle
        - 2 candles ago = inside candle (within mother range)
        - Last candle = breakout candle
        """
        if df is None or len(df) < 4:
            return None

        mother = df.iloc[-4]
        child  = df.iloc[-3]
        curr   = df.iloc[-2]

        mh = float(mother['High']); ml = float(mother['Low'])
        ch = float(child['High']);  cl = float(child['Low'])
        crh= float(curr['High']);   crl= float(curr['Low'])

        # Child must be inside mother
        if not (ch <= mh and cl >= ml): return None
        if ch == mh or cl == ml:        return None

        mother_range = mh - ml
        if mother_range < 30: return None

        # Breakout direction
        if crh > mh:
            return {
                'direction': 'LONG',
                'strategy':  'INSIDE',
                'entry':     mh,
                'sl':        ml - 10,
                'target':    mh + mother_range * 1.5,
                'reason':    f'Inside bar breakout above mother high {mh:.0f}'
            }
        elif crl < ml:
            return {
                'direction': 'SHORT',
                'strategy':  'INSIDE',
                'entry':     ml,
                'sl':        mh + 10,
                'target':    ml - mother_range * 1.5,
                'reason':    f'Inside bar breakout below mother low {ml:.0f}'
            }
        return None

    @staticmethod
    def check_round_number(df: pd.DataFrame, ltp: float) -> Optional[dict]:
        """Round number rejection at dynamic levels."""
        if df is None or len(df) < 2 or ltp == 0:
            return None

        prev = df.iloc[-2]
        ph = float(prev['High']); pl = float(prev['Low'])
        po = float(prev['Open']); pc = float(prev['Close'])
        candle_range = ph - pl
        if candle_range < 20: return None

        # Dynamic round levels near current price
        base   = round(ltp / 500) * 500
        levels = [base + x * 500 for x in range(-3, 4)]

        for level in levels:
            if level <= 0: continue
            # Check if prev candle touched level
            if pl <= level <= ph:
                # Bearish rejection: touched but closed below
                if pc < po and pc < level:
                    return {
                        'direction': 'SHORT',
                        'strategy':  'ROUND',
                        'entry':     pc,
                        'sl':        ph + 10,
                        'target':    pc - candle_range * 1.5,
                        'reason':    f'Bearish rejection at round level {level:,.0f}'
                    }
                # Bullish rejection: touched but closed above
                elif pc > po and pc > level:
                    return {
                        'direction': 'LONG',
                        'strategy':  'ROUND',
                        'entry':     pc,
                        'sl':        pl - 10,
                        'target':    pc + candle_range * 1.5,
                        'reason':    f'Bullish rejection at round level {level:,.0f}'
                    }
        return None



    @staticmethod
    def check_fibonacci_oi(df: pd.DataFrame, prev_high: float,
                            prev_low: float, pcr: float = 1.0) -> Optional[dict]:
        """
        Fibonacci + OI Strategy (our backtested system).
        Entry at 61.8% Fibonacci retracement of previous day range.
        Confirmed by PCR (>1.0 bullish, <1.0 bearish).
        Target: 76.4% extension. SL: below 78.6% level.
        """
        if df is None or len(df) < 3 or prev_high == 0 or prev_low == 0:
            return None

        diff = prev_high - prev_low
        if diff < 50:  # Skip tiny ranges
            return None

        # Current price
        current = float(df.iloc[-1]['Close'])

        # Fibonacci levels
        fib_618_from_low  = prev_low  + diff * 0.618  # LONG entry zone
        fib_618_from_high = prev_high - diff * 0.618  # SHORT entry zone
        fib_764_from_low  = prev_low  + diff * 0.764  # LONG SL
        fib_764_from_high = prev_high - diff * 0.764  # SHORT SL
        fib_ext_up        = prev_high + diff * 0.382  # LONG target
        fib_ext_down      = prev_low  - diff * 0.382  # SHORT target

        direction = None
        entry = target = sl = 0
        fib_level = ""

        # LONG: price near 61.8% from low + PCR bullish
        dist_up = abs(current - fib_618_from_low) / diff
        if dist_up < 0.40 and pcr >= 1.0:
            direction = "LONG"
            entry     = current
            target    = fib_ext_up
            sl        = fib_764_from_low - 10
            fib_level = f"61.8% support at {fib_618_from_low:,.0f}"

        # SHORT: price near 61.8% from high + PCR bearish
        dist_down = abs(current - fib_618_from_high) / diff
        if dist_down < 0.40 and pcr < 1.0 and direction is None:
            direction = "SHORT"
            entry     = current
            target    = fib_ext_down
            sl        = fib_764_from_high + 10
            fib_level = f"61.8% resistance at {fib_618_from_high:,.0f}"

        if direction is None:
            return None

        risk   = abs(entry - sl)
        reward = abs(target - entry)
        if risk == 0: return None
        rr = reward / risk
        if rr < 1.5: return None

        return {
            'direction': direction,
            'strategy':  'FIB+OI',
            'entry':     round(entry, 2),
            'sl':        round(sl, 2),
            'target':    round(target, 2),
            'reason':    f"Fib 61.8% | {fib_level} | PCR={pcr:.2f} | R:R 1:{rr:.1f}"
        }

    @staticmethod
    def check_orb(df: pd.DataFrame, orb_minutes: int = 15) -> Optional[dict]:
        """
        Opening Range Breakout (ORB).
        Marks the high/low of the first `orb_minutes` candles after 9:15 AM.
        Signals a breakout + retest above/below that range.
        Requires VWAP direction to agree (approximated via price vs session mean).
        """
        if df is None or len(df) < orb_minutes + 2:
            return None

        try:
            idx = df.index
            # Build session open range (first orb_minutes 1-min bars, or available candles)
            orb_bars = df.iloc[:orb_minutes]
            orb_high = float(orb_bars['High'].max())
            orb_low  = float(orb_bars['Low'].min())
            orb_range = orb_high - orb_low

            if orb_range < 20:  # too tight, skip
                return None

            # Current state (use last 2 candles)
            prev = df.iloc[-2]
            curr = df.iloc[-1]
            prev_close = float(prev['Close'])
            curr_close = float(curr['Close'])
            curr_high  = float(curr['High'])
            curr_low   = float(curr['Low'])

            # VWAP approximation: session mean price
            vwap_approx = float(df['Close'].mean())

            direction = None
            entry = sl = target = 0.0
            reason = ""

            # Bullish: prev candle closed above ORB high (breakout), current near ORB high
            if prev_close > orb_high and curr_close > orb_high and curr_close > vwap_approx:
                # Retest / continuation — enter near ORB high
                direction = "LONG"
                entry  = curr_close
                sl     = orb_low - 5
                target = orb_high + orb_range * 1.5   # 1.5× range extension
                reason = f"ORB breakout LONG | Range {orb_high:,.0f}–{orb_low:,.0f} | VWAP OK"

            # Bearish: prev candle closed below ORB low, current below VWAP
            elif prev_close < orb_low and curr_close < orb_low and curr_close < vwap_approx:
                direction = "SHORT"
                entry  = curr_close
                sl     = orb_high + 5
                target = orb_low - orb_range * 1.5
                reason = f"ORB breakout SHORT | Range {orb_high:,.0f}–{orb_low:,.0f} | VWAP OK"

            if direction is None:
                return None

            risk   = abs(entry - sl)
            reward = abs(target - entry)
            if risk == 0 or reward / risk < 1.8:
                return None

            return {
                'direction': direction,
                'strategy':  'ORB',
                'entry':     round(entry, 2),
                'sl':        round(sl, 2),
                'target':    round(target, 2),
                'reason':    reason
            }
        except Exception:
            return None

    @staticmethod
    def check_liquidity_sweep(df: pd.DataFrame, prev_high: float, prev_low: float) -> Optional[dict]:
        """
        Liquidity Sweep Reversal (Smart Money Trap).
        Detects when price spikes beyond the previous day high/low (stop hunt),
        then sharply rejects — classic smart-money manipulation pattern.
        Volume spike = exhaustion confirmation.
        """
        if df is None or len(df) < 3 or prev_high == 0 or prev_low == 0:
            return None

        try:
            prev2 = df.iloc[-3]
            prev1 = df.iloc[-2]
            curr  = df.iloc[-1]

            p1_high  = float(prev1['High'])
            p1_low   = float(prev1['Low'])
            p1_close = float(prev1['Close'])
            p1_open  = float(prev1['Open'])
            c_high   = float(curr['High'])
            c_low    = float(curr['Low'])
            c_close  = float(curr['Close'])
            c_vol    = float(curr.get('Volume', 0))
            avg_vol  = float(df['Volume'].iloc[-10:].mean()) if 'Volume' in df.columns else 0

            vwap_approx = float(df['Close'].mean())
            diff = prev_high - prev_low
            if diff < 50:
                return None

            direction = None
            entry = sl = target = 0.0
            reason = ""

            # Bearish sweep: previous candle spiked ABOVE prev_day high then closed below it
            wick_above = p1_high > prev_high + 5          # swept the high
            rejected   = p1_close < prev_high             # closed back below
            vol_spike  = (avg_vol > 0 and c_vol > avg_vol * 1.3) or avg_vol == 0
            if wick_above and rejected and c_close < vwap_approx and vol_spike:
                direction = "SHORT"
                entry  = c_close
                sl     = p1_high + 10
                target = vwap_approx - (vwap_approx - prev_low) * 0.5
                reason = f"Liq. sweep SHORT | Wick {p1_high:,.0f} > PDH {prev_high:,.0f}, rejected | Vol spike"

            # Bullish sweep: previous candle swept BELOW prev_day low then closed above it
            wick_below    = p1_low < prev_low - 5
            rejected_low  = p1_close > prev_low
            if wick_below and rejected_low and c_close > vwap_approx and vol_spike and direction is None:
                direction = "LONG"
                entry  = c_close
                sl     = p1_low - 10
                target = vwap_approx + (prev_high - vwap_approx) * 0.5
                reason = f"Liq. sweep LONG | Wick {p1_low:,.0f} < PDL {prev_low:,.0f}, rejected | Vol spike"

            if direction is None:
                return None

            risk   = abs(entry - sl)
            reward = abs(target - entry)
            if risk == 0 or reward / risk < 1.5:
                return None

            return {
                'direction': direction,
                'strategy':  'LIQUIDITY_SWEEP',
                'entry':     round(entry, 2),
                'sl':        round(sl, 2),
                'target':    round(target, 2),
                'reason':    reason
            }
        except Exception:
            return None

    @staticmethod
    def check_vwap_trend(df: pd.DataFrame) -> Optional[dict]:
        """
        VWAP Trend Continuation.
        Price trending strongly above/below VWAP; pullback to VWAP or 5EMA,
        then continuation signal. Requires volume spike at pullback point.
        """
        if df is None or len(df) < 10:
            return None

        try:
            closes  = df['Close'].astype(float)
            highs   = df['High'].astype(float)
            lows    = df['Low'].astype(float)
            volumes = df['Volume'].astype(float) if 'Volume' in df.columns else pd.Series([0]*len(df))

            vwap   = float(closes.mean())
            ema5   = float(closes.ewm(span=5, adjust=False).mean().iloc[-1])
            avg_vol = float(volumes.iloc[-10:].mean())

            curr_close = float(closes.iloc[-1])
            curr_vol   = float(volumes.iloc[-1])
            prev_close = float(closes.iloc[-2])

            vol_spike = (avg_vol > 0 and curr_vol > avg_vol * 1.2) or avg_vol == 0

            direction = None
            entry = sl = target = 0.0
            reason = ""

            # Bullish: price above VWAP, pulled back close to VWAP/EMA5, now bouncing
            above_vwap_regime = float(closes.iloc[-5:].mean()) > vwap
            near_support = curr_close <= max(vwap, ema5) * 1.005   # within 0.5% of VWAP or EMA5
            if above_vwap_regime and near_support and curr_close > prev_close and vol_spike:
                direction = "LONG"
                entry  = curr_close
                sl     = min(vwap, ema5) - 10
                target = curr_close + (curr_close - sl) * 2.0
                reason = f"VWAP trend LONG | EMA5={ema5:,.0f} VWAP={vwap:,.0f} | Vol spike"

            # Bearish: price below VWAP, pulled back near VWAP/EMA5, now rejecting
            below_vwap_regime = float(closes.iloc[-5:].mean()) < vwap
            near_resistance = curr_close >= min(vwap, ema5) * 0.995
            if below_vwap_regime and near_resistance and curr_close < prev_close and vol_spike and direction is None:
                direction = "SHORT"
                entry  = curr_close
                sl     = max(vwap, ema5) + 10
                target = curr_close - (sl - curr_close) * 2.0
                reason = f"VWAP trend SHORT | EMA5={ema5:,.0f} VWAP={vwap:,.0f} | Vol spike"

            if direction is None:
                return None

            risk   = abs(entry - sl)
            reward = abs(target - entry)
            if risk == 0 or reward / risk < 1.8:
                return None

            return {
                'direction': direction,
                'strategy':  'VWAP_TREND',
                'entry':     round(entry, 2),
                'sl':        round(sl, 2),
                'target':    round(target, 2),
                'reason':    reason
            }
        except Exception:
            return None

    # ── 25 new strategy detectors ─────────────────────────────────────────────

    @staticmethod
    def check_gap_fill(df: pd.DataFrame, prev_close: float) -> Optional[dict]:
        if df is None or len(df) < 5 or prev_close == 0:
            return None
        try:
            open_price = float(df.iloc[0]['Open'])
            gap = open_price - prev_close
            gap_pct = abs(gap) / prev_close
            if gap_pct < 0.001 or gap_pct > 0.015:
                return None
            curr = df.iloc[-2]
            cc = float(curr['Close']); co = float(curr['Open'])
            if gap > 0 and cc < co and cc < open_price:
                return {'direction':'SHORT','strategy':'GAP_FILL','entry':cc,
                        'sl':open_price+15,'target':prev_close,
                        'reason':f'Gap fill SHORT | Gap {gap:.0f}pts | Target {prev_close:,.0f}'}
            if gap < 0 and cc > co and cc > open_price:
                return {'direction':'LONG','strategy':'GAP_FILL','entry':cc,
                        'sl':open_price-15,'target':prev_close,
                        'reason':f'Gap fill LONG | Gap {gap:.0f}pts | Target {prev_close:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_tail_reversal(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 4:
            return None
        try:
            c = df.iloc[-2]
            ch = float(c['High']); cl = float(c['Low'])
            co = float(c['Open']); cc = float(c['Close'])
            body = abs(cc - co); rng = ch - cl
            if rng < 20 or body == 0:
                return None
            upper_wick = ch - max(co, cc)
            lower_wick = min(co, cc) - cl
            if upper_wick > body * 2 and upper_wick > rng * 0.5:
                return {'direction':'SHORT','strategy':'TAIL_REVERSAL','entry':cc,
                        'sl':ch+10,'target':cc-upper_wick*1.5,
                        'reason':f'Bearish tail | Wick {upper_wick:.0f} > body {body:.0f}'}
            if lower_wick > body * 2 and lower_wick > rng * 0.5:
                return {'direction':'LONG','strategy':'TAIL_REVERSAL','entry':cc,
                        'sl':cl-10,'target':cc+lower_wick*1.5,
                        'reason':f'Bullish tail | Wick {lower_wick:.0f} > body {body:.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_gap_and_go(df: pd.DataFrame, prev_close: float) -> Optional[dict]:
        if df is None or len(df) < 6 or prev_close == 0:
            return None
        try:
            open_price = float(df.iloc[0]['Open'])
            gap_pct = (open_price - prev_close) / prev_close
            if abs(gap_pct) < 0.003:
                return None
            ema9 = df['Close'].ewm(span=9, adjust=False).mean()
            curr = df.iloc[-2]
            cc = float(curr['Close']); e9 = float(ema9.iloc[-2])
            vols = df['Volume'].astype(float) if 'Volume' in df.columns else None
            avg_v = float(vols.iloc[-10:].mean()) if vols is not None else 0
            cv = float(curr.get('Volume', 0))
            vol_ok = avg_v == 0 or cv > avg_v * 1.3
            if gap_pct > 0 and cc > e9 and vol_ok:
                sl = e9 - 10
                return {'direction':'LONG','strategy':'GAP_AND_GO','entry':cc,
                        'sl':sl,'target':cc+(cc-sl)*2,
                        'reason':f'Gap+Go LONG | Gap {gap_pct*100:.1f}% | 9EMA {e9:.0f}'}
            if gap_pct < 0 and cc < e9 and vol_ok:
                sl = e9 + 10
                return {'direction':'SHORT','strategy':'GAP_AND_GO','entry':cc,
                        'sl':sl,'target':cc-(sl-cc)*2,
                        'reason':f'Gap+Go SHORT | Gap {gap_pct*100:.1f}% | 9EMA {e9:.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_hod_break(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 6:
            return None
        try:
            hist = df.iloc[:-1]
            hod = float(hist['High'].max()); lod = float(hist['Low'].min())
            day_range = hod - lod
            if day_range < 30:
                return None
            curr = df.iloc[-2]; cc = float(curr['Close'])
            vols = df['Volume'].astype(float) if 'Volume' in df.columns else None
            avg_v = float(vols.iloc[-10:].mean()) if vols is not None else 0
            cv = float(curr.get('Volume', 0))
            vol_ok = avg_v == 0 or cv > avg_v * 1.8
            if cc > hod and vol_ok:
                return {'direction':'LONG','strategy':'HOD_BREAK','entry':cc,
                        'sl':hod-10,'target':cc+day_range*0.5,
                        'reason':f'HOD break LONG | {hod:,.0f} → {cc:,.0f}'}
            if cc < lod and vol_ok:
                return {'direction':'SHORT','strategy':'HOD_BREAK','entry':cc,
                        'sl':lod+10,'target':cc-day_range*0.5,
                        'reason':f'LOD break SHORT | {lod:,.0f} → {cc:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_volume_climax(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 10 or 'Volume' not in df.columns:
            return None
        try:
            vols = df['Volume'].astype(float)
            avg_v = float(vols.iloc[-10:].mean())
            if avg_v == 0:
                return None
            prev = df.iloc[-2]; curr = df.iloc[-1]
            if float(prev.get('Volume', 0)) < avg_v * 2.5:
                return None
            ph = float(prev['High']); pl = float(prev['Low'])
            po = float(prev['Open']); pc = float(prev['Close'])
            cc = float(curr['Close']); co = float(curr['Open'])
            if ph - pl < 20:
                return None
            vwap = float(df['Close'].mean())
            if pc > po and cc < co:
                return {'direction':'SHORT','strategy':'VOL_CLIMAX','entry':cc,
                        'sl':ph+10,'target':vwap,
                        'reason':f'Vol climax SHORT | {float(prev.get("Volume",0))/avg_v:.1f}x avg vol'}
            if pc < po and cc > co:
                return {'direction':'LONG','strategy':'VOL_CLIMAX','entry':cc,
                        'sl':pl-10,'target':vwap,
                        'reason':f'Vol climax LONG | {float(prev.get("Volume",0))/avg_v:.1f}x avg vol'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_doji_reversal(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 5:
            return None
        try:
            doji = df.iloc[-3]; conf = df.iloc[-2]
            dh = float(doji['High']); dl = float(doji['Low'])
            do = float(doji['Open']); dc = float(doji['Close'])
            body = abs(dc - do); rng = dh - dl
            if rng < 15 or body > rng * 0.25:
                return None
            ch = float(conf['High']); cl = float(conf['Low']); cc = float(conf['Close'])
            if ch > dh and cc > dc:
                return {'direction':'LONG','strategy':'DOJI_REV','entry':cc,
                        'sl':dl-10,'target':cc+rng*1.5,
                        'reason':f'Bullish doji breakout | Doji {dl:.0f}-{dh:.0f}'}
            if cl < dl and cc < dc:
                return {'direction':'SHORT','strategy':'DOJI_REV','entry':cc,
                        'sl':dh+10,'target':cc-rng*1.5,
                        'reason':f'Bearish doji breakdown | Doji {dl:.0f}-{dh:.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_3bar_reversal(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 6:
            return None
        try:
            b1,b2,b3,b4 = df.iloc[-5],df.iloc[-4],df.iloc[-3],df.iloc[-2]
            c1=float(b1['Close'])-float(b1['Open'])
            c2=float(b2['Close'])-float(b2['Open'])
            c3=float(b3['Close'])-float(b3['Open'])
            c4=float(b4['Close'])-float(b4['Open'])
            if c1>0 and c2>0 and c3>0 and c4<0:
                move=float(b3['High'])-float(b1['Low'])
                if move < 20: return None
                return {'direction':'SHORT','strategy':'3BAR_REV',
                        'entry':float(b4['Close']),'sl':float(b3['High'])+10,
                        'target':float(b4['Close'])-move,
                        'reason':f'3-bar bull run + reversal | Move {move:.0f}pts'}
            if c1<0 and c2<0 and c3<0 and c4>0:
                move=float(b1['High'])-float(b3['Low'])
                if move < 20: return None
                return {'direction':'LONG','strategy':'3BAR_REV',
                        'entry':float(b4['Close']),'sl':float(b3['Low'])-10,
                        'target':float(b4['Close'])+move,
                        'reason':f'3-bar bear run + reversal | Move {move:.0f}pts'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_bb_squeeze(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 25:
            return None
        try:
            closes = df['Close'].astype(float)
            ma20 = closes.rolling(20).mean()
            std20 = closes.rolling(20).std()
            bbu = ma20 + 2*std20; bbl = ma20 - 2*std20
            bw = (bbu - bbl) / ma20
            avg_bw = float(bw.iloc[-20:].mean())
            curr_bw = float(bw.iloc[-2])
            if curr_bw > avg_bw * 0.7:
                return None
            curr = df.iloc[-2]; cc = float(curr['Close'])
            pub = float(bbu.iloc[-2]); plb = float(bbl.iloc[-2])
            sl_d = (pub - plb) * 0.3
            if cc > pub:
                return {'direction':'LONG','strategy':'BB_SQUEEZE','entry':cc,
                        'sl':cc-sl_d,'target':cc+sl_d*2,
                        'reason':f'BB squeeze breakout | BW {curr_bw:.3f} < avg {avg_bw:.3f}'}
            if cc < plb:
                return {'direction':'SHORT','strategy':'BB_SQUEEZE','entry':cc,
                        'sl':cc+sl_d,'target':cc-sl_d*2,
                        'reason':f'BB squeeze breakdown | BW {curr_bw:.3f} < avg {avg_bw:.3f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_support_bounce(df: pd.DataFrame, prev_low: float) -> Optional[dict]:
        if df is None or len(df) < 8 or prev_low == 0:
            return None
        try:
            swing_low = float(df['Low'].iloc[-10:].min())
            support = min(prev_low, swing_low)
            curr = df.iloc[-2]
            cc=float(curr['Close']); co=float(curr['Open']); cl=float(curr['Low'])
            if cl <= support*1.002 and cc > co and cc > support:
                sl = support - 15
                return {'direction':'LONG','strategy':'SUPPORT_BOUNCE','entry':cc,
                        'sl':sl,'target':cc+(cc-sl)*2,
                        'reason':f'Support bounce LONG | Support {support:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_ma_crossover(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 55:
            return None
        try:
            closes = df['Close'].astype(float)
            e20 = closes.ewm(span=20, adjust=False).mean()
            e50 = closes.ewm(span=50, adjust=False).mean()
            p20=float(e20.iloc[-3]);p50=float(e50.iloc[-3])
            c20=float(e20.iloc[-2]);c50=float(e50.iloc[-2])
            cc=float(closes.iloc[-2])
            sl_d=abs(c20-c50)+20
            if p20<p50 and c20>c50:
                return {'direction':'LONG','strategy':'MA_CROSS','entry':cc,
                        'sl':c50-20,'target':cc+sl_d*2,
                        'reason':f'MA20/50 bull cross | EMA20 {c20:.0f} > EMA50 {c50:.0f}'}
            if p20>p50 and c20<c50:
                return {'direction':'SHORT','strategy':'MA_CROSS','entry':cc,
                        'sl':c50+20,'target':cc-sl_d*2,
                        'reason':f'MA20/50 bear cross | EMA20 {c20:.0f} < EMA50 {c50:.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_mean_reversion(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 25:
            return None
        try:
            closes = df['Close'].astype(float)
            ma20=float(closes.rolling(20).mean().iloc[-2])
            std20=float(closes.rolling(20).std().iloc[-2])
            if std20 == 0: return None
            cc=float(closes.iloc[-2]); z=(cc-ma20)/std20
            curr=df.iloc[-2]
            if z>2.0 and float(curr['Close'])<float(curr['Open']):
                return {'direction':'SHORT','strategy':'MEAN_REV','entry':cc,
                        'sl':cc+std20,'target':ma20,
                        'reason':f'Mean rev SHORT | Z {z:.1f} | MA {ma20:,.0f}'}
            if z<-2.0 and float(curr['Close'])>float(curr['Open']):
                return {'direction':'LONG','strategy':'MEAN_REV','entry':cc,
                        'sl':cc-std20,'target':ma20,
                        'reason':f'Mean rev LONG | Z {z:.1f} | MA {ma20:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_bull_flag(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 10:
            return None
        try:
            closes = df['Close'].astype(float)
            pole = float(closes.iloc[-6]) - float(closes.iloc[-10])
            if abs(pole) < 40: return None
            flag_bars = df.iloc[-6:-1]
            fh=float(flag_bars['High'].max()); fl=float(flag_bars['Low'].min())
            if (fh-fl) > abs(pole)*0.5: return None
            curr=df.iloc[-2]; cc=float(curr['Close'])
            ch=float(curr['High']); cl=float(curr['Low'])
            if pole>0 and ch>fh:
                return {'direction':'LONG','strategy':'BULL_FLAG','entry':cc,
                        'sl':fl-10,'target':cc+abs(pole),
                        'reason':f'Bull flag breakout | Pole {pole:.0f}pts'}
            if pole<0 and cl<fl:
                return {'direction':'SHORT','strategy':'BULL_FLAG','entry':cc,
                        'sl':fh+10,'target':cc-abs(pole),
                        'reason':f'Bear flag breakdown | Pole {pole:.0f}pts'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_rsi_divergence(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 20:
            return None
        try:
            closes = df['Close'].astype(float)
            delta = closes.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rsi = 100 - (100/(1+gain/loss.replace(0, 1e-9)))
            p1p=float(closes.iloc[-12]); p2p=float(closes.iloc[-2])
            p1r=float(rsi.iloc[-12]);   p2r=float(rsi.iloc[-2])
            atr=float((df['High'].astype(float)-df['Low'].astype(float)).rolling(14).mean().iloc[-2])
            if atr==0: return None
            cc=float(closes.iloc[-2])
            if p2p>p1p*1.001 and p2r<p1r-3 and p2r>60:
                return {'direction':'SHORT','strategy':'RSI_DIV','entry':cc,
                        'sl':cc+atr,'target':cc-atr*2,
                        'reason':f'Bearish RSI div | Price {p1p:.0f}→{p2p:.0f} | RSI {p1r:.0f}→{p2r:.0f}'}
            if p2p<p1p*0.999 and p2r>p1r+3 and p2r<40:
                return {'direction':'LONG','strategy':'RSI_DIV','entry':cc,
                        'sl':cc-atr,'target':cc+atr*2,
                        'reason':f'Bullish RSI div | Price {p1p:.0f}→{p2p:.0f} | RSI {p1r:.0f}→{p2r:.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_macd_reversal(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 30:
            return None
        try:
            closes = df['Close'].astype(float)
            macd = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
            hist = macd - macd.ewm(span=9, adjust=False).mean()
            h2=float(hist.iloc[-3]); h1=float(hist.iloc[-2])
            atr=float((df['High'].astype(float)-df['Low'].astype(float)).rolling(14).mean().iloc[-2])
            if atr==0: return None
            cc=float(closes.iloc[-2])
            if h2>0 and h1<h2 and h2>atr*0.3:
                return {'direction':'SHORT','strategy':'MACD_REV','entry':cc,
                        'sl':cc+atr,'target':cc-atr*2,
                        'reason':f'MACD hist bearish flip | {h2:.1f}→{h1:.1f}'}
            if h2<0 and h1>h2 and abs(h2)>atr*0.3:
                return {'direction':'LONG','strategy':'MACD_REV','entry':cc,
                        'sl':cc-atr,'target':cc+atr*2,
                        'reason':f'MACD hist bullish flip | {h2:.1f}→{h1:.1f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> Optional[dict]:
        if df is None or len(df) < period+5:
            return None
        try:
            hi=df['High'].astype(float); lo=df['Low'].astype(float); cl=df['Close'].astype(float)
            tr=pd.concat([hi-lo,(hi-cl.shift()).abs(),(lo-cl.shift()).abs()],axis=1).max(axis=1)
            atr=tr.rolling(period).mean()
            hl2=(hi+lo)/2
            upper=hl2+mult*atr; lower=hl2-mult*atr
            dirs=pd.Series(1,index=df.index)
            for i in range(period,len(df)):
                dirs.iloc[i]=1 if cl.iloc[i]>upper.iloc[i-1] else (-1 if cl.iloc[i]<lower.iloc[i-1] else dirs.iloc[i-1])
            pd_=int(dirs.iloc[-3]); cd_=int(dirs.iloc[-2])
            cc=float(cl.iloc[-2]); av=float(atr.iloc[-2])
            if av==0: return None
            if pd_==-1 and cd_==1:
                return {'direction':'LONG','strategy':'SUPERTREND','entry':cc,
                        'sl':cc-av*mult,'target':cc+av*mult*2,
                        'reason':f'Supertrend flip LONG | ATR {av:.0f}'}
            if pd_==1 and cd_==-1:
                return {'direction':'SHORT','strategy':'SUPERTREND','entry':cc,
                        'sl':cc+av*mult,'target':cc-av*mult*2,
                        'reason':f'Supertrend flip SHORT | ATR {av:.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_fibonacci_cluster(df: pd.DataFrame, prev_high: float, prev_low: float) -> Optional[dict]:
        if df is None or len(df) < 10 or prev_high==0 or prev_low==0:
            return None
        try:
            closes=df['Close'].astype(float); cc=float(closes.iloc[-2])
            diff=prev_high-prev_low
            if diff<50: return None
            fibs=[prev_low+diff*r for r in (0.382,0.5,0.618,0.786)]
            diff2=float(df['High'].iloc[-20:].max())-float(df['Low'].iloc[-20:].min()) if len(df)>=20 else 0
            if diff2>30:
                base2=float(df['Low'].iloc[-20:].min())
                fibs+=[base2+diff2*r for r in (0.382,0.5,0.618)]
            near=[f for f in fibs if abs(f-cc)<cc*0.003]
            if len(near)<2: return None
            atr=float((df['High'].astype(float)-df['Low'].astype(float)).rolling(10).mean().iloc[-2])
            cl_lvl=sum(near)/len(near)
            curr=df.iloc[-2]
            if float(curr['Close'])>float(curr['Open']) and cc>cl_lvl:
                return {'direction':'LONG','strategy':'FIB_CLUSTER','entry':cc,
                        'sl':cl_lvl-atr,'target':cc+atr*2,
                        'reason':f'Fib cluster LONG | {len(near)} Fibs @ {cl_lvl:,.0f}'}
            if float(curr['Close'])<float(curr['Open']) and cc<cl_lvl:
                return {'direction':'SHORT','strategy':'FIB_CLUSTER','entry':cc,
                        'sl':cl_lvl+atr,'target':cc-atr*2,
                        'reason':f'Fib cluster SHORT | {len(near)} Fibs @ {cl_lvl:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_momentum_scalp(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 8:
            return None
        try:
            ema9=df['Close'].ewm(span=9,adjust=False).mean()
            b1,b2,b3=df.iloc[-4],df.iloc[-3],df.iloc[-2]
            c1=float(b1['Close'])-float(b1['Open'])
            c2=float(b2['Close'])-float(b2['Open'])
            move=float(b3['Close'])-float(b1['Open'])
            if abs(move)<30: return None
            cc=float(b3['Close']); e9=float(ema9.iloc[-2])
            vols=df['Volume'].astype(float) if 'Volume' in df.columns else None
            avg_v=float(vols.iloc[-10:].mean()) if vols is not None else 0
            cv=float(b3.get('Volume',0))
            vol_ok=avg_v==0 or cv>avg_v*1.2
            if c1>0 and c2>0 and move>0 and cc>e9 and vol_ok:
                return {'direction':'LONG','strategy':'MOM_SCALP','entry':cc,
                        'sl':e9-10,'target':cc+move*0.5,
                        'reason':f'Momentum scalp LONG | Push {move:.0f}pts'}
            if c1<0 and c2<0 and move<0 and cc<e9 and vol_ok:
                return {'direction':'SHORT','strategy':'MOM_SCALP','entry':cc,
                        'sl':e9+10,'target':cc-abs(move)*0.5,
                        'reason':f'Momentum scalp SHORT | Push {move:.0f}pts'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_open_drive(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 5:
            return None
        try:
            b1,b2,b3=df.iloc[0],df.iloc[1],df.iloc[2]
            c1=float(b1['Close'])-float(b1['Open'])
            c2=float(b2['Close'])-float(b2['Open'])
            c3=float(b3['Close'])-float(b3['Open'])
            if c1==0 or c2==0 or c3==0: return None
            drive=float(b3['Close'])-float(b1['Open'])
            if abs(drive)<30: return None
            curr=df.iloc[-2]; cc=float(curr['Close'])
            if c1>0 and c2>0 and c3>0:
                return {'direction':'LONG','strategy':'OPEN_DRIVE','entry':cc,
                        'sl':float(b1['Low'])-10,'target':cc+abs(drive),
                        'reason':f'NSE open drive LONG | Drive {drive:.0f}pts'}
            if c1<0 and c2<0 and c3<0:
                return {'direction':'SHORT','strategy':'OPEN_DRIVE','entry':cc,
                        'sl':float(b1['High'])+10,'target':cc-abs(drive),
                        'reason':f'NSE open drive SHORT | Drive {drive:.0f}pts'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_power_hour(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 6:
            return None
        try:
            from datetime import timezone, timedelta, datetime as _dt
            _ist_tz=timezone(timedelta(hours=5,minutes=30))
            _now=_dt.now(_ist_tz)
            if not (14 <= _now.hour < 16):
                return None
            hist=df.iloc[:-1]
            hod=float(hist['High'].max()); lod=float(hist['Low'].min())
            day_range=hod-lod
            if day_range<40: return None
            curr=df.iloc[-2]; cc=float(curr['Close'])
            vols=df['Volume'].astype(float) if 'Volume' in df.columns else None
            avg_v=float(vols.iloc[-10:].mean()) if vols is not None else 0
            cv=float(curr.get('Volume',0))
            vol_ok=avg_v==0 or cv>avg_v*1.5
            if cc>hod and vol_ok:
                return {'direction':'LONG','strategy':'POWER_HOUR','entry':cc,
                        'sl':hod-15,'target':cc+day_range*0.4,
                        'reason':f'Power hour HOD break | HOD {hod:,.0f}'}
            if cc<lod and vol_ok:
                return {'direction':'SHORT','strategy':'POWER_HOUR','entry':cc,
                        'sl':lod+15,'target':cc-day_range*0.4,
                        'reason':f'Power hour LOD break | LOD {lod:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_50ema_support(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 55:
            return None
        try:
            closes=df['Close'].astype(float)
            e50=closes.ewm(span=50,adjust=False).mean()
            e200=closes.ewm(span=200,adjust=False).mean() if len(df)>=200 else e50
            e50v=float(e50.iloc[-2]); e200v=float(e200.iloc[-2])
            curr=df.iloc[-2]
            cc=float(curr['Close']); cl=float(curr['Low']); ch=float(curr['High'])
            co=float(curr['Open'])
            atr=float((df['High'].astype(float)-df['Low'].astype(float)).rolling(14).mean().iloc[-2])
            if atr==0: return None
            if cc>e200v and cl<=e50v*1.001 and cc>co:
                return {'direction':'LONG','strategy':'EMA50_SUPPORT','entry':cc,
                        'sl':e50v-atr*0.5,'target':cc+atr*2,
                        'reason':f'50 EMA bounce LONG | EMA50 {e50v:,.0f}'}
            if cc<e200v and ch>=e50v*0.999 and cc<co:
                return {'direction':'SHORT','strategy':'EMA50_SUPPORT','entry':cc,
                        'sl':e50v+atr*0.5,'target':cc-atr*2,
                        'reason':f'50 EMA rejection SHORT | EMA50 {e50v:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_trend_follow(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 55:
            return None
        try:
            closes=df['Close'].astype(float)
            e20=closes.ewm(span=20,adjust=False).mean()
            e50=closes.ewm(span=50,adjust=False).mean()
            e20v=float(e20.iloc[-2]); e50v=float(e50.iloc[-2])
            cc=float(closes.iloc[-2])
            atr=float((df['High'].astype(float)-df['Low'].astype(float)).rolling(14).mean().iloc[-2])
            curr=df.iloc[-2]
            if e20v>e50v and float(curr['Low'])<=e20v*1.001 and float(curr['Close'])>float(curr['Open']):
                return {'direction':'LONG','strategy':'TREND_FOLLOW','entry':cc,
                        'sl':e50v-10,'target':cc+atr*2,
                        'reason':f'Trend follow LONG | EMA20({e20v:.0f})>EMA50({e50v:.0f})'}
            if e20v<e50v and float(curr['High'])>=e20v*0.999 and float(curr['Close'])<float(curr['Open']):
                return {'direction':'SHORT','strategy':'TREND_FOLLOW','entry':cc,
                        'sl':e50v+10,'target':cc-atr*2,
                        'reason':f'Trend follow SHORT | EMA20({e20v:.0f})<EMA50({e50v:.0f})'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_scalp_sr(df: pd.DataFrame, prev_high: float, prev_low: float) -> Optional[dict]:
        if df is None or len(df) < 5 or prev_high==0 or prev_low==0:
            return None
        try:
            b1=df.iloc[-3]; b2=df.iloc[-2]
            b1h=float(b1['High']); b1l=float(b1['Low']); b1c=float(b1['Close'])
            b2h=float(b2['High']); b2l=float(b2['Low']); b2c=float(b2['Close']); b2o=float(b2['Open'])
            atr=float((df['High'].astype(float)-df['Low'].astype(float)).rolling(10).mean().iloc[-2])
            if b1h>=prev_high*0.998 and b2c<b2o and b2c<b1c:
                return {'direction':'SHORT','strategy':'SCALP_SR','entry':b2c,
                        'sl':prev_high+15,'target':b2c-atr*1.5,
                        'reason':f'Scalp SHORT at PDH {prev_high:,.0f}'}
            if b1l<=prev_low*1.002 and b2c>b2o and b2c>b1c:
                return {'direction':'LONG','strategy':'SCALP_SR','entry':b2c,
                        'sl':prev_low-15,'target':b2c+atr*1.5,
                        'reason':f'Scalp LONG at PDL {prev_low:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_earnings_momentum(df: pd.DataFrame, prev_close: float) -> Optional[dict]:
        if df is None or len(df) < 5 or prev_close==0:
            return None
        try:
            open_price=float(df.iloc[0]['Open'])
            gap_pct=(open_price-prev_close)/prev_close
            if abs(gap_pct)<0.008: return None
            ema9=df['Close'].ewm(span=9,adjust=False).mean()
            curr=df.iloc[-2]; cc=float(curr['Close']); e9=float(ema9.iloc[-2])
            vols=df['Volume'].astype(float) if 'Volume' in df.columns else None
            avg_v=float(vols.iloc[-10:].mean()) if vols is not None else 0
            cv=float(curr.get('Volume',0))
            vol_ok=avg_v==0 or cv>avg_v*2.0
            if gap_pct>0 and cc>e9 and vol_ok and cc>open_price*0.998:
                return {'direction':'LONG','strategy':'EARNINGS_MOM','entry':cc,
                        'sl':open_price*0.995,'target':cc+abs(open_price-prev_close)*0.5,
                        'reason':f'Earnings gap+go LONG | Gap {gap_pct*100:.1f}%'}
            if gap_pct<0 and cc<e9 and vol_ok and cc<open_price*1.002:
                return {'direction':'SHORT','strategy':'EARNINGS_MOM','entry':cc,
                        'sl':open_price*1.005,'target':cc-abs(open_price-prev_close)*0.5,
                        'reason':f'Earnings gap+go SHORT | Gap {gap_pct*100:.1f}%'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_vix_reversal(vix_value: float, df: pd.DataFrame) -> Optional[dict]:
        if vix_value<=0 or df is None or len(df)<5:
            return None
        try:
            curr=df.iloc[-2]; cc=float(curr['Close']); co=float(curr['Open'])
            atr=float((df['High'].astype(float)-df['Low'].astype(float)).rolling(10).mean().iloc[-2])
            if atr==0: return None
            if vix_value>20 and cc>co:
                return {'direction':'LONG','strategy':'VIX_REV','entry':cc,
                        'sl':cc-atr,'target':cc+atr*2,
                        'reason':f'VIX spike reversal LONG | VIX {vix_value:.1f}'}
            if vix_value<12 and cc<co:
                return {'direction':'SHORT','strategy':'VIX_REV','entry':cc,
                        'sl':cc+atr,'target':cc-atr*2,
                        'reason':f'VIX complacency SHORT | VIX {vix_value:.1f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_lunch_fade(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 6:
            return None
        try:
            from datetime import timezone, timedelta, datetime as _dt
            _ist_tz=timezone(timedelta(hours=5,minutes=30))
            _now=_dt.now(_ist_tz)
            if not ((_now.hour==12) or (_now.hour==13 and _now.minute<=30)):
                return None
            hist=df.iloc[:-1]
            hod=float(hist['High'].max()); lod=float(hist['Low'].min())
            if hod-lod<30: return None
            curr=df.iloc[-2]; cc=float(curr['Close']); co=float(curr['Open'])
            vwap=float(df['Close'].mean())
            if float(curr['High'])>=hod*0.998 and cc<co:
                return {'direction':'SHORT','strategy':'LUNCH_FADE','entry':cc,
                        'sl':hod+10,'target':vwap,
                        'reason':f'Lunch fade SHORT | HOD {hod:,.0f} → VWAP {vwap:,.0f}'}
            if float(curr['Low'])<=lod*1.002 and cc>co:
                return {'direction':'LONG','strategy':'LUNCH_FADE','entry':cc,
                        'sl':lod-10,'target':vwap,
                        'reason':f'Lunch fade LONG | LOD {lod:,.0f} → VWAP {vwap:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_momentum_swing(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 25:
            return None
        try:
            closes=df['Close'].astype(float)
            e20=closes.ewm(span=20,adjust=False).mean()
            delta=closes.diff()
            gain=delta.clip(lower=0).rolling(14).mean()
            loss=(-delta.clip(upper=0)).rolling(14).mean()
            rsi=100-(100/(1+gain/loss.replace(0,1e-9)))
            e20v=float(e20.iloc[-2]); curr_rsi=float(rsi.iloc[-2])
            cc=float(closes.iloc[-2])
            atr=float((df['High'].astype(float)-df['Low'].astype(float)).rolling(14).mean().iloc[-2])
            curr=df.iloc[-2]
            if cc>e20v and curr_rsi>55 and float(curr['Low'])<=e20v*1.002 and float(curr['Close'])>float(curr['Open']):
                return {'direction':'LONG','strategy':'MOM_SWING','entry':cc,
                        'sl':e20v-atr*0.5,'target':cc+atr*2,
                        'reason':f'Momentum swing LONG | RSI {curr_rsi:.0f} | EMA20 {e20v:.0f}'}
            if cc<e20v and curr_rsi<45 and float(curr['High'])>=e20v*0.998 and float(curr['Close'])<float(curr['Open']):
                return {'direction':'SHORT','strategy':'MOM_SWING','entry':cc,
                        'sl':e20v+atr*0.5,'target':cc-atr*2,
                        'reason':f'Momentum swing SHORT | RSI {curr_rsi:.0f} | EMA20 {e20v:.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_breakout_momentum(df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < 10:
            return None
        try:
            rh=float(df['High'].iloc[-10:].max()); rl=float(df['Low'].iloc[-10:].min())
            consol=rh-rl
            if consol<30: return None
            curr=df.iloc[-2]; cc=float(curr['Close'])
            vols=df['Volume'].astype(float) if 'Volume' in df.columns else None
            avg_v=float(vols.iloc[-10:].mean()) if vols is not None else 0
            cv=float(curr.get('Volume',0))
            vol_ok=avg_v==0 or cv>avg_v*2.0
            if cc>rh and vol_ok:
                return {'direction':'LONG','strategy':'BREAKOUT_MOM','entry':cc,
                        'sl':rh-10,'target':cc+consol,
                        'reason':f'Breakout LONG | Range {rl:.0f}-{rh:.0f} | Vol {cv/max(avg_v,1):.1f}x'}
            if cc<rl and vol_ok:
                return {'direction':'SHORT','strategy':'BREAKOUT_MOM','entry':cc,
                        'sl':rl+10,'target':cc-consol,
                        'reason':f'Breakdown SHORT | Range {rl:.0f}-{rh:.0f} | Vol {cv/max(avg_v,1):.1f}x'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_premarket_range(df: pd.DataFrame, prev_high: float, prev_low: float) -> Optional[dict]:
        """Pre-market / Gift Nifty range break — first 2 bars define pre-open range."""
        if df is None or len(df) < 4:
            return None
        try:
            # Use first 2 candles as pre-market range proxy
            pm_high = float(df.iloc[:2]['High'].max())
            pm_low  = float(df.iloc[:2]['Low'].min())
            pm_range = pm_high - pm_low
            if pm_range < 15:
                return None
            curr = df.iloc[-2]; cc = float(curr['Close'])
            if cc > pm_high:
                return {'direction':'LONG','strategy':'PREMARKET_BREAK','entry':cc,
                        'sl':pm_high-10,'target':cc+pm_range*2,
                        'reason':f'Pre-market break LONG | PM H {pm_high:,.0f} | Range {pm_range:.0f}pts'}
            if cc < pm_low:
                return {'direction':'SHORT','strategy':'PREMARKET_BREAK','entry':cc,
                        'sl':pm_low+10,'target':cc-pm_range*2,
                        'reason':f'Pre-market break SHORT | PM L {pm_low:,.0f} | Range {pm_range:.0f}pts'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_globex_hl_break(df: pd.DataFrame, prev_high: float, prev_low: float) -> Optional[dict]:
        """Globex / overnight H-L break at RTH open — fires only in first 30 min."""
        if df is None or len(df) < 3 or prev_high == 0 or prev_low == 0:
            return None
        try:
            from datetime import timezone, timedelta, datetime as _dt
            _now = _dt.now(timezone(timedelta(hours=5, minutes=30)))
            if not (9 <= _now.hour < 10):  # only 9:15–10:00 IST
                return None
            curr = df.iloc[-2]; cc = float(curr['Close'])
            rng = prev_high - prev_low
            if rng < 30:
                return None
            if cc > prev_high:
                return {'direction':'LONG','strategy':'GLOBEX_HL_BREAK','entry':cc,
                        'sl':prev_high-10,'target':cc+rng*0.7,
                        'reason':f'Globex H break LONG | PDH {prev_high:,.0f}'}
            if cc < prev_low:
                return {'direction':'SHORT','strategy':'GLOBEX_HL_BREAK','entry':cc,
                        'sl':prev_low+10,'target':cc-rng*0.7,
                        'reason':f'Globex L break SHORT | PDL {prev_low:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_cum_delta(df: pd.DataFrame) -> Optional[dict]:
        """Cumulative delta divergence — price vs buy-sell volume proxy diverge."""
        if df is None or len(df) < 15 or 'Volume' not in df.columns:
            return None
        try:
            closes = df['Close'].astype(float)
            opens  = df['Open'].astype(float)
            vols   = df['Volume'].astype(float)
            # Proxy delta: positive candle = buy vol, negative = sell vol
            delta_raw = vols * closes.sub(opens).apply(lambda x: 1 if x >= 0 else -1)
            cum_delta = delta_raw.cumsum()
            # Compare price highs vs cum-delta highs over last 10 bars
            price_high1 = float(df['High'].iloc[-10:-5].max())
            price_high2 = float(df['High'].iloc[-5:].max())
            cd1 = float(cum_delta.iloc[-10:-5].max())
            cd2 = float(cum_delta.iloc[-5:].max())
            price_low1 = float(df['Low'].iloc[-10:-5].min())
            price_low2 = float(df['Low'].iloc[-5:].min())
            cd_low1 = float(cum_delta.iloc[-10:-5].min())
            cd_low2 = float(cum_delta.iloc[-5:].min())
            cc = float(closes.iloc[-2])
            atr = float((df['High'].astype(float)-df['Low'].astype(float)).rolling(10).mean().iloc[-2])
            if atr == 0: return None
            # Bearish div: price higher high but delta lower high
            if price_high2 > price_high1 and cd2 < cd1:
                return {'direction':'SHORT','strategy':'CUM_DELTA','entry':cc,
                        'sl':cc+atr,'target':cc-atr*2,
                        'reason':f'Cum delta bearish div | Price {price_high1:.0f}→{price_high2:.0f} | Delta falling'}
            # Bullish div: price lower low but delta higher low
            if price_low2 < price_low1 and cd_low2 > cd_low1:
                return {'direction':'LONG','strategy':'CUM_DELTA','entry':cc,
                        'sl':cc-atr,'target':cc+atr*2,
                        'reason':f'Cum delta bullish div | Price {price_low1:.0f}→{price_low2:.0f} | Delta rising'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_volume_poc(df: pd.DataFrame) -> Optional[dict]:
        """Volume Profile POC — trade VAH/VAL using volume-weighted price levels."""
        if df is None or len(df) < 15 or 'Volume' not in df.columns:
            return None
        try:
            closes = df['Close'].astype(float)
            vols   = df['Volume'].astype(float)
            total_vol = float(vols.sum())
            if total_vol == 0:
                return None
            vwap = float((closes * vols).sum() / total_vol)
            # VAH = vwap + 0.5×std, VAL = vwap - 0.5×std (70% volume approximation)
            std_v = float(closes.std())
            vah = vwap + std_v * 0.5
            val = vwap - std_v * 0.5
            curr = df.iloc[-2]; cc = float(curr['Close']); co = float(curr['Open'])
            atr = float((df['High'].astype(float)-df['Low'].astype(float)).rolling(10).mean().iloc[-2])
            if atr == 0: return None
            # Rejection at VAH → SHORT
            if float(curr['High']) >= vah and cc < co:
                return {'direction':'SHORT','strategy':'VOL_POC','entry':cc,
                        'sl':vah+atr*0.5,'target':vwap,
                        'reason':f'VOL POC SHORT at VAH {vah:,.0f} | VWAP {vwap:,.0f}'}
            # Bounce at VAL → LONG
            if float(curr['Low']) <= val and cc > co:
                return {'direction':'LONG','strategy':'VOL_POC','entry':cc,
                        'sl':val-atr*0.5,'target':vwap,
                        'reason':f'VOL POC LONG at VAL {val:,.0f} | VWAP {vwap:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_market_profile(df: pd.DataFrame) -> Optional[dict]:
        """Market Profile TPO — VAH/VAL boundaries with 30-min TPO approximation."""
        if df is None or len(df) < 20:
            return None
        try:
            closes = df['Close'].astype(float)
            highs  = df['High'].astype(float)
            lows   = df['Low'].astype(float)
            day_high = float(highs.max()); day_low = float(lows.min())
            day_range = day_high - day_low
            if day_range < 30:
                return None
            # POC: price level with most time spent (highest density)
            poc = float(closes.value_counts(bins=20).idxmax().mid)
            vah = poc + day_range * 0.15
            val = poc - day_range * 0.15
            curr = df.iloc[-2]; cc = float(curr['Close']); co = float(curr['Open'])
            atr = float((highs-lows).rolling(10).mean().iloc[-2])
            if atr == 0: return None
            # Short at VAH, long at VAL
            if float(curr['High']) >= vah and cc < co:
                return {'direction':'SHORT','strategy':'MKT_PROFILE','entry':cc,
                        'sl':vah+atr*0.5,'target':poc,
                        'reason':f'Market profile SHORT at VAH {vah:,.0f} | POC {poc:,.0f}'}
            if float(curr['Low']) <= val and cc > co:
                return {'direction':'LONG','strategy':'MKT_PROFILE','entry':cc,
                        'sl':val-atr*0.5,'target':poc,
                        'reason':f'Market profile LONG at VAL {val:,.0f} | POC {poc:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_volume_node_rejection(df: pd.DataFrame) -> Optional[dict]:
        """Volume Node Rejection — price stalls at high-volume node then reverses."""
        if df is None or len(df) < 15 or 'Volume' not in df.columns:
            return None
        try:
            closes = df['Close'].astype(float)
            vols   = df['Volume'].astype(float)
            # Find high-volume price node (bucket with most volume)
            min_p = float(closes.min()); max_p = float(closes.max())
            if max_p == min_p: return None
            bucket_size = (max_p - min_p) / 10
            vol_by_bucket = {}
            for p, v in zip(closes, vols):
                b = int((p - min_p) / bucket_size)
                vol_by_bucket[b] = vol_by_bucket.get(b, 0) + v
            peak_bucket = max(vol_by_bucket, key=vol_by_bucket.get)
            hvn = min_p + (peak_bucket + 0.5) * bucket_size
            curr = df.iloc[-2]; cc = float(curr['Close']); co = float(curr['Open'])
            atr = float((df['High'].astype(float)-df['Low'].astype(float)).rolling(10).mean().iloc[-2])
            if atr == 0: return None
            near_hvn = abs(cc - hvn) < atr * 0.5
            if near_hvn and cc < co:
                return {'direction':'SHORT','strategy':'VOL_NODE','entry':cc,
                        'sl':hvn+atr,'target':cc-atr*2,
                        'reason':f'Vol node rejection SHORT | HVN {hvn:,.0f}'}
            if near_hvn and cc > co:
                return {'direction':'LONG','strategy':'VOL_NODE','entry':cc,
                        'sl':hvn-atr,'target':cc+atr*2,
                        'reason':f'Vol node bounce LONG | HVN {hvn:,.0f}'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_banknifty_momentum(df: pd.DataFrame, instrument: str = "BANKNIFTY") -> Optional[dict]:
        """BankNifty / NQ Tech Momentum — sector-driven amplified directional move."""
        if df is None or len(df) < 8:
            return None
        try:
            closes = df['Close'].astype(float)
            ema9 = closes.ewm(span=9, adjust=False).mean()
            vwap = float((closes * df['Volume'].astype(float)).sum() / max(float(df['Volume'].astype(float).sum()), 1)) if 'Volume' in df.columns else float(closes.mean())
            cc = float(closes.iloc[-2]); e9 = float(ema9.iloc[-2])
            curr = df.iloc[-2]
            co = float(curr['Open']); ch = float(curr['High']); cl = float(curr['Low'])
            atr = float((df['High'].astype(float)-df['Low'].astype(float)).rolling(10).mean().iloc[-2])
            if atr == 0: return None
            vols = df['Volume'].astype(float) if 'Volume' in df.columns else None
            avg_v = float(vols.iloc[-10:].mean()) if vols is not None else 0
            cv = float(curr.get('Volume', 0))
            vol_ok = avg_v == 0 or cv > avg_v * 1.4
            # Strong momentum: above 9EMA + VWAP + volume
            if cc > e9 and cc > vwap and cc > co and vol_ok:
                return {'direction':'LONG','strategy':'BNTECH_MOM','entry':cc,
                        'sl':min(e9, vwap)-10,'target':cc+atr*1.5,
                        'reason':f'BankNifty momentum LONG | EMA9 {e9:.0f} | Vol {cv/max(avg_v,1):.1f}x'}
            if cc < e9 and cc < vwap and cc < co and vol_ok:
                return {'direction':'SHORT','strategy':'BNTECH_MOM','entry':cc,
                        'sl':max(e9, vwap)+10,'target':cc-atr*1.5,
                        'reason':f'BankNifty momentum SHORT | EMA9 {e9:.0f} | Vol {cv/max(avg_v,1):.1f}x'}
        except Exception:
            pass
        return None

    @staticmethod
    def check_sensex_divergence(nifty_df: pd.DataFrame, sensex_df: pd.DataFrame) -> Optional[dict]:
        """Sensex vs Nifty divergence — trade the lagging index short."""
        if nifty_df is None or sensex_df is None or len(nifty_df) < 5 or len(sensex_df) < 5:
            return None
        try:
            nf_ret  = (float(nifty_df['Close'].iloc[-1]) / float(nifty_df['Close'].iloc[-5]) - 1) * 100
            sx_ret  = (float(sensex_df['Close'].iloc[-1]) / float(sensex_df['Close'].iloc[-5]) - 1) * 100
            div     = nf_ret - sx_ret
            cc = float(nifty_df['Close'].iloc[-2])
            atr = float((nifty_df['High'].astype(float) - nifty_df['Low'].astype(float)).rolling(10).mean().iloc[-2])
            if atr == 0: return None
            if div > 0.5:   # Nifty outperforming — trade Sensex short
                return {'direction':'SHORT','strategy':'SENSEX_DIV','entry':cc,
                        'sl':cc+atr,'target':cc-atr*2,
                        'reason':f'Sensex divergence | NF {nf_ret:.2f}% vs SX {sx_ret:.2f}% | Trade laggard'}
            if div < -0.5:  # Sensex outperforming — trade Nifty short
                return {'direction':'SHORT','strategy':'SENSEX_DIV','entry':cc,
                        'sl':cc+atr,'target':cc-atr*2,
                        'reason':f'Nifty divergence | SX {sx_ret:.2f}% vs NF {nf_ret:.2f}% | Fade laggard'}
        except Exception:
            pass
        return None

# ─── Main Auto Trader ─────────────────────────────────────────────────────────

class POSAutoTrader:
    """
    Automated Power of Stocks trader.
    Runs every 5 minutes (after each candle close).
    
    Safety rules:
    1. Only trades 9:20 AM - 3:15 PM IST
    2. Max 1 trade per day
    3. Auto square off at 3:15 PM
    4. SL always set immediately
    5. Paper trade by default
    """

    def __init__(self, instrument: str = "NIFTY",
                 strategies: list = None,
                 qty: int = 75,
                 paper_trade: bool = True,
                 timeframe: str = "5m",
                 angel_client=None):
        self.instrument = instrument
        self.strategies = strategies or [
            # Original
            "5EMA","INSIDE","ROUND","FIB_OI","ORB","LIQUIDITY_SWEEP","VWAP_TREND",
            "FABIO_AAA","FABIO_SQUEEZE","FABIO_DELTA",
            # Reversal / candle patterns
            "TAIL_REVERSAL","DOJI_REV","3BAR_REV","VOL_CLIMAX","SCALP_SR",
            # Gap strategies
            "GAP_FILL","GAP_AND_GO","EARNINGS_MOM","PREMARKET_BREAK",
            # Breakout / momentum
            "HOD_BREAK","OPEN_DRIVE","GLOBEX_HL_BREAK","OVERNIGHT_RANGE",
            "MOM_SCALP","POWER_HOUR","BREAKOUT_MOM","BNTECH_MOM",
            # Trend follow / EMA
            "MA_CROSS","EMA50_SUPPORT","TREND_FOLLOW","MOM_SWING","BULL_FLAG",
            # Oscillator signals
            "RSI_DIV","MACD_REV","SUPERTREND","MEAN_REV","BB_SQUEEZE",
            # Volume / profile
            "VOL_POC","VOL_NODE","MKT_PROFILE","CUM_DELTA",
            # Support / Fibonacci
            "SUPPORT_BOUNCE","FIB_CLUSTER","INSIDE_DAY",
            # Time-of-day
            "LUNCH_FADE",
            # Multi-instrument (fetches own data)
            "SENSEX_DIV",
            # VIX
            "VIX_REV",
        ]
        self.qty        = qty
        self.paper      = paper_trade
        _tf_norm = {"5M":"5m","15M":"15m","1H":"1h","4H":"1h"}
        self.timeframe  = _tf_norm.get(timeframe, timeframe.lower())
        self.client     = angel_client
        self.data_eng   = LiveDataEngine(angel_client)
        self.detector   = POSSignalDetector()
        self.state      = POSState.load(self.state_key)
        self._running   = False
        self._thread    = None

    @property
    def state_key(self) -> str:
        """Unique key combining instrument and timeframe — used for isolated state files."""
        return f"{self.instrument}_{self.timeframe}"

    def start(self) -> str:
        if self._running:
            return "Already running!"
        self._running        = True
        self.state.status    = "RUNNING"
        self.state.log_msg(f"🚀 POS Auto Trader STARTED | {self.instrument} | TF: {self.timeframe} | Strategies: {self.strategies} | {'PAPER' if self.paper else 'LIVE'}")
        self.state.save(self.state_key)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return "Started!"

    def stop(self) -> str:
        self._running     = False
        self.state.status = "STOPPED"
        self.state.log_msg("🛑 POS Auto Trader STOPPED")
        self.state.save(self.state_key)
        return "Stopped!"

    def _loop(self):
        """Main loop — runs every candle close (5m=300s, 15m=900s)."""
        sleep_secs = 900 if self.timeframe == "15m" else 300
        while self._running:
            try:
                if self.state.status == "RUNNING":
                    self._scan()
                time.sleep(sleep_secs)
            except Exception as e:
                self.state.log_msg(f"Loop error: {e}", "ERROR")
                time.sleep(60)

    @staticmethod
    def get_ist_time():
        """Always returns current IST time regardless of local timezone."""
        return datetime.now(_IST)

    def _is_market_hours(self) -> bool:
        now_ist = self.get_ist_time().time()
        return dtime(9, 20) <= now_ist <= dtime(15, 15)

    def _scan(self):
        """Scan for signals and manage open trade."""
        from datetime import timezone, timedelta
        _ist_tz = timezone(timedelta(hours=5, minutes=30))
        _now_ist = datetime.now(_ist_tz)
        now_str = _now_ist.strftime('%H:%M:%S')
        self.state.last_scan = now_str

        # EOD square-off: close any open trade between 3:15 PM and 3:30 PM IST
        if dtime(15, 15) <= _now_ist.time() <= dtime(15, 30):
            if self.state.open_trade:
                self._square_off("EOD_SQUAREOFF")
                self.state.save(self.state_key)
            return

        # Check market hours (9:20 AM – 3:15 PM IST)
        if not self._is_market_hours():
            self.state.log_msg(f"⏰ Outside market hours ({now_str}) — waiting")
            self.state.save(self.state_key)
            return

        # Get live data
        ltp = self.data_eng.get_ltp(self.instrument)
        df  = self.data_eng.get_candles(self.instrument, 20, interval=self.timeframe)

        if ltp == 0 or df is None:
            self.state.log_msg("⚠️ No live data — retrying", "WARNING")
            return

        self.state.log_msg(f"🔍 Scanning {self.instrument} | LTP: {ltp:,.2f}")

        # Monitor open trade — one trade at a time in both paper and live mode
        if self.state.open_trade:
            self._monitor_trade(ltp)
            self.state.save(self.state_key)
            return

        # Max 1 trade per day in LIVE mode
        if not self.paper and self.state.trades_today >= 1:
            self.state.log_msg("✅ 1 trade already done today (LIVE) — waiting for tomorrow")
            return

        # Pre-compute prev-day OHLC + VIX for all strategies that need them
        _NEEDS_PREV = {
            "FIB_OI","FABIO_AAA","FABIO_SQUEEZE","FABIO_DELTA","LIQUIDITY_SWEEP",
            "GAP_FILL","GAP_AND_GO","EARNINGS_MOM","FIB_CLUSTER","INSIDE_DAY",
            "SCALP_SR","SUPPORT_BOUNCE","BREAKOUT_MOM","OVERNIGHT_RANGE",
        }
        _prev_high = _prev_low = _prev_close = 0.0
        _vix_value = 0.0
        if any(s in self.strategies for s in _NEEDS_PREV) or "VIX_REV" in self.strategies:
            try:
                import yfinance as yf
                _sym_map = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN"}
                _tk = yf.Ticker(_sym_map.get(self.instrument,"^NSEI"))
                _daily = _tk.history(period="5d",interval="1d")
                if len(_daily) >= 2:
                    _prev_high  = float(_daily.iloc[-2]["High"])
                    _prev_low   = float(_daily.iloc[-2]["Low"])
                    _prev_close = float(_daily.iloc[-2]["Close"])
            except Exception:
                pass
            if "VIX_REV" in self.strategies:
                try:
                    import yfinance as _yf2
                    _vix_tk = _yf2.Ticker("^INDIAVIX")
                    _vix_hist = _vix_tk.history(period="2d",interval="1d")
                    if len(_vix_hist) >= 1:
                        _vix_value = float(_vix_hist.iloc[-1]["Close"])
                except Exception:
                    pass

        # Fabio signal generator (lazy import)
        _fabio_gen = None
        if any(s.startswith("FABIO_") for s in self.strategies):
            try:
                from fabio_nse import FabioNSESignalGenerator
                _fabio_gen = FabioNSESignalGenerator()
            except Exception as _e:
                self.state.log_msg(f"Fabio import error: {_e}", "WARNING")

        # Scan for signals
        signal = None
        for strat in self.strategies:
            if strat in ("5EMA","5EMA_REJECT","5EMA_CROSS","5EMA_MULTI","INSIDE_EMA"):
                signal = self.detector.check_5ema(df)
            elif strat == "INSIDE":
                signal = self.detector.check_inside_candle(df)
            elif strat == "ROUND":
                signal = self.detector.check_round_number(df, ltp)
            elif strat == "FIB_OI" and _prev_high > 0:
                signal = self.detector.check_fibonacci_oi(df, _prev_high, _prev_low)
            elif strat in ("FABIO_AAA","FABIO_SQUEEZE","FABIO_DELTA") and _fabio_gen and _prev_high > 0:
                try:
                    _today_df = df.copy()
                    _vwap = float(_today_df["Close"].mean())
                    _fab_sig = _fabio_gen.get_signal(
                        df_today=_today_df, prev_high=_prev_high, prev_low=_prev_low,
                        prev_close=_prev_close, pcr=1.0, cpr_tc=_prev_high,
                        cpr_bc=_prev_low, vwap=_vwap, spot=ltp,
                    )
                    if _fab_sig:
                        setup_map = {"FABIO_AAA":"AAA_FADE","FABIO_SQUEEZE":"SQUEEZE","FABIO_DELTA":"DELTA_ABS"}
                        if _fab_sig.setup == setup_map.get(strat,"") or strat in ("FABIO_AAA","FABIO_SQUEEZE","FABIO_DELTA"):
                            signal = {'direction':_fab_sig.direction,'strategy':f"FABIO_{_fab_sig.setup}",
                                      'entry':round(_fab_sig.entry,2),'sl':round(_fab_sig.sl,2),
                                      'target':round(_fab_sig.target,2),'reason':_fab_sig.reason}
                except Exception as _fe:
                    self.state.log_msg(f"Fabio scan error: {_fe}", "WARNING")
            elif strat == "ORB":
                signal = self.detector.check_orb(df)
            elif strat == "LIQUIDITY_SWEEP" and _prev_high > 0:
                signal = self.detector.check_liquidity_sweep(df, _prev_high, _prev_low)
            elif strat == "VWAP_TREND":
                signal = self.detector.check_vwap_trend(df)
            # ── New strategies ────────────────────────────────────────────────
            elif strat == "GAP_FILL" and _prev_close > 0:
                signal = self.detector.check_gap_fill(df, _prev_close)
            elif strat == "TAIL_REVERSAL":
                signal = self.detector.check_tail_reversal(df)
            elif strat == "GAP_AND_GO" and _prev_close > 0:
                signal = self.detector.check_gap_and_go(df, _prev_close)
            elif strat == "HOD_BREAK":
                signal = self.detector.check_hod_break(df)
            elif strat == "VOL_CLIMAX":
                signal = self.detector.check_volume_climax(df)
            elif strat == "DOJI_REV":
                signal = self.detector.check_doji_reversal(df)
            elif strat == "3BAR_REV":
                signal = self.detector.check_3bar_reversal(df)
            elif strat == "BB_SQUEEZE":
                signal = self.detector.check_bb_squeeze(df)
            elif strat == "SUPPORT_BOUNCE" and _prev_low > 0:
                signal = self.detector.check_support_bounce(df, _prev_low)
            elif strat == "MA_CROSS":
                signal = self.detector.check_ma_crossover(df)
            elif strat == "MEAN_REV":
                signal = self.detector.check_mean_reversion(df)
            elif strat == "BULL_FLAG":
                signal = self.detector.check_bull_flag(df)
            elif strat == "RSI_DIV":
                signal = self.detector.check_rsi_divergence(df)
            elif strat == "MACD_REV":
                signal = self.detector.check_macd_reversal(df)
            elif strat == "SUPERTREND":
                signal = self.detector.check_supertrend(df)
            elif strat == "FIB_CLUSTER" and _prev_high > 0:
                signal = self.detector.check_fibonacci_cluster(df, _prev_high, _prev_low)
            elif strat == "OVERNIGHT_RANGE" and _prev_high > 0:
                signal = self.detector.check_overnight_range(df, _prev_high, _prev_low)
            elif strat == "MOM_SCALP":
                signal = self.detector.check_momentum_scalp(df)
            elif strat == "OPEN_DRIVE":
                signal = self.detector.check_open_drive(df)
            elif strat == "POWER_HOUR":
                signal = self.detector.check_power_hour(df)
            elif strat == "EMA50_SUPPORT":
                signal = self.detector.check_50ema_support(df)
            elif strat == "INSIDE_DAY" and _prev_high > 0:
                signal = self.detector.check_inside_day(df, _prev_high, _prev_low)
            elif strat == "TREND_FOLLOW":
                signal = self.detector.check_trend_follow(df)
            elif strat == "SCALP_SR" and _prev_high > 0:
                signal = self.detector.check_scalp_sr(df, _prev_high, _prev_low)
            elif strat == "EARNINGS_MOM" and _prev_close > 0:
                signal = self.detector.check_earnings_momentum(df, _prev_close)
            elif strat == "VIX_REV":
                signal = self.detector.check_vix_reversal(_vix_value, df)
            elif strat == "LUNCH_FADE":
                signal = self.detector.check_lunch_fade(df)
            elif strat == "MOM_SWING":
                signal = self.detector.check_momentum_swing(df)
            elif strat == "BREAKOUT_MOM":
                signal = self.detector.check_breakout_momentum(df)
            elif strat == "PREMARKET_BREAK":
                signal = self.detector.check_premarket_range(df, _prev_high, _prev_low)
            elif strat == "GLOBEX_HL_BREAK" and _prev_high > 0:
                signal = self.detector.check_globex_hl_break(df, _prev_high, _prev_low)
            elif strat == "CUM_DELTA":
                signal = self.detector.check_cum_delta(df)
            elif strat == "VOL_POC":
                signal = self.detector.check_volume_poc(df)
            elif strat == "MKT_PROFILE":
                signal = self.detector.check_market_profile(df)
            elif strat == "VOL_NODE":
                signal = self.detector.check_volume_node_rejection(df)
            elif strat == "BNTECH_MOM":
                signal = self.detector.check_banknifty_momentum(df, self.instrument)
            elif strat == "SENSEX_DIV":
                # Needs second instrument data — fetch on demand
                try:
                    import yfinance as _yf_sx
                    _sx_sym = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN"}
                    _nf_df2 = _yf_sx.download("^NSEI",  period="5d", interval="5m", progress=False, auto_adjust=True)
                    _sx_df2 = _yf_sx.download("^BSESN", period="5d", interval="5m", progress=False, auto_adjust=True)
                    if isinstance(_nf_df2.columns, __import__('pandas').MultiIndex): _nf_df2.columns = _nf_df2.columns.get_level_values(0)
                    if isinstance(_sx_df2.columns, __import__('pandas').MultiIndex): _sx_df2.columns = _sx_df2.columns.get_level_values(0)
                    signal = self.detector.check_sensex_divergence(_nf_df2.tail(20), _sx_df2.tail(20))
                except Exception:
                    pass
            if signal:
                break

        if signal:
            rr = abs(signal['target'] - signal['entry']) / abs(signal['entry'] - signal['sl'])
            if rr >= 1.8:
                self.state.log_msg(
                    f"🎯 SIGNAL: {signal['direction']} {self.instrument} | "
                    f"Entry: {signal['entry']:,.0f} | "
                    f"Target: {signal['target']:,.0f} | "
                    f"SL: {signal['sl']:,.0f} | "
                    f"R:R 1:{rr:.1f} | {signal['reason']}",
                    "TRADE"
                )
                self._execute_trade(signal, ltp)
            else:
                self.state.log_msg(f"⚠️ Signal found but R:R {rr:.1f} < 1.8 — skipped")
        else:
            self.state.log_msg(f"⚪ No signal this scan")
            self.state.last_signal = "No signal"

        self.state.save(self.state_key)

    def _execute_trade(self, signal: dict, ltp: float):
        """Place the trade."""
        # ── Daily capital cap check ──────────────────────────────────────────
        _this_invest = (round(ltp * self.qty * 0.1, 2)
                        if self.instrument.upper() in _IDX_SET
                        else round(ltp * self.qty, 2))
        _deployed = _get_today_deployed_rs()
        if _deployed + _this_invest > DAILY_CAP_RS:
            self.state.log_msg(
                f"🚫 Daily cap ₹{DAILY_CAP_RS:,.0f} reached! "
                f"Deployed: ₹{_deployed:,.0f} | This trade: ₹{_this_invest:,.0f} — skipped.",
                "WARNING"
            )
            return

        _ist_now = POSAutoTrader.get_ist_time()
        trade_id = f"POS_{_ist_now.strftime('%Y%m%d_%H%M%S')}"

        if self.paper:
            self.state.log_msg(f"📋 PAPER TRADE: {signal['direction']} {self.instrument} @ {ltp:,.0f}", "PAPER")
        else:
            # Real order via Angel One
            if self.client and self.client.connected:
                side = "BUY" if signal['direction'] == "LONG" else "SELL"
                self.state.log_msg(f"🔴 REAL ORDER: {side} {self.instrument} {self.qty} qty")
                # Place order
                try:
                    result = self.client.place_order(
                        symbol=self.instrument,
                        option_type="",
                        strike=0,
                        expiry="",
                        side=side,
                        qty=self.qty,
                        paper=False
                    )
                    if result:
                        self.state.log_msg(f"✅ Order placed: {result}", "TRADE")
                    else:
                        self.state.log_msg("❌ Order failed!", "ERROR")
                        return
                except Exception as e:
                    self.state.log_msg(f"❌ Order error: {e}", "ERROR")
                    return

        trade = POSTrade(
            id=trade_id,
            date=_ist_now.strftime('%Y-%m-%d'),
            time=_ist_now.strftime('%H:%M:%S'),
            strategy=signal['strategy'],
            instrument=self.instrument,
            direction=signal['direction'],
            entry=round(ltp, 2),  # Use actual LTP as entry
            target=round(signal['target'], 2),
            stop_loss=round(signal['sl'], 2),
            qty=self.qty,
            paper=self.paper
        )

        self.state.open_trade   = trade
        self.state.trades_today += 1
        self.state.total_trades += 1
        self.state.last_signal  = f"{signal['direction']} @ {ltp:,.0f}"
        self.state.last_action  = f"Opened {signal['direction']} {self.instrument}"

    def _monitor_trade(self, ltp: float):
        """Check if open trade hit target or SL."""
        trade = self.state.open_trade
        if not trade: return

        # Calculate unrealized PnL
        if trade.direction == "LONG":
            pnl = ltp - trade.entry
            hit_target = ltp >= trade.target
            hit_sl     = ltp <= trade.stop_loss
        else:
            pnl = trade.entry - ltp
            hit_target = ltp <= trade.target
            hit_sl     = ltp >= trade.stop_loss

        unreal_pnl = pnl * trade.qty
        self.state.log_msg(
            f"📊 {trade.direction} {trade.instrument} | "
            f"Entry: {trade.entry:,.0f} | LTP: {ltp:,.0f} | "
            f"Unrealized: {pnl:+.0f}pts (Rs.{unreal_pnl:+,.0f})"
        )

        if hit_target:
            self._square_off("TARGET_HIT", ltp)
        elif hit_sl:
            self._square_off("STOP_LOSS", ltp)

    def _square_off(self, reason: str, exit_price: float = 0.0):
        """Close the open trade."""
        trade = self.state.open_trade
        if not trade: return

        if exit_price == 0.0:
            exit_price = self.data_eng.get_ltp(self.instrument)

        if trade.direction == "LONG":
            pnl_pts = exit_price - trade.entry
        else:
            pnl_pts = trade.entry - exit_price

        pnl_rs = pnl_pts * trade.qty
        won    = pnl_pts > 0

        trade.status     = "CLOSED"
        trade.exit_price = round(exit_price, 2)
        trade.exit_reason= reason
        trade.pnl_pts    = round(pnl_pts, 2)
        trade.pnl_rs     = round(pnl_rs, 2)

        emoji = "✅ WIN" if won else "❌ LOSS"
        self.state.log_msg(
            f"{emoji}: Closed {trade.direction} {trade.instrument} | "
            f"Entry: {trade.entry:,.0f} → Exit: {exit_price:,.0f} | "
            f"PnL: {pnl_pts:+.0f}pts (Rs.{pnl_rs:+,.0f}) | {reason}",
            "TRADE"
        )

        if won: self.state.winning += 1
        self.state.total_pnl  += pnl_rs
        self.state.history.append(trade)
        self.state.open_trade  = None
        self.state.last_action = f"Closed {pnl_pts:+.0f}pts (Rs.{pnl_rs:+,.0f})"

        # Real square off
        if not self.paper and self.client and self.client.connected:
            try:
                side = "SELL" if trade.direction == "LONG" else "BUY"
                self.client.place_order(
                    symbol=trade.instrument, option_type="",
                    strike=0, expiry="", side=side,
                    qty=trade.qty, paper=False
                )
            except Exception as e:
                self.state.log_msg(f"Square off error: {e}", "ERROR")

        self.state.save(self.state_key)

    def run_once(self):
        """Manually trigger one scan."""
        self._scan()
        return self.state

    def get_pnl_today(self) -> float:
        """Get today's realized PnL."""
        today = _ist_date_str()
        return sum(t.pnl_rs for t in self.state.history
                   if t.date == today and t.status == "CLOSED")

    def get_unrealized_pnl(self) -> tuple:
        """Get unrealized PnL on open trade."""
        if not self.state.open_trade:
            return 0.0, 0.0
        trade = self.state.open_trade
        ltp   = self.data_eng.get_ltp(trade.instrument)
        if ltp == 0: return 0.0, 0.0
        pts = (ltp - trade.entry) if trade.direction == "LONG" else (trade.entry - ltp)
        return pts, pts * trade.qty
