"""
DLE Strategy Engine — Direction, Location, Execution (Smart Money Concepts)
HTF=4H bias → MTF=1H POI (Order Blocks) → LTF=15M execution confluences
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DLETrade:
    date:         str
    time:         str
    symbol:       str
    htf_bias:     str   # BULL / BEAR
    equil:        float
    ob_zone_low:  float
    ob_zone_high: float
    entry:        float
    stop:         float
    target:       float
    risk_pts:     float
    rr:           float  # always 3.0
    exit_price:   float
    exit_time:    str
    exit_reason:  str   # TP / SL / OPEN
    pnl_pts:      float
    won:          bool
    c1:           str   # confluence 1 label
    c2:           str   # confluence 2 label
    c3:           str   # confluence 3 label


@dataclass
class DLEResult:
    symbol:           str
    total_trades:     int
    winning_trades:   int
    losing_trades:    int
    win_rate:         float
    total_pnl_pts:    float
    avg_win_pts:      float
    avg_loss_pts:     float
    profit_factor:    float
    max_drawdown_pts: float
    trades:           List[DLETrade] = field(default_factory=list)
    equity_curve:     List[float]   = field(default_factory=list)
    current_bias:     str  = "NEUTRAL"
    current_equil:    float = 0.0
    current_sw_low:   float = 0.0
    current_sw_high:  float = 0.0
    active_obs:       list  = field(default_factory=list)


# ── Engine ────────────────────────────────────────────────────────────────────

class DLEStrategyEngine:

    INDICES = {
        "NIFTY":     {"yf": "^NSEI",   "lot": 75},
        "BANKNIFTY": {"yf": "^NSEBANK","lot": 35},
        "SENSEX":    {"yf": "^BSESN",  "lot": 20},
    }

    def __init__(self, index: str = "NIFTY", days: int = 90,
                 swing_n: int = 5, min_impulse: int = 2):
        meta = self.INDICES.get(index, self.INDICES["NIFTY"])
        self.index      = index
        self.yf_sym     = meta["yf"]
        self.lot        = meta["lot"]
        self.days       = days
        self.swing_n    = swing_n    # candles each side to confirm swing H/L
        self.min_impulse = min_impulse  # consecutive candles to confirm impulse

    # ── data fetch ─────────────────────────────────────────────────────────────

    def _fetch(self, interval: str, days: int) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data.  4H is not a yfinance interval — we fetch 1H and resample.
        yfinance limits: 15m ≤ 60d, 1h ≤ 730d.
        """
        import yfinance as yf

        # 4H: download 1H and resample — yfinance has no '4h' interval
        if interval == "4h":
            df1h = self._fetch("1h", min(days + 30, 729))
            if df1h is None or df1h.empty:
                return None
            df = (df1h.resample("4h")
                  .agg({"Open": "first", "High": "max",
                        "Low": "min", "Close": "last", "Volume": "sum"})
                  .dropna())
            return df

        cap   = {"15m": 57, "30m": 57, "1h": 725}
        d     = min(days, cap.get(interval, days))
        end   = datetime.now()
        buf   = 1 if interval in ("15m", "30m") else 5   # 15m hard limit ~60d
        start = end - timedelta(days=d + buf)
        try:
            df = yf.download(self.yf_sym,
                             start=start.strftime('%Y-%m-%d'),
                             end=end.strftime('%Y-%m-%d'),
                             interval=interval, progress=False, auto_adjust=True)
            if df is None or df.empty:
                return None
            # Flatten MultiIndex columns (newer yfinance returns ticker as second level)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            return df
        except Exception:
            return None

    # ── swing high / low detection ─────────────────────────────────────────────

    def _swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        n   = self.swing_n
        df  = df.copy()
        sh  = [False] * len(df)
        sl  = [False] * len(df)
        for i in range(n, len(df) - n):
            window_h = df['High'].iloc[i - n: i + n + 1]
            window_l = df['Low'].iloc[i - n: i + n + 1]
            if df['High'].iloc[i] == window_h.max():
                sh[i] = True
            if df['Low'].iloc[i] == window_l.min():
                sl[i] = True
        df['is_sh'] = sh
        df['is_sl'] = sl
        return df

    # ── Step 1: 4H Break of Structure ─────────────────────────────────────────

    def _detect_bos(self, df4h: pd.DataFrame) -> List[Dict]:
        df   = self._swing_points(df4h)
        bos  = []
        last_sh_price = last_sl_price = None
        last_sh_idx   = last_sl_idx   = None

        for i in range(len(df)):
            row = df.iloc[i]
            if row['is_sh']:
                last_sh_price = row['High']
                last_sh_idx   = i
            if row['is_sl']:
                last_sl_price = row['Low']
                last_sl_idx   = i

            # Bullish BOS: close above last confirmed swing high
            if last_sh_price and row['Close'] > last_sh_price:
                swing_low = last_sl_price if last_sl_price else row['Low']
                if not bos or bos[-1]['type'] != 'BULL' or bos[-1]['swing_high'] != last_sh_price:
                    bos.append({
                        'ts': df.index[i], 'idx': i, 'type': 'BULL',
                        'swing_low':  swing_low,
                        'swing_high': last_sh_price,
                    })
                last_sh_price = None  # consumed

            # Bearish BOS: close below last confirmed swing low
            elif last_sl_price and row['Close'] < last_sl_price:
                swing_high = last_sh_price if last_sh_price else row['High']
                if not bos or bos[-1]['type'] != 'BEAR' or bos[-1]['swing_low'] != last_sl_price:
                    bos.append({
                        'ts': df.index[i], 'idx': i, 'type': 'BEAR',
                        'swing_high': swing_high,
                        'swing_low':  last_sl_price,
                    })
                last_sl_price = None

        return bos

    # ── Step 2: 1H Order Blocks ────────────────────────────────────────────────

    def _find_obs(self, df1h: pd.DataFrame,
                  equil: float, bias: str) -> List[Dict]:
        """
        1H Order Blocks in the correct premium/discount zone.
        An OB is 'consumed' only when a 1H candle CLOSES through the zone body —
        a mere wick/touch is NOT consumption (that is exactly when we execute).
        Returns OBs that have NOT been fully consumed yet.

        BULL → Demand OB: last bearish 1H candle before a bullish impulse, below equil.
        BEAR → Supply OB: last bullish 1H candle before a bearish impulse, above equil.
        """
        obs = []
        n   = self.min_impulse

        for i in range(1, len(df1h) - n - 1):
            row = df1h.iloc[i]

            if bias == 'BULL':
                if row['Close'] >= row['Open']:      # must be a bearish candle
                    continue
                ob_low  = min(row['Open'], row['Close'])
                ob_high = max(row['Open'], row['Close'])
                if ob_high > equil:                  # must sit in discount zone
                    continue
                # At least 1 bullish candle in the next n candles (impulse away)
                after = df1h.iloc[i + 1: i + 1 + n]
                if len(after) == 0:
                    continue
                if (after['Close'] > after['Open']).sum() < 1:
                    continue
                # OB consumed only if a future close goes BELOW ob_low
                future   = df1h.iloc[i + 1:]
                consumed = (future['Close'] < ob_low).any()

            else:  # BEAR
                if row['Close'] <= row['Open']:      # must be a bullish candle
                    continue
                ob_low  = min(row['Open'], row['Close'])
                ob_high = max(row['Open'], row['Close'])
                if ob_low < equil:                   # must sit in premium zone
                    continue
                after = df1h.iloc[i + 1: i + 1 + n]
                if len(after) == 0:
                    continue
                if (after['Close'] < after['Open']).sum() < 1:
                    continue
                # OB consumed only if a future close goes ABOVE ob_high
                future   = df1h.iloc[i + 1:]
                consumed = (future['Close'] > ob_high).any()

            if not consumed:
                obs.append({
                    'ts':         df1h.index[i],
                    'bias':       bias,
                    'zone_low':   round(ob_low, 2),
                    'zone_high':  round(ob_high, 2),
                })

        return obs

    # ── Step 3: 15M execution confluences ─────────────────────────────────────

    def _check_confluences(self, df15m: pd.DataFrame, start_idx: int,
                           ob_low: float, ob_high: float,
                           bias: str) -> Tuple[bool, Optional[Dict]]:
        """
        Returns (True, trade_info) when ALL 3 confluences fire inside the OB.
        Looks at up to 30 candles from start_idx.
        """
        window = df15m.iloc[start_idx: start_idx + 30]
        if len(window) < 4:
            return False, None

        for j in range(2, len(window) - 2):
            curr = window.iloc[j]
            prev = window.iloc[j - 1]

            # Price must touch the OB zone
            if not (curr['Low'] <= ob_high and curr['High'] >= ob_low):
                continue

            # ── Confluence 1: Rejection candle (Engulfing or Pin Bar) ──────────
            c1 = None
            body      = abs(curr['Close'] - curr['Open'])
            if body == 0:
                continue

            if bias == 'BULL':
                # Bullish engulfing
                bull_eng = (curr['Close'] > curr['Open']
                            and prev['Close'] < prev['Open']
                            and curr['Close'] > prev['Open']
                            and curr['Open']  < prev['Close'])
                # Bullish pin bar (hammer): lower wick > 2× body
                lower_wick = min(curr['Open'], curr['Close']) - curr['Low']
                upper_wick = curr['High'] - max(curr['Open'], curr['Close'])
                pin        = (lower_wick >= 2 * body and lower_wick > upper_wick)
                if bull_eng:
                    c1 = "Bullish Engulfing"
                elif pin:
                    c1 = "Bullish Pin Bar (Hammer)"
            else:  # BEAR
                # Bearish engulfing
                bear_eng = (curr['Close'] < curr['Open']
                            and prev['Close'] > prev['Open']
                            and curr['Close'] < prev['Open']
                            and curr['Open']  > prev['Close'])
                # Bearish pin bar (shooting star): upper wick > 2× body
                upper_wick = curr['High'] - max(curr['Open'], curr['Close'])
                lower_wick = min(curr['Open'], curr['Close']) - curr['Low']
                pin        = (upper_wick >= 2 * body and upper_wick > lower_wick)
                if bear_eng:
                    c1 = "Bearish Engulfing"
                elif pin:
                    c1 = "Bearish Pin Bar (Shooting Star)"

            if not c1:
                continue

            rej_low  = curr['Low']
            rej_high = curr['High']

            # ── Confluence 2: Market Structure Shift (MSS) ──────────────────
            recent = window.iloc[max(0, j - 6): j]
            if len(recent) < 2:
                continue
            c2 = None
            if bias == 'BULL':
                recent_sh = recent['High'].max()
                if curr['Close'] > recent_sh:
                    c2 = f"MSS: close {curr['Close']:.2f} > local high {recent_sh:.2f}"
            else:
                recent_sl = recent['Low'].min()
                if curr['Close'] < recent_sl:
                    c2 = f"MSS: close {curr['Close']:.2f} < local low {recent_sl:.2f}"

            if not c2:
                continue

            # ── Confluence 3: Failed Counter-Move ────────────────────────────
            nxt = window.iloc[j + 1]
            c3  = None
            if bias == 'BULL':
                if nxt['Close'] >= rej_low:   # counter-move failed to close below rej low
                    c3 = f"Counter-move held above rej low {rej_low:.2f}"
            else:
                if nxt['Close'] <= rej_high:  # counter-move failed to close above rej high
                    c3 = f"Counter-move held below rej high {rej_high:.2f}"

            if not c3:
                continue

            # ── All 3 confluences met → define trade ─────────────────────────
            entry_bar = window.iloc[j + 1]
            entry     = float(entry_bar['Open'])

            if bias == 'BULL':
                sl    = round(rej_low  - body * 0.1, 2)   # tiny buffer
                risk  = entry - sl
            else:
                sl    = round(rej_high + body * 0.1, 2)
                risk  = sl - entry

            if risk <= 0:
                continue

            tp = round(entry + risk * 3, 2) if bias == 'BULL' else round(entry - risk * 3, 2)

            return True, {
                'entry':     round(entry, 2),
                'sl':        sl,
                'tp':        tp,
                'risk':      round(risk, 2),
                'entry_ts':  entry_bar.name,
                'rej_low':   rej_low,
                'rej_high':  rej_high,
                'c1': c1, 'c2': c2, 'c3': c3,
            }

        return False, None

    # ── Simulate exit on forward bars ──────────────────────────────────────────

    def _simulate_exit(self, df15m: pd.DataFrame, after_ts,
                       entry: float, sl: float, tp: float,
                       bias: str) -> Tuple[float, str, str]:
        future = df15m[df15m.index > after_ts]
        for ts, row in future.iterrows():
            if bias == 'BULL':
                if row['Low']  <= sl: return sl, "SL", str(ts)[:16]
                if row['High'] >= tp: return tp, "TP", str(ts)[:16]
            else:
                if row['High'] >= sl: return sl, "SL", str(ts)[:16]
                if row['Low']  <= tp: return tp, "TP", str(ts)[:16]
        last = future.iloc[-1] if len(future) > 0 else None
        ep   = float(last['Close']) if last is not None else entry
        return ep, "OPEN", str(future.index[-1])[:16] if len(future) > 0 else ""

    # ── Main run ───────────────────────────────────────────────────────────────

    def run(self) -> Optional[DLEResult]:
        # Fetch all 3 timeframes
        df4h  = self._fetch("4h",  self.days + 30)
        df1h  = self._fetch("1h",  self.days + 10)
        df15m = self._fetch("15m", min(self.days, 58))

        if df4h is None or df1h is None or df15m is None:
            return None
        if len(df4h) < 10 or len(df1h) < 10 or len(df15m) < 20:
            return None

        # ── Step 1: 4H BOS events ─────────────────────────────────────────────
        bos_events = self._detect_bos(df4h)
        if not bos_events:
            return self._empty()

        trades: List[DLETrade] = []
        equity = [100_000.0]
        cur_eq = 100_000.0

        for bi, bos in enumerate(bos_events):
            bias       = bos['type']
            swing_low  = bos['swing_low']
            swing_high = bos['swing_high']
            equil      = (swing_low + swing_high) / 2.0
            bos_ts     = bos['ts']
            next_ts    = bos_events[bi + 1]['ts'] if bi + 1 < len(bos_events) else df4h.index[-1]

            # ── Step 2: 1H Order Blocks valid for this bias period ────────────
            df1h_win = df1h[(df1h.index >= bos_ts) & (df1h.index < next_ts)]
            if len(df1h_win) < 4:
                continue

            obs = self._find_obs(df1h_win, equil, bias)
            if not obs:
                continue

            for ob in obs:
                ob_ts   = ob['ts']
                ob_low  = ob['zone_low']
                ob_high = ob['zone_high']

                # ── Step 3: 15M confluences ───────────────────────────────────
                df15m_win = df15m[(df15m.index >= ob_ts) & (df15m.index < next_ts)]
                if len(df15m_win) < 5:
                    continue

                # Find first bar where price touches the OB
                for k in range(len(df15m_win) - 4):
                    bar = df15m_win.iloc[k]
                    if not (bar['Low'] <= ob_high and bar['High'] >= ob_low):
                        continue

                    ok, info = self._check_confluences(
                        df15m_win, k, ob_low, ob_high, bias)
                    if not ok:
                        continue

                    entry     = info['entry']
                    sl        = info['sl']
                    tp        = info['tp']
                    risk      = info['risk']
                    entry_ts  = info['entry_ts']

                    exit_p, exit_r, exit_t = self._simulate_exit(
                        df15m, entry_ts, entry, sl, tp, bias)

                    pnl = (exit_p - entry) if bias == 'BULL' else (entry - exit_p)
                    won = pnl > 0
                    cur_eq += pnl * self.lot
                    equity.append(cur_eq)

                    trades.append(DLETrade(
                        date=str(pd.Timestamp(entry_ts).date()),
                        time=str(pd.Timestamp(entry_ts).time())[:5],
                        symbol=self.index,
                        htf_bias=bias,
                        equil=round(equil, 2),
                        ob_zone_low=ob_low,
                        ob_zone_high=ob_high,
                        entry=entry, stop=sl, target=tp,
                        risk_pts=risk, rr=3.0,
                        exit_price=round(exit_p, 2),
                        exit_time=exit_t,
                        exit_reason=exit_r,
                        pnl_pts=round(pnl, 2),
                        won=won,
                        c1=info['c1'], c2=info['c2'], c3=info['c3'],
                    ))
                    break   # one trade per OB

        if not trades:
            return self._empty()

        return self._compute(trades, equity)

    def _empty(self) -> DLEResult:
        return DLEResult(
            symbol=self.index, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, total_pnl_pts=0, avg_win_pts=0, avg_loss_pts=0,
            profit_factor=0, max_drawdown_pts=0)

    def _compute(self, trades: List[DLETrade],
                 equity: List[float]) -> DLEResult:
        wins   = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        wr     = len(wins) / len(trades) * 100 if trades else 0
        avg_w  = float(np.mean([t.pnl_pts for t in wins]))   if wins   else 0.0
        avg_l  = float(abs(np.mean([t.pnl_pts for t in losses]))) if losses else 0.0
        pf     = (len(wins) * avg_w) / (len(losses) * avg_l) if losses and avg_l > 0 else 99.0
        eq     = np.array(equity)
        peak   = np.maximum.accumulate(eq)
        dd     = float(np.max(peak - eq))
        return DLEResult(
            symbol=self.index,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=round(wr, 1),
            total_pnl_pts=round(sum(t.pnl_pts for t in trades), 2),
            avg_win_pts=round(avg_w, 2),
            avg_loss_pts=round(avg_l, 2),
            profit_factor=round(pf, 2),
            max_drawdown_pts=round(dd, 2),
            trades=trades,
            equity_curve=equity,
            current_bias=trades[-1].htf_bias if trades else "NEUTRAL",
            current_equil=trades[-1].equil if trades else 0.0,
        )
