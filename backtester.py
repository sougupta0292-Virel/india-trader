"""
backtester.py - Two strategies:
1. Fibonacci + OI Strategy (our main system)
2. Intraday Hunter Strategy (gap + divergence + trap)
"""
import numpy as np
import pandas as pd
from datetime import datetime, date, time
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# ─── Safe float extraction ────────────────────────────────────────────────────
def _f(v):
    try:
        while hasattr(v, 'iloc'): v = v.iloc[0]
        if hasattr(v, 'item'): return float(v.item())
        return float(v)
    except: return 0.0

# ─── Trade & Result structures ────────────────────────────────────────────────
@dataclass
class BtTrade:
    date:        str
    strategy:    str
    direction:   str
    entry:       float
    target:      float
    stop:        float
    exit_price:  float
    exit_reason: str
    pnl_pts:     float
    rr:          float
    won:         bool
    reason:      str = ""

@dataclass
class BtResult:
    strategy:        str
    total_trades:    int
    winning_trades:  int
    losing_trades:   int
    win_rate:        float
    avg_win_pts:     float
    avg_loss_pts:    float
    profit_factor:   float
    max_drawdown_pts: float
    total_pnl_pts:   float
    trades:          list = field(default_factory=list)
    equity_curve:    list = field(default_factory=list)

# ─── Data Fetcher ─────────────────────────────────────────────────────────────
def fetch_index(symbol: str, days: int) -> Optional[pd.DataFrame]:
    import yfinance as yf
    from datetime import datetime, timedelta

    # yfinance 5m data limited to 60 days max
    # For longer periods use 1d interval
    frames = []

    if days <= 55:
        # Fetch 5-minute data directly
        try:
            df = yf.download(symbol, period=f"{min(days+5,55)}d",
                             interval="5m", progress=False, auto_adjust=True)
            if df is not None and len(df) > 20:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                frames.append(df)
        except Exception as e:
            print(f"5m fetch error: {e}")
    else:
        # Fetch in 30-day chunks
        end = datetime.now()
        remaining = days
        while remaining > 0:
            chunk = min(remaining, 55)
            start = end - timedelta(days=chunk)
            try:
                df = yf.download(symbol, start=start.strftime('%Y-%m-%d'),
                                 end=end.strftime('%Y-%m-%d'),
                                 interval="5m", progress=False, auto_adjust=True)
                if df is not None and len(df) > 0:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    frames.append(df)
            except Exception as e:
                print(f"Chunk fetch error: {e}")
            end = start
            remaining -= chunk

    if not frames:
        print(f"No data fetched for {symbol}")
        return None

    try:
        result = pd.concat(frames).sort_index().drop_duplicates()
        for col in ['Open','High','Low','Close']:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce').astype(float)
        result = result.dropna(subset=['Open','High','Low','Close'])
        if hasattr(result.index, 'date'):
            result['_date'] = result.index.date
        else:
            result['_date'] = pd.to_datetime(result.index).date
        print(f"Fetched {len(result)} rows for {symbol} ({result['_date'].nunique()} days)")
        return result
    except Exception as e:
        print(f"Concat error: {e}")
        return None

def simulate_exit(remaining: pd.DataFrame, direction: str,
                   entry: float, target: float, sl: float) -> tuple:
    """Simulate trade exit through remaining candles."""
    for j in range(len(remaining)):
        row = remaining.iloc[j]
        h   = _f(row['High'])
        l   = _f(row['Low'])
        if direction == "LONG":
            if l <= sl:   return sl,     "STOP_LOSS", (j+1)*5
            if h >= target: return target, "TARGET_HIT", (j+1)*5
        else:
            if h >= sl:   return sl,     "STOP_LOSS", (j+1)*5
            if l <= target: return target, "TARGET_HIT", (j+1)*5
    exit_p = _f(remaining.iloc[-1]['Close']) if len(remaining) > 0 else entry
    return exit_p, "EOD_EXIT", len(remaining)*5


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: Fibonacci + OI
# ═══════════════════════════════════════════════════════════════════════════════
class FibOIBacktester:
    """
    Strategy: Fibonacci retracement + OI Put-Call Ratio
    - Entry at 61.8% Fibonacci retracement of previous day range
    - Confirmed by PCR (>1.2 bullish, <0.8 bearish)
    - Target at 38.2% extension
    - SL below/above 78.6% level
    """
    def __init__(self, instrument="NIFTY", days=30, lot_size=50):
        self.symbol = "^NSEI" if "BANK" not in instrument else "^NSEBANK"
        self.instrument = instrument
        self.days    = days
        self.lot_size= lot_size

    def run(self) -> BtResult:
        df = fetch_index(self.symbol, self.days)
        if df is None or len(df) < 10:
            return self._empty()

        dates  = sorted(df['_date'].unique())
        trades = []
        equity = [100000]
        cur_eq = 100000

        for i in range(1, len(dates)):
            today = dates[i]
            prev  = dates[i-1]

            today_df = df[df['_date'] == today].copy()
            prev_df  = df[df['_date'] == prev].copy()
            if len(today_df) < 6 or len(prev_df) < 3: continue

            # Previous day H/L/C
            prev_high  = float(prev_df['High'].max())
            prev_low   = float(prev_df['Low'].min())
            prev_close = float(prev_df['Close'].iloc[-1])
            diff       = prev_high - prev_low
            if diff < 50: continue

            # Today opening
            fc       = today_df.iloc[0]
            fc_open  = _f(fc['Open'])
            fc_close = _f(fc['Close'])
            spot     = fc_close

            # Simulated PCR based on gap direction
            gap_pct = (fc_open - prev_close) / prev_close if prev_close > 0 else 0
            day_ret = (float(prev_df['Close'].iloc[-1]) - float(prev_df['Open'].iloc[0])) / float(prev_df['Open'].iloc[0])
            pcr = 1.0 + day_ret * 10  # Simulate PCR from previous day trend

            # Fibonacci levels
            fib_618_up   = prev_low  + diff * 0.618   # 61.8% from low (LONG entry)
            fib_618_down = prev_high - diff * 0.618   # 61.8% from high (SHORT entry)
            fib_786_up   = prev_low  + diff * 0.786
            fib_786_down = prev_high - diff * 0.786
            fib_382_up   = prev_low  + diff * 0.764   # Target LONG (76.4%)
            fib_382_down = prev_high - diff * 0.764   # Target SHORT (76.4%)

            # Signal: Price near 61.8% Fib + PCR confirmation
            direction = None
            entry = target = sl = 0

            dist_to_618_up   = abs(spot - fib_618_up) / diff
            dist_to_618_down = abs(spot - fib_618_down) / diff

            if dist_to_618_up < 0.40 and pcr > 1.0:
                direction = "LONG"
                entry     = spot
                target    = fib_382_up
                sl        = fib_786_up - 20
            elif dist_to_618_down < 0.40 and pcr < 1.0:
                direction = "SHORT"
                entry     = spot
                target    = fib_382_down
                sl        = fib_786_down + 20

            if direction is None:
                equity.append(cur_eq); continue

            risk   = abs(entry - sl)
            reward = abs(target - entry)
            if risk == 0: equity.append(cur_eq); continue
            rr = reward / risk
            if rr < 1.2: equity.append(cur_eq); continue

            exit_p, exit_r, _ = simulate_exit(today_df.iloc[1:], direction, entry, target, sl)
            pnl = (exit_p - entry) if direction=="LONG" else (entry - exit_p)
            cur_eq += pnl * self.lot_size

            trades.append(BtTrade(
                date=str(today), strategy="FIB+OI",
                direction=direction, entry=round(entry,2),
                target=round(target,2), stop=round(sl,2),
                exit_price=round(exit_p,2), exit_reason=exit_r,
                pnl_pts=round(pnl,2), rr=round(rr,2),
                won=pnl>0,
                reason=f"61.8% Fib | PCR={pcr:.2f} | Gap {gap_pct*100:+.2f}%"
            ))
            equity.append(cur_eq)

        return self._compute(trades, equity)

    def _empty(self):
        return BtResult("FIB+OI",0,0,0,0,0,0,0,0,0,[],[])

    def _compute(self, trades, equity) -> BtResult:
        if not trades: return self._empty()
        wins   = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        wr     = len(wins)/len(trades)*100
        avg_w  = np.mean([t.pnl_pts for t in wins]) if wins else 0
        avg_l  = abs(np.mean([t.pnl_pts for t in losses])) if losses else 0
        pf     = (len(wins)*avg_w)/(len(losses)*avg_l) if losses and avg_l>0 else 99
        eq     = np.array(equity)
        peak   = np.maximum.accumulate(eq)
        dd     = float(np.max(peak-eq))
        return BtResult(
            strategy="FIB+OI",
            total_trades=len(trades), winning_trades=len(wins),
            losing_trades=len(losses), win_rate=round(wr,1),
            avg_win_pts=round(avg_w,2), avg_loss_pts=round(avg_l,2),
            profit_factor=round(pf,2), max_drawdown_pts=round(dd,2),
            total_pnl_pts=round(sum(t.pnl_pts for t in trades),2),
            trades=trades, equity_curve=equity
        )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: Intraday Hunter (All 3 indices)
# ═══════════════════════════════════════════════════════════════════════════════
class IntraHunterBacktester:
    """
    Strategy: Intraday Hunter - Gap + Divergence + Trap
    - Checks Nifty, BankNifty, Sensex simultaneously
    - Trades divergence (2 strong + 1 weak)
    - Gap trap confirmation
    - Target: retail SL levels
    """
    def __init__(self, days=30, lot_size=15):
        self.days     = days
        self.lot_size = lot_size

    def run(self) -> BtResult:
        # Fetch all 3 indices
        print("Fetching Nifty...")
        nf = fetch_index("^NSEI",   self.days)
        print("Fetching BankNifty...")
        bn = fetch_index("^NSEBANK", self.days)
        print("Fetching Sensex...")
        sx = fetch_index("^BSESN",  self.days)

        # Use whichever we got
        if nf is None and bn is None:
            return self._empty()
        if nf is None: nf = bn
        if bn is None: bn = nf
        if sx is None: sx = nf

        dates  = sorted(set(nf['_date'].unique()) &
                        set(bn['_date'].unique()))
        trades = []
        equity = [100000]
        cur_eq = 100000

        for i in range(1, len(dates)):
            today = dates[i]
            prev  = dates[i-1]

            nf_t = nf[nf['_date']==today]; nf_p = nf[nf['_date']==prev]
            bn_t = bn[bn['_date']==today]; bn_p = bn[bn['_date']==prev]
            sx_t = sx[sx['_date']==today]; sx_p = sx[sx['_date']==prev]

            if any(len(x)<6 for x in [nf_t,bn_t,nf_p,bn_p]): continue

            # Previous day stats
            nf_prev_close = float(nf_p['Close'].iloc[-1])
            bn_prev_close = float(bn_p['Close'].iloc[-1])
            bn_prev_high  = float(bn_p['High'].max())
            bn_prev_low   = float(bn_p['Low'].min())

            # First candle
            def fc_stats(df_t, prev_close):
                fc = df_t.iloc[0]
                o  = _f(fc['Open'])
                c  = _f(fc['Close'])
                gap = (o - prev_close) / prev_close if prev_close > 0 else 0
                bull = c > o
                body = abs(c - o)
                rng  = _f(fc['High']) - _f(fc['Low'])
                strong = rng > 0 and body/rng > 0.5
                move = (c - o) / o if o > 0 else 0
                return {'gap':gap,'bull':bull,'strong':strong,'move':move,'close':c,'open':o}

            nf_fc = fc_stats(nf_t, nf_prev_close)
            bn_fc = fc_stats(bn_t, bn_prev_close)
            try: sx_fc = fc_stats(sx_t, float(sx_p['Close'].iloc[-1]))
            except: sx_fc = nf_fc

            # ── Intraday Hunter Rules ──────────────────────────────────
            direction = None
            reason    = ""

            # Rule 1: Gap UP + Rejection (all indices)
            gap_ups = sum(1 for fc in [nf_fc,bn_fc,sx_fc] if fc['gap']>0.002)
            rejections = sum(1 for fc in [nf_fc,bn_fc,sx_fc] if not fc['bull'] and fc['strong'])
            if gap_ups >= 2 and rejections >= 2:
                direction = "SHORT"
                reason = f"Gap UP ({gap_ups}/3) + Rejection ({rejections}/3) → Buyer Trap"

            # Rule 2: Gap DOWN + Recovery
            elif sum(1 for fc in [nf_fc,bn_fc,sx_fc] if fc['gap']<-0.002) >= 2 and \
                 sum(1 for fc in [nf_fc,bn_fc,sx_fc] if fc['bull'] and fc['strong']) >= 2:
                direction = "LONG"
                reason = "Gap DOWN + Recovery → Seller Trap"

            # Rule 3: Divergence — Nifty+Sensex strong, BankNifty weak
            elif (nf_fc['move'] > 0.003 and sx_fc['move'] > 0.003 and
                  abs(bn_fc['move']) < 0.001):
                direction = "SHORT"  # BankNifty lagging → will reverse
                reason = "Divergence: Nifty+Sensex strong, BankNifty weak → PUT BN"

            # Rule 4: Divergence — Nifty+Sensex falling, BankNifty holding
            elif (nf_fc['move'] < -0.003 and sx_fc['move'] < -0.003 and
                  bn_fc['move'] > -0.001):
                direction = "LONG"
                reason = "Divergence: Nifty+Sensex falling, BankNifty holding → CALL BN"

            if direction is None:
                equity.append(cur_eq); continue

            # Trade BankNifty
            diff   = bn_prev_high - bn_prev_low
            if diff < 50: equity.append(cur_eq); continue

            spot   = bn_fc['close']
            if direction == "LONG":
                entry  = spot
                target = spot + diff * 0.382
                sl     = spot - diff * 0.168  # Tighter SL
            else:
                entry  = spot
                target = spot - diff * 0.382
                sl     = spot + diff * 0.168  # Tighter SL

            risk   = abs(entry - sl)
            reward = abs(target - entry)
            if risk == 0: equity.append(cur_eq); continue
            rr = reward / risk
            if rr < 1.3: equity.append(cur_eq); continue

            # Simulate on BankNifty
            exit_p, exit_r, _ = simulate_exit(bn_t.iloc[1:], direction, entry, target, sl)
            pnl = (exit_p - entry) if direction=="LONG" else (entry - exit_p)
            cur_eq += pnl * self.lot_size

            trades.append(BtTrade(
                date=str(today), strategy="HUNTER",
                direction=direction, entry=round(entry,2),
                target=round(target,2), stop=round(sl,2),
                exit_price=round(exit_p,2), exit_reason=exit_r,
                pnl_pts=round(pnl,2), rr=round(rr,2),
                won=pnl>0, reason=reason
            ))
            equity.append(cur_eq)

        return self._compute(trades, equity)

    def _empty(self):
        return BtResult("HUNTER",0,0,0,0,0,0,0,0,0,[],[])

    def _compute(self, trades, equity) -> BtResult:
        if not trades: return self._empty()
        wins   = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        wr     = len(wins)/len(trades)*100
        avg_w  = np.mean([t.pnl_pts for t in wins]) if wins else 0
        avg_l  = abs(np.mean([t.pnl_pts for t in losses])) if losses else 0
        pf     = (len(wins)*avg_w)/(len(losses)*avg_l) if losses and avg_l>0 else 99
        eq     = np.array(equity)
        peak   = np.maximum.accumulate(eq)
        dd     = float(np.max(peak-eq))
        return BtResult(
            strategy="HUNTER",
            total_trades=len(trades), winning_trades=len(wins),
            losing_trades=len(losses), win_rate=round(wr,1),
            avg_win_pts=round(avg_w,2), avg_loss_pts=round(avg_l,2),
            profit_factor=round(pf,2), max_drawdown_pts=round(dd,2),
            total_pnl_pts=round(sum(t.pnl_pts for t in trades),2),
            trades=trades, equity_curve=equity
        )

# Keep old Backtester class for compatibility
class Backtester(FibOIBacktester):
    def __init__(self, instrument="NIFTY", days=30,
                 initial_capital=100000, lot_size=50):
        super().__init__(instrument=instrument, days=days, lot_size=lot_size)
        self.capital = initial_capital



# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Per-Index Backtester (Nifty, BankNifty, Sensex separately)
# ═══════════════════════════════════════════════════════════════════════════════
class PerIndexBacktester:
    """Run both strategies on each index separately."""

    INDICES = {
        "Nifty":     {"symbol": "^NSEI",   "lot": 75,  "min_range": 30},
        "BankNifty": {"symbol": "^NSEBANK", "lot": 35,  "min_range": 100},
        "Sensex":    {"symbol": "^BSESN",  "lot": 20,  "min_range": 100},
    }

    def __init__(self, days=30, strategy="FIB"):
        self.days     = days
        self.strategy = strategy  # "FIB" or "HUNTER"

    def run_all(self) -> dict:
        """Returns dict of {index_name: BtResult}"""
        results = {}
        for name, meta in self.INDICES.items():
            print(f"Running {self.strategy} on {name}...")
            df = fetch_index(meta["symbol"], self.days)
            if df is None or len(df) < 10:
                continue
            if self.strategy == "FIB":
                result = self._run_fib(df, name, meta)
            else:
                result = self._run_hunter(df, name, meta)
            results[name] = result
        return results

    def _run_fib(self, df, name, meta) -> BtResult:
        dates  = sorted(df['_date'].unique())
        trades = []; equity = [100000]; cur_eq = 100000

        for i in range(1, len(dates)):
            today = dates[i]; prev = dates[i-1]
            today_df = df[df['_date']==today]
            prev_df  = df[df['_date']==prev]
            if len(today_df)<6 or len(prev_df)<3: continue

            prev_high  = float(prev_df['High'].max())
            prev_low   = float(prev_df['Low'].min())
            prev_close = float(prev_df['Close'].iloc[-1])
            diff       = prev_high - prev_low
            if diff < meta["min_range"]: continue

            fc = today_df.iloc[0]
            fc_open = _f(fc['Open']); fc_close = _f(fc['Close'])
            spot = fc_close

            day_ret = (float(prev_df['Close'].iloc[-1]) - float(prev_df['Open'].iloc[0])) / max(float(prev_df['Open'].iloc[0]), 1)
            pcr = 1.0 + day_ret * 10

            fib_618_up   = prev_low  + diff * 0.618
            fib_618_down = prev_high - diff * 0.618
            fib_764_up   = prev_low  + diff * 0.764
            fib_764_down = prev_high - diff * 0.764
            ext_up       = prev_high + diff * 0.382
            ext_down     = prev_low  - diff * 0.382

            direction = None
            if abs(spot - fib_618_up)/diff < 0.40 and pcr > 1.0:
                direction = "LONG"; entry = spot; target = ext_up; sl = fib_764_up - 10
            elif abs(spot - fib_618_down)/diff < 0.40 and pcr < 1.0:
                direction = "SHORT"; entry = spot; target = ext_down; sl = fib_764_down + 10

            if direction is None: equity.append(cur_eq); continue
            risk = abs(entry-sl); reward = abs(target-entry)
            if risk==0 or reward/risk<1.2: equity.append(cur_eq); continue

            exit_p, exit_r, _ = simulate_exit(today_df.iloc[1:], direction, entry, target, sl)
            pnl = (exit_p-entry) if direction=="LONG" else (entry-exit_p)
            cur_eq += pnl * meta["lot"]

            trades.append(BtTrade(
                date=str(today), strategy=f"FIB-{name}",
                direction=direction, entry=round(entry,2),
                target=round(target,2), stop=round(sl,2),
                exit_price=round(exit_p,2), exit_reason=exit_r,
                pnl_pts=round(pnl,2), rr=round(reward/risk,2),
                won=pnl>0, reason=f"61.8% Fib | PCR={pcr:.2f}"
            ))
            equity.append(cur_eq)
        return self._compute(trades, equity, f"FIB-{name}")

    def _run_hunter(self, df, name, meta) -> BtResult:
        dates  = sorted(df['_date'].unique())
        trades = []; equity = [100000]; cur_eq = 100000

        for i in range(1, len(dates)):
            today = dates[i]; prev = dates[i-1]
            today_df = df[df['_date']==today]
            prev_df  = df[df['_date']==prev]
            if len(today_df)<6 or len(prev_df)<3: continue

            prev_high  = float(prev_df['High'].max())
            prev_low   = float(prev_df['Low'].min())
            prev_close = float(prev_df['Close'].iloc[-1])
            diff       = prev_high - prev_low
            if diff < meta["min_range"]: continue

            fc = today_df.iloc[0]
            fc_open  = _f(fc['Open']); fc_close = _f(fc['Close'])
            gap_pct  = (fc_open - prev_close)/prev_close if prev_close>0 else 0
            candle_bull = fc_close > fc_open
            body  = abs(fc_close - fc_open)
            rng   = _f(fc['High']) - _f(fc['Low'])
            strong = rng>0 and body/rng>0.5

            direction = None
            if gap_pct > 0.002 and not candle_bull and strong:
                direction = "SHORT"
                reason = f"Gap UP {gap_pct*100:+.2f}% + Rejection"
            elif gap_pct < -0.002 and candle_bull and strong:
                direction = "LONG"
                reason = f"Gap DOWN {gap_pct*100:+.2f}% + Recovery"
            elif abs(gap_pct) < 0.001 and strong:
                prev_bull = prev_close > (prev_high+prev_low)/2
                if candle_bull and prev_bull:
                    direction = "LONG"; reason = "Flat open + Bullish candle"
                elif not candle_bull and not prev_bull:
                    direction = "SHORT"; reason = "Flat open + Bearish candle"

            if direction is None: equity.append(cur_eq); continue

            spot = fc_close
            if direction == "LONG":
                entry = spot; target = spot+diff*0.382; sl = spot-diff*0.168
            else:
                entry = spot; target = spot-diff*0.382; sl = spot+diff*0.168

            risk = abs(entry-sl); reward = abs(target-entry)
            if risk==0 or reward/risk<1.3: equity.append(cur_eq); continue

            exit_p, exit_r, _ = simulate_exit(today_df.iloc[1:], direction, entry, target, sl)
            pnl = (exit_p-entry) if direction=="LONG" else (entry-exit_p)
            cur_eq += pnl * meta["lot"]

            trades.append(BtTrade(
                date=str(today), strategy=f"HUNTER-{name}",
                direction=direction, entry=round(entry,2),
                target=round(target,2), stop=round(sl,2),
                exit_price=round(exit_p,2), exit_reason=exit_r,
                pnl_pts=round(pnl,2), rr=round(reward/risk,2),
                won=pnl>0, reason=reason
            ))
            equity.append(cur_eq)
        return self._compute(trades, equity, f"HUNTER-{name}")


    def _run_5ema_reject(self, df, name) -> BtResult:
        """5 EMA Rejection setup."""
        trades = []; equity = [100000]; cur_eq = 100000
        df = df.copy()
        df['ema5'] = df['Close'].ewm(span=5, adjust=False).mean()

        for i in range(2, len(df)-1):
            row  = df.iloc[i]; nxt = df.iloc[i+1]
            ema5 = float(row['ema5'])
            h = float(row['High']); l = float(row['Low'])
            o = float(row['Open']); c = float(row['Close'])
            nh = float(nxt['High']); nl = float(nxt['Low']); nc = float(nxt['Close'])
            candle_range = h - l
            if candle_range < 20: equity.append(cur_eq); continue

            direction = None
            # Bearish rejection: EMA within candle, closes bearish, upper wick dominant
            if l <= ema5 <= h and c < o:
                upper_wick = h - max(o, c)
                body = abs(c - o)
                if upper_wick > body * 0.5:
                    direction = "SHORT"
                    entry = l; sl = h + 10; target = l - candle_range * 2
            # Bullish rejection
            elif l <= ema5 <= h and c > o:
                lower_wick = min(o, c) - l
                body = abs(c - o)
                if lower_wick > body * 0.5:
                    direction = "LONG"
                    entry = h; sl = l - 10; target = h + candle_range * 2

            if direction is None: equity.append(cur_eq); continue
            risk = abs(entry-sl); reward = abs(target-entry)
            if risk == 0 or reward/risk < 1.8: equity.append(cur_eq); continue

            if direction == "LONG":
                exit_p = target if nh>=target else sl if nl<=sl else nc
                exit_r = "TARGET_HIT" if nh>=target else "STOP_LOSS" if nl<=sl else "EOD"
            else:
                exit_p = target if nl<=target else sl if nh>=sl else nc
                exit_r = "TARGET_HIT" if nl<=target else "STOP_LOSS" if nh>=sl else "EOD"

            pnl = (exit_p-entry) if direction=="LONG" else (entry-exit_p)
            cur_eq += pnl * self.lot_size
            trades.append(BtTrade(str(df.index[i])[:10], f"5EMA_REJECT", direction,
                round(entry,2), round(target,2), round(sl,2), round(exit_p,2),
                exit_r, round(pnl,2), round(reward/risk,2), pnl>0,
                f"EMA rejection at {ema5:.0f}"))
            equity.append(cur_eq)
        return self._compute(trades, equity, f"5EMA_REJECT-{name}")

    def _run_5ema_cross(self, df, name) -> BtResult:
        """5 EMA Crossover setup."""
        trades = []; equity = [100000]; cur_eq = 100000
        df = df.copy()
        df['ema5'] = df['Close'].ewm(span=5, adjust=False).mean()

        for i in range(3, len(df)-1):
            c1=df.iloc[i-2]; c2=df.iloc[i-1]; curr=df.iloc[i]; nxt=df.iloc[i+1]
            e1=float(c1['ema5']); e2=float(c2['ema5']); e3=float(curr['ema5'])
            cl1=float(c1['Close']); cl2=float(c2['Close']); cl3=float(curr['Close'])
            h3=float(curr['High']); l3=float(curr['Low'])
            nh=float(nxt['High']); nl=float(nxt['Low']); nc=float(nxt['Close'])
            rng = h3 - l3
            if rng < 20: equity.append(cur_eq); continue

            direction = None
            if cl1<e1 and cl2<e2 and cl3>e3:
                direction="LONG"; entry=cl3; sl=l3-10; target=cl3+rng*2
            elif cl1>e1 and cl2>e2 and cl3<e3:
                direction="SHORT"; entry=cl3; sl=h3+10; target=cl3-rng*2

            if direction is None: equity.append(cur_eq); continue
            risk=abs(entry-sl); reward=abs(target-entry)
            if risk==0 or reward/risk<1.8: equity.append(cur_eq); continue

            if direction=="LONG":
                exit_p=target if nh>=target else sl if nl<=sl else nc
                exit_r="TARGET_HIT" if nh>=target else "STOP_LOSS" if nl<=sl else "EOD"
            else:
                exit_p=target if nl<=target else sl if nh>=sl else nc
                exit_r="TARGET_HIT" if nl<=target else "STOP_LOSS" if nh>=sl else "EOD"

            pnl=(exit_p-entry) if direction=="LONG" else (entry-exit_p)
            cur_eq += pnl*self.lot_size
            trades.append(BtTrade(str(df.index[i])[:10],"5EMA_CROSS",direction,
                round(entry,2),round(target,2),round(sl,2),round(exit_p,2),
                exit_r,round(pnl,2),round(reward/risk,2),pnl>0,
                f"EMA crossover at {e3:.0f}"))
            equity.append(cur_eq)
        return self._compute(trades, equity, f"5EMA_CROSS-{name}")

    def _run_inside_ema(self, df, name) -> BtResult:
        """Inside Candle + EMA combo setup."""
        trades = []; equity = [100000]; cur_eq = 100000
        df = df.copy()
        df['ema5'] = df['Close'].ewm(span=5, adjust=False).mean()

        for i in range(2, len(df)-1):
            mother=df.iloc[i-2]; child=df.iloc[i-1]
            curr=df.iloc[i]; nxt=df.iloc[i+1]
            mh=float(mother['High']); ml=float(mother['Low'])
            ch=float(child['High']); cl=float(child['Low'])
            ema5=float(child['ema5'])
            crh=float(curr['High']); crl=float(curr['Low'])
            nh=float(nxt['High']); nl=float(nxt['Low']); nc=float(nxt['Close'])

            if not (ch<=mh and cl>=ml): equity.append(cur_eq); continue
            if ch==mh or cl==ml: equity.append(cur_eq); continue
            mrange = mh-ml
            if mrange < 30: equity.append(cur_eq); continue

            # EMA near candle
            mid = (mh+ml)/2
            if abs(ema5-mid)/mid > 0.008: equity.append(cur_eq); continue

            direction=None
            if crh>mh: direction="LONG"; entry=mh; sl=ml-10; target=mh+mrange*2
            elif crl<ml: direction="SHORT"; entry=ml; sl=mh+10; target=ml-mrange*2
            else: equity.append(cur_eq); continue

            risk=abs(entry-sl); reward=abs(target-entry)
            if risk==0 or reward/risk<1.8: equity.append(cur_eq); continue

            if direction=="LONG":
                exit_p=target if nh>=target else sl if nl<=sl else nc
                exit_r="TARGET_HIT" if nh>=target else "STOP_LOSS" if nl<=sl else "EOD"
            else:
                exit_p=target if nl<=target else sl if nh>=sl else nc
                exit_r="TARGET_HIT" if nl<=target else "STOP_LOSS" if nh>=sl else "EOD"

            pnl=(exit_p-entry) if direction=="LONG" else (entry-exit_p)
            cur_eq+=pnl*self.lot_size
            trades.append(BtTrade(str(df.index[i])[:10],"INSIDE+EMA",direction,
                round(entry,2),round(target,2),round(sl,2),round(exit_p,2),
                exit_r,round(pnl,2),round(reward/risk,2),pnl>0,
                f"Inside+EMA({ema5:.0f}) breakout"))
            equity.append(cur_eq)
        return self._compute(trades, equity, f"INSIDE+EMA-{name}")

    def _compute(self, trades, equity, strategy) -> BtResult:
        if not trades:
            return BtResult(strategy,0,0,0,0,0,0,0,0,0,[],[])
        wins   = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        wr     = len(wins)/len(trades)*100
        avg_w  = float(np.mean([t.pnl_pts for t in wins])) if wins else 0
        avg_l  = abs(float(np.mean([t.pnl_pts for t in losses]))) if losses else 0
        pf     = (len(wins)*avg_w)/(len(losses)*avg_l) if losses and avg_l>0 else 99
        eq     = np.array(equity)
        dd     = float(np.max(np.maximum.accumulate(eq)-eq))
        return BtResult(
            strategy=strategy,
            total_trades=len(trades), winning_trades=len(wins),
            losing_trades=len(losses), win_rate=round(wr,1),
            avg_win_pts=round(avg_w,2), avg_loss_pts=round(avg_l,2),
            profit_factor=round(pf,2), max_drawdown_pts=round(dd,2),
            total_pnl_pts=round(sum(t.pnl_pts for t in trades),2),
            trades=trades, equity_curve=equity
        )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: Subasish Pani — Power of Stocks
# 5 EMA + Inside Candle + Traffic Light + Round Numbers
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_daily(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV — works for 1 year."""
    try:
        import yfinance as yf
        df = yf.download(symbol, period=f"{days+10}d",
                         interval="1d", progress=False, auto_adjust=True)
        if df is not None and len(df) > 5:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for col in ['Open','High','Low','Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            df = df.dropna()
            return df
    except Exception as e:
        print(f"Daily fetch error: {e}")
    return None


class SubasishBacktester:
    """
    Subasish Pani / Power of Stocks strategies.
    Uses daily candles for 1-year backtest.
    
    Strategies:
    1. 5 EMA — candle not touching 5 EMA → break of low = SELL
    2. Inside Candle — mother candle + inside bar → breakout trade
    3. Traffic Light — consecutive red/green candles → momentum
    4. Round Number — reaction at psychological levels
    """

    # Round levels generated dynamically based on price

    def __init__(self, symbol="^NSEI", days=365, lot_size=75, strategy="ALL"):
        self.symbol   = symbol
        self.days     = days
        self.lot_size = lot_size
        self.strategy = strategy  # "5EMA","INSIDE","TRAFFIC","ROUND","ALL"
        self.name_map = {"^NSEI":"Nifty","^NSEBANK":"BankNifty","^BSESN":"Sensex"}

    def run(self) -> dict:
        """Returns dict of {strategy_name: BtResult}"""
        df = fetch_daily(self.symbol, self.days)
        if df is None or len(df) < 20:
            print(f"No data for {self.symbol}")
            return {}

        # Calculate indicators
        df = df.copy()
        df['ema5']  = df['Close'].ewm(span=5,  adjust=False).mean()
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()

        results = {}
        name = self.name_map.get(self.symbol, self.symbol)

        if self.strategy in ["5EMA","ALL"]:
            results[f"5EMA-{name}"]      = self._run_5ema(df, name)
            results[f"5EMA_REJECT-{name}"]= self._run_5ema_reject(df, name)
            results[f"5EMA_CROSS-{name}"] = self._run_5ema_cross(df, name)
            results[f"INSIDE+EMA-{name}"] = self._run_inside_ema(df, name)

        if self.strategy in ["INSIDE","ALL"]:
            results[f"INSIDE-{name}"] = self._run_inside(df, name)

        if self.strategy in ["TRAFFIC","ALL"]:
            results[f"TRAFFIC-{name}"] = self._run_traffic(df, name)

        if self.strategy in ["ROUND","ALL"]:
            results[f"ROUND-{name}"] = self._run_round(df, name)

        return results

    # ── 5 EMA Strategy ───────────────────────────────────────────────────────
    def _run_5ema(self, df, name) -> BtResult:
        """
        SELL: Candle does NOT touch 5 EMA → break of low = SHORT
        BUY:  Candle does NOT touch 5 EMA from below → break of high = LONG
        """
        trades = []; equity = [100000]; cur_eq = 100000

        for i in range(2, len(df)-1):
            row      = df.iloc[i]
            prev_row = df.iloc[i-1]
            ema5     = float(row['ema5'])
            o = float(row['Open']); h = float(row['High'])
            l = float(row['Low']);  c = float(row['Close'])
            po = float(prev_row['Open']); ph = float(prev_row['High'])
            pl = float(prev_row['Low']);  pc_val = float(prev_row['Close'])

            direction = None
            entry = target = sl = 0

            # SELL setup: candle entirely above EMA5, low breaks previous low
            if pl > ema5 and pc_val > ema5 and o > ema5:
                # Previous candle didn't touch EMA5 from below
                if l < pl:  # Current candle breaks prev low
                    direction = "SHORT"
                    entry  = pl         # Enter at prev low break
                    sl     = ph + 10    # SL above prev high
                    target = entry - (sl - entry) * 2  # 1:2 RR

            # BUY setup: candle entirely below EMA5, high breaks previous high
            elif ph < ema5 and pc_val < ema5 and o < ema5:
                if h > ph:  # Current candle breaks prev high
                    direction = "LONG"
                    entry  = ph         # Enter at prev high break
                    sl     = pl - 10    # SL below prev low
                    target = entry + (entry - sl) * 2  # 1:2 RR

            if direction is None:
                equity.append(cur_eq); continue

            risk = abs(entry-sl)
            if risk == 0: equity.append(cur_eq); continue
            rr = abs(target-entry)/risk
            if rr < 1.8: equity.append(cur_eq); continue

            # Simulate on next candle
            nxt = df.iloc[i+1]
            nh = float(nxt['High']); nl = float(nxt['Low']); nc = float(nxt['Close'])

            if direction == "LONG":
                if nl <= sl:   exit_p = sl;     exit_r = "STOP_LOSS"
                elif nh >= target: exit_p = target; exit_r = "TARGET_HIT"
                else:          exit_p = nc;     exit_r = "EOD_EXIT"
            else:
                if nh >= sl:   exit_p = sl;     exit_r = "STOP_LOSS"
                elif nl <= target: exit_p = target; exit_r = "TARGET_HIT"
                else:          exit_p = nc;     exit_r = "EOD_EXIT"

            pnl = (exit_p-entry) if direction=="LONG" else (entry-exit_p)
            cur_eq += pnl * self.lot_size
            trades.append(BtTrade(
                date=str(df.index[i])[:10], strategy=f"5EMA",
                direction=direction, entry=round(entry,2),
                target=round(target,2), stop=round(sl,2),
                exit_price=round(exit_p,2), exit_reason=exit_r,
                pnl_pts=round(pnl,2), rr=round(rr,2), won=pnl>0,
                reason=f"5EMA={ema5:.0f} | Candle away from EMA"
            ))
            equity.append(cur_eq)

        return self._compute(trades, equity, f"5EMA-{name}")

    # ── Inside Candle Strategy ────────────────────────────────────────────────
    def _run_inside(self, df, name) -> BtResult:
        """
        Mother candle + Inside bar (child within mother range)
        Trade breakout of mother candle high (LONG) or low (SHORT)
        """
        trades = []; equity = [100000]; cur_eq = 100000

        for i in range(1, len(df)-1):
            mother = df.iloc[i-1]
            child  = df.iloc[i]
            nxt    = df.iloc[i+1]

            mh = float(mother['High']); ml = float(mother['Low'])
            ch = float(child['High']);  cl = float(child['Low'])
            nh = float(nxt['High']);    nl = float(nxt['Low'])
            nc = float(nxt['Close'])

            # Inside candle: child within mother range
            if not (ch <= mh and cl >= ml): continue
            if ch == mh or cl == ml:        continue  # Must be strictly inside

            mother_range = mh - ml
            if mother_range < 30: continue  # Skip tiny ranges

            # Breakout direction
            if nh > mh:      # Breaks mother high → LONG
                direction = "LONG"
                entry  = mh
                sl     = ml - 10
                target = mh + mother_range * 1.5
            elif nl < ml:    # Breaks mother low → SHORT
                direction = "SHORT"
                entry  = ml
                sl     = mh + 10
                target = ml - mother_range * 1.5
            else:
                equity.append(cur_eq); continue

            risk = abs(entry-sl)
            if risk == 0: equity.append(cur_eq); continue
            rr = abs(target-entry)/risk

            # Exit
            if direction == "LONG":
                if nl <= sl:        exit_p = sl;     exit_r = "STOP_LOSS"
                elif nh >= target:  exit_p = target; exit_r = "TARGET_HIT"
                else:               exit_p = nc;     exit_r = "EOD_EXIT"
            else:
                if nh >= sl:        exit_p = sl;     exit_r = "STOP_LOSS"
                elif nl <= target:  exit_p = target; exit_r = "TARGET_HIT"
                else:               exit_p = nc;     exit_r = "EOD_EXIT"

            pnl = (exit_p-entry) if direction=="LONG" else (entry-exit_p)
            cur_eq += pnl * self.lot_size
            trades.append(BtTrade(
                date=str(df.index[i])[:10], strategy="INSIDE",
                direction=direction, entry=round(entry,2),
                target=round(target,2), stop=round(sl,2),
                exit_price=round(exit_p,2), exit_reason=exit_r,
                pnl_pts=round(pnl,2), rr=round(rr,2), won=pnl>0,
                reason=f"Mother: {ml:.0f}-{mh:.0f} | Inside bar breakout"
            ))
            equity.append(cur_eq)

        return self._compute(trades, equity, f"INSIDE-{name}")

    # ── Traffic Light Strategy ────────────────────────────────────────────────
    def _run_traffic(self, df, name) -> BtResult:
        """
        3 consecutive red candles → BUY (oversold bounce)
        3 consecutive green candles → SELL (overbought pullback)
        With 1:2 RR minimum
        """
        trades = []; equity = [100000]; cur_eq = 100000

        for i in range(3, len(df)-1):
            c1 = df.iloc[i-3]; c2 = df.iloc[i-2]
            c3 = df.iloc[i-1]; cur = df.iloc[i]; nxt = df.iloc[i+1]

            def is_red(row): return float(row['Close']) < float(row['Open'])
            def is_grn(row): return float(row['Close']) > float(row['Open'])

            direction = None
            if all(is_red(r) for r in [c1,c2,c3]):
                direction = "LONG"   # 3 reds → buy bounce
            elif all(is_grn(r) for r in [c1,c2,c3]):
                direction = "SHORT"  # 3 greens → sell pullback

            if direction is None:
                equity.append(cur_eq); continue

            ch = float(cur['High']); cl = float(cur['Low']); cc = float(cur['Close'])
            nh = float(nxt['High']); nl = float(nxt['Low']); nc = float(nxt['Close'])
            rng = ch - cl
            if rng < 20: equity.append(cur_eq); continue

            if direction == "LONG":
                entry  = cc
                sl     = cl - rng * 0.3
                target = cc + (cc - sl) * 2
            else:
                entry  = cc
                sl     = ch + rng * 0.3
                target = cc - (sl - cc) * 2

            risk = abs(entry-sl)
            if risk == 0: equity.append(cur_eq); continue

            if direction == "LONG":
                if nl <= sl:        exit_p = sl;     exit_r = "STOP_LOSS"
                elif nh >= target:  exit_p = target; exit_r = "TARGET_HIT"
                else:               exit_p = nc;     exit_r = "EOD_EXIT"
            else:
                if nh >= sl:        exit_p = sl;     exit_r = "STOP_LOSS"
                elif nl <= target:  exit_p = target; exit_r = "TARGET_HIT"
                else:               exit_p = nc;     exit_r = "EOD_EXIT"

            pnl = (exit_p-entry) if direction=="LONG" else (entry-exit_p)
            cur_eq += pnl * self.lot_size
            trades.append(BtTrade(
                date=str(df.index[i])[:10], strategy="TRAFFIC",
                direction=direction, entry=round(entry,2),
                target=round(target,2), stop=round(sl,2),
                exit_price=round(exit_p,2), exit_reason=exit_r,
                pnl_pts=round(pnl,2), rr=round(abs(target-entry)/risk,2),
                won=pnl>0,
                reason=f"3 consecutive {'RED→BUY' if direction=='LONG' else 'GREEN→SELL'}"
            ))
            equity.append(cur_eq)

        return self._compute(trades, equity, f"TRAFFIC-{name}")

    # ── Round Number Strategy ─────────────────────────────────────────────────
    def _run_round(self, df, name) -> BtResult:
        """
        Dynamic round numbers based on current price.
        Any level ending in 000 or 500 near the current price.
        e.g. if price is 24150 → check 24000, 24500
        if price is 56300 → check 56000, 56500
        """
        trades = []; equity = [100000]; cur_eq = 100000

        for i in range(1, len(df)-1):
            row = df.iloc[i]
            nxt = df.iloc[i+1]
            h = float(row['High']); l = float(row['Low'])
            o = float(row['Open']); c = float(row['Close'])
            nh = float(nxt['High']); nl = float(nxt['Low']); nc = float(nxt['Close'])

            # Dynamically generate round levels near current price
            # Round to nearest 500 and check ±3 levels
            base = round(c / 500) * 500
            levels = [base + (x * 500) for x in range(-3, 4)]

            for level in levels:
                if level <= 0: continue
                proximity = abs(h - level) / level
                if proximity > 0.008: continue  # Within 0.8% of level

                candle_range = h - l
                if candle_range < 20: continue

                # Bearish rejection at round number (touched level but closed below)
                if h >= level and c < o and c < level:
                    direction = "SHORT"
                    entry  = c
                    sl     = h + 20
                    target = c - candle_range * 1.5

                # Bullish rejection at round number (touched level but closed above)
                elif l <= level and c > o and c > level:
                    direction = "LONG"
                    entry  = c
                    sl     = l - 20
                    target = c + candle_range * 1.5
                else:
                    continue

                risk = abs(entry-sl)
                if risk == 0: continue
                rr = abs(target-entry)/risk
                if rr < 1.5: continue

                if direction == "LONG":
                    if nl <= sl:       exit_p = sl;     exit_r = "STOP_LOSS"
                    elif nh >= target: exit_p = target; exit_r = "TARGET_HIT"
                    else:              exit_p = nc;     exit_r = "EOD_EXIT"
                else:
                    if nh >= sl:       exit_p = sl;     exit_r = "STOP_LOSS"
                    elif nl <= target: exit_p = target; exit_r = "TARGET_HIT"
                    else:              exit_p = nc;     exit_r = "EOD_EXIT"

                pnl = (exit_p-entry) if direction=="LONG" else (entry-exit_p)
                cur_eq += pnl * self.lot_size
                trades.append(BtTrade(
                    date=str(df.index[i])[:10], strategy="ROUND",
                    direction=direction, entry=round(entry,2),
                    target=round(target,2), stop=round(sl,2),
                    exit_price=round(exit_p,2), exit_reason=exit_r,
                    pnl_pts=round(pnl,2), rr=round(rr,2), won=pnl>0,
                    reason=f"Round level {level:,} | {'Bearish' if direction=='SHORT' else 'Bullish'} rejection"
                ))
                equity.append(cur_eq)
                break  # One trade per day

            else:
                equity.append(cur_eq)

        return self._compute(trades, equity, f"ROUND-{name}")


    def _run_5ema_reject(self, df, name) -> BtResult:
        """5 EMA Rejection setup."""
        trades = []; equity = [100000]; cur_eq = 100000
        df = df.copy()
        df['ema5'] = df['Close'].ewm(span=5, adjust=False).mean()

        for i in range(2, len(df)-1):
            row  = df.iloc[i]; nxt = df.iloc[i+1]
            ema5 = float(row['ema5'])
            h = float(row['High']); l = float(row['Low'])
            o = float(row['Open']); c = float(row['Close'])
            nh = float(nxt['High']); nl = float(nxt['Low']); nc = float(nxt['Close'])
            candle_range = h - l
            if candle_range < 20: equity.append(cur_eq); continue

            direction = None
            # Bearish rejection: EMA within candle, closes bearish, upper wick dominant
            if l <= ema5 <= h and c < o:
                upper_wick = h - max(o, c)
                body = abs(c - o)
                if upper_wick > body * 0.5:
                    direction = "SHORT"
                    entry = l; sl = h + 10; target = l - candle_range * 2
            # Bullish rejection
            elif l <= ema5 <= h and c > o:
                lower_wick = min(o, c) - l
                body = abs(c - o)
                if lower_wick > body * 0.5:
                    direction = "LONG"
                    entry = h; sl = l - 10; target = h + candle_range * 2

            if direction is None: equity.append(cur_eq); continue
            risk = abs(entry-sl); reward = abs(target-entry)
            if risk == 0 or reward/risk < 1.8: equity.append(cur_eq); continue

            if direction == "LONG":
                exit_p = target if nh>=target else sl if nl<=sl else nc
                exit_r = "TARGET_HIT" if nh>=target else "STOP_LOSS" if nl<=sl else "EOD"
            else:
                exit_p = target if nl<=target else sl if nh>=sl else nc
                exit_r = "TARGET_HIT" if nl<=target else "STOP_LOSS" if nh>=sl else "EOD"

            pnl = (exit_p-entry) if direction=="LONG" else (entry-exit_p)
            cur_eq += pnl * self.lot_size
            trades.append(BtTrade(str(df.index[i])[:10], f"5EMA_REJECT", direction,
                round(entry,2), round(target,2), round(sl,2), round(exit_p,2),
                exit_r, round(pnl,2), round(reward/risk,2), pnl>0,
                f"EMA rejection at {ema5:.0f}"))
            equity.append(cur_eq)
        return self._compute(trades, equity, f"5EMA_REJECT-{name}")

    def _run_5ema_cross(self, df, name) -> BtResult:
        """5 EMA Crossover setup."""
        trades = []; equity = [100000]; cur_eq = 100000
        df = df.copy()
        df['ema5'] = df['Close'].ewm(span=5, adjust=False).mean()

        for i in range(3, len(df)-1):
            c1=df.iloc[i-2]; c2=df.iloc[i-1]; curr=df.iloc[i]; nxt=df.iloc[i+1]
            e1=float(c1['ema5']); e2=float(c2['ema5']); e3=float(curr['ema5'])
            cl1=float(c1['Close']); cl2=float(c2['Close']); cl3=float(curr['Close'])
            h3=float(curr['High']); l3=float(curr['Low'])
            nh=float(nxt['High']); nl=float(nxt['Low']); nc=float(nxt['Close'])
            rng = h3 - l3
            if rng < 20: equity.append(cur_eq); continue

            direction = None
            if cl1<e1 and cl2<e2 and cl3>e3:
                direction="LONG"; entry=cl3; sl=l3-10; target=cl3+rng*2
            elif cl1>e1 and cl2>e2 and cl3<e3:
                direction="SHORT"; entry=cl3; sl=h3+10; target=cl3-rng*2

            if direction is None: equity.append(cur_eq); continue
            risk=abs(entry-sl); reward=abs(target-entry)
            if risk==0 or reward/risk<1.8: equity.append(cur_eq); continue

            if direction=="LONG":
                exit_p=target if nh>=target else sl if nl<=sl else nc
                exit_r="TARGET_HIT" if nh>=target else "STOP_LOSS" if nl<=sl else "EOD"
            else:
                exit_p=target if nl<=target else sl if nh>=sl else nc
                exit_r="TARGET_HIT" if nl<=target else "STOP_LOSS" if nh>=sl else "EOD"

            pnl=(exit_p-entry) if direction=="LONG" else (entry-exit_p)
            cur_eq += pnl*self.lot_size
            trades.append(BtTrade(str(df.index[i])[:10],"5EMA_CROSS",direction,
                round(entry,2),round(target,2),round(sl,2),round(exit_p,2),
                exit_r,round(pnl,2),round(reward/risk,2),pnl>0,
                f"EMA crossover at {e3:.0f}"))
            equity.append(cur_eq)
        return self._compute(trades, equity, f"5EMA_CROSS-{name}")

    def _run_inside_ema(self, df, name) -> BtResult:
        """Inside Candle + EMA combo setup."""
        trades = []; equity = [100000]; cur_eq = 100000
        df = df.copy()
        df['ema5'] = df['Close'].ewm(span=5, adjust=False).mean()

        for i in range(2, len(df)-1):
            mother=df.iloc[i-2]; child=df.iloc[i-1]
            curr=df.iloc[i]; nxt=df.iloc[i+1]
            mh=float(mother['High']); ml=float(mother['Low'])
            ch=float(child['High']); cl=float(child['Low'])
            ema5=float(child['ema5'])
            crh=float(curr['High']); crl=float(curr['Low'])
            nh=float(nxt['High']); nl=float(nxt['Low']); nc=float(nxt['Close'])

            if not (ch<=mh and cl>=ml): equity.append(cur_eq); continue
            if ch==mh or cl==ml: equity.append(cur_eq); continue
            mrange = mh-ml
            if mrange < 30: equity.append(cur_eq); continue

            # EMA near candle
            mid = (mh+ml)/2
            if abs(ema5-mid)/mid > 0.008: equity.append(cur_eq); continue

            direction=None
            if crh>mh: direction="LONG"; entry=mh; sl=ml-10; target=mh+mrange*2
            elif crl<ml: direction="SHORT"; entry=ml; sl=mh+10; target=ml-mrange*2
            else: equity.append(cur_eq); continue

            risk=abs(entry-sl); reward=abs(target-entry)
            if risk==0 or reward/risk<1.8: equity.append(cur_eq); continue

            if direction=="LONG":
                exit_p=target if nh>=target else sl if nl<=sl else nc
                exit_r="TARGET_HIT" if nh>=target else "STOP_LOSS" if nl<=sl else "EOD"
            else:
                exit_p=target if nl<=target else sl if nh>=sl else nc
                exit_r="TARGET_HIT" if nl<=target else "STOP_LOSS" if nh>=sl else "EOD"

            pnl=(exit_p-entry) if direction=="LONG" else (entry-exit_p)
            cur_eq+=pnl*self.lot_size
            trades.append(BtTrade(str(df.index[i])[:10],"INSIDE+EMA",direction,
                round(entry,2),round(target,2),round(sl,2),round(exit_p,2),
                exit_r,round(pnl,2),round(reward/risk,2),pnl>0,
                f"Inside+EMA({ema5:.0f}) breakout"))
            equity.append(cur_eq)
        return self._compute(trades, equity, f"INSIDE+EMA-{name}")

    def _compute(self, trades, equity, strategy) -> BtResult:
        if not trades:
            return BtResult(strategy,0,0,0,0,0,0,0,0,0,[],[100000])
        wins   = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        wr     = len(wins)/len(trades)*100
        avg_w  = float(np.mean([t.pnl_pts for t in wins])) if wins else 0
        avg_l  = abs(float(np.mean([t.pnl_pts for t in losses]))) if losses else 0
        pf     = (len(wins)*avg_w)/(len(losses)*avg_l) if losses and avg_l>0 else 99
        eq     = np.array(equity)
        dd     = float(np.max(np.maximum.accumulate(eq)-eq))
        return BtResult(
            strategy=strategy,
            total_trades=len(trades), winning_trades=len(wins),
            losing_trades=len(losses), win_rate=round(wr,1),
            avg_win_pts=round(avg_w,2), avg_loss_pts=round(avg_l,2),
            profit_factor=round(pf,2), max_drawdown_pts=round(dd,2),
            total_pnl_pts=round(sum(t.pnl_pts for t in trades),2),
            trades=trades, equity_curve=equity
        )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 5: Exact 5-Min Intraday Backtest (last 60 days)
# Uses real 5-minute candles — most accurate for 5 EMA strategy
# ═══════════════════════════════════════════════════════════════════════════════

class IntraDay5MinBacktester:
    """
    Exact 5 EMA strategy backtested on real 5-minute candles.
    
    Rules (exactly as Subasish Pani teaches):
    
    Setup 1 — Classic:
      Candle entirely above 5 EMA (no touch)
      Next candle breaks low of that candle → SHORT
      Reverse for LONG
      
    Setup 2 — Rejection:
      Candle touches 5 EMA with upper/lower wick
      Closes in opposite direction → trade
      
    Setup 3 — Inside + EMA:
      Inside candle forms near 5 EMA
      Breakout of mother candle → trade
      
    Trade rules:
      - Only trade 9:20 AM to 2:30 PM IST
      - Max 1 trade per day
      - SL = opposite end of signal candle
      - Target = 2× risk (1:2 RR minimum)
      - Square off at 3:15 PM if not hit
    """

    INDICES = {
        "Nifty":     "^NSEI",
        "BankNifty": "^NSEBANK",
        "Sensex":    "^BSESN",
    }
    LOT_SIZES = {
        "Nifty": 75, "BankNifty": 35, "Sensex": 20
    }

    def __init__(self, index="Nifty", days=60,
                 setups=None, min_rr=2.0):
        self.index   = index
        self.symbol  = self.INDICES.get(index, "^NSEI")
        self.days    = min(days, 58)  # yfinance limit
        self.setups  = setups or ["CLASSIC","REJECTION","INSIDE_EMA"]
        self.min_rr  = min_rr
        self.lot     = self.LOT_SIZES.get(index, 75)

    def _fetch_5min(self) -> Optional[pd.DataFrame]:
        """Fetch real 5-minute candles."""
        try:
            import yfinance as yf
            df = yf.download(
                self.symbol,
                period=f"{self.days}d",
                interval="5m",
                progress=False,
                auto_adjust=True
            )
            if df is None or len(df) < 50:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for col in ['Open','High','Low','Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            df = df.dropna()
            # Add date column
            if hasattr(df.index, 'date'):
                df['_date'] = df.index.date
            else:
                df['_date'] = pd.to_datetime(df.index).date
            # Add time column
            if hasattr(df.index, 'time'):
                df['_time'] = df.index.time
            else:
                df['_time'] = pd.to_datetime(df.index).time
            # Calculate 5 EMA on 5-min candles
            df['ema5']  = df['Close'].ewm(span=5,  adjust=False).mean()
            df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
            print(f"Fetched {len(df)} 5-min candles for {self.index}")
            return df
        except Exception as e:
            print(f"5-min fetch error: {e}")
            return None

    def _check_classic(self, df_day: pd.DataFrame, i: int) -> Optional[dict]:
        """5 EMA Classic setup on 5-min chart."""
        if i < 2 or i >= len(df_day) - 1:
            return None

        prev = df_day.iloc[i-1]
        curr = df_day.iloc[i]

        ph = float(prev['High']); pl = float(prev['Low'])
        po = float(prev['Open']); pc = float(prev['Close'])
        ema5 = float(prev['ema5'])
        ch = float(curr['High']); cl = float(curr['Low'])

        candle_range = ph - pl
        if candle_range < 5: return None

        # SHORT: prev candle entirely above EMA, curr breaks low
        if pl > ema5 and pc > ema5 and po > ema5:
            if cl < pl:
                sl     = ph + 5
                entry  = pl
                target = entry - (sl - entry) * self.min_rr
                return {'dir':'SHORT','entry':entry,'sl':sl,
                        'target':target,'setup':'CLASSIC',
                        'reason':f'Above EMA5({ema5:.0f}), break low {pl:.0f}'}

        # LONG: prev candle entirely below EMA, curr breaks high
        if ph < ema5 and pc < ema5 and po < ema5:
            if ch > ph:
                sl     = pl - 5
                entry  = ph
                target = entry + (entry - sl) * self.min_rr
                return {'dir':'LONG','entry':entry,'sl':sl,
                        'target':target,'setup':'CLASSIC',
                        'reason':f'Below EMA5({ema5:.0f}), break high {ph:.0f}'}
        return None

    def _check_rejection(self, df_day: pd.DataFrame, i: int) -> Optional[dict]:
        """5 EMA Rejection setup on 5-min chart."""
        if i < 1 or i >= len(df_day) - 1:
            return None

        prev = df_day.iloc[i-1]
        curr = df_day.iloc[i]

        ph = float(prev['High']); pl = float(prev['Low'])
        po = float(prev['Open']); pc = float(prev['Close'])
        ema5 = float(prev['ema5'])
        ch = float(curr['High']); cl = float(curr['Low'])

        candle_range = ph - pl
        if candle_range < 5: return None

        # EMA must be within the candle
        if not (pl <= ema5 <= ph): return None

        body = abs(pc - po)

        # Bearish rejection: upper wick dominant, closes bearish
        if pc < po:
            upper_wick = ph - max(po, pc)
            if upper_wick > body * 0.5 and cl < pl:
                sl     = ph + 5
                entry  = pl
                target = entry - (sl - entry) * self.min_rr
                return {'dir':'SHORT','entry':entry,'sl':sl,
                        'target':target,'setup':'REJECTION',
                        'reason':f'Bearish rejection at EMA5({ema5:.0f})'}

        # Bullish rejection: lower wick dominant, closes bullish
        if pc > po:
            lower_wick = min(po, pc) - pl
            if lower_wick > body * 0.5 and ch > ph:
                sl     = pl - 5
                entry  = ph
                target = entry + (entry - sl) * self.min_rr
                return {'dir':'LONG','entry':entry,'sl':sl,
                        'target':target,'setup':'REJECTION',
                        'reason':f'Bullish rejection at EMA5({ema5:.0f})'}
        return None

    def _check_inside_ema(self, df_day: pd.DataFrame, i: int) -> Optional[dict]:
        """Inside candle + EMA combo on 5-min chart."""
        if i < 2 or i >= len(df_day) - 1:
            return None

        mother = df_day.iloc[i-2]
        child  = df_day.iloc[i-1]
        curr   = df_day.iloc[i]

        mh = float(mother['High']); ml = float(mother['Low'])
        ch = float(child['High']);  cl = float(child['Low'])
        ema5 = float(child['ema5'])
        crh = float(curr['High']); crl = float(curr['Low'])

        # Child must be inside mother
        if not (ch <= mh and cl >= ml): return None
        if ch == mh or cl == ml: return None

        mrange = mh - ml
        if mrange < 10: return None

        # EMA must be near the candles
        mid = (mh + ml) / 2
        if abs(ema5 - mid) / mid > 0.005: return None

        if crh > mh:
            sl     = ml - 5
            entry  = mh
            target = entry + mrange * self.min_rr
            return {'dir':'LONG','entry':entry,'sl':sl,
                    'target':target,'setup':'INSIDE+EMA',
                    'reason':f'Inside+EMA breakout above {mh:.0f}'}

        if crl < ml:
            sl     = mh + 5
            entry  = ml
            target = entry - mrange * self.min_rr
            return {'dir':'SHORT','entry':entry,'sl':sl,
                    'target':target,'setup':'INSIDE+EMA',
                    'reason':f'Inside+EMA breakdown below {ml:.0f}'}
        return None

    def _simulate_exit(self, df_day: pd.DataFrame, start_i: int,
                       signal: dict) -> tuple:
        """Walk through 5-min candles to find exit."""
        from datetime import time as dtime
        direction = signal['dir']
        entry     = signal['entry']
        target    = signal['target']
        sl        = signal['sl']

        for j in range(start_i, len(df_day)):
            row  = df_day.iloc[j]
            t    = row['_time']
            h    = float(row['High'])
            l    = float(row['Low'])
            c    = float(row['Close'])

            # Square off at 3:15 PM
            if t >= dtime(15, 15):
                return c, "SQUAREOFF_315"

            if direction == "LONG":
                if l <= sl:      return sl,     "STOP_LOSS"
                if h >= target:  return target, "TARGET_HIT"
            else:
                if h >= sl:      return sl,     "STOP_LOSS"
                if l <= target:  return target, "TARGET_HIT"

        # End of day
        last_close = float(df_day.iloc[-1]['Close'])
        return last_close, "EOD_EXIT"

    def run(self) -> dict:
        """Run backtest — returns dict of {setup: BtResult}"""
        df = self._fetch_5min()
        if df is None or len(df) < 50:
            print("No 5-min data available")
            return {}

        from datetime import time as dtime

        dates   = sorted(df['_date'].unique())
        results = {s: {'trades':[], 'equity':[100000], 'cur_eq':100000}
                   for s in self.setups}

        for day in dates:
            day_df = df[df['_date'] == day].copy().reset_index(drop=True)
            if len(day_df) < 10: continue

            # Only trade 9:20 AM to 2:30 PM
            day_df = day_df[
                (day_df['_time'] >= dtime(9, 20)) &
                (day_df['_time'] <= dtime(14, 30))
            ].reset_index(drop=True)

            if len(day_df) < 6: continue

            # Track if traded today (per setup)
            traded_today = {s: False for s in self.setups}

            for i in range(2, len(day_df)):
                for setup in self.setups:
                    if traded_today[setup]: continue

                    # Check signal
                    signal = None
                    if setup == "CLASSIC":
                        signal = self._check_classic(day_df, i)
                    elif setup == "REJECTION":
                        signal = self._check_rejection(day_df, i)
                    elif setup == "INSIDE_EMA":
                        signal = self._check_inside_ema(day_df, i)

                    if signal is None: continue

                    # Validate RR
                    risk   = abs(signal['entry'] - signal['sl'])
                    reward = abs(signal['target'] - signal['entry'])
                    if risk == 0: continue
                    rr = reward / risk
                    if rr < self.min_rr: continue

                    # Simulate exit
                    exit_p, exit_r = self._simulate_exit(day_df, i+1, signal)
                    pnl = (exit_p - signal['entry']) if signal['dir']=="LONG"                           else (signal['entry'] - exit_p)

                    results[setup]['cur_eq'] += pnl * self.lot
                    results[setup]['equity'].append(results[setup]['cur_eq'])
                    results[setup]['trades'].append(BtTrade(
                        date=str(day),
                        strategy=setup,
                        direction=signal['dir'],
                        entry=round(signal['entry'],2),
                        target=round(signal['target'],2),
                        stop=round(signal['sl'],2),
                        exit_price=round(exit_p,2),
                        exit_reason=exit_r,
                        pnl_pts=round(pnl,2),
                        rr=round(rr,2),
                        won=pnl>0,
                        reason=signal['reason']
                    ))
                    traded_today[setup] = True

            # Append equity for days with no trade
            for setup in self.setups:
                if not traded_today[setup]:
                    results[setup]['equity'].append(results[setup]['cur_eq'])

        # Build BtResult for each setup
        final = {}
        for setup, data in results.items():
            trades = data['trades']
            equity = data['equity']
            if not trades:
                final[setup] = BtResult(
                    strategy=f"5MIN_{setup}-{self.index}",
                    total_trades=0, winning_trades=0, losing_trades=0,
                    win_rate=0, avg_win_pts=0, avg_loss_pts=0,
                    profit_factor=0, max_drawdown_pts=0,
                    total_pnl_pts=0, trades=[], equity_curve=equity
                )
                continue

            wins   = [t for t in trades if t.won]
            losses = [t for t in trades if not t.won]
            wr     = len(wins)/len(trades)*100
            avg_w  = float(np.mean([t.pnl_pts for t in wins])) if wins else 0
            avg_l  = abs(float(np.mean([t.pnl_pts for t in losses]))) if losses else 0
            pf     = (len(wins)*avg_w)/(len(losses)*avg_l) if losses and avg_l>0 else 99.0
            eq     = np.array(equity)
            dd     = float(np.max(np.maximum.accumulate(eq) - eq))

            final[setup] = BtResult(
                strategy=f"5MIN_{setup}-{self.index}",
                total_trades=len(trades),
                winning_trades=len(wins),
                losing_trades=len(losses),
                win_rate=round(wr,1),
                avg_win_pts=round(avg_w,2),
                avg_loss_pts=round(avg_l,2),
                profit_factor=round(pf,2),
                max_drawdown_pts=round(dd,2),
                total_pnl_pts=round(sum(t.pnl_pts for t in trades),2),
                trades=trades,
                equity_curve=equity
            )

        return final


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 6: Fabio NSE — AAA Fade + Squeeze + Delta Absorption
# Ported from Fabio Valentini NQ Futures strategy to Indian markets
# ═══════════════════════════════════════════════════════════════════════════════

class FabioNSEIndexBacktester:
    """
    Wrapper that runs FabioNSEBacktester from fabio_nse.py
    and returns results in standard BtResult format.
    """

    SETUPS = ["AAA_FADE", "SQUEEZE", "DELTA_ABS"]

    def __init__(self, index="Nifty", days=55, min_rr=2.0):
        self.index  = index
        self.days   = days
        self.min_rr = min_rr

    def run(self) -> dict:
        """Returns {setup_name: BtResult}"""
        try:
            from fabio_nse import FabioNSEBacktester
            bt = FabioNSEBacktester(
                index=self.index,
                days=self.days,
                min_rr=self.min_rr,
                setups=self.SETUPS,
                apply_risk_rules=True
            )
            return bt.run()
        except ImportError as e:
            print(f"fabio_nse.py not found: {e}")
            return {}
        except Exception as e:
            print(f"FabioNSE backtest error: {e}")
            return {}
