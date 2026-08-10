"""
sl_hunter.py - Stop Loss Hunter Strategy
Based on Nageswar Rao's Intraday Hunter methodology
Detects liquidity grabs at PDH/PDL and trades the reversal
"""
import numpy as np
import pandas as pd
from datetime import datetime, date, time as dtime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

IST = timezone(timedelta(hours=5, minutes=30))

def get_ist(): return datetime.now(IST)

# ─── Data structures ──────────────────────────────────────────────────────────
@dataclass
class SLHuntTrade:
    date:        str
    instrument:  str
    direction:   str       # LONG/SHORT
    hunt_level:  float     # PDH or PDL that was hunted
    hunt_type:   str       # PDH_HUNT or PDL_HUNT
    entry:       float
    stop:        float     # Tip of wick
    target:      float     # Opposite level
    exit_price:  float = 0.0
    exit_reason: str = ""
    pnl_pts:     float = 0.0
    rr:          float = 0.0
    won:         bool = False
    reason:      str = ""
    # Options fields
    option_type:   str = ""
    strike:        int = 0
    entry_premium: float = 0.0
    exit_premium:  float = 0.0
    pnl_options:   float = 0.0

    def to_dict(self): return self.__dict__

@dataclass 
class SLHuntResult:
    instrument:      str
    total_trades:    int
    winning_trades:  int
    losing_trades:   int
    win_rate:        float
    avg_win_pts:     float
    avg_loss_pts:    float
    profit_factor:   float
    max_drawdown_pts: float
    total_pnl_pts:   float
    total_pnl_options: float = 0.0
    trades:          list = field(default_factory=list)
    equity_curve:    list = field(default_factory=list)
    equity_options:  list = field(default_factory=list)


# ─── Signal Detector ─────────────────────────────────────────────────────────
class SLHuntDetector:
    """
    Detects SL hunting patterns at PDH/PDL.
    Uses 1-min or 5-min candles.
    """

    @staticmethod
    def is_hammer(candle, direction="BULL") -> bool:
        """
        Bull Hammer: Long lower wick, closes near high
        Bear Shooting Star: Long upper wick, closes near low
        Rule: Wick > 2x body
        """
        h  = float(candle['High'])
        l  = float(candle['Low'])
        o  = float(candle['Open'])
        c  = float(candle['Close'])
        body = abs(c - o)
        if body == 0: body = 0.01

        if direction == "BULL":
            lower_wick = min(o, c) - l
            return lower_wick > 2 * body and c > o  # Bullish close
        else:
            upper_wick = h - max(o, c)
            return upper_wick > 2 * body and c < o  # Bearish close

    @staticmethod
    def check_pdl_hunt(df_slice: pd.DataFrame, pdl: float,
                        vix: float = 14.0) -> Optional[dict]:
        """
        PDL Hunt (Buy Setup):
        1. Price wicks BELOW PDL (0.05-0.1% buffer)
        2. Candle closes ABOVE PDL (rejection)
        3. Bullish hammer/pin bar
        4. Next candle breaks rejection candle high = ENTRY
        """
        if len(df_slice) < 2: return None

        hunt_candle = df_slice.iloc[-2]
        curr_candle = df_slice.iloc[-1]

        hc_low   = float(hunt_candle['Low'])
        hc_high  = float(hunt_candle['High'])
        hc_close = float(hunt_candle['Close'])
        hc_open  = float(hunt_candle['Open'])
        cc_high  = float(curr_candle['High'])

        # Buffer based on VIX
        buffer_pct = 0.002 if vix <= 15 else 0.003 if vix <= 20 else 0.005
        buffer     = pdl * buffer_pct

        # 1. Wick broke below PDL (within buffer)
        if hc_low >= pdl: return None  # Must breach PDL

        # 2. Candle closed ABOVE PDL (rejection)
        if hc_close <= pdl: return None

        # 3. Bullish hammer check
        lower_wick = min(hc_open, hc_close) - hc_low
        body       = abs(hc_close - hc_open)
        if body == 0: body = 0.1
        if lower_wick < 1.5 * body: return None

        # Entry = hunt candle high (with or without confirmation)
        entry  = hc_high
        sl     = hc_low - 5          # Tip of wick + 5pt buffer
        # Target = PDH (opposite liquidity zone)
        risk   = entry - sl
        target = entry + risk * 2.0  # Min 1:2.0

        return {
            'direction':  'LONG',
            'hunt_type':  'PDL_HUNT',
            'hunt_level': pdl,
            'entry':      round(entry, 2),
            'stop':       round(sl, 2),
            'target':     round(target, 2),
            'wick_low':   round(hc_low, 2),
            'reason':     f"PDL Hunt: Wick to {hc_low:.0f} below PDL {pdl:.0f} | Rejection close {hc_close:.0f}"
        }

    @staticmethod
    def check_pdh_hunt(df_slice: pd.DataFrame, pdh: float,
                        vix: float = 14.0) -> Optional[dict]:
        """
        PDH Hunt (Sell Setup) with proper filters.
        """
        if len(df_slice) < 3: return None

        hunt = df_slice.iloc[-2]
        conf = df_slice.iloc[-1]

        h = float(hunt['High']); l = float(hunt['Low'])
        o = float(hunt['Open']); c = float(hunt['Close'])
        ch = float(conf['High']); cl = float(conf['Low'])
        cc = float(conf['Close'])

        # Volume filter
        if 'Volume' in df_slice.columns:
            avg_vol  = float(df_slice.iloc[:-2]['Volume'].mean()) if len(df_slice) > 2 else 0
            hunt_vol = float(hunt['Volume'])
            if avg_vol > 0 and hunt_vol < avg_vol * 0.8:
                return None

        # 1. Hunt candle wicked above PDH
        if h <= pdh: return None

        # 2. Hunt candle closed BELOW PDH
        if c >= pdh: return None

        # 3. Shooting star: upper wick > 1.5x body
        upper_wick = h - max(o, c)
        body       = abs(c - o)
        if body < 0.5: body = 0.5
        if upper_wick < 1.5 * body: return None

        # 4. Confirmation: next candle is bearish
        if cc >= c: return None

        # 5. Hunt depth filter
        hunt_depth = (h - pdh) / pdh
        if hunt_depth > 0.015: return None

        entry  = cc
        sl     = h + 5
        risk   = sl - entry
        if risk <= 0: return None
        target = entry - risk * 2.0

        rr = (entry - target) / risk
        if rr < 1.8: return None

        return {
            'direction':  'SHORT',
            'hunt_type':  'PDH_HUNT',
            'hunt_level': pdh,
            'entry':      round(entry, 2),
            'stop':       round(sl, 2),
            'target':     round(target, 2),
            'wick_high':  round(h, 2),
            'reason':     f"PDH Hunt ↑{h:.0f} PDH:{pdh:.0f} | ShootingStar+Confirm | Depth:{hunt_depth*100:.2f}%"
        }

    @staticmethod
    def check_opening_range_hunt(df_day: pd.DataFrame,
                                  vix: float = 14.0) -> Optional[dict]:
        """
        Opening Range Hunt (9:15-9:30 AM):
        Mark high/low of first 15 min (3 × 5-min candles)
        Hunt above OR high = entry
        """
        if len(df_day) < 5: return None

        # First 8 candles = opening range (9:15-9:30 on 2-min chart)
        or_candles = df_day.iloc[:8]
        or_high    = float(or_candles['High'].max())
        or_low     = float(or_candles['Low'].min())
        or_range   = or_high - or_low

        if or_range < 20: return None  # Skip tiny ranges

        # Check 4th and 5th candles for hunt
        if len(df_day) < 5: return None
        hunt = df_day.iloc[3]
        conf = df_day.iloc[4]

        # PDL Hunt on OR low
        signal = SLHuntDetector.check_pdl_hunt(
            df_day.iloc[2:5], or_low, vix)
        if signal:
            signal['hunt_type'] = 'OR_LOW_HUNT'
            signal['reason']    = f"Opening Range Low Hunt: OR={or_low:.0f}-{or_high:.0f}"
            return signal

        # PDH Hunt on OR high
        signal = SLHuntDetector.check_pdh_hunt(
            df_day.iloc[2:5], or_high, vix)
        if signal:
            signal['hunt_type'] = 'OR_HIGH_HUNT'
            signal['reason']    = f"Opening Range High Hunt: OR={or_low:.0f}-{or_high:.0f}"
            return signal

        return None


# ─── Option Premium Estimator ─────────────────────────────────────────────────
class SLHuntOptionPricer:
    LOT_SIZES = {"NIFTY": 75, "BANKNIFTY": 35, "SENSEX": 20}

    @staticmethod
    def get_atm_strike(spot: float, instrument: str) -> int:
        step = 50 if instrument == "NIFTY" else 100
        return round(spot / step) * step

    @staticmethod
    def estimate_premium(spot: float, instrument: str,
                          option_type: str, dte: int = 3,
                          vix: float = 14.0) -> float:
        import math
        iv     = vix / 100
        T      = max(dte, 1) / 365
        strike = SLHuntOptionPricer.get_atm_strike(spot, instrument)
        prem   = spot * iv * math.sqrt(T) * 0.4
        if option_type == "CE" and spot > strike:
            prem += (spot - strike)
        elif option_type == "PE" and spot < strike:
            prem += (strike - spot)
        return round(max(prem, 5.0), 2)


# ─── Backtester ───────────────────────────────────────────────────────────────
class SLHuntBacktester:
    """
    Backtests SL Hunt strategy on real 5-min data.
    Tests on Nifty, BankNifty, Sensex.

    Rules:
    - Identify PDH/PDL from previous day
    - Look for hunt + rejection in first 90 min (9:15-10:45 AM)
    - Enter on confirmation candle
    - SL = tip of hunt wick
    - Target = opposite level (min 1:2.5 RR)
    - Also calculates options PnL (buy CE/PE)
    """

    SYMBOLS = {
        "Nifty":     "^NSEI",
        "BankNifty": "^NSEBANK",
        "Sensex":    "^BSESN",
    }
    LOT_SIZES = {"Nifty": 75, "BankNifty": 35, "Sensex": 20}

    def __init__(self, days: int = 45, vix: float = 14.0,
                 options: bool = True):
        self.days    = min(days, 55)
        self.vix     = vix
        self.options = options

    def _fetch(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            df = yf.download(symbol, period=f"{self.days+5}d",
                             interval="2m", progress=False, auto_adjust=True)
            if df is None or len(df) < 50: return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for col in ['Open','High','Low','Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            df = df.dropna()
            df['_date'] = pd.to_datetime(df.index).date
            df['_time'] = pd.to_datetime(df.index).time
            return df
        except Exception as e:
            print(f"Fetch error: {e}")
            return None

    def run_all(self) -> dict:
        """Run backtest on all 3 indices."""
        results = {}
        for name, symbol in self.SYMBOLS.items():
            print(f"Running SL Hunt on {name}...")
            df = self._fetch(symbol)
            if df is None:
                print(f"  No data for {name}")
                continue
            results[name] = self._backtest(df, name)
            t = results[name]
            print(f"  {name}: {t.total_trades} trades | {t.win_rate:.1f}% WR | {t.total_pnl_pts:+.0f} pts")
        return results

    def _backtest(self, df: pd.DataFrame, name: str) -> SLHuntResult:
        dates    = sorted(df['_date'].unique())
        lot      = self.LOT_SIZES.get(name, 75)
        trades   = []
        equity   = [100000]; cur_eq   = 100000
        eq_opt   = [100000]; cur_opt  = 100000

        for i in range(1, len(dates)):
            today = dates[i]; prev = dates[i-1]
            today_df = df[df['_date'] == today].copy().reset_index(drop=True)
            prev_df  = df[df['_date'] == prev].copy()

            if len(today_df) < 6 or len(prev_df) < 3: continue

            # Previous day levels
            pdh = float(prev_df['High'].max())
            pdl = float(prev_df['Low'].min())

            # Only trade first 90 minutes: 9:15-10:45 AM
            hunt_window = today_df[
                (today_df['_time'] >= dtime(9, 15)) &
                (today_df['_time'] <= dtime(10, 45))
            ].reset_index(drop=True)

            if len(hunt_window) < 8: continue

            # Check for signals across entire trading day (not just 90 min)
            signal = None

            # Also check previous week high/low for hunt zones
            prev_week_high = float(prev_df['High'].tail(10).max()) if len(prev_df) >= 10 else pdh
            prev_week_low  = float(prev_df['Low'].tail(10).min())  if len(prev_df) >= 10 else pdl

            # Round number levels near PDH/PDL
            round_level_h = round(pdh / 500) * 500
            round_level_l = round(pdl / 500) * 500

            hunt_zones_h = list(set([pdh, prev_week_high, round_level_h]))
            hunt_zones_l = list(set([pdl, prev_week_low,  round_level_l]))

            # Scan entire day (not just 90 min) for better signals
            full_window = today_df[
                (today_df['_time'] >= dtime(9, 20)) &
                (today_df['_time'] <= dtime(14, 30))
            ].reset_index(drop=True)

            if len(full_window) < 5:
                equity.append(cur_eq)
                eq_opt.append(cur_opt)
                continue

            # Check all hunt zones
            for j in range(2, min(len(full_window), 100)):
                slice_df = full_window.iloc[max(0,j-2):j+1]

                # Check PDL and nearby levels
                for level in hunt_zones_l:
                    sig = SLHuntDetector.check_pdl_hunt(slice_df, level, self.vix)
                    if sig:
                        signal = sig
                        break

                if signal: break

                # Check PDH and nearby levels
                for level in hunt_zones_h:
                    sig = SLHuntDetector.check_pdh_hunt(slice_df, level, self.vix)
                    if sig:
                        signal = sig
                        break

                if signal: break

            if signal is None:
                equity.append(cur_eq)
                eq_opt.append(cur_opt)
                continue

            # Validate RR
            risk   = abs(signal['entry'] - signal['stop'])
            reward = abs(signal['target'] - signal['entry'])
            if risk == 0: equity.append(cur_eq); eq_opt.append(cur_opt); continue
            rr = reward / risk
            if rr < 1.8: equity.append(cur_eq); eq_opt.append(cur_opt); continue

            # Simulate trade exit on remaining candles
            remaining = today_df[
                today_df['_time'] > dtime(10, 45)
            ].reset_index(drop=True)

            exit_p = float(today_df['Close'].iloc[-1])
            exit_r = "EOD_EXIT"

            for j in range(len(remaining)):
                row = remaining.iloc[j]
                h   = float(row['High'])
                l   = float(row['Low'])
                t   = row['_time']

                # Square off at 3:15 PM
                if t >= dtime(15, 15):
                    exit_p = float(row['Close'])
                    exit_r = "SQUAREOFF_315"
                    break

                if signal['direction'] == "LONG":
                    if l <= signal['stop']:
                        exit_p = signal['stop']; exit_r = "STOP_LOSS"; break
                    if h >= signal['target']:
                        exit_p = signal['target']; exit_r = "TARGET_HIT"; break
                else:
                    if h >= signal['stop']:
                        exit_p = signal['stop']; exit_r = "STOP_LOSS"; break
                    if l <= signal['target']:
                        exit_p = signal['target']; exit_r = "TARGET_HIT"; break

            pnl = (exit_p - signal['entry']) if signal['direction']=="LONG" \
                  else (signal['entry'] - exit_p)
            won = pnl > 0
            cur_eq += pnl * lot

            # Options PnL
            opt_pnl = 0.0
            opt_entry = 0.0; opt_exit = 0.0
            opt_type  = "CE" if signal['direction']=="LONG" else "PE"
            strike    = SLHuntOptionPricer.get_atm_strike(signal['entry'], name.replace("Nifty","NIFTY").replace("BankNifty","BANKNIFTY").replace("Sensex","SENSEX"))
            dte       = 3

            if self.options:
                opt_entry = SLHuntOptionPricer.estimate_premium(
                    signal['entry'], name.upper().replace("BANKNIFTY","BANKNIFTY"), opt_type, dte, self.vix)
                # Estimate exit premium based on spot move
                exit_spot_move = pnl if signal['direction']=="LONG" else -pnl
                opt_exit = max(opt_entry + exit_spot_move * 0.5, 0)
                opt_pnl  = (opt_exit - opt_entry) * lot
                cur_opt += opt_pnl

            trades.append(SLHuntTrade(
                date=str(today),
                instrument=name,
                direction=signal['direction'],
                hunt_level=signal['hunt_level'],
                hunt_type=signal['hunt_type'],
                entry=round(signal['entry'],2),
                stop=round(signal['stop'],2),
                target=round(signal['target'],2),
                exit_price=round(exit_p,2),
                exit_reason=exit_r,
                pnl_pts=round(pnl,2),
                rr=round(rr,2),
                won=won,
                reason=signal['reason'],
                option_type=opt_type,
                strike=strike,
                entry_premium=round(opt_entry,2),
                exit_premium=round(opt_exit,2),
                pnl_options=round(opt_pnl,2)
            ))
            equity.append(cur_eq)
            eq_opt.append(cur_opt)

        # Compute stats
        if not trades:
            return SLHuntResult(
                instrument=name, total_trades=0, winning_trades=0,
                losing_trades=0, win_rate=0, avg_win_pts=0,
                avg_loss_pts=0, profit_factor=0, max_drawdown_pts=0,
                total_pnl_pts=0, trades=[], equity_curve=equity,
                equity_options=eq_opt
            )

        wins   = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        wr     = len(wins)/len(trades)*100
        avg_w  = float(np.mean([t.pnl_pts for t in wins])) if wins else 0
        avg_l  = abs(float(np.mean([t.pnl_pts for t in losses]))) if losses else 0
        pf     = (len(wins)*avg_w)/(len(losses)*avg_l) if losses and avg_l>0 else 99.0
        eq_arr = np.array(equity)
        dd     = float(np.max(np.maximum.accumulate(eq_arr)-eq_arr))

        return SLHuntResult(
            instrument=name,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=round(wr,1),
            avg_win_pts=round(avg_w,2),
            avg_loss_pts=round(avg_l,2),
            profit_factor=round(pf,2),
            max_drawdown_pts=round(dd,2),
            total_pnl_pts=round(sum(t.pnl_pts for t in trades),2),
            total_pnl_options=round(sum(t.pnl_options for t in trades),2),
            trades=trades,
            equity_curve=equity,
            equity_options=eq_opt
        )


# ─── Live Signal Scanner ──────────────────────────────────────────────────────
class SLHuntScanner:
    """Live signal scanner for Hunter Alerts tab."""

    def __init__(self, vix: float = 14.0):
        self.vix = vix

    def scan_all(self) -> dict:
        """Scan all 3 indices for live SL hunt signals."""
        symbols = {
            "Nifty":     "^NSEI",
            "BankNifty": "^NSEBANK",
            "Sensex":    "^BSESN",
        }
        signals = {}
        for name, sym in symbols.items():
            try:
                import yfinance as yf
                df = yf.download(sym, period="5d", interval="2m",
                                  progress=False, auto_adjust=True)
                if df is None or len(df) < 10: continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df['_date'] = pd.to_datetime(df.index).date
                df['_time'] = pd.to_datetime(df.index).time

                dates = sorted(df['_date'].unique())
                if len(dates) < 2: continue

                today = dates[-1]; prev = dates[-2]
                today_df = df[df['_date']==today].reset_index(drop=True)
                prev_df  = df[df['_date']==prev]

                pdh = float(prev_df['High'].max())
                pdl = float(prev_df['Low'].min())

                # Check last few candles for signal
                if len(today_df) >= 3:
                    sig = SLHuntDetector.check_pdl_hunt(
                        today_df.iloc[-3:], pdl, self.vix)
                    if sig:
                        sig['instrument'] = name
                        sig['pdh'] = pdh; sig['pdl'] = pdl
                        signals[name] = sig
                        continue

                    sig = SLHuntDetector.check_pdh_hunt(
                        today_df.iloc[-3:], pdh, self.vix)
                    if sig:
                        sig['instrument'] = name
                        sig['pdh'] = pdh; sig['pdl'] = pdl
                        signals[name] = sig

            except Exception as e:
                print(f"Scan error {name}: {e}")

        return signals
