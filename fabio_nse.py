"""
fabio_nse.py
============
Fabio Valentini Strategy — Ported to NSE/BSE India

Original: NQ Futures (Nasdaq) — Fabio Valentini
Ported:   Indian Indices (Nifty, BankNifty, Sensex) + Nifty 50 Stocks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP A — AAA VALUE AREA FADE  (adapted from VAH/VAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original: Trade rejection at Value Area High (VAH) / Value Area Low (VAL)
NSE Adaptation:
  - Use PREVIOUS DAY High (PDH) and Previous Day Low (PDL) as VAH/VAL
  - Use OI buildup at ATM strikes as volume proxy for value area
  - Session windows: 9:15–10:15 IST (morning open) + 14:00–15:20 IST (closing hour)
  - Look for rejection (long wick, engulfing) at PDH/PDL levels

Signal:
  SELL: Price rallies to PDH + stalls (upper wick, OI buildup at CE)
  BUY:  Price drops to PDL + bounces (lower wick, OI buildup at PE)
  Filter: PCR confirms direction. Narrow CPR = trending day.
  SL: 3 points beyond PDH/PDL
  Target: 2× risk (R:R 1:2 minimum)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP B — MOMENTUM SQUEEZE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original: Bollinger Band squeeze + Keltner Channel contraction + delta confirmation
NSE Adaptation:
  - Bollinger Band (20, 2) + Keltner Channel (20, 1.5 × ATR) squeeze
  - EMA5 direction confirms breakout
  - Volume spike (current candle vol > 1.5× 20-bar avg) = delta proxy
  - PCR direction confirms
  Session: Any time 9:20–14:30 IST

Signal:
  BUY:  BB inside KC, volume spike up, EMA5 rising, PCR bullish
  SELL: BB inside KC, volume spike down, EMA5 falling, PCR bearish
  SL: Opposite BB band
  Target: 2.5× ATR from entry

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP C — CVD DELTA ABSORPTION (order flow proxy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original: Cumulative Volume Delta — absorption at key levels
NSE Adaptation:
  - OHLC candle delta proxy: (Close-Low)/(High-Low) = buying pressure
  - Bull absorption: price drops to support, candle closes near high (delta > 0.7)
  - Bear absorption: price rallies to resistance, closes near low (delta < 0.3)
  - Must occur at PDH/PDL/CPR/VWAP levels
  Session: Any time 9:20–14:30 IST

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK MANAGEMENT (adapted from Fabio)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 3-strike rule: If 3 consecutive losses → STOP trading for day
  - Scale-in: If 1R in profit → add 0.5 position
  - Min R:R: 2:1 (originally 3:1 on NQ, relaxed for Indian volatility)
  - Target win rate: 45–55% (vs 43–49% on NQ)
  - Max 3 trades per session
  - Hard square-off: 3:15 PM IST
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, time as dtime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List

# ─── IST helpers ─────────────────────────────────────────────────────────────

MARKET_OPEN  = dtime(9, 15)
MORNING_END  = dtime(10, 15)  # AAA morning session end
AFTERNOON_ST = dtime(14, 0)   # Closing hour start
SQUAREOFF    = dtime(15, 15)  # Hard square-off
TRADE_START  = dtime(9, 20)
TRADE_END    = dtime(14, 30)

# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class FabioSignal:
    setup:      str           # AAA_FADE / SQUEEZE / DELTA_ABS
    direction:  str           # LONG / SHORT
    entry:      float
    target:     float
    sl:         float
    rr:         float
    confidence: int           # 1-5
    reason:     str
    session:    str           # MORNING / CLOSING / ANYTIME
    key_level:  float         # The PDH/PDL/CPR/VWAP level that triggered
    level_name: str           # "PDH" / "PDL" / "TC" / "BC" / "VWAP"


@dataclass
class FabioSessionState:
    """Tracks Fabio risk management rules during live trading."""
    date:              str  = ""
    consecutive_losses: int = 0
    trades_today:      int  = 0
    max_trades:        int  = 3      # Per session
    stopped_by_rule:   bool = False  # 3-strike rule triggered
    pnl_pts_today:     float = 0.0

    def can_trade(self) -> tuple[bool, str]:
        """Returns (can_trade, reason)."""
        if self.stopped_by_rule:
            return False, "3-STRIKE RULE: 3 consecutive losses — done for today"
        if self.trades_today >= self.max_trades:
            return False, f"Max {self.max_trades} trades reached today"
        return True, "OK"

    def record_result(self, won: bool, pnl_pts: float):
        self.trades_today += 1
        self.pnl_pts_today += pnl_pts
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 3:
                self.stopped_by_rule = True


# ─── Indicator Helpers ────────────────────────────────────────────────────────

def _bollinger_keltner_squeeze(df: pd.DataFrame, bb_period=20,
                                bb_std=2.0, kc_mult=1.5) -> pd.Series:
    """
    True if Bollinger Bands are INSIDE Keltner Channel (squeeze ON).
    Fabio's core squeeze indicator — adapted to OHLCV pandas data.
    """
    close = df['Close']
    high  = df['High']
    low   = df['Low']

    # Bollinger Bands
    bb_mid = close.rolling(bb_period).mean()
    bb_std_val = close.rolling(bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_val
    bb_lower = bb_mid - bb_std * bb_std_val

    # Keltner Channel (using EMA + ATR)
    ema   = close.ewm(span=bb_period, adjust=False).mean()
    tr    = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr   = tr.rolling(bb_period).mean()
    kc_upper = ema + kc_mult * atr
    kc_lower = ema - kc_mult * atr

    # Squeeze = BB inside KC
    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    return squeeze


def _atr(df: pd.DataFrame, period=14) -> pd.Series:
    """Average True Range."""
    h = df['High']; l = df['Low']; c = df['Close']
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _candle_delta(row) -> float:
    """
    OHLCV candle buying pressure proxy.
    0.0 = pure selling, 1.0 = pure buying, 0.5 = neutral
    """
    rng = float(row['High']) - float(row['Low'])
    if rng == 0: return 0.5
    return (float(row['Close']) - float(row['Low'])) / rng


def _volume_spike(df: pd.DataFrame, lookback=20) -> pd.Series:
    """Volume relative to rolling average. >1.5 = spike."""
    vol_avg = df['Volume'].rolling(lookback).mean()
    return df['Volume'] / vol_avg.replace(0, np.nan).fillna(1)


# ─── SETUP A: AAA Value Area Fade ────────────────────────────────────────────

class AAAValueAreaFade:
    """
    Fabio's #1 setup — Rejection at key previous-day levels.
    NSE: Previous Day High (PDH) and Previous Day Low (PDL) = VAH / VAL
    """

    PROXIMITY_PCT  = 0.003   # 0.3% proximity to level = "at level"
    WICK_RATIO     = 0.55    # Wick must be >55% of candle range
    MIN_RANGE_PCT  = 0.001   # Candle must move at least 0.1%

    @staticmethod
    def check(df_today: pd.DataFrame, prev_high: float, prev_low: float,
              prev_close: float, pcr: float,
              cpr_tc: float, cpr_bc: float, vwap: float,
              spot: float, candle_idx: int = -1) -> Optional[FabioSignal]:
        """
        Check for AAA fade setup at current candle.
        df_today: intraday 5-min candles for today (from 9:15 AM)
        """
        if candle_idx == -1:
            candle_idx = len(df_today) - 1
        if candle_idx < 1 or candle_idx >= len(df_today):
            return None

        row   = df_today.iloc[candle_idx]
        h     = float(row['High'])
        l     = float(row['Low'])
        o     = float(row['Open'])
        c     = float(row['Close'])
        rng   = h - l
        # Use 0.1% of close as minimum range — works for 5-min (Nifty ~3pts)
        # AND daily stock candles (₹500 stock → ₹0.5 min, not hardcoded 3)
        min_rng = max(1.0, float(c) * 0.001)
        if rng < min_rng: return None

        # Check time window — prefer morning session or closing hour
        row_time = None
        try:
            if hasattr(df_today.index, 'time'):
                row_time = df_today.index[candle_idx].time()
            elif '_time' in df_today.columns:
                row_time = df_today.iloc[candle_idx]['_time']
        except: pass

        session = "ANYTIME"
        if row_time:
            if row_time <= MORNING_END:
                session = "MORNING"
            elif row_time >= AFTERNOON_ST:
                session = "CLOSING"

        # All key levels to check
        levels = [
            (prev_high,  "PDH", "SHORT"),
            (prev_low,   "PDL", "LONG"),
            (cpr_tc,     "TC",  "SHORT"),
            (cpr_bc,     "BC",  "LONG"),
            (vwap,       "VWAP","SHORT"),  # price at VWAP from above → could go either
        ]

        for level, lname, preferred_dir in levels:
            if level <= 0: continue
            proximity = abs(c - level) / level

            if proximity > AAAValueAreaFade.PROXIMITY_PCT:
                continue  # Not near enough to this level

            # BEARISH FADE at level (SHORT setup)
            if preferred_dir == "SHORT" or lname == "VWAP":
                upper_wick = h - max(o, c)
                body       = abs(c - o)
                if (upper_wick > rng * AAAValueAreaFade.WICK_RATIO and
                        c < o and  # bearish close
                        pcr < 1.1):  # not strongly bullish PCR
                    entry  = c
                    sl     = h + (rng * 0.3)
                    target = entry - abs(entry - sl) * 2.0
                    rr     = abs(target - entry) / abs(entry - sl)
                    if rr < 1.8: continue

                    conf = AAAValueAreaFade._confidence(
                        "SHORT", pcr, session, lname, body, rng)

                    return FabioSignal(
                        setup="AAA_FADE", direction="SHORT",
                        entry=round(entry, 2), target=round(target, 2),
                        sl=round(sl, 2), rr=round(rr, 2),
                        confidence=conf, session=session,
                        key_level=level, level_name=lname,
                        reason=(f"Bearish rejection at {lname} {level:,.0f} | "
                                f"Wick {upper_wick:.0f}pts | PCR={pcr:.2f}")
                    )

            # BULLISH FADE at level (LONG setup)
            if preferred_dir == "LONG" or lname == "VWAP":
                lower_wick = min(o, c) - l
                body       = abs(c - o)
                if (lower_wick > rng * AAAValueAreaFade.WICK_RATIO and
                        c > o and  # bullish close
                        pcr > 0.9):  # not strongly bearish PCR
                    entry  = c
                    sl     = l - (rng * 0.3)
                    target = entry + abs(entry - sl) * 2.0
                    rr     = abs(target - entry) / abs(entry - sl)
                    if rr < 1.8: continue

                    conf = AAAValueAreaFade._confidence(
                        "LONG", pcr, session, lname, body, rng)

                    return FabioSignal(
                        setup="AAA_FADE", direction="LONG",
                        entry=round(entry, 2), target=round(target, 2),
                        sl=round(sl, 2), rr=round(rr, 2),
                        confidence=conf, session=session,
                        key_level=level, level_name=lname,
                        reason=(f"Bullish absorption at {lname} {level:,.0f} | "
                                f"Wick {lower_wick:.0f}pts | PCR={pcr:.2f}")
                    )
        return None

    @staticmethod
    def _confidence(direction, pcr, session, level_name, body, rng) -> int:
        score = 1
        # PCR confirmation
        if direction == "LONG"  and pcr > 1.2: score += 1
        if direction == "SHORT" and pcr < 0.8: score += 1
        # Session bonus
        if session in ("MORNING", "CLOSING"): score += 1
        # PDH/PDL = strongest levels
        if level_name in ("PDH", "PDL"): score += 1
        return min(score, 5)


# ─── SETUP B: Momentum Squeeze ───────────────────────────────────────────────

class MomentumSqueeze:
    """
    Bollinger Band / Keltner Channel squeeze + volume breakout.
    Fabio's #2 setup — consolidation into explosive move.
    """

    MIN_SQUEEZE_BARS = 3    # Squeeze must hold for at least 3 bars
    VOL_SPIKE_RATIO  = 1.5  # Volume spike threshold

    @staticmethod
    def check(df: pd.DataFrame, pcr: float,
              candle_idx: int = -1) -> Optional[FabioSignal]:
        """
        df: 5-min candles with at least 30 rows.
        candle_idx: candle to check (-1 = last).
        """
        if len(df) < 25: return None
        if candle_idx == -1: candle_idx = len(df) - 1
        if candle_idx < 22: return None

        df_work = df.iloc[:candle_idx + 1].copy()

        squeeze    = _bollinger_keltner_squeeze(df_work)
        vol_ratio  = _volume_spike(df_work)
        atr_series = _atr(df_work)

        # Need at least MIN_SQUEEZE_BARS consecutive squeeze before this candle
        prev_n = squeeze.iloc[-1 - MomentumSqueeze.MIN_SQUEEZE_BARS: -1]
        if not prev_n.all(): return None

        # Current candle must have broken OUT of squeeze (squeeze OFF)
        if squeeze.iloc[-1]: return None  # Still in squeeze, not a breakout yet

        curr_row   = df_work.iloc[-1]
        curr_vol   = float(vol_ratio.iloc[-1])
        curr_atr   = float(atr_series.iloc[-1])
        c          = float(curr_row['Close'])
        o          = float(curr_row['Open'])
        h          = float(curr_row['High'])
        l          = float(curr_row['Low'])
        ema5       = float(df_work['Close'].ewm(span=5, adjust=False).mean().iloc[-1])
        ema5_prev  = float(df_work['Close'].ewm(span=5, adjust=False).mean().iloc[-2])

        if curr_vol < MomentumSqueeze.VOL_SPIKE_RATIO: return None
        if curr_atr == 0: return None

        direction = None

        # Bullish squeeze breakout
        if (c > o and c > ema5 and ema5 > ema5_prev and pcr > 0.85):
            direction = "LONG"
            entry  = c
            sl     = l - curr_atr * 0.3
            target = entry + curr_atr * 2.5
            rr     = abs(target - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0

        # Bearish squeeze breakout
        elif (c < o and c < ema5 and ema5 < ema5_prev and pcr < 1.15):
            direction = "SHORT"
            entry  = c
            sl     = h + curr_atr * 0.3
            target = entry - curr_atr * 2.5
            rr     = abs(target - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0

        if direction is None: return None
        if rr < 1.8: return None

        # Session
        session = "ANYTIME"
        try:
            if hasattr(df.index, 'time'):
                t = df.index[candle_idx].time()
            elif '_time' in df.columns:
                t = df.iloc[candle_idx]['_time']
            else:
                t = None
            if t:
                if t <= MORNING_END:   session = "MORNING"
                elif t >= AFTERNOON_ST: session = "CLOSING"
        except: pass

        conf = 2
        if pcr > 1.3 and direction == "LONG":   conf += 1
        if pcr < 0.7 and direction == "SHORT":  conf += 1
        if curr_vol > 2.0: conf += 1
        conf = min(conf, 5)

        return FabioSignal(
            setup="SQUEEZE", direction=direction,
            entry=round(entry, 2), target=round(target, 2),
            sl=round(sl, 2), rr=round(rr, 2),
            confidence=conf, session=session,
            key_level=round(ema5, 2), level_name="EMA5",
            reason=(f"BB/KC Squeeze breakout | Vol×{curr_vol:.1f} | "
                    f"ATR={curr_atr:.0f} | EMA5={'↑' if direction=='LONG' else '↓'} | "
                    f"PCR={pcr:.2f}")
        )


# ─── SETUP C: Delta Absorption ───────────────────────────────────────────────

class DeltaAbsorption:
    """
    Order flow absorption — price reaches key level but big players absorb it.
    NSE proxy: candle delta (close vs range) at key support/resistance.
    """

    BULL_DELTA_THRESH = 0.70   # Close in top 30% of range = buyers absorbed sellers
    BEAR_DELTA_THRESH = 0.30   # Close in bottom 30% of range = sellers absorbed buyers
    PROXIMITY_PCT     = 0.004  # 0.4% to key level

    @staticmethod
    def check(df_today: pd.DataFrame, prev_high: float, prev_low: float,
              cpr_tc: float, cpr_bc: float, vwap: float,
              pcr: float, candle_idx: int = -1) -> Optional[FabioSignal]:
        """Detect absorption candle at key level."""
        if len(df_today) < 3: return None
        if candle_idx == -1: candle_idx = len(df_today) - 1
        if candle_idx < 2: return None

        row   = df_today.iloc[candle_idx]
        prev1 = df_today.iloc[candle_idx - 1]
        delta = _candle_delta(row)
        h = float(row['High']); l = float(row['Low'])
        c = float(row['Close']); o = float(row['Open'])
        rng = h - l
        min_rng = max(1.0, float(c) * 0.001)
        if rng < min_rng: return None

        # Need volume higher than previous candle (absorption confirmation)
        vol_curr = float(row.get('Volume', 0)) if 'Volume' in row.index else 1
        vol_prev = float(prev1.get('Volume', 0)) if 'Volume' in prev1.index else 1
        if vol_curr < vol_prev * 0.8: return None  # Volume must not drop

        levels = [
            (prev_high,  "PDH"),
            (prev_low,   "PDL"),
            (cpr_tc,     "TC"),
            (cpr_bc,     "BC"),
            (vwap,       "VWAP"),
        ]

        for level, lname in levels:
            if level <= 0: continue
            prox = abs(c - level) / level
            if prox > DeltaAbsorption.PROXIMITY_PCT: continue

            # BULL ABSORPTION: price touches PDL/BC area, closes near HIGH
            if lname in ("PDL", "BC") and delta >= DeltaAbsorption.BULL_DELTA_THRESH:
                entry  = c
                sl     = l - rng * 0.25
                target = entry + rng * 2.5
                rr     = abs(target - entry) / abs(entry - sl)
                if rr < 1.8: continue
                if pcr < 0.8: continue  # Need bullish OI

                return FabioSignal(
                    setup="DELTA_ABS", direction="LONG",
                    entry=round(entry, 2), target=round(target, 2),
                    sl=round(sl, 2), rr=round(rr, 2),
                    confidence=4, session="ANYTIME",
                    key_level=level, level_name=lname,
                    reason=(f"Bull absorption at {lname} {level:,.0f} | "
                            f"Delta={delta:.2f} | PCR={pcr:.2f}")
                )

            # BEAR ABSORPTION: price touches PDH/TC area, closes near LOW
            if lname in ("PDH", "TC") and delta <= DeltaAbsorption.BEAR_DELTA_THRESH:
                entry  = c
                sl     = h + rng * 0.25
                target = entry - rng * 2.5
                rr     = abs(target - entry) / abs(entry - sl)
                if rr < 1.8: continue
                if pcr > 1.2: continue  # Need bearish OI

                return FabioSignal(
                    setup="DELTA_ABS", direction="SHORT",
                    entry=round(entry, 2), target=round(target, 2),
                    sl=round(sl, 2), rr=round(rr, 2),
                    confidence=4, session="ANYTIME",
                    key_level=level, level_name=lname,
                    reason=(f"Bear absorption at {lname} {level:,.0f} | "
                            f"Delta={delta:.2f} | PCR={pcr:.2f}")
                )
        return None


# ─── Master Signal Generator ─────────────────────────────────────────────────

class FabioNSESignalGenerator:
    """
    Combines all 3 Fabio setups and scores them.
    Returns the best signal for the current candle.
    """

    def __init__(self):
        self.aaa     = AAAValueAreaFade()
        self.squeeze = MomentumSqueeze()
        self.delta   = DeltaAbsorption()

    def get_signal(self,
                   df_today:   pd.DataFrame,
                   prev_high:  float,
                   prev_low:   float,
                   prev_close: float,
                   pcr:        float,
                   cpr_tc:     float,
                   cpr_bc:     float,
                   vwap:       float,
                   spot:       float,
                   candle_idx: int = -1) -> Optional[FabioSignal]:
        """
        Run all 3 setups and return highest-confidence signal.
        Returns None if no valid signal found.
        """
        signals = []

        s1 = AAAValueAreaFade.check(
            df_today, prev_high, prev_low, prev_close,
            pcr, cpr_tc, cpr_bc, vwap, spot, candle_idx)
        if s1: signals.append(s1)

        s2 = MomentumSqueeze.check(df_today, pcr, candle_idx)
        if s2: signals.append(s2)

        s3 = DeltaAbsorption.check(
            df_today, prev_high, prev_low,
            cpr_tc, cpr_bc, vwap, pcr, candle_idx)
        if s3: signals.append(s3)

        if not signals: return None

        # Prefer confluence: if 2+ signals agree on direction, boost confidence
        long_sigs  = [s for s in signals if s.direction == "LONG"]
        short_sigs = [s for s in signals if s.direction == "SHORT"]

        if len(long_sigs) >= 2:
            best = max(long_sigs, key=lambda s: s.confidence)
            best.confidence = min(best.confidence + 1, 5)
            best.reason = f"[CONFLUENCE {len(long_sigs)} setups] " + best.reason
            return best
        if len(short_sigs) >= 2:
            best = max(short_sigs, key=lambda s: s.confidence)
            best.confidence = min(best.confidence + 1, 5)
            best.reason = f"[CONFLUENCE {len(short_sigs)} setups] " + best.reason
            return best

        # Single signal — only take if confidence >= 3
        best = max(signals, key=lambda s: s.confidence)
        return best if best.confidence >= 3 else None

    def interpret(self, signal: FabioSignal, spot: float) -> list:
        """Human-readable interpretation for the UI."""
        lines = []
        setup_names = {
            "AAA_FADE":  "🎯 Fabio AAA Value Area Fade",
            "SQUEEZE":   "💥 Fabio Momentum Squeeze",
            "DELTA_ABS": "🌊 Fabio Delta Absorption",
        }
        name = setup_names.get(signal.setup, signal.setup)
        dir_emoji = "🟢 LONG" if signal.direction == "LONG" else "🔴 SHORT"

        lines.append(f"{name}")
        lines.append(f"{dir_emoji} | Confidence: {'⭐'*signal.confidence} ({signal.confidence}/5)")
        lines.append(f"📍 Entry: {signal.entry:,.2f} | SL: {signal.sl:,.2f} | Target: {signal.target:,.2f}")
        lines.append(f"📐 R:R = 1:{signal.rr:.1f} | Level: {signal.level_name} @ {signal.key_level:,.2f}")
        lines.append(f"⏱ Session: {signal.session}")
        lines.append(f"📝 {signal.reason}")

        risk_pts = abs(signal.entry - signal.sl)
        lines.append(f"⚠️ Risk: {risk_pts:.0f} pts | Reward: {abs(signal.target-signal.entry):.0f} pts")

        return lines


# ─── Backtester for Fabio NSE Setups ─────────────────────────────────────────

class FabioNSEBacktester:
    """
    Backtest all 3 Fabio NSE setups on 5-min intraday data.
    Uses real yfinance 5-min candles.

    Respects Fabio risk management:
    - 3-strike rule per day
    - Max 3 trades per day
    - Square off at 3:15 PM IST
    - Scale-in when 1R in profit (simulated)
    """

    # ── Keys here are INDEX names (5-min intraday), not stock names.
    # ── For Nifty 50 constituent stocks (daily candles) use FabioNSEStockBacktester.
    INDICES = {
        "NiftyIndex":  {"symbol": "^NSEI",   "lot": 75},   # Nifty 50 Index (^NSEI)
        "BankNifty":   {"symbol": "^NSEBANK", "lot": 35},
        "Sensex":      {"symbol": "^BSESN",  "lot": 20},
    }

    def __init__(self, index="NiftyIndex", days=55, min_rr=2.0,
                 setups=None, apply_risk_rules=True):
        self.index     = index
        self.symbol    = self.INDICES.get(index, {"symbol": "^NSEI"})["symbol"]
        self.lot       = self.INDICES.get(index, {"lot": 75})["lot"]
        self.days      = min(days, 55)  # yfinance 5m limit
        self.min_rr    = min_rr
        self.setups    = setups or ["AAA_FADE", "SQUEEZE", "DELTA_ABS"]
        self.apply_rules = apply_risk_rules
        self.gen       = FabioNSESignalGenerator()

    def _fetch(self) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            df = yf.download(self.symbol, period=f"{self.days}d",
                             interval="5m", progress=False, auto_adjust=True)
            if df is None or len(df) < 50: return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for col in ['Open','High','Low','Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            df = df.dropna()
            if hasattr(df.index, 'date'):
                df['_date'] = df.index.date
                df['_time'] = df.index.time
            else:
                dt_idx = pd.to_datetime(df.index)
                df['_date'] = dt_idx.date
                df['_time'] = dt_idx.time
            print(f"[Fabio] Fetched {len(df)} 5-min rows for {self.index}")
            return df
        except Exception as e:
            print(f"[Fabio] Fetch error: {e}")
            return None

    def run(self) -> dict:
        """Returns {setup_name: BtResult}"""
        from backtester import BtTrade, BtResult
        df = self._fetch()
        if df is None:
            return {}

        dates = sorted(df['_date'].unique())
        results = {s: {'trades': [], 'equity': [100000], 'cur': 100000}
                   for s in self.setups}

        for di in range(1, len(dates)):
            today    = dates[di]
            prev_d   = dates[di - 1]
            today_df = df[df['_date'] == today].copy().reset_index(drop=True)
            prev_df  = df[df['_date'] == prev_d].copy()
            if len(today_df) < 15 or len(prev_df) < 10: continue

            # Previous day levels
            prev_high  = float(prev_df['High'].max())
            prev_low   = float(prev_df['Low'].min())
            prev_close = float(prev_df['Close'].iloc[-1])
            diff       = prev_high - prev_low

            # CPR
            pivot  = (prev_high + prev_low + prev_close) / 3
            bc_cpr = (prev_high + prev_low) / 2
            tc_cpr = (pivot - bc_cpr) + pivot
            if tc_cpr < bc_cpr: tc_cpr, bc_cpr = bc_cpr, tc_cpr

            # VWAP estimate (use opening as proxy if no full data)
            vwap_est = float(today_df['Close'].iloc[:6].mean()) if len(today_df) >= 6 else float(today_df['Close'].iloc[0])

            # Simulated PCR from previous day trend
            prev_ret = (float(prev_df['Close'].iloc[-1]) - float(prev_df['Open'].iloc[0]))
            prev_op  = float(prev_df['Open'].iloc[0])
            pcr      = 1.0 + (prev_ret / prev_op * 8) if prev_op > 0 else 1.0
            pcr      = max(0.4, min(2.5, pcr))

            # Filter 9:20 to 14:30
            trade_window = today_df[
                (today_df['_time'] >= TRADE_START) &
                (today_df['_time'] <= TRADE_END)
            ].copy().reset_index(drop=True)
            if len(trade_window) < 10: continue

            # Per-setup daily state (3-strike rule)
            daily_state  = {s: FabioSessionState(date=str(today)) for s in self.setups}
            traded_today = {s: False for s in self.setups}  # track no-trade days

            for ci in range(3, len(trade_window) - 1):
                df_so_far = trade_window.iloc[:ci + 1].copy()

                for setup in self.setups:
                    can, _ = daily_state[setup].can_trade()
                    if not can and self.apply_rules: continue

                    signal = None
                    try:
                        if setup == "AAA_FADE":
                            signal = AAAValueAreaFade.check(
                                df_so_far, prev_high, prev_low, prev_close,
                                pcr, tc_cpr, bc_cpr, vwap_est, float(df_so_far['Close'].iloc[-1]))
                        elif setup == "SQUEEZE":
                            if len(df_so_far) >= 25:
                                signal = MomentumSqueeze.check(df_so_far, pcr)
                        elif setup == "DELTA_ABS":
                            signal = DeltaAbsorption.check(
                                df_so_far, prev_high, prev_low,
                                tc_cpr, bc_cpr, vwap_est, pcr)
                    except:
                        continue

                    if signal is None: continue
                    if signal.rr < self.min_rr: continue

                    # Simulate exit on remaining candles
                    remaining = trade_window.iloc[ci + 1:]
                    exit_p, exit_r = self._simulate_exit(
                        remaining, signal.direction, signal.entry,
                        signal.target, signal.sl)

                    pnl = (exit_p - signal.entry) if signal.direction == "LONG" \
                          else (signal.entry - exit_p)

                    # Scale-in: if hit 1R profit, count 1.5x (Fabio scale-in)
                    scale = 1.5 if pnl > abs(signal.entry - signal.sl) else 1.0
                    results[setup]['cur'] += pnl * self.lot * scale
                    results[setup]['equity'].append(results[setup]['cur'])
                    results[setup]['trades'].append(BtTrade(
                        date=str(today), strategy=f"FABIO_{setup}",
                        direction=signal.direction,
                        entry=signal.entry, target=signal.target, stop=signal.sl,
                        exit_price=round(exit_p, 2), exit_reason=exit_r,
                        pnl_pts=round(pnl, 2), rr=round(signal.rr, 2),
                        won=pnl > 0,
                        reason=f"[{signal.level_name}] {signal.reason}"
                    ))

                    won = pnl > 0
                    daily_state[setup].record_result(won, pnl)
                    traded_today[setup] = True

            # No-trade days: keep equity flat
            for setup in self.setups:
                if not traded_today[setup]:
                    results[setup]['equity'].append(results[setup]['cur'])

        # Build BtResult objects
        from backtester import BtResult
        final = {}
        for setup, data in results.items():
            trades = data['trades']
            equity = data['equity']
            if not trades:
                final[setup] = BtResult(
                    strategy=f"FABIO_{setup}-{self.index}",
                    total_trades=0, winning_trades=0, losing_trades=0,
                    win_rate=0, avg_win_pts=0, avg_loss_pts=0,
                    profit_factor=0, max_drawdown_pts=0,
                    total_pnl_pts=0, trades=[], equity_curve=equity)
                continue

            wins   = [t for t in trades if t.won]
            losses = [t for t in trades if not t.won]
            wr     = len(wins) / len(trades) * 100
            avg_w  = float(np.mean([t.pnl_pts for t in wins])) if wins else 0
            avg_l  = abs(float(np.mean([t.pnl_pts for t in losses]))) if losses else 0
            pf     = (len(wins) * avg_w) / (len(losses) * avg_l) \
                     if losses and avg_l > 0 else 99.0
            eq_arr = np.array(equity)
            dd     = float(np.max(np.maximum.accumulate(eq_arr) - eq_arr))

            final[setup] = BtResult(
                strategy=f"FABIO_{setup}-{self.index}",
                total_trades=len(trades),
                winning_trades=len(wins),
                losing_trades=len(losses),
                win_rate=round(wr, 1),
                avg_win_pts=round(avg_w, 2),
                avg_loss_pts=round(avg_l, 2),
                profit_factor=round(pf, 2),
                max_drawdown_pts=round(dd, 2),
                total_pnl_pts=round(sum(t.pnl_pts for t in trades), 2),
                trades=trades, equity_curve=equity
            )
        return final

    def _simulate_exit(self, remaining: pd.DataFrame, direction: str,
                       entry: float, target: float, sl: float) -> tuple:
        for j in range(len(remaining)):
            row = remaining.iloc[j]
            h   = float(row['High'])
            l   = float(row['Low'])
            c   = float(row['Close'])
            t   = row.get('_time', None)

            # Square off at 3:15 PM
            if t and t >= SQUAREOFF:
                return c, "SQUAREOFF_315"

            if direction == "LONG":
                if l <= sl:     return sl,     "STOP_LOSS"
                if h >= target: return target, "TARGET_HIT"
            else:
                if h >= sl:     return sl,     "STOP_LOSS"
                if l <= target: return target, "TARGET_HIT"

        last = float(remaining.iloc[-1]['Close']) if len(remaining) > 0 else entry
        return last, "EOD_EXIT"


# ─── Fabio NSE for Stock Backtester ──────────────────────────────────────────

class FabioNSEStockBacktester:
    """
    Run Fabio setups on individual NSE stocks (Nifty50).
    Uses daily candles (no 5-min limit issue).
    """

    def __init__(self, symbol: str, days: int = 180):
        self.symbol = symbol
        self.days   = days
        self.gen    = FabioNSESignalGenerator()

    def run(self) -> dict:
        """Returns {setup: StockBtResult} (compatible with nse_stocks_engine)"""
        try:
            import yfinance as yf
            from nse_stocks_engine import NIFTY50_STOCKS, StockBtResult
            meta = NIFTY50_STOCKS.get(self.symbol, {})
            yf_sym = meta.get("yf", f"{self.symbol}.NS")
            df = yf.download(yf_sym, period=f"{self.days + 10}d",
                             interval="1d", progress=False, auto_adjust=True)
            if df is None or len(df) < 20: return {}
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for col in ['Open','High','Low','Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            df = df.dropna()
        except Exception as e:
            print(f"[FabioStock] Fetch error: {e}")
            return {}

        results = {s: {'trades': [], 'equity': [100000], 'cur': 100000}
                   for s in ["AAA_FADE", "SQUEEZE", "DELTA_ABS"]}

        for i in range(1, len(df) - 1):
            row  = df.iloc[i]
            prev = df.iloc[i - 1]
            nxt  = df.iloc[i + 1]

            prev_h = float(prev['High']); prev_l = float(prev['Low'])
            prev_c = float(prev['Close'])
            diff   = prev_h - prev_l
            # Use 0.3% of prev close as minimum range instead of hardcoded 5pts
            if diff < prev_c * 0.003: continue

            c = float(row['Close']); h = float(row['High']); l = float(row['Low'])
            o = float(row['Open'])

            prev_ret = prev_c - float(prev['Open'])
            pcr = 1.0 + prev_ret / float(prev['Open']) * 6 if float(prev['Open']) > 0 else 1.0
            pcr = max(0.4, min(2.5, pcr))

            pivot  = (prev_h + prev_l + prev_c) / 3
            bc_cpr = (prev_h + prev_l) / 2
            tc_cpr = (pivot - bc_cpr) + pivot
            if tc_cpr < bc_cpr: tc_cpr, bc_cpr = bc_cpr, tc_cpr

            vwap_est = c  # Same-day close as proxy

            # Build a mini intraday DataFrame from daily OHLCV for signal checks
            mini_df = df.iloc[max(0, i-20): i+1].copy()

            nh = float(nxt['High']); nl = float(nxt['Low']); nc = float(nxt['Close'])

            for setup in ["AAA_FADE", "SQUEEZE", "DELTA_ABS"]:
                signal = None
                try:
                    if setup == "AAA_FADE":
                        signal = AAAValueAreaFade.check(
                            mini_df, prev_h, prev_l, prev_c,
                            pcr, tc_cpr, bc_cpr, vwap_est, c)
                    elif setup == "SQUEEZE" and len(mini_df) >= 22:
                        signal = MomentumSqueeze.check(mini_df, pcr)
                    elif setup == "DELTA_ABS":
                        signal = DeltaAbsorption.check(
                            mini_df, prev_h, prev_l, tc_cpr, bc_cpr, vwap_est, pcr)
                except: continue

                if signal is None: continue
                if signal.rr < 2.0: continue

                # Simulate next-candle exit
                if signal.direction == "LONG":
                    if nl <= signal.sl:     exit_p, exit_r = signal.sl, "STOP_LOSS"
                    elif nh >= signal.target: exit_p, exit_r = signal.target, "TARGET_HIT"
                    else:                   exit_p, exit_r = nc, "EOD_EXIT"
                else:
                    if nh >= signal.sl:     exit_p, exit_r = signal.sl, "STOP_LOSS"
                    elif nl <= signal.target: exit_p, exit_r = signal.target, "TARGET_HIT"
                    else:                   exit_p, exit_r = nc, "EOD_EXIT"

                pnl = (exit_p - signal.entry) if signal.direction == "LONG" \
                      else (signal.entry - exit_p)
                pnl_pct = pnl / signal.entry * 100

                results[setup]['cur'] += pnl_pct * 1000  # normalized units
                results[setup]['equity'].append(results[setup]['cur'])
                results[setup]['trades'].append({
                    'date': str(df.index[i])[:10],
                    'direction': signal.direction,
                    'entry': round(signal.entry, 2),
                    'target': round(signal.target, 2),
                    'sl': round(signal.sl, 2),
                    'exit': round(exit_p, 2),
                    'exit_reason': exit_r,
                    'pnl_pct': round(pnl_pct, 2),
                    'rr': round(signal.rr, 2),
                    'won': pnl > 0,
                    'reason': signal.reason,
                    'setup': setup
                })

            # No-trade bar: keep equity flat for each setup that didn't trade today
            today_str = str(df.index[i])[:10]
            for setup in ["AAA_FADE", "SQUEEZE", "DELTA_ABS"]:
                if not results[setup]['trades'] or \
                   results[setup]['trades'][-1]['date'] != today_str:
                    results[setup]['equity'].append(results[setup]['cur'])

        return results


# ─── Quick Signal for Options Bot ────────────────────────────────────────────

def get_fabio_signal_for_options(df_5min: pd.DataFrame,
                                  instrument: str = "NIFTY") -> Optional[dict]:
    """
    Quick wrapper — returns signal dict compatible with OptionsAutoTrader.
    Called from options_auto_trader.py to add FABIO_NSE strategy.
    """
    if df_5min is None or len(df_5min) < 20: return None

    try:
        # Get previous day levels from the data
        today_date = df_5min.index[-1].date() if hasattr(df_5min.index, 'date') else \
                     pd.to_datetime(df_5min.index[-1]).date()

        if '_date' in df_5min.columns:
            dates_in_df = sorted(df_5min['_date'].unique())
            if len(dates_in_df) < 2: return None
            prev_date = dates_in_df[-2]
            prev_df   = df_5min[df_5min['_date'] == prev_date]
            today_df  = df_5min[df_5min['_date'] == dates_in_df[-1]]
        else:
            prev_df  = df_5min.iloc[:-30]
            today_df = df_5min.iloc[-30:]

        if len(prev_df) < 5 or len(today_df) < 5: return None

        prev_high  = float(prev_df['High'].max())
        prev_low   = float(prev_df['Low'].min())
        prev_close = float(prev_df['Close'].iloc[-1])

        pivot  = (prev_high + prev_low + prev_close) / 3
        bc_cpr = (prev_high + prev_low) / 2
        tc_cpr = (pivot - bc_cpr) + pivot
        if tc_cpr < bc_cpr: tc_cpr, bc_cpr = bc_cpr, tc_cpr

        spot     = float(today_df['Close'].iloc[-1])
        vwap_est = float(today_df['Close'].mean())

        prev_ret = prev_close - float(prev_df['Open'].iloc[0])
        pcr = 1.0 + prev_ret / float(prev_df['Open'].iloc[0]) * 6 \
              if float(prev_df['Open'].iloc[0]) > 0 else 1.0
        pcr = max(0.4, min(2.5, pcr))

        gen    = FabioNSESignalGenerator()
        signal = gen.get_signal(
            today_df, prev_high, prev_low, prev_close,
            pcr, tc_cpr, bc_cpr, vwap_est, spot)

        if signal is None: return None

        return {
            'direction': signal.direction,
            'entry':     signal.entry,
            'target':    signal.target,
            'sl':        signal.sl,
            'rr':        signal.rr,
            'strategy':  f"FABIO_{signal.setup}",
            'reason':    signal.reason,
            'confidence': signal.confidence,
        }

    except Exception as e:
        print(f"[FabioOptions] Signal error: {e}")
        return None
