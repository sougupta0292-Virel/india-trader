"""
vwap_cpr.py
===========
VWAP and CPR (Central Pivot Range) calculations.
These are the two most widely used tools by professional Indian intraday traders.

VWAP — Volume Weighted Average Price
  - The average price weighted by volume
  - Price above VWAP = bullish, below = bearish
  - Used by institutions to benchmark trades
  - Resets every day at 9:15 AM

CPR — Central Pivot Range
  - Calculated from PREVIOUS day's High, Low, Close
  - Gives you Pivot Point + support/resistance levels for the day
  - Narrow CPR = trending day expected
  - Wide CPR = sideways/choppy day expected
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─── CPR Data Structure ───────────────────────────────────────────────────────

@dataclass
class CPRLevels:
    # Core CPR
    pivot:  float   # (H + L + C) / 3
    bc:     float   # Bottom Central = (H + L) / 2
    tc:     float   # Top Central = (Pivot - BC) + Pivot
    cpr_width: float  # TC - BC (narrow = trending, wide = sideways)
    cpr_width_pct: float  # Width as % of spot

    # Standard Support / Resistance
    r1: float   # First resistance
    r2: float   # Second resistance
    r3: float   # Third resistance
    s1: float   # First support
    s2: float   # Second support
    s3: float   # Third support

    # Day type prediction
    day_type: str   # TRENDING / SIDEWAYS
    bias:     str   # BULLISH / BEARISH / NEUTRAL
    cpr_position: str  # ABOVE_CPR / BELOW_CPR / INSIDE_CPR


@dataclass
class VWAPLevels:
    vwap:       float
    vwap_upper1: float  # VWAP + 1 std dev
    vwap_lower1: float  # VWAP - 1 std dev
    vwap_upper2: float  # VWAP + 2 std dev
    vwap_lower2: float  # VWAP - 2 std dev
    price_vs_vwap: str  # ABOVE / BELOW / AT
    signal:     str     # BULLISH / BEARISH / NEUTRAL


# ─── CPR Calculator ───────────────────────────────────────────────────────────

class CPRCalculator:
    """
    Central Pivot Range — calculated from previous day OHLC.
    The most important pre-market calculation for Indian intraday traders.
    """

    # CPR width threshold — below this = trending day expected
    NARROW_CPR_PCT = 0.15   # 0.15% of spot price

    @staticmethod
    def calculate(prev_high: float, prev_low: float,
                  prev_close: float, spot: float) -> CPRLevels:
        """
        Calculate all CPR levels from previous day data.
        spot = current market price (to determine position vs CPR)
        """
        # Core CPR
        pivot = (prev_high + prev_low + prev_close) / 3
        bc    = (prev_high + prev_low) / 2
        tc    = (pivot - bc) + pivot

        # Ensure TC > BC
        if tc < bc:
            tc, bc = bc, tc

        cpr_width     = round(tc - bc, 2)
        cpr_width_pct = round((cpr_width / spot) * 100, 3) if spot > 0 else 0.0

        # Standard Pivot Support / Resistance
        r1 = (2 * pivot) - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)

        s1 = (2 * pivot) - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)

        # Day type
        day_type = "TRENDING" if cpr_width_pct < CPRCalculator.NARROW_CPR_PCT else "SIDEWAYS"

        # Bias from spot vs CPR
        if spot > tc:
            bias = "BULLISH"
            cpr_position = "ABOVE_CPR"
        elif spot < bc:
            bias = "BEARISH"
            cpr_position = "BELOW_CPR"
        else:
            bias = "NEUTRAL"
            cpr_position = "INSIDE_CPR"

        return CPRLevels(
            pivot=round(pivot, 2),
            bc=round(bc, 2),
            tc=round(tc, 2),
            cpr_width=cpr_width,
            cpr_width_pct=cpr_width_pct,
            r1=round(r1, 2), r2=round(r2, 2), r3=round(r3, 2),
            s1=round(s1, 2), s2=round(s2, 2), s3=round(s3, 2),
            day_type=day_type,
            bias=bias,
            cpr_position=cpr_position
        )

    @staticmethod
    def interpret(cpr: CPRLevels) -> list[str]:
        """Human-readable interpretation of CPR levels."""
        lines = []

        if cpr.day_type == "TRENDING":
            lines.append(f"✅ NARROW CPR ({cpr.cpr_width_pct:.2f}%) → Trending day expected — good for our strategy")
        else:
            lines.append(f"⚠️ WIDE CPR ({cpr.cpr_width_pct:.2f}%) → Sideways/choppy day likely — be careful")

        if cpr.cpr_position == "ABOVE_CPR":
            lines.append(f"🟢 Spot ABOVE CPR → Bullish bias | Look for LONG trades")
            lines.append(f"   Support at BC {cpr.bc:,.2f} → if price comes back here, strong buy zone")
        elif cpr.cpr_position == "BELOW_CPR":
            lines.append(f"🔴 Spot BELOW CPR → Bearish bias | Look for SHORT trades")
            lines.append(f"   Resistance at BC {cpr.bc:,.2f} → if price bounces here, strong sell zone")
        else:
            lines.append(f"⚪ Spot INSIDE CPR ({cpr.bc:,.2f} - {cpr.tc:,.2f}) → Wait for breakout direction")

        lines.append(f"📍 Key levels: R1={cpr.r1:,.2f} | R2={cpr.r2:,.2f} | S1={cpr.s1:,.2f} | S2={cpr.s2:,.2f}")
        return lines


# ─── VWAP Calculator ──────────────────────────────────────────────────────────

class VWAPCalculator:
    """
    Volume Weighted Average Price.
    Calculated from intraday 5-min candles from 9:15 AM onwards.
    """

    @staticmethod
    def calculate_from_candles(candles: pd.DataFrame,
                                spot: float) -> VWAPLevels:
        """
        candles: DataFrame with columns Open, High, Low, Close, Volume
        Indexed by datetime (5-min candles from 9:15 AM)
        """
        if candles is None or len(candles) == 0:
            return VWAPCalculator._estimate_vwap(spot)

        df = candles.copy()

        # Typical price = (H + L + C) / 3
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['tp_vol'] = df['typical_price'] * df['Volume']

        cumulative_tp_vol = df['tp_vol'].cumsum()
        cumulative_vol    = df['Volume'].cumsum()

        # Guard: if all volume is zero, fall back to estimate
        if cumulative_vol.iloc[-1] == 0:
            return VWAPCalculator._estimate_vwap(spot)

        # VWAP at each point
        df['vwap'] = cumulative_tp_vol / cumulative_vol

        current_vwap = float(df['vwap'].iloc[-1])

        # Standard deviation bands
        df['vwap_diff_sq'] = (df['typical_price'] - df['vwap']) ** 2
        variance = float(df['vwap_diff_sq'].mean())
        std_dev  = np.sqrt(variance)

        upper1 = current_vwap + std_dev
        lower1 = current_vwap - std_dev
        upper2 = current_vwap + (2 * std_dev)
        lower2 = current_vwap - (2 * std_dev)

        # Signal
        tolerance = current_vwap * 0.001  # 0.1% tolerance
        if spot > current_vwap + tolerance:
            price_vs_vwap = "ABOVE"
            signal = "BULLISH"
        elif spot < current_vwap - tolerance:
            price_vs_vwap = "BELOW"
            signal = "BEARISH"
        else:
            price_vs_vwap = "AT"
            signal = "NEUTRAL"

        return VWAPLevels(
            vwap=round(current_vwap, 2),
            vwap_upper1=round(upper1, 2),
            vwap_lower1=round(lower1, 2),
            vwap_upper2=round(upper2, 2),
            vwap_lower2=round(lower2, 2),
            price_vs_vwap=price_vs_vwap,
            signal=signal
        )

    @staticmethod
    def _estimate_vwap(spot: float) -> VWAPLevels:
        """Estimate VWAP when no candle data available."""
        std = spot * 0.003  # 0.3% estimated std dev
        return VWAPLevels(
            vwap=round(spot * 0.9995, 2),
            vwap_upper1=round(spot * 0.9995 + std, 2),
            vwap_lower1=round(spot * 0.9995 - std, 2),
            vwap_upper2=round(spot * 0.9995 + 2*std, 2),
            vwap_lower2=round(spot * 0.9995 - 2*std, 2),
            price_vs_vwap="AT",
            signal="NEUTRAL"
        )

    @staticmethod
    def generate_sample_candles(spot: float, n: int = 15) -> pd.DataFrame:
        """Generate sample intraday candles for testing."""
        np.random.seed(int(spot) % 100)
        import pandas as pd
        from datetime import datetime, timedelta

        rows = []
        price = spot * 0.998
        base_time = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)

        for i in range(n):
            price += np.random.normal(0, spot * 0.001)
            h = price + abs(np.random.normal(0, spot * 0.0005))
            l = price - abs(np.random.normal(0, spot * 0.0005))
            rows.append({
                'High': h, 'Low': l,
                'Close': price,
                'Volume': np.random.randint(100000, 500000),
                'Time': base_time + timedelta(minutes=5*i)
            })

        df = pd.DataFrame(rows).set_index('Time')
        return df

    @staticmethod
    def interpret(vwap: VWAPLevels, spot: float) -> list[str]:
        """Human-readable VWAP interpretation."""
        lines = []

        if not vwap.vwap or vwap.vwap == 0:
            lines.append("⚠️ VWAP unavailable — no candle data yet (market just opened?)")
            return lines

        if not spot or spot == 0:
            lines.append("⚠️ Spot price unavailable — cannot interpret VWAP")
            return lines

        dist = ((spot - vwap.vwap) / vwap.vwap) * 100

        if vwap.price_vs_vwap == "ABOVE":
            lines.append(f"🟢 Price ABOVE VWAP by {dist:+.2f}% → Bullish intraday trend")
            lines.append(f"   VWAP {vwap.vwap:,.2f} acts as support — buy on dips to VWAP")
        elif vwap.price_vs_vwap == "BELOW":
            lines.append(f"🔴 Price BELOW VWAP by {dist:+.2f}% → Bearish intraday trend")
            lines.append(f"   VWAP {vwap.vwap:,.2f} acts as resistance — sell on bounces to VWAP")
        else:
            lines.append(f"⚪ Price AT VWAP → No clear trend — wait for breakout")

        lines.append(f"📊 VWAP Bands: Upper={vwap.vwap_upper1:,.2f} | Lower={vwap.vwap_lower1:,.2f}")
        lines.append(f"   Extreme bands: Upper2={vwap.vwap_upper2:,.2f} | Lower2={vwap.vwap_lower2:,.2f}")
        return lines


# ─── Combined Signal with VWAP + CPR ─────────────────────────────────────────

class EnhancedSignalGenerator:
    """
    Enhanced version of SignalGenerator that adds VWAP + CPR confluence.
    Now scores out of 14 instead of 10.
    Target: 60-70% accuracy with 8+ score required.
    """

    MIN_SCORE = 8  # Higher bar with more factors

    @staticmethod
    def score_confluence(
        global_bias: str,
        global_score: float,
        oi_signal,
        first_candle: dict,
        fib_entry_pct: float,
        cpr: CPRLevels,
        vwap: VWAPLevels,
        direction: str,
        rr: float
    ) -> tuple[int, list[str]]:
        """
        Score all confluence factors. Returns (score, reasons).
        Maximum score: 14
        """
        score = 0
        reasons = []

        # 1. Global bias (0-2)
        if global_bias != "NEUTRAL":
            if (direction == "LONG" and global_bias == "BULLISH") or \
               (direction == "SHORT" and global_bias == "BEARISH"):
                score += 1
                reasons.append(f"✅ Global bias {global_bias} (+1)")
                if abs(global_score) > 0.3:
                    score += 1
                    reasons.append(f"✅ Strong global signal ({global_score:+.2f}) (+1)")

        # 2. OI confirmation (0-2)
        oi_aligned = (direction == "LONG" and oi_signal.signal == "BULLISH") or \
                     (direction == "SHORT" and oi_signal.signal == "BEARISH")
        if oi_aligned:
            score += 1
            reasons.append(f"✅ OI signal aligned — {oi_signal.signal} (+1)")
            pcr_good = (direction == "LONG" and oi_signal.pcr > 1.2) or \
                       (direction == "SHORT" and oi_signal.pcr < 0.8)
            if pcr_good:
                score += 1
                reasons.append(f"✅ PCR confirms — {oi_signal.pcr:.2f} (+1)")

        # 3. CPR confluence (0-3) ← NEW
        cpr_aligned = (direction == "LONG" and cpr.bias == "BULLISH") or \
                      (direction == "SHORT" and cpr.bias == "BEARISH")
        if cpr_aligned:
            score += 1
            reasons.append(f"✅ CPR bias {cpr.bias} — spot {cpr.cpr_position} (+1)")
        if cpr.day_type == "TRENDING":
            score += 1
            reasons.append(f"✅ Narrow CPR ({cpr.cpr_width_pct:.2f}%) → trending day (+1)")
        # Price bouncing from CPR level
        if direction == "LONG" and cpr.cpr_position == "ABOVE_CPR":
            score += 1
            reasons.append(f"✅ Spot holding above TC {cpr.tc:,.2f} — bullish (+1)")
        elif direction == "SHORT" and cpr.cpr_position == "BELOW_CPR":
            score += 1
            reasons.append(f"✅ Spot held below BC {cpr.bc:,.2f} — bearish (+1)")

        # 4. VWAP confluence (0-2) ← NEW
        vwap_aligned = (direction == "LONG" and vwap.signal == "BULLISH") or \
                       (direction == "SHORT" and vwap.signal == "BEARISH")
        if vwap_aligned:
            score += 1
            reasons.append(f"✅ Price {vwap.price_vs_vwap} VWAP {vwap.vwap:,.2f} (+1)")
            score += 1
            reasons.append(f"✅ VWAP trend confirms direction (+1)")

        # 5. First candle (0-2)
        candle_body  = abs(first_candle['close'] - first_candle['open'])
        candle_range = first_candle['high'] - first_candle['low']
        candle_strong = candle_body > candle_range * 0.6
        candle_bull   = first_candle['close'] > first_candle['open']

        if candle_strong:
            score += 1
            reasons.append(f"✅ Strong first candle (body {candle_body:.1f} pts) (+1)")
        if (direction == "LONG" and candle_bull) or (direction == "SHORT" and not candle_bull):
            score += 1
            reasons.append(f"✅ First candle direction matches (+1)")

        # 6. Fibonacci level (0-2)
        if fib_entry_pct in (0.618, 0.382):
            score += 2
            reasons.append(f"✅ Entry at key Fibonacci level ({fib_entry_pct*100:.1f}%) (+2)")
        elif fib_entry_pct in (0.5, 0.236):
            score += 1
            reasons.append(f"✅ Entry at Fibonacci level ({fib_entry_pct*100:.1f}%) (+1)")

        # 7. Risk:Reward (0-1)
        if rr >= 2.0:
            score += 1
            reasons.append(f"✅ Excellent R:R 1:{rr:.1f} (+1)")

        return score, reasons
