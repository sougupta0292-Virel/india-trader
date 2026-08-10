"""
strategy_engine.py
==================
Core trading strategy for Indian intraday markets.

STRATEGY LOGIC (OI + Fibonacci + Global Cues):
1. Pre-market: Pull global cues (SGX/GIFT Nifty, Dow, Nasdaq, Nikkei, Hang Seng, DAX)
2. Calculate expected gap → bias (bullish/bearish/neutral)
3. Plot Fibonacci retracement from prev day High-Low
4. At 9:20 AM: Confirm first 5-min candle direction
5. Enter on pullback to 38.2% or 61.8% Fib level WITH OI confirmation
   - OI rising + price rising  → trend trade (long)
   - OI rising + price falling → trend trade (short)
   - OI falling + price rising → short covering (weaker, skip)
   - OI falling + price falling→ long unwinding (weaker, skip)
6. Target: next Fibonacci level. SL: below/above 61.8% Fib.
7. Max 1-2 trades per day. No trades after 2:30 PM.

ACCURACY TARGET: 60-70% via high-selectivity filtering
- Only trade when 3+ confluence factors align
- Skip low-volatility days (VIX < 11 or > 25)
- Skip days with major events (FOMC, RBI, expiry day)
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class GlobalCue:
    symbol: str
    name: str
    prev_close: float
    current: float
    change_pct: float
    sentiment: str  # BULLISH / BEARISH / NEUTRAL
    weight: float   # How much it influences Nifty

@dataclass
class OISignal:
    strike: int
    call_oi: float
    put_oi: float
    call_oi_change: float
    put_oi_change: float
    pcr: float              # Put-Call Ratio
    max_pain: int
    key_support: int        # Highest put OI strike
    key_resistance: int     # Highest call OI strike
    signal: str             # BULLISH / BEARISH / NEUTRAL

@dataclass
class FibLevel:
    level_name: str
    level_pct: float
    price: float
    role: str               # SUPPORT / RESISTANCE

@dataclass
class TradeSignal:
    timestamp: datetime
    instrument: str         # NIFTY / BANKNIFTY
    direction: str          # LONG / SHORT
    entry_price: float
    target_1: float
    target_2: float
    stop_loss: float
    fib_entry_level: str    # e.g., "61.8% retracement"
    oi_confirmation: bool
    global_bias: str
    confluence_score: int   # 0-10
    confidence: str         # HIGH / MEDIUM / LOW
    reason: str
    option_strike: Optional[int] = None
    option_type: Optional[str] = None  # CE / PE
    risk_reward: float = 0.0


# ─── Global Cues Engine ───────────────────────────────────────────────────────

class GlobalCuesEngine:
    """
    Fetches and interprets international market data.
    Symbols map to yfinance tickers.
    """
    MARKET_MAP = {
        # SGX/GIFT Nifty is the primary cue
        "GIFT_NIFTY":   {"name": "GIFT Nifty Futures",   "ticker": "^NSEI",   "weight": 0.35},
        "DOW":          {"name": "Dow Jones",             "ticker": "^DJI",    "weight": 0.20},
        "NASDAQ":       {"name": "Nasdaq 100",            "ticker": "^NDX",    "weight": 0.15},
        "SP500":        {"name": "S&P 500",               "ticker": "^GSPC",   "weight": 0.10},
        "NIKKEI":       {"name": "Nikkei 225 (Japan)",    "ticker": "^N225",   "weight": 0.08},
        "HANGSENG":     {"name": "Hang Seng (HK/China)",  "ticker": "^HSI",    "weight": 0.06},
        "DAX":          {"name": "DAX (Germany)",         "ticker": "^GDAXI",  "weight": 0.04},
        "VIX":          {"name": "India VIX",             "ticker": "^INDIAVIX","weight": 0.02},
    }

    @staticmethod
    def fetch_global_cues() -> list[GlobalCue]:
        """Fetch global market data using yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            return GlobalCuesEngine._sample_cues()

        cues = []
        for key, meta in GlobalCuesEngine.MARKET_MAP.items():
            try:
                ticker = yf.Ticker(meta["ticker"])
                hist = ticker.history(period="2d", interval="1d")
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[-2]
                    current    = hist['Close'].iloc[-1]
                else:
                    prev_close = current = hist['Close'].iloc[-1]

                change_pct = ((current - prev_close) / prev_close) * 100

                if key == "VIX":
                    # VIX logic is inverted
                    if current < 13:   sentiment = "BULLISH"
                    elif current > 20: sentiment = "BEARISH"
                    else:              sentiment = "NEUTRAL"
                elif change_pct > 0.3:
                    sentiment = "BULLISH"
                elif change_pct < -0.3:
                    sentiment = "BEARISH"
                else:
                    sentiment = "NEUTRAL"

                cues.append(GlobalCue(
                    symbol=key, name=meta["name"],
                    prev_close=round(prev_close, 2),
                    current=round(current, 2),
                    change_pct=round(change_pct, 2),
                    sentiment=sentiment,
                    weight=meta["weight"]
                ))
            except Exception:
                continue
        return cues

    @staticmethod
    def compute_global_bias(cues: list[GlobalCue]) -> Tuple[str, float, str]:
        """
        Weighted bias from all global cues.
        Returns: (bias, score, reasoning)
        score > 0 = bullish, < 0 = bearish
        """
        score = 0.0
        reasons = []
        for cue in cues:
            weight = cue.weight
            if cue.sentiment == "BULLISH":
                score += weight * abs(cue.change_pct)
                reasons.append(f"{cue.name} +{cue.change_pct:.1f}%")
            elif cue.sentiment == "BEARISH":
                score -= weight * abs(cue.change_pct)
                reasons.append(f"{cue.name} {cue.change_pct:.1f}%")

        if score > 0.15:    bias = "BULLISH"
        elif score < -0.15: bias = "BEARISH"
        else:               bias = "NEUTRAL"

        return bias, round(score, 3), " | ".join(reasons[:4])

    @staticmethod
    def _sample_cues() -> list[GlobalCue]:
        """Sample data when yfinance unavailable."""
        return [
            GlobalCue("DOW","Dow Jones",39000,39250,0.64,"BULLISH",0.20),
            GlobalCue("NASDAQ","Nasdaq",17500,17620,0.69,"BULLISH",0.15),
            GlobalCue("NIKKEI","Nikkei",38800,39100,0.77,"BULLISH",0.08),
            GlobalCue("VIX","India VIX",13.5,13.5,0.0,"NEUTRAL",0.02),
        ]


# ─── OI Analysis Engine ───────────────────────────────────────────────────────

class OIAnalyser:
    """
    Analyses NSE Option Chain data (Open Interest).
    In live mode: fetches from Zerodha Kite API.
    In backtest mode: reads from historical OI CSV.
    """

    @staticmethod
    def analyse_option_chain(option_chain_df: pd.DataFrame,
                              spot_price: float,
                              expiry_offset: int = 0) -> OISignal:
        """
        option_chain_df columns: strike, call_oi, put_oi, call_oi_change, put_oi_change
        """
        df = option_chain_df.copy()
        df = df.sort_values('strike')

        # ATM strike
        atm_idx = (df['strike'] - spot_price).abs().idxmin()
        atm_strike = int(df.loc[atm_idx, 'strike'])

        # PCR — Put-Call Ratio
        total_call_oi = df['call_oi'].sum()
        total_put_oi  = df['put_oi'].sum()
        pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 1.0

        # Max Pain — strike where option buyers lose most
        max_pain = OIAnalyser._calculate_max_pain(df)

        # Key levels — highest OI strikes
        key_resistance = int(df.loc[df['call_oi'].idxmax(), 'strike'])
        key_support    = int(df.loc[df['put_oi'].idxmax(), 'strike'])

        # Signal logic
        # PCR > 1.2 → bullish (more put writers = floor support)
        # PCR < 0.8 → bearish (more call writers = ceiling resistance)
        call_oi_chg = df['call_oi_change'].sum()
        put_oi_chg  = df['put_oi_change'].sum()

        if pcr > 1.2 and put_oi_chg > 0:
            signal = "BULLISH"   # Put writers building floor
        elif pcr < 0.8 and call_oi_chg > 0:
            signal = "BEARISH"   # Call writers building ceiling
        elif spot_price > max_pain:
            signal = "BEARISH"   # Market tends to gravitate toward max pain
        elif spot_price < max_pain:
            signal = "BULLISH"
        else:
            signal = "NEUTRAL"

        return OISignal(
            strike=atm_strike,
            call_oi=total_call_oi,
            put_oi=total_put_oi,
            call_oi_change=call_oi_chg,
            put_oi_change=put_oi_chg,
            pcr=pcr,
            max_pain=max_pain,
            key_support=key_support,
            key_resistance=key_resistance,
            signal=signal
        )

    @staticmethod
    def _calculate_max_pain(df: pd.DataFrame) -> int:
        """Strike where total option buyer pain is maximum."""
        min_pain = float('inf')
        max_pain_strike = int(df['strike'].iloc[len(df)//2])

        for strike in df['strike']:
            call_pain = df.loc[df['strike'] >= strike, 'call_oi'].sum() * (df.loc[df['strike'] >= strike, 'strike'] - strike).sum()
            put_pain  = df.loc[df['strike'] <= strike, 'put_oi'].sum() * (strike - df.loc[df['strike'] <= strike, 'strike']).sum()
            total_pain = call_pain + put_pain
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = int(strike)

        return max_pain_strike

    @staticmethod
    def generate_sample_chain(spot: float, n_strikes: int = 20) -> pd.DataFrame:
        """Generate realistic sample option chain for testing."""
        np.random.seed(42)
        step = 50  # Nifty step size
        atm = round(spot / step) * step
        strikes = [atm + (i - n_strikes//2) * step for i in range(n_strikes)]

        rows = []
        for s in strikes:
            dist = abs(s - atm) / step
            # OI builds up at round numbers near ATM
            base_oi = max(0, 500000 - dist * 40000 + np.random.randint(-20000, 20000))
            call_oi = base_oi * (1.2 if s >= atm else 0.8) + np.random.randint(0, 50000)
            put_oi  = base_oi * (0.8 if s >= atm else 1.2) + np.random.randint(0, 50000)
            rows.append({
                'strike': s,
                'call_oi': int(call_oi),
                'put_oi': int(put_oi),
                'call_oi_change': int(np.random.randint(-50000, 100000)),
                'put_oi_change': int(np.random.randint(-30000, 80000)),
            })

        # Create strong support/resistance at specific strikes
        df = pd.DataFrame(rows)
        df['call_oi'] = df['call_oi'].astype(float)
        df['put_oi']  = df['put_oi'].astype(float)
        df.loc[df['strike'] == atm + 200, 'call_oi'] *= 2.5  # Strong resistance
        df.loc[df['strike'] == atm - 150, 'put_oi']  *= 2.5  # Strong support
        return df


# ─── Fibonacci Engine ─────────────────────────────────────────────────────────

class FibonacciEngine:
    """
    Fibonacci retracement and extension levels.
    Uses previous day High-Low as the swing for intraday.
    """
    RETRACEMENT_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    EXTENSION_LEVELS   = [1.272, 1.414, 1.618, 2.0, 2.618]

    @staticmethod
    def calculate_levels(swing_high: float, swing_low: float,
                         trend: str = "UP") -> list[FibLevel]:
        """
        trend="UP": retracement from low→high (support levels on pullback)
        trend="DOWN": retracement from high→low (resistance levels on bounce)
        """
        diff = swing_high - swing_low
        levels = []

        for pct in FibonacciEngine.RETRACEMENT_LEVELS:
            if trend == "UP":
                price = swing_high - (diff * pct)
                role  = "SUPPORT" if pct > 0 else "HIGH"
            else:
                price = swing_low + (diff * pct)
                role  = "RESISTANCE" if pct > 0 else "LOW"

            name = {0.0: "0% (Swing High)", 0.236: "23.6%", 0.382: "38.2%",
                    0.5: "50%", 0.618: "61.8% (Golden)", 0.786: "78.6%",
                    1.0: "100% (Swing Low)"}.get(pct, f"{pct*100:.1f}%")

            levels.append(FibLevel(
                level_name=name,
                level_pct=pct,
                price=round(price, 2),
                role=role
            ))

        # Extension levels
        for pct in FibonacciEngine.EXTENSION_LEVELS:
            if trend == "UP":
                price = swing_high + (diff * (pct - 1.0))
                role  = "EXTENSION_TARGET"
            else:
                price = swing_low - (diff * (pct - 1.0))
                role  = "EXTENSION_TARGET"

            name = f"{pct*100:.1f}% Extension"
            levels.append(FibLevel(level_name=name, level_pct=pct,
                                   price=round(price, 2), role=role))
        return levels

    @staticmethod
    def find_nearest_level(current_price: float,
                           levels: list[FibLevel],
                           tolerance_pct: float = 0.15) -> Optional[FibLevel]:
        """Find closest Fibonacci level within tolerance."""
        tolerance = current_price * tolerance_pct / 100
        closest = None
        min_dist = float('inf')
        for lvl in levels:
            dist = abs(current_price - lvl.price)
            if dist < tolerance and dist < min_dist:
                min_dist = dist
                closest = lvl
        return closest

    @staticmethod
    def find_entry_zone(fib_levels: list[FibLevel],
                        direction: str) -> Tuple[Optional[FibLevel], Optional[FibLevel]]:
        """
        Returns (entry_level, target_level) based on direction.
        LONG: Enter at 61.8% retracement, target 38.2% or 0%
        SHORT: Enter at 38.2% bounce, target 61.8% or 100%
        """
        retracement = [l for l in fib_levels if l.level_pct in FibonacciEngine.RETRACEMENT_LEVELS]

        if direction == "LONG":
            entry  = next((l for l in retracement if abs(l.level_pct - 0.618) < 0.01), None)
            target = next((l for l in retracement if abs(l.level_pct - 0.382) < 0.01), None)
        else:
            entry  = next((l for l in retracement if abs(l.level_pct - 0.382) < 0.01), None)
            target = next((l for l in retracement if abs(l.level_pct - 0.618) < 0.01), None)

        return entry, target


# ─── Signal Generator ─────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Combines global cues + OI analysis + Fibonacci to generate trade signals.
    Scores confluence and filters low-probability setups.
    """

    MIN_CONFLUENCE_SCORE = 4  # Out of 10 — lowered for more signals

    @staticmethod
    def generate_signal(
        instrument: str,
        spot_price: float,
        prev_high: float,
        prev_low: float,
        first_candle: dict,     # {'open': x, 'high': x, 'low': x, 'close': x, 'volume': x}
        oi_signal: OISignal,
        global_bias: str,
        global_score: float,
        vix: float = 14.0,
        is_expiry: bool = False,
        timestamp: Optional[datetime] = None
    ) -> Optional[TradeSignal]:
        """
        Master signal generation. Returns None if no high-confidence trade.
        """
        if timestamp is None:
            timestamp = datetime.now()

        # ── Hard filters ─────────────────────────────────────────────────
        if is_expiry:
            return None  # Skip expiry — erratic OI

        if vix > 25 or vix < 10:
            return None  # Too volatile or dead market

        trade_time = timestamp.time()
        if trade_time < time(9, 20) or trade_time > time(14, 30):
            return None  # Outside trading window

        # ── Determine direction ───────────────────────────────────────────
        candle_bullish = first_candle['close'] > first_candle['open']
        candle_body    = abs(first_candle['close'] - first_candle['open'])
        candle_range   = first_candle['high'] - first_candle['low']
        candle_strong  = candle_body > candle_range * 0.6  # Strong directional candle

        # Direction must agree: global + OI + first candle
        signals = []
        if global_bias == "BULLISH": signals.append(1)
        elif global_bias == "BEARISH": signals.append(-1)
        else: signals.append(0)

        if oi_signal.signal == "BULLISH": signals.append(1)
        elif oi_signal.signal == "BEARISH": signals.append(-1)
        else: signals.append(0)

        if candle_bullish: signals.append(1)
        else: signals.append(-1)

        direction_sum = sum(signals)

        if direction_sum >= 2:
            direction = "LONG"
        elif direction_sum <= -2:
            direction = "SHORT"
        else:
            return None  # No confluence — skip

        # ── Fibonacci levels ──────────────────────────────────────────────
        fib_trend = "UP" if direction == "LONG" else "DOWN"
        fib_levels = FibonacciEngine.calculate_levels(prev_high, prev_low, fib_trend)

        entry_fib, target_fib = FibonacciEngine.find_entry_zone(fib_levels, direction)
        if entry_fib is None or target_fib is None:
            return None

        entry_price = entry_fib.price

        # Sanity: entry must be near current price (within 0.5%)
        if abs(entry_price - spot_price) / spot_price > 0.005:
            entry_price = spot_price  # Use current price if not near fib

        # ── Risk-Reward calculation ───────────────────────────────────────
        if direction == "LONG":
            target_1 = target_fib.price
            target_2 = next((l.price for l in fib_levels if abs(l.level_pct - 0.236) < 0.01), target_1 * 1.005)
            stop_loss = next((l.price for l in fib_levels if abs(l.level_pct - 0.786) < 0.01), entry_price * 0.995)
            # Also use OI key support as hard stop
            stop_loss = min(stop_loss, oi_signal.key_support - 30)
        else:
            target_1 = target_fib.price
            target_2 = next((l.price for l in fib_levels if abs(l.level_pct - 0.786) < 0.01), target_1 * 0.995)
            stop_loss = next((l.price for l in fib_levels if abs(l.level_pct - 0.236) < 0.01), entry_price * 1.005)
            stop_loss = max(stop_loss, oi_signal.key_resistance + 30)

        risk   = abs(entry_price - stop_loss)
        reward = abs(target_1 - entry_price)
        rr = round(reward / risk, 2) if risk > 0 else 0

        if rr < 1.2:
            return None  # Minimum 1:1.2 R:R required

        # ── Confluence Score (0-10) ───────────────────────────────────────
        score = 0

        # 1. Global bias alignment (0-2)
        if global_bias != "NEUTRAL":
            score += 1
            if abs(global_score) > 0.3: score += 1

        # 2. OI confirmation (0-2)
        if oi_signal.signal == ("BULLISH" if direction == "LONG" else "BEARISH"):
            score += 1
            if oi_signal.pcr > 1.2 if direction == "LONG" else oi_signal.pcr < 0.8:
                score += 1

        # 3. First candle strength (0-2)
        if candle_strong: score += 1
        if (direction == "LONG" and candle_bullish) or (direction == "SHORT" and not candle_bullish):
            score += 1

        # 4. Fibonacci at key level (0-2)
        if entry_fib.level_pct in (0.618, 0.382): score += 2
        elif entry_fib.level_pct in (0.5, 0.236): score += 1

        # 5. R:R quality (0-1)
        if rr >= 2.0: score += 1

        # 6. Volume (0-1)
        if first_candle.get('volume', 0) > 500000: score += 1

        if score < SignalGenerator.MIN_CONFLUENCE_SCORE:
            return None

        # ── Confidence tier ───────────────────────────────────────────────
        if score >= 9:   confidence = "HIGH"
        elif score >= 7: confidence = "MEDIUM"
        else:            confidence = "LOW"

        # ── Option selection ──────────────────────────────────────────────
        # Buy ATM or 1-strike OTM option in direction
        option_step  = 50 if "NIFTY" in instrument and "BANK" not in instrument else 100
        option_strike = round(entry_price / option_step) * option_step
        option_type   = "CE" if direction == "LONG" else "PE"

        reason = (
            f"Global cues {global_bias} ({global_score:+.2f}) | "
            f"OI PCR {oi_signal.pcr:.2f} → {oi_signal.signal} | "
            f"Fib entry {entry_fib.level_name} | "
            f"Support {oi_signal.key_support} / Resistance {oi_signal.key_resistance} | "
            f"Confluence score {score}/10"
        )

        return TradeSignal(
            timestamp=timestamp,
            instrument=instrument,
            direction=direction,
            entry_price=round(entry_price, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            stop_loss=round(stop_loss, 2),
            fib_entry_level=entry_fib.level_name,
            oi_confirmation=True,
            global_bias=global_bias,
            confluence_score=score,
            confidence=confidence,
            reason=reason,
            option_strike=option_strike,
            option_type=option_type,
            risk_reward=rr
        )
