"""
options_strategies_engine.py
Complete options strategies engine for India intraday trading app.
Pure Python — no scipy. Uses math.erf for Black-Scholes.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import datetime

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyLeg:
    action: str          # "BUY" or "SELL"
    option_type: str     # "CALL" or "PUT"
    strike_offset: int   # 0=ATM, +1=OTM by 1 step, -1=ITM by 1 step
    expiry: str          # "near" or "far"
    ratio: int = 1       # number of contracts relative to 1 base unit


@dataclass
class OptionsStrategy:
    name: str
    group: str
    legs: List[Dict[str, Any]]
    max_profit: str
    max_loss: str
    breakeven_formula: str
    ideal_market_view: str
    ideal_iv_view: str
    description: str
    risk_rating: int     # 1-5 (1=lowest risk)
    example: str = ""


@dataclass
class BacktestTrade:
    entry_date: str
    exit_date: str
    strategy: str
    entry_premium: float
    exit_premium: float
    pnl_per_lot: float
    total_pnl: float
    lot_size: int
    quantity: int
    exit_reason: str     # "expiry", "stop_loss", "target"


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES PRICER  (pure Python, no scipy)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Cumulative standard normal via math.erf."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "CALL") -> float:
    """
    Black-Scholes option pricer.
    S     = spot price
    K     = strike price
    T     = time to expiry in years
    r     = risk-free rate (e.g. 0.065 for 6.5%)
    sigma = implied volatility as decimal (e.g. 0.20 for 20%)
    """
    if T <= 0:
        if option_type.upper() == "CALL":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    if sigma <= 0:
        sigma = 0.0001

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.upper() == "CALL":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    return max(0.0, price)


def bs_iv_approx(option_price: float, S: float, K: float, T: float, r: float, option_type: str = "CALL") -> float:
    """Newton-Raphson IV solver (simplified)."""
    if T <= 0:
        return 0.20
    sigma = 0.20
    for _ in range(50):
        price = black_scholes(S, K, T, r, sigma, option_type)
        vega = S * math.sqrt(T) * math.exp(-0.5 * ((math.log(S / K) + (r + 0.5 * sigma ** 2) * T) /
               (sigma * math.sqrt(T))) ** 2) / math.sqrt(2 * math.pi)
        if vega < 1e-8:
            break
        diff = price - option_price
        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 5.0))
        if abs(diff) < 0.001:
            break
    return sigma


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def _build_strategies() -> Dict[str, OptionsStrategy]:
    s = {}

    # ── GROUP 1: DIRECTIONAL (VERTICAL) ──────────────────────────────────────

    s["Long Call"] = OptionsStrategy(
        name="Long Call", group="Directional",
        legs=[{"action": "BUY", "type": "CALL", "strike_offset": 0, "expiry": "near", "ratio": 1}],
        max_profit="Unlimited (as spot rises)",
        max_loss="Premium Paid",
        breakeven_formula="Strike + Premium Paid",
        ideal_market_view="Strongly Bullish",
        ideal_iv_view="Low IV (buy cheap options)",
        description="Buy an ATM Call option. Profits when spot rises above breakeven. "
                    "Limited loss (premium paid), unlimited upside.",
        risk_rating=2,
        example="NIFTY ATM=24500, Buy 24500CE @ ₹150. BEP=24650. Profit if NIFTY>24650 at expiry."
    )

    s["Long Put"] = OptionsStrategy(
        name="Long Put", group="Directional",
        legs=[{"action": "BUY", "type": "PUT", "strike_offset": 0, "expiry": "near", "ratio": 1}],
        max_profit="Unlimited (as spot falls)",
        max_loss="Premium Paid",
        breakeven_formula="Strike - Premium Paid",
        ideal_market_view="Strongly Bearish",
        ideal_iv_view="Low IV (buy cheap options)",
        description="Buy an ATM Put option. Profits when spot falls below breakeven. "
                    "Limited loss (premium paid), large downside capture.",
        risk_rating=2,
        example="NIFTY ATM=24500, Buy 24500PE @ ₹150. BEP=24350. Profit if NIFTY<24350 at expiry."
    )

    s["Short Call"] = OptionsStrategy(
        name="Short Call", group="Directional",
        legs=[{"action": "SELL", "type": "CALL", "strike_offset": 0, "expiry": "near", "ratio": 1}],
        max_profit="Premium Received",
        max_loss="Unlimited (theoretically)",
        breakeven_formula="Strike + Premium Received",
        ideal_market_view="Neutral to Bearish",
        ideal_iv_view="High IV (sell expensive options)",
        description="Sell an ATM Call (covered). Collect premium. Profits if spot stays below strike+premium. "
                    "Naked short call has unlimited risk — use only covered.",
        risk_rating=4,
        example="NIFTY ATM=24500, Sell 24500CE @ ₹150. BEP=24650. Profit if NIFTY<24650 at expiry."
    )

    s["Short Put"] = OptionsStrategy(
        name="Short Put", group="Directional",
        legs=[{"action": "SELL", "type": "PUT", "strike_offset": 0, "expiry": "near", "ratio": 1}],
        max_profit="Premium Received",
        max_loss="Strike - Premium (large but defined)",
        breakeven_formula="Strike - Premium Received",
        ideal_market_view="Neutral to Bullish",
        ideal_iv_view="High IV (sell expensive options)",
        description="Sell a cash-secured Put. Collect premium. Profits if spot stays above strike-premium. "
                    "Want to buy the underlying cheaper or earn income.",
        risk_rating=3,
        example="NIFTY ATM=24500, Sell 24500PE @ ₹150. BEP=24350. Profit if NIFTY>24350 at expiry."
    )

    s["Bull Call Spread"] = OptionsStrategy(
        name="Bull Call Spread", group="Directional",
        legs=[
            {"action": "BUY",  "type": "CALL", "strike_offset": 0,  "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "CALL", "strike_offset": 1,  "expiry": "near", "ratio": 1},
        ],
        max_profit="Width of Spread - Net Debit",
        max_loss="Net Debit Paid",
        breakeven_formula="Lower Strike + Net Debit",
        ideal_market_view="Moderately Bullish",
        ideal_iv_view="Low to Moderate IV",
        description="Buy lower strike Call, Sell higher strike Call. Defined risk/reward. "
                    "Profits moderately when spot rises. Cheaper than outright long call.",
        risk_rating=2,
        example="Buy 24500CE @ ₹150, Sell 24600CE @ ₹80. Net Debit=₹70. Max Profit=₹30 (spread 100-debit 70). BEP=24570."
    )

    s["Bear Put Spread"] = OptionsStrategy(
        name="Bear Put Spread", group="Directional",
        legs=[
            {"action": "BUY",  "type": "PUT", "strike_offset": 0,  "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "PUT", "strike_offset": -1, "expiry": "near", "ratio": 1},
        ],
        max_profit="Width of Spread - Net Debit",
        max_loss="Net Debit Paid",
        breakeven_formula="Higher Strike - Net Debit",
        ideal_market_view="Moderately Bearish",
        ideal_iv_view="Low to Moderate IV",
        description="Buy higher strike Put, Sell lower strike Put. Defined risk/reward for bearish view. "
                    "Cheaper than outright long put.",
        risk_rating=2,
        example="Buy 24500PE @ ₹150, Sell 24400PE @ ₹80. Net Debit=₹70. Max Profit=₹30. BEP=24430."
    )

    s["Bull Put Spread"] = OptionsStrategy(
        name="Bull Put Spread", group="Directional",
        legs=[
            {"action": "SELL", "type": "PUT", "strike_offset": 0,  "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "PUT", "strike_offset": -1, "expiry": "near", "ratio": 1},
        ],
        max_profit="Net Credit Received",
        max_loss="Width of Spread - Net Credit",
        breakeven_formula="Short Strike - Net Credit",
        ideal_market_view="Neutral to Bullish",
        ideal_iv_view="High IV (sell premium)",
        description="Sell higher strike Put, Buy lower strike Put for protection. Credit strategy. "
                    "Profits if spot stays above short strike. Popular income strategy.",
        risk_rating=2,
        example="Sell 24500PE @ ₹150, Buy 24400PE @ ₹80. Credit=₹70. Max Loss=₹30. BEP=24430."
    )

    s["Bear Call Spread"] = OptionsStrategy(
        name="Bear Call Spread", group="Directional",
        legs=[
            {"action": "SELL", "type": "CALL", "strike_offset": 0, "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "CALL", "strike_offset": 1, "expiry": "near", "ratio": 1},
        ],
        max_profit="Net Credit Received",
        max_loss="Width of Spread - Net Credit",
        breakeven_formula="Short Strike + Net Credit",
        ideal_market_view="Neutral to Bearish",
        ideal_iv_view="High IV (sell premium)",
        description="Sell lower strike Call, Buy higher strike Call for protection. Credit strategy. "
                    "Profits if spot stays below short call strike.",
        risk_rating=2,
        example="Sell 24500CE @ ₹150, Buy 24600CE @ ₹80. Credit=₹70. Max Loss=₹30. BEP=24570."
    )

    # ── GROUP 2: INCOME & RANGE (THETA) ──────────────────────────────────────

    s["Short Straddle"] = OptionsStrategy(
        name="Short Straddle", group="Income & Range",
        legs=[
            {"action": "SELL", "type": "CALL", "strike_offset": 0, "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "PUT",  "strike_offset": 0, "expiry": "near", "ratio": 1},
        ],
        max_profit="Total Premium Received (CE + PE)",
        max_loss="Unlimited both sides",
        breakeven_formula="Upper: Strike + Total Premium | Lower: Strike - Total Premium",
        ideal_market_view="Strictly Sideways",
        ideal_iv_view="High IV (IV crush expected)",
        description="Sell ATM Call and ATM Put. Maximum theta decay. Profits when spot stays "
                    "near strike. High risk — unlimited loss on big moves.",
        risk_rating=5,
        example="Sell 24500CE @ ₹150 + Sell 24500PE @ ₹150. Total Credit=₹300. BEP: 24200 / 24800."
    )

    s["Short Strangle"] = OptionsStrategy(
        name="Short Strangle", group="Income & Range",
        legs=[
            {"action": "SELL", "type": "CALL", "strike_offset": 1, "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "PUT",  "strike_offset": -1,"expiry": "near", "ratio": 1},
        ],
        max_profit="Total OTM Premium Received",
        max_loss="Unlimited both sides",
        breakeven_formula="Upper: Call Strike + Total Premium | Lower: Put Strike - Total Premium",
        ideal_market_view="Sideways with Wide Range",
        ideal_iv_view="High IV (IV crush expected)",
        description="Sell OTM Call and OTM Put. Wider profit zone than straddle but lower credit. "
                    "Still unlimited risk on large moves.",
        risk_rating=4,
        example="Sell 24600CE @ ₹80 + Sell 24400PE @ ₹80. Credit=₹160. BEP: 24240 / 24760."
    )

    s["Iron Condor"] = OptionsStrategy(
        name="Iron Condor", group="Income & Range",
        legs=[
            {"action": "SELL", "type": "PUT",  "strike_offset": -1, "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "PUT",  "strike_offset": -2, "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "CALL", "strike_offset": 1,  "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "CALL", "strike_offset": 2,  "expiry": "near", "ratio": 1},
        ],
        max_profit="Net Credit Received",
        max_loss="Width of Wing - Net Credit",
        breakeven_formula="Put BEP: Short Put - Credit | Call BEP: Short Call + Credit",
        ideal_market_view="Sideways / Range Bound",
        ideal_iv_view="High IV (collect premium, expect IV crush)",
        description="Sell OTM Put Spread + Sell OTM Call Spread. Four legs. Defined risk on both sides. "
                    "The most popular index selling strategy in India.",
        risk_rating=2,
        example="Sell 24400PE, Buy 24300PE, Sell 24600CE, Buy 24700CE. Net Credit ≈₹120. Max Loss ≈₹80."
    )

    s["Iron Butterfly"] = OptionsStrategy(
        name="Iron Butterfly", group="Income & Range",
        legs=[
            {"action": "BUY",  "type": "PUT",  "strike_offset": -1, "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "PUT",  "strike_offset": 0,  "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "CALL", "strike_offset": 0,  "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "CALL", "strike_offset": 1,  "expiry": "near", "ratio": 1},
        ],
        max_profit="Net Credit (body strikes sold ATM)",
        max_loss="Wing Width - Net Credit",
        breakeven_formula="ATM Strike ± Net Credit",
        ideal_market_view="Pinning at ATM",
        ideal_iv_view="Very High IV (maximum theta)",
        description="Sell ATM Straddle + Buy OTM wings for protection. Maximum profit if spot pins at ATM. "
                    "Defined risk unlike naked straddle.",
        risk_rating=3,
        example="Buy 24400PE, Sell 24500PE+CE, Buy 24600CE. Body credit higher than wing cost."
    )

    s["Jade Lizard"] = OptionsStrategy(
        name="Jade Lizard", group="Income & Range",
        legs=[
            {"action": "SELL", "type": "PUT",  "strike_offset": -1, "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "CALL", "strike_offset": 1,  "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "CALL", "strike_offset": 2,  "expiry": "near", "ratio": 1},
        ],
        max_profit="Total Net Credit",
        max_loss="Put Strike - Net Credit (downside) | Capped on upside",
        breakeven_formula="Short Put - Total Credit (downside) | No upside risk",
        ideal_market_view="Neutral to Bullish",
        ideal_iv_view="High IV",
        description="Sell OTM Put + Sell OTM Call Spread. No upside risk (credit > call spread width). "
                    "Only downside risk. Bullish bias with income.",
        risk_rating=3,
        example="Sell 24400PE@₹80, Sell 24600CE@₹80, Buy 24700CE@₹40. Total credit=₹120. No upside risk."
    )

    s["Long Straddle"] = OptionsStrategy(
        name="Long Straddle", group="Income & Range",
        legs=[
            {"action": "BUY", "type": "CALL", "strike_offset": 0, "expiry": "near", "ratio": 1},
            {"action": "BUY", "type": "PUT",  "strike_offset": 0, "expiry": "near", "ratio": 1},
        ],
        max_profit="Unlimited (either direction)",
        max_loss="Total Premium Paid (CE + PE)",
        breakeven_formula="Upper: Strike + Total Premium | Lower: Strike - Total Premium",
        ideal_market_view="Big Move Expected (direction unknown)",
        ideal_iv_view="Low IV before event (buy cheap, sell after IV spike)",
        description="Buy ATM Call + Buy ATM Put. Profits on any large move. Ideal before events. "
                    "Theta decay is the enemy — time it well.",
        risk_rating=3,
        example="Buy 24500CE@₹150 + 24500PE@₹150. Total Cost=₹300. BEP: 24200 / 24800."
    )

    s["Long Strangle"] = OptionsStrategy(
        name="Long Strangle", group="Income & Range",
        legs=[
            {"action": "BUY", "type": "CALL", "strike_offset": 1,  "expiry": "near", "ratio": 1},
            {"action": "BUY", "type": "PUT",  "strike_offset": -1, "expiry": "near", "ratio": 1},
        ],
        max_profit="Unlimited (either direction)",
        max_loss="Total OTM Premium Paid",
        breakeven_formula="Upper: Call Strike + Premium | Lower: Put Strike - Premium",
        ideal_market_view="Big Move Expected (direction unknown)",
        ideal_iv_view="Low IV (cheaper than straddle)",
        description="Buy OTM Call + Buy OTM Put. Cheaper than straddle but needs larger move to profit. "
                    "Good before volatile events.",
        risk_rating=3,
        example="Buy 24600CE@₹80 + 24400PE@₹80. Total Cost=₹160. BEP: 24240 / 24760."
    )

    # ── GROUP 3: PINNING (BUTTERFLIES) ───────────────────────────────────────

    s["Long Call Butterfly"] = OptionsStrategy(
        name="Long Call Butterfly", group="Pinning",
        legs=[
            {"action": "BUY",  "type": "CALL", "strike_offset": -1, "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "CALL", "strike_offset": 0,  "expiry": "near", "ratio": 2},
            {"action": "BUY",  "type": "CALL", "strike_offset": 1,  "expiry": "near", "ratio": 1},
        ],
        max_profit="Width - Net Debit (at middle strike)",
        max_loss="Net Debit Paid",
        breakeven_formula="Lower Strike + Net Debit | Upper Strike - Net Debit",
        ideal_market_view="Pinning at Middle Strike",
        ideal_iv_view="Low IV (cheap wings) or IV expected to fall",
        description="Buy lower Call, Sell 2x middle Call, Buy upper Call. Profits if spot pins "
                    "near middle strike at expiry. Very cheap strategy.",
        risk_rating=1,
        example="Buy 24400CE@₹200, Sell 2x24500CE@₹150, Buy 24600CE@₹100. Net Cost=₹0. Max Profit=₹100."
    )

    s["Long Put Butterfly"] = OptionsStrategy(
        name="Long Put Butterfly", group="Pinning",
        legs=[
            {"action": "BUY",  "type": "PUT", "strike_offset": 1,  "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "PUT", "strike_offset": 0,  "expiry": "near", "ratio": 2},
            {"action": "BUY",  "type": "PUT", "strike_offset": -1, "expiry": "near", "ratio": 1},
        ],
        max_profit="Width - Net Debit (at middle strike)",
        max_loss="Net Debit Paid",
        breakeven_formula="Upper Strike - Net Debit | Lower Strike + Net Debit",
        ideal_market_view="Pinning at Middle Strike",
        ideal_iv_view="Low IV",
        description="Buy upper Put, Sell 2x middle Put, Buy lower Put. Put version of butterfly. "
                    "Profits when spot pins at middle strike.",
        risk_rating=1,
        example="Buy 24600PE@₹200, Sell 2x24500PE@₹150, Buy 24400PE@₹100. Net Cost=₹0. Max Profit=₹100."
    )

    s["Call Condor"] = OptionsStrategy(
        name="Call Condor", group="Pinning",
        legs=[
            {"action": "BUY",  "type": "CALL", "strike_offset": -1, "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "CALL", "strike_offset": 0,  "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "CALL", "strike_offset": 1,  "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "CALL", "strike_offset": 2,  "expiry": "near", "ratio": 1},
        ],
        max_profit="Inner Width - Net Debit",
        max_loss="Net Debit Paid",
        breakeven_formula="Lowest Strike + Net Debit | Highest Strike - Net Debit",
        ideal_market_view="Moderate Range (wider than butterfly)",
        ideal_iv_view="Low to Moderate IV",
        description="Buy 1 lower Call, Sell 1 call, Sell 1 call, Buy 1 upper Call (4 legs). "
                    "Wider profit zone than butterfly. All calls same expiry.",
        risk_rating=2,
        example="Buy 24400CE, Sell 24500CE, Sell 24600CE, Buy 24700CE. Net Debit small."
    )

    s["Put Condor"] = OptionsStrategy(
        name="Put Condor", group="Pinning",
        legs=[
            {"action": "BUY",  "type": "PUT", "strike_offset": 2,  "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "PUT", "strike_offset": 1,  "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "PUT", "strike_offset": 0,  "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "PUT", "strike_offset": -1, "expiry": "near", "ratio": 1},
        ],
        max_profit="Inner Width - Net Debit",
        max_loss="Net Debit Paid",
        breakeven_formula="Highest Strike - Net Debit | Lowest Strike + Net Debit",
        ideal_market_view="Moderate Range",
        ideal_iv_view="Low to Moderate IV",
        description="Put version of Call Condor. Four put legs spanning same range. "
                    "Profits in a moderate range around current spot.",
        risk_rating=2,
        example="Buy 24700PE, Sell 24600PE, Sell 24500PE, Buy 24400PE. Net Debit small."
    )

    # ── GROUP 4: RATIO (VOLATILITY) ───────────────────────────────────────────

    s["Call Ratio Back Spread"] = OptionsStrategy(
        name="Call Ratio Back Spread", group="Ratio / Volatility",
        legs=[
            {"action": "SELL", "type": "CALL", "strike_offset": 0, "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "CALL", "strike_offset": 1, "expiry": "near", "ratio": 2},
        ],
        max_profit="Unlimited above upper breakeven",
        max_loss="Width between strikes - Net Credit (or + Net Debit)",
        breakeven_formula="Upper: Short Strike + (Width + Net Debit) | Lower: Short Strike",
        ideal_market_view="Volatile / Big UP move expected",
        ideal_iv_view="Low IV (buy 2 OTM calls cheap)",
        description="Sell 1 ATM Call, Buy 2 OTM Calls. Net debit or credit. Profits hugely on "
                    "large upside. Small loss in range. Good black-swan protection.",
        risk_rating=3,
        example="Sell 1x24500CE@₹150, Buy 2x24600CE@₹80. Net Credit=₹-10. Max Loss=₹90 (near 24600)."
    )

    s["Put Ratio Back Spread"] = OptionsStrategy(
        name="Put Ratio Back Spread", group="Ratio / Volatility",
        legs=[
            {"action": "SELL", "type": "PUT", "strike_offset": 0,  "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "PUT", "strike_offset": -1, "expiry": "near", "ratio": 2},
        ],
        max_profit="Unlimited below lower breakeven",
        max_loss="Width between strikes - Net Credit",
        breakeven_formula="Lower: Short Strike - (Width + Net Debit) | Upper: Short Strike",
        ideal_market_view="Volatile / Big DOWN move expected",
        ideal_iv_view="Low IV (buy 2 OTM puts cheap)",
        description="Sell 1 ATM Put, Buy 2 OTM Puts. Profits hugely on large downside. "
                    "Black-swan / crash protection strategy.",
        risk_rating=3,
        example="Sell 1x24500PE@₹150, Buy 2x24400PE@₹80. Net Credit=₹-10. Max Loss=₹90 (near 24400)."
    )

    s["Call Ratio Spread"] = OptionsStrategy(
        name="Call Ratio Spread", group="Ratio / Volatility",
        legs=[
            {"action": "BUY",  "type": "CALL", "strike_offset": 0, "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "CALL", "strike_offset": 1, "expiry": "near", "ratio": 2},
        ],
        max_profit="Width of strikes + Net Credit",
        max_loss="Unlimited above upper breakeven",
        breakeven_formula="Lower Strike + Net Debit | Upper: Short Strike × 2 - Long Strike + Net Debit",
        ideal_market_view="Mildly Bullish (but NOT strongly bullish)",
        ideal_iv_view="High IV on OTM calls to sell",
        description="Buy 1 ATM Call, Sell 2 OTM Calls. Profits from moderate rise. "
                    "Unlimited risk if spot rockets. Use cautiously.",
        risk_rating=4,
        example="Buy 1x24500CE@₹150, Sell 2x24600CE@₹80. Net Credit=₹10. Max Profit at 24600."
    )

    s["Put Ratio Spread"] = OptionsStrategy(
        name="Put Ratio Spread", group="Ratio / Volatility",
        legs=[
            {"action": "BUY",  "type": "PUT", "strike_offset": 0,  "expiry": "near", "ratio": 1},
            {"action": "SELL", "type": "PUT", "strike_offset": -1, "expiry": "near", "ratio": 2},
        ],
        max_profit="Width of strikes + Net Credit",
        max_loss="Unlimited below lower breakeven",
        breakeven_formula="Upper Strike - Net Debit | Lower: Short Strike × 2 - Long Strike - Net Debit",
        ideal_market_view="Mildly Bearish (but NOT strongly bearish)",
        ideal_iv_view="High IV on OTM puts to sell",
        description="Buy 1 ATM Put, Sell 2 OTM Puts. Profits from moderate fall. "
                    "Unlimited risk if spot crashes. Use cautiously.",
        risk_rating=4,
        example="Buy 1x24500PE@₹150, Sell 2x24400PE@₹80. Net Credit=₹10. Max Profit at 24400."
    )

    # ── GROUP 5: TIME (CALENDAR) ──────────────────────────────────────────────

    s["Call Calendar Spread"] = OptionsStrategy(
        name="Call Calendar Spread", group="Time / Calendar",
        legs=[
            {"action": "SELL", "type": "CALL", "strike_offset": 0, "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "CALL", "strike_offset": 0, "expiry": "far",  "ratio": 1},
        ],
        max_profit="When near-term expires worthless and far-term retains value",
        max_loss="Net Debit Paid (far - near premium)",
        breakeven_formula="Approximate: Strike ± Width of profit zone (depends on IV)",
        ideal_market_view="Neutral to Slightly Bullish (near-term sideways, longer bullish)",
        ideal_iv_view="Low near-term IV, High far-term IV (or flat)",
        description="Sell near-month Call, Buy far-month Call at same strike. "
                    "Exploit time decay differential. Profits when spot pins at strike near expiry.",
        risk_rating=2,
        example="Sell May 24500CE@₹150, Buy Jun 24500CE@₹220. Net Debit=₹70. Profit if May expires worthless."
    )

    s["Put Calendar Spread"] = OptionsStrategy(
        name="Put Calendar Spread", group="Time / Calendar",
        legs=[
            {"action": "SELL", "type": "PUT", "strike_offset": 0, "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "PUT", "strike_offset": 0, "expiry": "far",  "ratio": 1},
        ],
        max_profit="When near-term expires worthless and far-term retains value",
        max_loss="Net Debit Paid",
        breakeven_formula="Approximate: Strike ± Width of profit zone",
        ideal_market_view="Neutral to Slightly Bearish (near-term sideways)",
        ideal_iv_view="Low near-term IV, High far-term IV",
        description="Sell near-month Put, Buy far-month Put at same strike. "
                    "Time decay strategy for mildly bearish outlook.",
        risk_rating=2,
        example="Sell May 24500PE@₹150, Buy Jun 24500PE@₹220. Net Debit=₹70."
    )

    s["Diagonal Spread"] = OptionsStrategy(
        name="Diagonal Spread", group="Time / Calendar",
        legs=[
            {"action": "SELL", "type": "CALL", "strike_offset": 1, "expiry": "near", "ratio": 1},
            {"action": "BUY",  "type": "CALL", "strike_offset": 0, "expiry": "far",  "ratio": 1},
        ],
        max_profit="When near OTM expires worthless, far ITM retained",
        max_loss="Net Debit Paid",
        breakeven_formula="Depends on strikes and expiry gap",
        ideal_market_view="Mildly Bullish (near-term range, longer bullish)",
        ideal_iv_view="High near-term IV (sell expensive), Low far-term IV (buy cheap)",
        description="Sell near-month OTM Call, Buy far-month ATM Call. Diagonal = different strike + expiry. "
                    "Best of calendar + vertical spread.",
        risk_rating=2,
        example="Sell May 24600CE@₹80, Buy Jun 24500CE@₹220. Net Debit=₹140."
    )

    return s


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class StrategyEngine:
    """Main engine for options strategies — simulation, backtesting, decision matrix."""

    LOT_SIZES = {"NIFTY": 50, "BANKNIFTY": 15, "SENSEX": 20}
    STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100, "SENSEX": 100}
    RISK_FREE_RATE = 0.065   # 6.5% India

    def __init__(self):
        self._strategies = _build_strategies()

    # ── Strategy Retrieval ────────────────────────────────────────────────────

    def all_strategy_names(self) -> List[str]:
        return list(self._strategies.keys())

    def all_strategies(self) -> Dict[str, OptionsStrategy]:
        return self._strategies

    def get_strategy(self, name: str) -> Optional[OptionsStrategy]:
        return self._strategies.get(name)

    def strategies_by_group(self) -> Dict[str, List[OptionsStrategy]]:
        groups: Dict[str, List[OptionsStrategy]] = {}
        for s in self._strategies.values():
            groups.setdefault(s.group, []).append(s)
        return groups

    # ── PnL Simulation ───────────────────────────────────────────────────────

    def simulate_pnl(
        self,
        strategy,          # OptionsStrategy object OR strategy name string
        spot: float,
        strike: float,
        premium_ce: float,
        premium_pe: float,
        lot_size: int,
        quantity: int,
        strike_step: float = 50.0,
        iv: float = 0.15,
        dte: int = 7,
    ) -> Dict[str, Any]:
        """
        Simulate PnL at expiry across a range of spot prices.
        Returns dict with:
          - spot_range: list of spot prices
          - pnl: list of PnL values in ₹ (per lot_size * quantity)
          - max_profit: float
          - max_loss: float
          - breakevens: list of approximate breakeven spot prices
          - legs_detail: list of per-leg info
        """
        if isinstance(strategy, str):
            strategy = self.get_strategy(strategy)

        # Build spot range: ±15% of current spot in 100 steps
        width = spot * 0.15
        spot_range = [spot - width + (2 * width / 200) * i for i in range(201)]

        # Resolve actual strikes for each leg
        leg_strikes = []
        for leg in strategy.legs:
            offset = leg.get("strike_offset", 0)
            if leg.get("type") == "CALL":
                lk = strike + offset * strike_step
            else:
                lk = strike - offset * strike_step  # negative offset means lower strike for puts
                # Normalise: for put offset convention
                lk = strike + offset * strike_step
            lk = round(lk / strike_step) * strike_step
            leg_strikes.append(max(lk, strike_step))

        # PnL at expiry = intrinsic value difference × lots
        total_lots = lot_size * quantity
        pnl_curve = []
        for s_exp in spot_range:
            total_pnl_pt = 0.0
            for i, leg in enumerate(strategy.legs):
                lk = leg_strikes[i]
                action = leg.get("action", "BUY")
                otype = leg.get("type", "CALL")
                ratio = leg.get("ratio", 1)
                # Use market premium for first leg, approximate others
                if i == 0:
                    prem = premium_ce if otype == "CALL" else premium_pe
                elif i == 1:
                    prem = premium_pe if otype == "PUT" else premium_ce
                else:
                    # Approximate via BS
                    T = max(dte / 365.0, 0.001)
                    prem = black_scholes(spot, lk, T, self.RISK_FREE_RATE, iv, otype)

                intrinsic = max(0.0, s_exp - lk) if otype == "CALL" else max(0.0, lk - s_exp)
                if action == "BUY":
                    leg_pnl = (intrinsic - prem) * ratio
                else:
                    leg_pnl = (prem - intrinsic) * ratio
                total_pnl_pt += leg_pnl
            pnl_curve.append(total_pnl_pt * total_lots)

        # Find breakevens (sign changes)
        breakevens = []
        for i in range(len(pnl_curve) - 1):
            if pnl_curve[i] * pnl_curve[i + 1] < 0:
                # Linear interpolation
                bep = spot_range[i] + (spot_range[i + 1] - spot_range[i]) * (
                    -pnl_curve[i] / (pnl_curve[i + 1] - pnl_curve[i])
                )
                breakevens.append(round(bep, 2))

        max_profit = max(pnl_curve)
        max_loss = min(pnl_curve)

        # Legs detail table
        T_entry = max(dte / 365.0, 0.001)
        legs_detail = []
        for i, leg in enumerate(strategy.legs):
            lk = leg_strikes[i]
            otype = leg.get("type", "CALL")
            action = leg.get("action", "BUY")
            ratio = leg.get("ratio", 1)
            if i == 0:
                prem = premium_ce if otype == "CALL" else premium_pe
            elif i == 1:
                prem = premium_pe if otype == "PUT" else premium_ce
            else:
                prem = black_scholes(spot, lk, T_entry, self.RISK_FREE_RATE, iv, otype)
            legs_detail.append({
                "leg": i + 1,
                "action": action,
                "type": otype,
                "strike": lk,
                "premium": round(prem, 2),
                "ratio": ratio,
                "expiry": leg.get("expiry", "near"),
            })

        return {
            "spot_range": spot_range,
            "pnl": pnl_curve,
            "max_profit": round(max_profit, 2),
            "max_loss": round(max_loss, 2),
            "breakevens": breakevens,
            "legs_detail": legs_detail,
        }

    # ── Backtesting ──────────────────────────────────────────────────────────

    def backtest_strategy(
        self,
        strategy,              # OptionsStrategy object OR strategy name string
        df,                    # pandas DataFrame: Date, Open, High, Low, Close, Volume
        lot_size: int,
        quantity: int,
        days_to_expiry: int = 7,
        iv: float = 0.18,
    ) -> List[BacktestTrade]:
        """
        Simple backtest: enter at open each Monday (or first day of week),
        exit at expiry (days_to_expiry later) or stop-loss / target.
        Uses Black-Scholes for option pricing.
        """
        if isinstance(strategy, str):
            strategy = self.get_strategy(strategy)
        import pandas as pd
        trades: List[BacktestTrade] = []

        try:
            df = df.reset_index(drop=False)
            if "Date" not in df.columns and "index" in df.columns:
                df = df.rename(columns={"index": "Date"})
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
        except Exception:
            return trades

        total_lots = lot_size * quantity
        r = self.RISK_FREE_RATE
        strike_step = 50.0  # default NIFTY

        i = 0
        while i < len(df) - days_to_expiry:
            row = df.iloc[i]
            spot = float(row.get("Close", row.get("close", 100)))
            open_price = float(row.get("Open", row.get("open", spot)))
            entry_date = str(row["Date"])[:10]

            # Only enter weekly (every 5th row approx)
            if i % 5 != 0:
                i += 1
                continue

            # ATM strike
            atm_strike = round(spot / strike_step) * strike_step
            T_entry = days_to_expiry / 365.0

            # Compute net premium for the strategy
            entry_prem = 0.0
            for leg in strategy.legs:
                lk = atm_strike + leg.get("strike_offset", 0) * strike_step
                otype = leg.get("type", "CALL")
                action = leg.get("action", "BUY")
                ratio = leg.get("ratio", 1)
                p = black_scholes(spot, lk, T_entry, r, iv, otype)
                if action == "BUY":
                    entry_prem -= p * ratio     # debit
                else:
                    entry_prem += p * ratio     # credit

            # Exit at expiry row
            exit_idx = min(i + days_to_expiry, len(df) - 1)
            exit_row = df.iloc[exit_idx]
            exit_spot = float(exit_row.get("Close", exit_row.get("close", spot)))
            exit_date = str(exit_row["Date"])[:10]

            # PnL at expiry (intrinsic)
            exit_prem = 0.0
            for leg in strategy.legs:
                lk = atm_strike + leg.get("strike_offset", 0) * strike_step
                otype = leg.get("type", "CALL")
                action = leg.get("action", "BUY")
                ratio = leg.get("ratio", 1)
                intrinsic = max(0.0, exit_spot - lk) if otype == "CALL" else max(0.0, lk - exit_spot)
                if action == "BUY":
                    exit_prem -= intrinsic * ratio
                else:
                    exit_prem += intrinsic * ratio

            # Check stop loss / target during the period
            exit_reason = "expiry"
            daily_pnl_check = entry_prem
            for j in range(i + 1, exit_idx):
                mid_row = df.iloc[j]
                mid_spot = float(mid_row.get("Close", mid_row.get("close", spot)))
                T_mid = max((days_to_expiry - (j - i)) / 365.0, 0.001)
                mid_prem = 0.0
                for leg in strategy.legs:
                    lk = atm_strike + leg.get("strike_offset", 0) * strike_step
                    otype = leg.get("type", "CALL")
                    action = leg.get("action", "BUY")
                    ratio = leg.get("ratio", 1)
                    p = black_scholes(mid_spot, lk, T_mid, r, iv, otype)
                    if action == "BUY":
                        mid_prem -= p * ratio
                    else:
                        mid_prem += p * ratio

                # Credit strategy SL: 50% of credit collected
                if entry_prem > 0:  # credit received
                    if mid_prem < -entry_prem * 0.5:
                        exit_prem = mid_prem
                        exit_date = str(mid_row["Date"])[:10]
                        exit_reason = "stop_loss"
                        break
                    if mid_prem > entry_prem * 0.75:
                        exit_prem = mid_prem
                        exit_date = str(mid_row["Date"])[:10]
                        exit_reason = "target"
                        break
                else:  # debit paid
                    if mid_prem < entry_prem * 2.0:  # 2x loss
                        exit_prem = mid_prem
                        exit_date = str(mid_row["Date"])[:10]
                        exit_reason = "stop_loss"
                        break
                    if mid_prem > abs(entry_prem) * 0.8:  # 80% gain
                        exit_prem = mid_prem
                        exit_date = str(mid_row["Date"])[:10]
                        exit_reason = "target"
                        break

            pnl_per_lot = (exit_prem - entry_prem) * lot_size if entry_prem <= 0 else (exit_prem) * lot_size
            # Simpler: PnL = change in net premium × lot_size
            pnl_per_lot = (exit_prem - entry_prem) * lot_size
            total_pnl = pnl_per_lot * quantity

            trades.append(BacktestTrade(
                entry_date=entry_date,
                exit_date=exit_date,
                strategy=strategy.name,
                entry_premium=round(entry_prem, 2),
                exit_premium=round(exit_prem, 2),
                pnl_per_lot=round(pnl_per_lot, 2),
                total_pnl=round(total_pnl, 2),
                lot_size=lot_size,
                quantity=quantity,
                exit_reason=exit_reason,
            ))

            i += days_to_expiry  # next cycle

        return trades

    # ── Decision Matrix ───────────────────────────────────────────────────────

    def get_decision_matrix(
        self,
        trend: str,           # "UP", "DOWN", "SIDEWAYS"
        iv_percentile: float, # 0-100
        days_to_event: int,   # 0 = event today, 30 = no event soon
    ) -> List[Dict[str, Any]]:
        """
        Returns ranked list of recommended strategies with explanation.
        Each item: {name, score, reason, strategy obj}
        """
        scores: Dict[str, float] = {n: 0.0 for n in self._strategies}
        reasons: Dict[str, str] = {n: "" for n in self._strategies}

        iv_high = iv_percentile >= 60
        iv_low  = iv_percentile <= 40
        iv_extreme_high = iv_percentile >= 80
        event_near = days_to_event <= 3
        event_tomorrow = days_to_event <= 1

        t_norm     = trend.strip().upper()
        trend_up   = t_norm in ("UP", "BULLISH")
        trend_dn   = t_norm in ("DOWN", "BEARISH")
        trend_side = t_norm in ("SIDEWAYS", "NEUTRAL", "RANGE")

        def add(name, pts, reason):
            if name in scores:
                scores[name] += pts
                if pts > 0:
                    reasons[name] = reason if not reasons[name] else reasons[name] + " | " + reason

        # Event near → volatility plays
        if event_tomorrow:
            add("Long Straddle", 10, "Event tomorrow → buy vol")
            add("Long Strangle", 9,  "Event tomorrow → buy vol (cheaper)")
            add("Call Calendar Spread", 5, "Event → calendar for IV expansion")

        if event_near and trend_up:
            add("Call Calendar Spread", 7, "Bullish event → call calendar")
            add("Long Call", 5, "Bullish event → directional")

        # Black swan / extreme fear
        if iv_extreme_high and not trend_up:
            add("Put Ratio Back Spread", 6, "High IV + bearish → backspread")
            add("Call Ratio Back Spread", 4, "High IV → hedge")

        # Trend UP
        if trend_up and iv_low:
            add("Bull Call Spread", 9, "Bullish + Low IV → debit spread")
            add("Long Call", 8, "Bullish + Low IV → buy call")
            add("Bull Put Spread", 6, "Bullish + Low IV → credit put spread")

        if trend_up and iv_high:
            add("Bull Put Spread", 10, "Bullish + High IV → sell puts")
            add("Jade Lizard", 9,  "Bullish + High IV → jade lizard")
            add("Short Put", 7,    "Bullish + High IV → sell put")
            add("Bear Call Spread", 3, "Hedge: sell OTM calls")

        if trend_up and not iv_low and not iv_high:
            add("Bull Call Spread", 7, "Bullish moderate IV")
            add("Bull Put Spread", 7,  "Bullish moderate IV")

        # Trend DOWN
        if trend_dn and iv_low:
            add("Bear Put Spread", 9, "Bearish + Low IV → debit spread")
            add("Long Put", 8,        "Bearish + Low IV → buy put")

        if trend_dn and iv_high:
            add("Bear Call Spread", 10, "Bearish + High IV → sell calls")
            add("Short Call", 6,        "Bearish + High IV → sell call")
            add("Put Ratio Back Spread", 5, "Bearish + High IV → backspread")

        if trend_dn and not iv_low and not iv_high:
            add("Bear Put Spread", 7, "Bearish moderate IV")
            add("Bear Call Spread", 7, "Bearish moderate IV")

        # SIDEWAYS
        if trend_side and iv_high:
            add("Iron Condor", 10,     "Sideways + High IV → condor")
            add("Short Strangle", 9,   "Sideways + High IV → strangle")
            add("Short Straddle", 8,   "Sideways + Very High IV → straddle")
            add("Iron Butterfly", 7,   "Sideways + High IV → iron fly")
            add("Jade Lizard", 6,      "Sideways + High IV → jade lizard")

        if trend_side and iv_low:
            add("Iron Butterfly", 9,   "Sideways + Low IV → iron fly (pinning)")
            add("Long Call Butterfly", 8, "Sideways + Low IV → call butterfly")
            add("Long Put Butterfly", 7,  "Sideways + Low IV → put butterfly")
            add("Call Calendar Spread", 7, "Sideways + Low IV → calendar")
            add("Put Calendar Spread", 6,  "Sideways + Low IV → calendar")
            add("Iron Condor", 5,       "Sideways + Low IV → condor")

        if trend_side and not iv_low and not iv_high:
            add("Iron Condor", 8,      "Sideways moderate IV → condor")
            add("Short Strangle", 6,   "Sideways moderate IV → strangle")
            add("Diagonal Spread", 5,  "Sideways moderate IV → diagonal")

        # Sort by score
        ranked = sorted(
            [{"name": n, "score": s, "reason": reasons[n], "strategy": self._strategies[n]}
             for n, s in scores.items() if s > 0],
            key=lambda x: -x["score"]
        )
        return ranked[:8]  # top 8


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST SUMMARY HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compute_backtest_summary(trades: List[BacktestTrade]) -> Dict[str, Any]:
    """Compute summary metrics from a list of BacktestTrade."""
    if not trades:
        return {}

    pnls = [t.total_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_profit = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    # Max drawdown
    equity = [0.0]
    for p in pnls:
        equity.append(equity[-1] + p)
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = peak - e
        if dd > max_dd:
            max_dd = dd

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_profit": round(avg_profit, 2),
        "avg_loss": round(avg_loss, 2),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(profit_factor, 2),
        "equity_curve": equity,
    }
