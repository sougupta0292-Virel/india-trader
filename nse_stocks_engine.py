"""
nse_stocks_engine.py
====================
NSE Top 50 (Nifty 50) Stocks Engine
- Live price fetching via yfinance + Angel One
- All strategies: Fabio OI+Fib, VWAP+CPR, 5EMA, Inside Candle, Hunter
- Options + Futures support for stock derivatives
- Full backtesting for each stock
- Angel One order placement for stocks/options/futures

NSE Symbol map → yfinance tickers: RELIANCE → RELIANCE.NS
F&O eligible: Most Nifty 50 stocks have monthly F&O
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime, timedelta, date
import warnings, logging, os, sys
warnings.filterwarnings('ignore')

# Silence yfinance's own noisy logger (404s, "possibly delisted", etc.)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)


def _yf_download(ticker: str, **kwargs) -> Optional[pd.DataFrame]:
    """Download from yfinance, silently trying .BO fallback if .NS fails."""
    import yfinance as yf
    tickers_to_try = [ticker]
    if ticker.endswith(".NS"):
        tickers_to_try.append(ticker[:-3] + ".BO")

    for t in tickers_to_try:
        try:
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            try:
                df = yf.download(t, progress=False, auto_adjust=True, **kwargs)
            finally:
                sys.stderr.close()
                sys.stderr = old_stderr
            if df is not None and len(df) >= 2:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception:
            pass
    return None


def _yf_ticker_history(ticker: str, **kwargs) -> Optional[pd.DataFrame]:
    """Get history from yfinance Ticker, silently trying .BO fallback if .NS fails."""
    import yfinance as yf
    tickers_to_try = [ticker]
    if ticker.endswith(".NS"):
        tickers_to_try.append(ticker[:-3] + ".BO")

    for t in tickers_to_try:
        try:
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            try:
                hist = yf.Ticker(t).history(**kwargs)
            finally:
                sys.stderr.close()
                sys.stderr = old_stderr
            if hist is not None and len(hist) >= 1:
                return hist
        except Exception:
            pass
    return None

# ─── Nifty 50 Stocks ──────────────────────────────────────────────────────────

NIFTY50_STOCKS = {
    "RELIANCE":     {"yf": "RELIANCE.NS",  "lot": 250,  "sector": "Energy",       "fo": True},
    "TCS":          {"yf": "TCS.NS",        "lot": 175,  "sector": "IT",           "fo": True},
    "HDFCBANK":     {"yf": "HDFCBANK.NS",   "lot": 550,  "sector": "Banking",      "fo": True},
    "INFY":         {"yf": "INFY.NS",        "lot": 400,  "sector": "IT",           "fo": True},
    "HINDUNILVR":   {"yf": "HINDUNILVR.NS", "lot": 300,  "sector": "FMCG",         "fo": True},
    "ICICIBANK":    {"yf": "ICICIBANK.NS",  "lot": 700,  "sector": "Banking",      "fo": True},
    "KOTAKBANK":    {"yf": "KOTAKBANK.NS",  "lot": 400,  "sector": "Banking",      "fo": True},
    "LT":           {"yf": "LT.NS",         "lot": 150,  "sector": "Infrastructure","fo": True},
    "SBIN":         {"yf": "SBIN.NS",       "lot": 750,  "sector": "Banking",      "fo": True},
    "BHARTIARTL":   {"yf": "BHARTIARTL.NS", "lot": 475,  "sector": "Telecom",      "fo": True},
    "AXISBANK":     {"yf": "AXISBANK.NS",   "lot": 625,  "sector": "Banking",      "fo": True},
    "WIPRO":        {"yf": "WIPRO.NS",       "lot": 1500, "sector": "IT",           "fo": True},
    "MARUTI":       {"yf": "MARUTI.NS",      "lot": 100,  "sector": "Auto",         "fo": True},
    "SUNPHARMA":    {"yf": "SUNPHARMA.NS",  "lot": 350,  "sector": "Pharma",       "fo": True},
    "TATAMOTORS":   {"yf": "TATAMOTORS.NS", "lot": 1425, "sector": "Auto",         "fo": True},
    "HCLTECH":      {"yf": "HCLTECH.NS",    "lot": 700,  "sector": "IT",           "fo": True},
    "BAJFINANCE":   {"yf": "BAJFINANCE.NS", "lot": 125,  "sector": "Finance",      "fo": True},
    "TITAN":        {"yf": "TITAN.NS",       "lot": 375,  "sector": "Consumer",     "fo": True},
    "ULTRACEMCO":   {"yf": "ULTRACEMCO.NS", "lot": 100,  "sector": "Cement",       "fo": True},
    "ASIANPAINT":   {"yf": "ASIANPAINT.NS", "lot": 300,  "sector": "Consumer",     "fo": True},
    "ADANIENT":     {"yf": "ADANIENT.NS",   "lot": 625,  "sector": "Infra",        "fo": True},
    "POWERGRID":    {"yf": "POWERGRID.NS",  "lot": 2700, "sector": "Power",        "fo": True},
    "NTPC":         {"yf": "NTPC.NS",        "lot": 2250, "sector": "Power",        "fo": True},
    "JSWSTEEL":     {"yf": "JSWSTEEL.NS",   "lot": 675,  "sector": "Metal",        "fo": True},
    "TATASTEEL":    {"yf": "TATASTEEL.NS",  "lot": 5500, "sector": "Metal",        "fo": True},
    "ONGC":         {"yf": "ONGC.NS",        "lot": 1925, "sector": "Energy",       "fo": True},
    "COALINDIA":    {"yf": "COALINDIA.NS",  "lot": 1400, "sector": "Mining",       "fo": True},
    "BPCL":         {"yf": "BPCL.NS",        "lot": 1800, "sector": "Energy",       "fo": True},
    "M&M":          {"yf": "M&M.NS",         "lot": 350,  "sector": "Auto",         "fo": True},
    "HINDALCO":     {"yf": "HINDALCO.NS",   "lot": 1075, "sector": "Metal",        "fo": True},
    "DRREDDY":      {"yf": "DRREDDY.NS",    "lot": 125,  "sector": "Pharma",       "fo": True},
    "CIPLA":        {"yf": "CIPLA.NS",       "lot": 650,  "sector": "Pharma",       "fo": True},
    "EICHERMOT":    {"yf": "EICHERMOT.NS",  "lot": 175,  "sector": "Auto",         "fo": True},
    "GRASIM":       {"yf": "GRASIM.NS",      "lot": 375,  "sector": "Diversified",  "fo": True},
    "HEROMOTOCO":   {"yf": "HEROMOTOCO.NS", "lot": 150,  "sector": "Auto",         "fo": True},
    "DIVISLAB":     {"yf": "DIVISLAB.NS",   "lot": 200,  "sector": "Pharma",       "fo": True},
    "INDUSINDBK":   {"yf": "INDUSINDBK.NS", "lot": 500,  "sector": "Banking",      "fo": True},
    "TECHM":        {"yf": "TECHM.NS",       "lot": 600,  "sector": "IT",           "fo": True},
    "APOLLOHOSP":   {"yf": "APOLLOHOSP.NS", "lot": 150,  "sector": "Healthcare",   "fo": True},
    "BAJAJFINSV":   {"yf": "BAJAJFINSV.NS", "lot": 500,  "sector": "Finance",      "fo": True},
    "TATACONSUM":   {"yf": "TATACONSUM.NS", "lot": 900,  "sector": "FMCG",         "fo": True},
    "ADANIPORTS":   {"yf": "ADANIPORTS.NS", "lot": 1150, "sector": "Infra",        "fo": True},
    "SBILIFE":      {"yf": "SBILIFE.NS",    "lot": 750,  "sector": "Insurance",    "fo": True},
    "HDFCLIFE":     {"yf": "HDFCLIFE.NS",   "lot": 1100, "sector": "Insurance",    "fo": True},
    "ITC":          {"yf": "ITC.NS",         "lot": 3200, "sector": "FMCG",         "fo": True},
    "NESTLEIND":    {"yf": "NESTLEIND.NS",  "lot": 50,   "sector": "FMCG",         "fo": True},
    "SHREECEM":     {"yf": "SHREECEM.NS",   "lot": 25,   "sector": "Cement",       "fo": True},
    "PIDILITIND":   {"yf": "PIDILITIND.NS", "lot": 375,  "sector": "Chemicals",    "fo": True},
    "TRENT":        {"yf": "TRENT.NS",       "lot": 350,  "sector": "Retail",       "fo": True},
    "BEL":          {"yf": "BEL.NS",         "lot": 4550, "sector": "Defence",      "fo": True},
}

SECTORS = sorted(set(v["sector"] for v in NIFTY50_STOCKS.values()))


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class StockQuote:
    symbol:      str
    ltp:         float
    prev_close:  float
    prev_high:   float
    prev_low:    float
    day_high:    float
    day_low:     float
    change_pct:  float
    volume:      int
    sector:      str
    lot_size:    int
    fo_eligible: bool
    # Computed signals
    ema5:        float = 0.0
    ema20:       float = 0.0
    vwap:        float = 0.0
    rsi:         float = 50.0
    signal:      str   = "NEUTRAL"   # BULLISH / BEARISH / NEUTRAL
    signal_score: int  = 0
    strategy_hit: str  = ""          # Which strategy triggered

@dataclass
class StockSignal:
    symbol:      str
    direction:   str          # LONG / SHORT
    strategy:    str          # FABIO_FIB / VWAP_CPR / EMA5 / HUNTER
    entry:       float
    target:      float
    stop_loss:   float
    rr:          float
    score:       int
    confidence:  str
    segment:     str          # EQUITY / OPTIONS / FUTURES
    option_strike: Optional[int] = None
    option_type:   Optional[str] = None  # CE / PE
    reason:      str = ""
    lot_size:    int = 1

@dataclass
class StockBtResult:
    symbol:      str
    strategy:    str
    total_trades: int
    winning:     int
    losing:      int
    win_rate:    float
    avg_win:     float
    avg_loss:    float
    profit_factor: float
    total_pnl_pts: float
    total_pnl_rs:  float  # In rupees (pts × lot size)
    max_drawdown:  float
    trades:      list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


# ─── Live Data Fetcher ────────────────────────────────────────────────────────

class StockDataFetcher:
    """Fetch live + historical stock data from NSE via yfinance."""

    _cache: Dict[str, dict] = {}
    _cache_time: Dict[str, float] = {}
    CACHE_SECS = 30

    @classmethod
    def fetch_quote(cls, symbol: str) -> Optional[StockQuote]:
        """Fetch live quote for a single NSE stock."""
        import time as _t
        meta = NIFTY50_STOCKS.get(symbol)
        if not meta:
            return None

        now = _t.time()
        if symbol in cls._cache and (now - cls._cache_time.get(symbol, 0)) < cls.CACHE_SECS:
            return cls._cache[symbol]

        try:
            hist = _yf_ticker_history(meta["yf"], period="5d", interval="1d")
            if hist is None or len(hist) < 2:
                return None

            prev = hist.iloc[-2]
            curr = hist.iloc[-1]
            ltp  = float(curr["Close"])
            pc   = float(prev["Close"])

            # 5-min intraday for EMA/VWAP
            intra = _yf_ticker_history(meta["yf"], period="2d", interval="5m") or pd.DataFrame()
            today_intra = intra[intra.index.date == intra.index[-1].date()] if len(intra) > 0 else pd.DataFrame()

            # EMA5 on daily
            closes = hist["Close"].astype(float)
            ema5  = float(closes.ewm(span=5,  adjust=False).mean().iloc[-1])
            ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])

            # VWAP from intraday
            vwap = ltp
            if len(today_intra) > 3:
                tp   = (today_intra["High"] + today_intra["Low"] + today_intra["Close"]) / 3
                vol  = today_intra["Volume"].replace(0, 1)
                vwap = float((tp * vol).cumsum().iloc[-1] / vol.cumsum().iloc[-1])

            # RSI 14 on daily
            delta = closes.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, 0.0001)
            rsi   = float(100 - 100 / (1 + rs.iloc[-1]))

            # Signal logic
            score  = 0
            sig    = "NEUTRAL"
            strat  = ""

            if ltp > ema5 > ema20:
                score += 2
                strat += "EMA_BULLISH "
            elif ltp < ema5 < ema20:
                score -= 2
                strat += "EMA_BEARISH "

            if ltp > vwap:
                score += 1
                strat += "ABOVE_VWAP "
            elif ltp < vwap:
                score -= 1
                strat += "BELOW_VWAP "

            if rsi < 35:
                score += 1
                strat += "RSI_OVERSOLD "
            elif rsi > 65:
                score -= 1
                strat += "RSI_OVERBOUGHT "

            # Price vs prev levels
            if ltp > float(prev["High"]):
                score += 2
                strat += "PREV_HIGH_BREAKOUT "
            elif ltp < float(prev["Low"]):
                score -= 2
                strat += "PREV_LOW_BREAKDOWN "

            sig = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "NEUTRAL"

            q = StockQuote(
                symbol=symbol,
                ltp=round(ltp, 2),
                prev_close=round(pc, 2),
                prev_high=round(float(prev["High"]), 2),
                prev_low=round(float(prev["Low"]), 2),
                day_high=round(float(curr["High"]), 2),
                day_low=round(float(curr["Low"]), 2),
                change_pct=round((ltp - pc) / pc * 100, 2),
                volume=int(curr["Volume"]),
                sector=meta["sector"],
                lot_size=meta["lot"],
                fo_eligible=meta["fo"],
                ema5=round(ema5, 2),
                ema20=round(ema20, 2),
                vwap=round(vwap, 2),
                rsi=round(rsi, 1),
                signal=sig,
                signal_score=score,
                strategy_hit=strat.strip(),
            )
            cls._cache[symbol]      = q
            cls._cache_time[symbol] = now
            return q

        except Exception as e:
            print(f"Quote error {symbol}: {e}")
            return None

    @classmethod
    def fetch_all_quotes(cls, symbols: Optional[List[str]] = None,
                          progress_cb=None) -> Dict[str, StockQuote]:
        """Fetch all Nifty 50 quotes. Returns dict of {symbol: StockQuote}."""
        symbols = symbols or list(NIFTY50_STOCKS.keys())
        results = {}
        for i, sym in enumerate(symbols):
            q = cls.fetch_quote(sym)
            if q:
                results[sym] = q
            if progress_cb:
                progress_cb(i + 1, len(symbols), sym)
        return results

    @classmethod
    def fetch_historical(cls, symbol: str, days: int = 90,
                          interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV for backtesting."""
        meta = NIFTY50_STOCKS.get(symbol)
        if not meta:
            return None
        try:
            period = f"{min(days + 10, 700)}d" if interval == "1d" else f"{min(days, 55)}d"
            df = _yf_download(meta["yf"], period=period, interval=interval)
            if df is None or len(df) < 10:
                return None
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            df = df.dropna()
            df["_date"] = df.index.date if hasattr(df.index, "date") else pd.to_datetime(df.index).date
            return df
        except Exception:
            return None


# ─── Signal Generator for Stocks ─────────────────────────────────────────────

class StockSignalGenerator:
    """
    Run all strategies on a stock quote and generate trading signals.
    Supports EQUITY / OPTIONS / FUTURES segments.
    """

    @staticmethod
    def generate_all_signals(quote: StockQuote,
                              segment: str = "EQUITY") -> List[StockSignal]:
        """Returns list of signals from all strategies."""
        signals = []

        # 1. Fabio OI + Fibonacci style
        fib_sig = StockSignalGenerator._fabio_fib(quote, segment)
        if fib_sig:
            signals.append(fib_sig)

        # 2. VWAP + CPR
        vwap_sig = StockSignalGenerator._vwap_cpr(quote, segment)
        if vwap_sig:
            signals.append(vwap_sig)

        # 3. 5 EMA
        ema_sig = StockSignalGenerator._ema5(quote, segment)
        if ema_sig:
            signals.append(ema_sig)

        # 4. Hunter (Gap + Momentum)
        hunt_sig = StockSignalGenerator._hunter(quote, segment)
        if hunt_sig:
            signals.append(hunt_sig)

        # 5. Fabio NSE — AAA Fade / Squeeze / Delta Absorption
        fabio_nse_sigs = StockSignalGenerator._fabio_nse(quote, segment)
        signals.extend(fabio_nse_sigs)

        return signals

    @staticmethod
    def _fabio_nse(q: StockQuote, segment: str) -> List[StockSignal]:
        """Fabio NSE setups: AAA Fade, Squeeze, Delta Absorption."""
        results = []
        try:
            from fabio_nse import (AAAValueAreaFade, MomentumSqueeze,
                                    DeltaAbsorption, FabioNSESignalGenerator)
            import pandas as pd

            if q.intraday_candles is None or len(q.intraday_candles) < 10:
                return []

            df = q.intraday_candles.copy()
            if 'ema5' not in df.columns:
                df['ema5'] = df['Close'].ewm(span=5, adjust=False).mean()

            pivot  = (q.prev_high + q.prev_low + q.prev_close) / 3
            bc_cpr = (q.prev_high + q.prev_low) / 2
            tc_cpr = (pivot - bc_cpr) + pivot
            if tc_cpr < bc_cpr: tc_cpr, bc_cpr = bc_cpr, tc_cpr

            vwap_est = q.vwap if hasattr(q, 'vwap') and q.vwap > 0 else q.ltp

            # Simulate PCR from RSI proxy
            pcr = 1.0
            if hasattr(q, 'rsi') and q.rsi:
                pcr = 1.0 + (50 - q.rsi) / 100

            gen = FabioNSESignalGenerator()
            sig = gen.get_signal(
                df, q.prev_high, q.prev_low, q.prev_close,
                pcr, tc_cpr, bc_cpr, vwap_est, q.ltp)

            if sig is None: return []

            step = _option_step(q.ltp)
            opt_type = "CE" if sig.direction == "LONG" else "PE"
            strike   = int(round(sig.entry / step) * step) if segment != "EQUITY" else None

            ss = StockSignal(
                symbol=q.symbol, ltp=q.ltp,
                direction=sig.direction,
                entry=sig.entry, target=sig.target, sl=sig.sl,
                rr=sig.rr, score=sig.confidence * 2,
                strategy=f"FABIO_{sig.setup}",
                reason=sig.reason,
                segment=segment,
                option_strike=strike, option_type=opt_type,
                confidence=f"{sig.confidence}/5 ({'⭐'*sig.confidence})"
            )
            results.append(ss)
        except Exception as e:
            print(f"[FabioNSE signal] {e}")
        return results


    @staticmethod
    def _fabio_fib(q: StockQuote, segment: str) -> Optional[StockSignal]:
        """Fibonacci retracement from prev H/L."""
        diff = q.prev_high - q.prev_low
        if diff < q.ltp * 0.005:  # Need at least 0.5% range
            return None

        fib_618_up   = q.prev_low  + diff * 0.618
        fib_618_down = q.prev_high - diff * 0.618
        fib_382_up   = q.prev_high + diff * 0.382   # Extension target
        fib_382_down = q.prev_low  - diff * 0.382

        direction = None
        entry = target = sl = 0.0

        if abs(q.ltp - fib_618_up) / diff < 0.35 and q.signal == "BULLISH":
            direction = "LONG"
            entry  = q.ltp
            target = fib_382_up
            sl     = q.prev_low + diff * 0.786 - diff * 0.05
        elif abs(q.ltp - fib_618_down) / diff < 0.35 and q.signal == "BEARISH":
            direction = "SHORT"
            entry  = q.ltp
            target = fib_382_down
            sl     = q.prev_high - diff * 0.786 + diff * 0.05

        if direction is None:
            return None

        risk   = abs(entry - sl)
        reward = abs(target - entry)
        if risk == 0:
            return None
        rr = round(reward / risk, 2)
        if rr < 1.2:
            return None

        score = q.signal_score + (2 if rr >= 2 else 0)

        # Options: ATM strike rounded to nearest step; Futures: no strike needed
        strike = None
        opt_type = None
        if segment == "OPTIONS":
            step = _option_step(q.ltp)
            strike = int(round(entry / step) * step)
            opt_type = "CE" if direction == "LONG" else "PE"

        return StockSignal(
            symbol=q.symbol, direction=direction,
            strategy="FABIO_FIB",
            entry=round(entry, 2), target=round(target, 2),
            stop_loss=round(sl, 2), rr=rr,
            score=score,
            confidence="HIGH" if score >= 5 else "MEDIUM" if score >= 3 else "LOW",
            segment=segment,
            option_strike=strike, option_type=opt_type,
            reason=f"61.8% Fib | {q.signal} | EMA5={q.ema5:.0f} | RSI={q.rsi:.0f}",
            lot_size=q.lot_size,
        )

    @staticmethod
    def _vwap_cpr(q: StockQuote, segment: str) -> Optional[StockSignal]:
        """VWAP + CPR confluence signal."""
        if q.vwap == 0:
            return None

        # CPR
        pivot = (q.prev_high + q.prev_low + q.prev_close) / 3
        bc    = (q.prev_high + q.prev_low) / 2
        tc    = (pivot - bc) + pivot
        if tc < bc:
            tc, bc = bc, tc

        direction = None
        entry = target = sl = 0.0
        reason_parts = []

        above_vwap  = q.ltp > q.vwap
        above_cpr   = q.ltp > tc
        below_cpr   = q.ltp < bc
        vwap_band   = abs(q.ltp - q.vwap) / q.vwap

        if above_vwap and above_cpr and q.ltp > q.ema5:
            direction = "LONG"
            entry   = q.ltp
            sl      = min(q.vwap, bc) - (q.ltp * 0.005)
            r1      = (2 * pivot) - q.prev_low
            target  = max(r1, entry + abs(entry - sl) * 1.5)
            reason_parts = [f"Above VWAP({q.vwap:.0f})", f"Above CPR({tc:.0f})", "EMA bullish"]

        elif not above_vwap and below_cpr and q.ltp < q.ema5:
            direction = "SHORT"
            entry   = q.ltp
            sl      = max(q.vwap, tc) + (q.ltp * 0.005)
            s1      = (2 * pivot) - q.prev_high
            target  = min(s1, entry - abs(sl - entry) * 1.5)
            reason_parts = [f"Below VWAP({q.vwap:.0f})", f"Below CPR({bc:.0f})", "EMA bearish"]

        if direction is None:
            return None

        risk   = abs(entry - sl)
        reward = abs(target - entry)
        if risk == 0:
            return None
        rr = round(reward / risk, 2)
        if rr < 1.2:
            return None

        strike = opt_type = None
        if segment == "OPTIONS":
            step = _option_step(q.ltp)
            strike   = int(round(entry / step) * step)
            opt_type = "CE" if direction == "LONG" else "PE"
        # FUTURES: entry/target/sl same as equity, no strike

        return StockSignal(
            symbol=q.symbol, direction=direction,
            strategy="VWAP_CPR",
            entry=round(entry, 2), target=round(target, 2),
            stop_loss=round(sl, 2), rr=rr,
            score=q.signal_score + 2,
            confidence="HIGH" if rr >= 2 else "MEDIUM",
            segment=segment,
            option_strike=strike, option_type=opt_type,
            reason=" | ".join(reason_parts),
            lot_size=q.lot_size,
        )

    @staticmethod
    def _ema5(q: StockQuote, segment: str) -> Optional[StockSignal]:
        """5 EMA breakout / breakdown signal."""
        if q.ema5 == 0:
            return None

        direction = None
        entry = target = sl = 0.0

        # Bullish: price crossed above ema5 and ema5 > ema20
        if q.ltp > q.ema5 and q.ema5 > q.ema20 and q.ltp > q.prev_high:
            direction = "LONG"
            entry  = q.ltp
            sl     = q.ema5 * 0.997
            target = entry + (entry - sl) * 2.0
            reason = f"Break prev high | EMA5={q.ema5:.0f} > EMA20={q.ema20:.0f}"

        elif q.ltp < q.ema5 and q.ema5 < q.ema20 and q.ltp < q.prev_low:
            direction = "SHORT"
            entry  = q.ltp
            sl     = q.ema5 * 1.003
            target = entry - (sl - entry) * 2.0
            reason = f"Break prev low | EMA5={q.ema5:.0f} < EMA20={q.ema20:.0f}"

        if direction is None:
            return None

        risk   = abs(entry - sl)
        reward = abs(target - entry)
        if risk == 0:
            return None
        rr = round(reward / risk, 2)

        strike = opt_type = None
        if segment == "OPTIONS":
            step = _option_step(q.ltp)
            strike   = int(round(entry / step) * step)
            opt_type = "CE" if direction == "LONG" else "PE"
        # FUTURES: same levels, no strike

        return StockSignal(
            symbol=q.symbol, direction=direction,
            strategy="5EMA",
            entry=round(entry, 2), target=round(target, 2),
            stop_loss=round(sl, 2), rr=rr,
            score=abs(q.signal_score) + 3,
            confidence="HIGH" if q.signal_score >= 3 else "MEDIUM",
            segment=segment,
            option_strike=strike, option_type=opt_type,
            reason=reason,
            lot_size=q.lot_size,
        )

    @staticmethod
    def _hunter(q: StockQuote, segment: str) -> Optional[StockSignal]:
        """Gap + Momentum hunter for stocks."""
        gap_pct = (q.ltp - q.prev_close) / q.prev_close * 100 if q.prev_close > 0 else 0
        direction = None
        entry = target = sl = 0.0
        reason = ""

        diff = q.prev_high - q.prev_low
        if diff < q.ltp * 0.003:
            return None

        # Gap up + still holding above prev high = momentum long
        if gap_pct > 0.5 and q.ltp > q.prev_high and q.ltp > q.vwap:
            direction = "LONG"
            entry  = q.ltp
            sl     = q.prev_high - diff * 0.1
            target = q.ltp + diff * 0.5
            reason = f"Gap UP +{gap_pct:.1f}% | Holding above prev high"

        # Gap down + still below prev low = momentum short
        elif gap_pct < -0.5 and q.ltp < q.prev_low and q.ltp < q.vwap:
            direction = "SHORT"
            entry  = q.ltp
            sl     = q.prev_low + diff * 0.1
            target = q.ltp - diff * 0.5
            reason = f"Gap DOWN {gap_pct:.1f}% | Holding below prev low"

        if direction is None:
            return None

        risk   = abs(entry - sl)
        reward = abs(target - entry)
        if risk == 0 or reward / risk < 1.2:
            return None
        rr = round(reward / risk, 2)

        strike = opt_type = None
        if segment == "OPTIONS":
            step = _option_step(q.ltp)
            strike   = int(round(entry / step) * step)
            opt_type = "CE" if direction == "LONG" else "PE"
        # FUTURES: no strike needed

        return StockSignal(
            symbol=q.symbol, direction=direction,
            strategy="HUNTER",
            entry=round(entry, 2), target=round(target, 2),
            stop_loss=round(sl, 2), rr=rr,
            score=3,
            confidence="MEDIUM",
            segment=segment,
            option_strike=strike, option_type=opt_type,
            reason=reason,
            lot_size=q.lot_size,
        )


# ─── Stock Backtester ─────────────────────────────────────────────────────────

class StockBacktester:
    """
    Backtest all strategies on a single NSE stock.
    Uses daily candles for reliable long-term testing.
    """

    STRATEGIES = ["FABIO_FIB", "FABIO_AAA", "FABIO_SQUEEZE", "FABIO_DELTA", "EMA5_CROSS", "INSIDE_CANDLE", "TRAFFIC_LIGHT", "VWAP_MOMENTUM"]

    def __init__(self, symbol: str, days: int = 365):
        self.symbol = symbol
        self.days   = days
        self.meta   = NIFTY50_STOCKS.get(symbol, {})
        self.lot    = self.meta.get("lot", 1)

    def run_all(self) -> Dict[str, StockBtResult]:
        """Run all strategies. Returns {strategy: StockBtResult}."""
        df = StockDataFetcher.fetch_historical(self.symbol, self.days, "1d")
        if df is None or len(df) < 20:
            return {}

        df = df.copy()
        df["ema5"]  = df["Close"].ewm(span=5,  adjust=False).mean()
        df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["rsi"]   = self._rsi(df["Close"])

        results = {}
        results["FABIO_FIB"]      = self._run_fabio_fib(df)
        results["EMA5_CROSS"]     = self._run_ema5_cross(df)
        results["INSIDE_CANDLE"]  = self._run_inside(df)
        results["TRAFFIC_LIGHT"]  = self._run_traffic(df)
        results["VWAP_MOMENTUM"]  = self._run_vwap_momentum(df)
        # Fabio NSE setups
        fabio_results = self._run_fabio_nse(df)
        results.update(fabio_results)
        return results

    def _rsi(self, close, period=14):
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, 0.0001)
        return 100 - 100 / (1 + rs)

    def _simulate_exit(self, df, start_i, direction, entry, target, sl):
        for j in range(start_i, min(start_i + 30, len(df))):
            row = df.iloc[j]
            h   = float(row["High"]); l = float(row["Low"]); c = float(row["Close"])
            if direction == "LONG":
                if l <= sl:     return sl,     "STOP_LOSS"
                if h >= target: return target, "TARGET_HIT"
            else:
                if h >= sl:     return sl,     "STOP_LOSS"
                if l <= target: return target, "TARGET_HIT"
        if start_i < len(df):
            return float(df.iloc[min(start_i + 29, len(df) - 1)]["Close"]), "TIMEOUT"
        return entry, "NO_EXIT"

    def _build_result(self, trades, equity, strategy):
        if not trades:
            return StockBtResult(self.symbol, strategy, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], [100000])
        wins   = [t for t in trades if t["won"]]
        losses = [t for t in trades if not t["won"]]
        wr     = len(wins) / len(trades) * 100
        avg_w  = float(np.mean([t["pnl"] for t in wins])) if wins else 0
        avg_l  = abs(float(np.mean([t["pnl"] for t in losses]))) if losses else 0
        pf     = (len(wins) * avg_w) / (len(losses) * avg_l) if losses and avg_l > 0 else 99
        eq     = np.array(equity)
        dd     = float(np.max(np.maximum.accumulate(eq) - eq))
        total_pts = sum(t["pnl"] for t in trades)
        return StockBtResult(
            symbol=self.symbol, strategy=strategy,
            total_trades=len(trades), winning=len(wins), losing=len(losses),
            win_rate=round(wr, 1), avg_win=round(avg_w, 2), avg_loss=round(avg_l, 2),
            profit_factor=round(pf, 2),
            total_pnl_pts=round(total_pts, 2),
            total_pnl_rs=round(total_pts * self.lot, 0),
            max_drawdown=round(dd, 2),
            trades=trades, equity_curve=equity,
        )

    def _run_fabio_fib(self, df) -> StockBtResult:
        trades = []; equity = [100000]; cur_eq = 100000
        for i in range(1, len(df) - 1):
            prev = df.iloc[i - 1]; curr = df.iloc[i]; nxt = df.iloc[i + 1]
            ph = float(prev["High"]); pl = float(prev["Low"]); pc = float(prev["Close"])
            diff = ph - pl
            if diff < float(curr["Close"]) * 0.005:
                equity.append(cur_eq); continue

            spot = float(curr["Close"])
            fib_618_up   = pl + diff * 0.618
            fib_618_down = ph - diff * 0.618
            ext_up       = ph + diff * 0.382
            ext_down     = pl - diff * 0.382
            ema5         = float(curr["ema5"])
            rsi          = float(curr["rsi"])

            direction = None
            if abs(spot - fib_618_up) / diff < 0.4 and spot > ema5 and rsi < 60:
                direction = "LONG"; entry = spot; target = ext_up;   sl = pl + diff * 0.786 - diff * 0.05
            elif abs(spot - fib_618_down) / diff < 0.4 and spot < ema5 and rsi > 40:
                direction = "SHORT"; entry = spot; target = ext_down; sl = ph - diff * 0.786 + diff * 0.05

            if direction is None: equity.append(cur_eq); continue
            risk = abs(entry - sl); reward = abs(target - entry)
            if risk == 0 or reward / risk < 1.2: equity.append(cur_eq); continue

            exit_p, exit_r = self._simulate_exit(df, i + 1, direction, entry, target, sl)
            pnl = (exit_p - entry) if direction == "LONG" else (entry - exit_p)
            cur_eq += pnl * self.lot
            trades.append({"date": str(df.index[i])[:10], "won": pnl > 0, "pnl": pnl,
                           "rr": round(reward / risk, 2), "exit": exit_r, "dir": direction})
            equity.append(cur_eq)
        return self._build_result(trades, equity, "FABIO_FIB")

    def _run_fabio_nse(self, df) -> Dict[str, "StockBtResult"]:
        """
        Run all 3 Fabio NSE setups (AAA Fade, Squeeze, Delta Absorption)
        on daily stock data. Returns dict of {setup_name: StockBtResult}.
        """
        setup_names = ["FABIO_AAA", "FABIO_SQUEEZE", "FABIO_DELTA"]
        empty = {s: self._build_result([], [100000], s) for s in setup_names}

        try:
            from fabio_nse import (AAAValueAreaFade, MomentumSqueeze,
                                    DeltaAbsorption)
        except ImportError:
            print("[StockBacktester] fabio_nse.py not found — FABIO_AAA/SQUEEZE/DELTA skipped")
            return empty
        setup_classes = {
            "FABIO_AAA":     AAAValueAreaFade,
            "FABIO_SQUEEZE": MomentumSqueeze,
            "FABIO_DELTA":   DeltaAbsorption,
        }
        data = {s: {"trades": [], "equity": [100000], "cur": 100000}
                for s in setup_names}

        for i in range(2, len(df) - 1):
            row  = df.iloc[i]; prev = df.iloc[i-1]; nxt = df.iloc[i+1]
            prev_h = float(prev["High"]); prev_l = float(prev["Low"])
            prev_c = float(prev["Close"]); prev_o = float(prev["Open"])
            diff = prev_h - prev_l
            if diff < prev_c * 0.003: continue  # skip if prev day range < 0.3% of price

            c = float(row["Close"]); h = float(row["High"])
            l = float(row["Low"]); o = float(row["Open"])
            nh = float(nxt["High"]); nl = float(nxt["Low"]); nc = float(nxt["Close"])

            day_ret = prev_c - prev_o
            pcr = 1.0 + day_ret / prev_o * 6 if prev_o > 0 else 1.0
            pcr = max(0.4, min(2.5, pcr))

            pivot  = (prev_h + prev_l + prev_c) / 3
            bc_cpr = (prev_h + prev_l) / 2
            tc_cpr = (pivot - bc_cpr) + pivot
            if tc_cpr < bc_cpr: tc_cpr, bc_cpr = bc_cpr, tc_cpr

            mini = df.iloc[max(0, i-20): i+1].copy()

            for sname in setup_names:
                signal = None
                try:
                    if sname == "FABIO_AAA":
                        signal = AAAValueAreaFade.check(
                            mini, prev_h, prev_l, prev_c,
                            pcr, tc_cpr, bc_cpr, c, c)
                    elif sname == "FABIO_SQUEEZE" and len(mini) >= 22:
                        signal = MomentumSqueeze.check(mini, pcr)
                    elif sname == "FABIO_DELTA":
                        signal = DeltaAbsorption.check(
                            mini, prev_h, prev_l, tc_cpr, bc_cpr, c, pcr)
                except:
                    continue

                if signal is None: continue
                if signal.rr < 2.0: continue

                if signal.direction == "LONG":
                    if nl <= signal.sl:        exit_p, exit_r = signal.sl,     "STOP_LOSS"
                    elif nh >= signal.target:  exit_p, exit_r = signal.target, "TARGET_HIT"
                    else:                      exit_p, exit_r = nc, "EOD_EXIT"
                else:
                    if nh >= signal.sl:        exit_p, exit_r = signal.sl,     "STOP_LOSS"
                    elif nl <= signal.target:  exit_p, exit_r = signal.target, "TARGET_HIT"
                    else:                      exit_p, exit_r = nc, "EOD_EXIT"

                pnl = (exit_p - signal.entry) if signal.direction == "LONG" \
                      else (signal.entry - exit_p)
                data[sname]["cur"] += pnl * self.lot
                data[sname]["equity"].append(data[sname]["cur"])
                data[sname]["trades"].append({
                    "date": str(df.index[i])[:10], "won": pnl > 0, "pnl": pnl,
                    "rr": round(signal.rr, 2), "exit": exit_r, "dir": signal.direction
                })

        results = {}
        for sname in setup_names:
            results[sname] = self._build_result(
                data[sname]["trades"], data[sname]["equity"], sname)
        return results

    def _run_ema5_cross(self, df) -> StockBtResult:
        trades = []; equity = [100000]; cur_eq = 100000
        for i in range(3, len(df) - 1):
            c1 = df.iloc[i-2]; c2 = df.iloc[i-1]; curr = df.iloc[i]
            cl1 = float(c1["Close"]); e1 = float(c1["ema5"])
            cl2 = float(c2["Close"]); e2 = float(c2["ema5"])
            cl3 = float(curr["Close"]); e3 = float(curr["ema5"])
            h = float(curr["High"]); l = float(curr["Low"])

            direction = None
            if cl1 < e1 and cl2 < e2 and cl3 > e3:
                direction = "LONG"; entry = cl3; sl = l - (h - l) * 0.1; target = cl3 + (cl3 - sl) * 2
            elif cl1 > e1 and cl2 > e2 and cl3 < e3:
                direction = "SHORT"; entry = cl3; sl = h + (h - l) * 0.1; target = cl3 - (sl - cl3) * 2

            if direction is None: equity.append(cur_eq); continue
            risk = abs(entry - sl)
            if risk == 0: equity.append(cur_eq); continue
            rr = abs(target - entry) / risk
            if rr < 1.8: equity.append(cur_eq); continue

            exit_p, exit_r = self._simulate_exit(df, i + 1, direction, entry, target, sl)
            pnl = (exit_p - entry) if direction == "LONG" else (entry - exit_p)
            cur_eq += pnl * self.lot
            trades.append({"date": str(df.index[i])[:10], "won": pnl > 0, "pnl": pnl,
                           "rr": round(rr, 2), "exit": exit_r, "dir": direction})
            equity.append(cur_eq)
        return self._build_result(trades, equity, "EMA5_CROSS")

    def _run_inside(self, df) -> StockBtResult:
        trades = []; equity = [100000]; cur_eq = 100000
        for i in range(1, len(df) - 1):
            mother = df.iloc[i - 1]; child = df.iloc[i]; nxt = df.iloc[i + 1]
            mh = float(mother["High"]); ml = float(mother["Low"])
            ch = float(child["High"]);  cl = float(child["Low"])
            nh = float(nxt["High"]); nl = float(nxt["Low"]); nc = float(nxt["Close"])
            if not (ch <= mh and cl >= ml) or ch == mh or cl == ml:
                equity.append(cur_eq); continue
            mrange = mh - ml
            if mrange < float(mother["Close"]) * 0.005: equity.append(cur_eq); continue

            if nh > mh:
                direction = "LONG"; entry = mh; sl = ml - mrange * 0.1; target = mh + mrange * 1.5
            elif nl < ml:
                direction = "SHORT"; entry = ml; sl = mh + mrange * 0.1; target = ml - mrange * 1.5
            else:
                equity.append(cur_eq); continue

            risk = abs(entry - sl)
            if risk == 0: equity.append(cur_eq); continue
            rr = abs(target - entry) / risk

            if direction == "LONG":
                exit_p = target if nh >= target else sl if nl <= sl else nc
                exit_r = "TARGET_HIT" if nh >= target else "STOP_LOSS" if nl <= sl else "EOD"
            else:
                exit_p = target if nl <= target else sl if nh >= sl else nc
                exit_r = "TARGET_HIT" if nl <= target else "STOP_LOSS" if nh >= sl else "EOD"

            pnl = (exit_p - entry) if direction == "LONG" else (entry - exit_p)
            cur_eq += pnl * self.lot
            trades.append({"date": str(df.index[i])[:10], "won": pnl > 0, "pnl": pnl,
                           "rr": round(rr, 2), "exit": exit_r, "dir": direction})
            equity.append(cur_eq)
        return self._build_result(trades, equity, "INSIDE_CANDLE")

    def _run_traffic(self, df) -> StockBtResult:
        trades = []; equity = [100000]; cur_eq = 100000
        for i in range(3, len(df) - 1):
            c1 = df.iloc[i-3]; c2 = df.iloc[i-2]; c3 = df.iloc[i-1]
            curr = df.iloc[i]; nxt = df.iloc[i+1]

            def is_red(r): return float(r["Close"]) < float(r["Open"])
            def is_grn(r): return float(r["Close"]) > float(r["Open"])

            direction = None
            if all(is_red(r) for r in [c1, c2, c3]): direction = "LONG"
            elif all(is_grn(r) for r in [c1, c2, c3]): direction = "SHORT"
            if direction is None: equity.append(cur_eq); continue

            c  = float(curr["Close"]); h = float(curr["High"]); l = float(curr["Low"])
            nh = float(nxt["High"]); nl = float(nxt["Low"]); nc = float(nxt["Close"])
            rng = h - l
            if rng < float(curr["Close"]) * 0.003: equity.append(cur_eq); continue

            if direction == "LONG":
                entry = c; sl = l - rng * 0.3; target = c + (c - sl) * 2
            else:
                entry = c; sl = h + rng * 0.3; target = c - (sl - c) * 2

            risk = abs(entry - sl)
            if risk == 0: equity.append(cur_eq); continue

            if direction == "LONG":
                exit_p = target if nh >= target else sl if nl <= sl else nc
                exit_r = "TARGET_HIT" if nh >= target else "STOP_LOSS" if nl <= sl else "EOD"
            else:
                exit_p = target if nl <= target else sl if nh >= sl else nc
                exit_r = "TARGET_HIT" if nl <= target else "STOP_LOSS" if nh >= sl else "EOD"

            pnl = (exit_p - entry) if direction == "LONG" else (entry - exit_p)
            cur_eq += pnl * self.lot
            trades.append({"date": str(df.index[i])[:10], "won": pnl > 0, "pnl": pnl,
                           "rr": round(abs(target - entry) / risk, 2), "exit": exit_r, "dir": direction})
            equity.append(cur_eq)
        return self._build_result(trades, equity, "TRAFFIC_LIGHT")

    def _run_vwap_momentum(self, df) -> StockBtResult:
        """VWAP approximate momentum on daily (using prior day's typical price)."""
        trades = []; equity = [100000]; cur_eq = 100000
        for i in range(5, len(df) - 1):
            curr = df.iloc[i]; prev = df.iloc[i-1]
            # Approximate VWAP as 5-day typical price avg
            window = df.iloc[i-5:i]
            tp_avg = float(((window["High"] + window["Low"] + window["Close"]) / 3).mean())

            ltp  = float(curr["Close"])
            ema5 = float(curr["ema5"])
            ph   = float(prev["High"]); pl = float(prev["Low"])
            diff = ph - pl
            if diff < ltp * 0.005: equity.append(cur_eq); continue

            direction = None
            if ltp > tp_avg * 1.005 and ltp > ema5 and ltp > ph:
                direction = "LONG"; entry = ltp; sl = tp_avg - diff * 0.1; target = ltp + diff * 0.5
            elif ltp < tp_avg * 0.995 and ltp < ema5 and ltp < pl:
                direction = "SHORT"; entry = ltp; sl = tp_avg + diff * 0.1; target = ltp - diff * 0.5

            if direction is None: equity.append(cur_eq); continue
            risk = abs(entry - sl)
            if risk == 0 or abs(target - entry) / risk < 1.2: equity.append(cur_eq); continue
            rr = abs(target - entry) / risk

            exit_p, exit_r = self._simulate_exit(df, i + 1, direction, entry, target, sl)
            pnl = (exit_p - entry) if direction == "LONG" else (entry - exit_p)
            cur_eq += pnl * self.lot
            trades.append({"date": str(df.index[i])[:10], "won": pnl > 0, "pnl": pnl,
                           "rr": round(rr, 2), "exit": exit_r, "dir": direction})
            equity.append(cur_eq)
        return self._build_result(trades, equity, "VWAP_MOMENTUM")


# ─── Angel One Order Placement for Stocks ────────────────────────────────────

class StockOrderManager:
    """
    Place equity / options / futures orders via Angel One API.
    """

    def __init__(self, angel_client):
        self.client = angel_client

    def place_equity_order(self, symbol: str, side: str,
                            qty: int, paper: bool = True) -> Optional[str]:
        """Buy/Sell NSE equity. side = BUY / SELL."""
        if paper:
            import time as _t
            return f"PAPER_EQ_{int(_t.time())}"
        if not self.client or not self.client.connected:
            return None
        try:
            import requests, time as _t
            url  = "https://apiconnect.angelone.in/rest/secure/angelbroking/order/v1/placeOrder"
            body = {
                "variety":         "NORMAL",
                "tradingsymbol":   symbol,
                "symboltoken":     "0",
                "transactiontype": side,
                "exchange":        "NSE",
                "ordertype":       "MARKET",
                "producttype":     "DELIVERY",   # CNC for delivery, INTRADAY for MIS
                "duration":        "DAY",
                "quantity":        str(qty),
                "price":           "0",
                "squareoff":       "0",
                "stoploss":        "0",
            }
            r = requests.post(url, json=body,
                              headers=self.client._headers(auth=True), timeout=10)
            d = r.json()
            return d["data"]["orderid"] if d.get("status") else None
        except Exception as e:
            print(f"Equity order error: {e}")
            return None

    def place_futures_order(self, symbol: str, expiry: str,
                             side: str, qty: int, paper: bool = True) -> Optional[str]:
        """Buy/Sell NSE stock futures."""
        if paper:
            import time as _t
            return f"PAPER_FUT_{int(_t.time())}"
        if not self.client or not self.client.connected:
            return None
        try:
            import requests
            ts   = f"{symbol}{expiry}FUT"
            url  = "https://apiconnect.angelone.in/rest/secure/angelbroking/order/v1/placeOrder"
            body = {
                "variety": "NORMAL", "tradingsymbol": ts,
                "symboltoken": "0", "transactiontype": side,
                "exchange": "NFO", "ordertype": "MARKET",
                "producttype": "INTRADAY", "duration": "DAY",
                "quantity": str(qty), "price": "0",
                "squareoff": "0", "stoploss": "0",
            }
            r = requests.post(url, json=body,
                              headers=self.client._headers(auth=True), timeout=10)
            d = r.json()
            return d["data"]["orderid"] if d.get("status") else None
        except Exception as e:
            print(f"Futures order error: {e}")
            return None

    def place_options_order(self, symbol: str, strike: int,
                             option_type: str, expiry: str,
                             side: str, qty: int, paper: bool = True) -> Optional[str]:
        """Buy/Sell NSE stock options. option_type = CE/PE."""
        if paper:
            import time as _t
            return f"PAPER_OPT_{int(_t.time())}"
        if not self.client or not self.client.connected:
            return None
        try:
            import requests
            ts   = f"{symbol}{expiry}{option_type}{strike}"
            url  = "https://apiconnect.angelone.in/rest/secure/angelbroking/order/v1/placeOrder"
            body = {
                "variety": "NORMAL", "tradingsymbol": ts,
                "symboltoken": "0", "transactiontype": side,
                "exchange": "NFO", "ordertype": "MARKET",
                "producttype": "INTRADAY", "duration": "DAY",
                "quantity": str(qty), "price": "0",
                "squareoff": "0", "stoploss": "0",
            }
            r = requests.post(url, json=body,
                              headers=self.client._headers(auth=True), timeout=10)
            d = r.json()
            return d["data"]["orderid"] if d.get("status") else None
        except Exception as e:
            print(f"Options order error: {e}")
            return None

    def get_stock_ltp(self, symbol: str) -> float:
        """Get live LTP of a stock."""
        q = StockDataFetcher.fetch_quote(symbol)
        return q.ltp if q else 0.0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _option_step(price: float) -> int:
    """Get option strike step for a stock based on price."""
    if price < 100:    return 5
    elif price < 500:  return 10
    elif price < 1000: return 20
    elif price < 2000: return 50
    elif price < 5000: return 100
    else:              return 200


def get_top_movers(quotes: Dict[str, StockQuote], n: int = 10) -> dict:
    """Returns top gainers, losers, and signals from a quotes dict."""
    all_q = list(quotes.values())
    gainers  = sorted(all_q, key=lambda q: q.change_pct, reverse=True)[:n]
    losers   = sorted(all_q, key=lambda q: q.change_pct)[:n]
    bullish  = [q for q in all_q if q.signal == "BULLISH"]
    bearish  = [q for q in all_q if q.signal == "BEARISH"]
    return {
        "gainers": gainers,
        "losers": losers,
        "bullish": bullish,
        "bearish": bearish,
    }


def sector_heatmap(quotes: Dict[str, StockQuote]) -> Dict[str, dict]:
    """Aggregate quotes by sector for heatmap."""
    sectors: Dict[str, dict] = {}
    for q in quotes.values():
        s = q.sector
        if s not in sectors:
            sectors[s] = {"count": 0, "avg_change": 0.0, "bullish": 0, "bearish": 0}
        sectors[s]["count"]      += 1
        sectors[s]["avg_change"] += q.change_pct
        if q.signal == "BULLISH":   sectors[s]["bullish"] += 1
        elif q.signal == "BEARISH": sectors[s]["bearish"] += 1

    for s in sectors:
        n = sectors[s]["count"]
        if n > 0:
            sectors[s]["avg_change"] = round(sectors[s]["avg_change"] / n, 2)
        sectors[s]["sentiment"] = (
            "BULLISH" if sectors[s]["bullish"] > sectors[s]["bearish"]
            else "BEARISH" if sectors[s]["bearish"] > sectors[s]["bullish"]
            else "NEUTRAL"
        )
    return sectors
