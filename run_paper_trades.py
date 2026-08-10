"""
run_paper_trades.py
Standalone paper trading engine — runs via GitHub Actions every 15 min during
NSE market hours (9:15 AM – 3:30 PM IST, Mon–Fri).

State is persisted in data/paper_trades.json so it survives across runs.
"""

import os, sys, json, uuid, logging
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

IST           = timezone(timedelta(hours=5, minutes=30))
DATA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATE_FILE    = os.path.join(DATA_DIR, "paper_trades.json")
CAPITAL_START = 100_000.0
MAX_OPEN_PER_SYMBOL = 1   # Only one open trade per symbol+tf+strategy

# ── Universe ──────────────────────────────────────────────────────────────────

INDEX_SCAN = {
    "NIFTY":     "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX":    "^BSESN",
}

# Top 15 liquid Nifty 50 stocks — fast enough for a 15-min CI job
STOCK_SCAN = {
    "RELIANCE":  {"yf": "RELIANCE.NS",  "lot": 250,  "seg": "Equity"},
    "TCS":       {"yf": "TCS.NS",       "lot": 175,  "seg": "Equity"},
    "HDFCBANK":  {"yf": "HDFCBANK.NS",  "lot": 550,  "seg": "Equity"},
    "INFY":      {"yf": "INFY.NS",      "lot": 400,  "seg": "Equity"},
    "ICICIBANK": {"yf": "ICICIBANK.NS", "lot": 700,  "seg": "Equity"},
    "SBIN":      {"yf": "SBIN.NS",      "lot": 750,  "seg": "Equity"},
    "AXISBANK":  {"yf": "AXISBANK.NS",  "lot": 625,  "seg": "Equity"},
    "WIPRO":     {"yf": "WIPRO.NS",     "lot": 1500, "seg": "Equity"},
    "MARUTI":    {"yf": "MARUTI.NS",    "lot": 100,  "seg": "Equity"},
    "TATAMOTORS":{"yf": "TATAMOTORS.NS","lot": 1425, "seg": "Equity"},
    "ITC":       {"yf": "ITC.NS",       "lot": 3200, "seg": "Equity"},
    "BAJFINANCE":{"yf": "BAJFINANCE.NS","lot": 125,  "seg": "Equity"},
    "LT":        {"yf": "LT.NS",        "lot": 150,  "seg": "Equity"},
    "HCLTECH":   {"yf": "HCLTECH.NS",   "lot": 700,  "seg": "Equity"},
    "SUNPHARMA": {"yf": "SUNPHARMA.NS", "lot": 350,  "seg": "Equity"},
}

TFs_INTRADAY = ["5M", "15M", "1H"]   # Skip 4H for intraday paper trading

# ── Market hours ──────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t

def is_near_close() -> bool:
    now = datetime.now(IST)
    close_t = now.replace(hour=15, minute=25, second=0, microsecond=0)
    return now >= close_t

# ── State I/O ─────────────────────────────────────────────────────────────────

def load_state() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "open_trades":   [],
        "closed_trades": [],
        "capital":       CAPITAL_START,
        "last_updated":  None,
        "runs":          0,
    }

def save_state(state: dict):
    state["last_updated"] = datetime.now(IST).isoformat()
    state["runs"] = state.get("runs", 0) + 1
    # Keep last 200 closed trades to prevent file bloat
    if len(state["closed_trades"]) > 200:
        state["closed_trades"] = state["closed_trades"][-200:]
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)

# ── Price fetching ────────────────────────────────────────────────────────────

def get_ltp(yf_ticker: str) -> float | None:
    import yfinance as yf
    tickers = [yf_ticker]
    if yf_ticker.endswith(".NS"):
        tickers.append(yf_ticker[:-3] + ".BO")
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="1d", interval="1m")
            if hist is not None and len(hist) > 0:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return None

# ── Open trade updater ────────────────────────────────────────────────────────

def update_open_trades(state: dict):
    remaining = []
    for trade in state["open_trades"]:
        price = get_ltp(trade["yf_ticker"])
        if price is None:
            trade["current_price"] = trade.get("current_price", trade["entry"])
            remaining.append(trade)
            continue

        trade["current_price"] = round(price, 2)
        direction = trade["direction"]
        entry, target, stop = trade["entry"], trade["target"], trade["stop"]

        exit_price, exit_reason = None, None

        # Force close at end of day
        if is_near_close():
            exit_price, exit_reason = price, "EOD_CLOSE"
        elif direction == "LONG":
            if price <= stop:
                exit_price, exit_reason = stop, "STOP_LOSS"
            elif price >= target:
                exit_price, exit_reason = target, "TARGET_HIT"
        else:
            if price >= stop:
                exit_price, exit_reason = stop, "STOP_LOSS"
            elif price <= target:
                exit_price, exit_reason = target, "TARGET_HIT"

        if exit_price is not None:
            pnl_pts = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
            lot     = trade.get("lot_size", 1)
            pnl_rs  = round(pnl_pts * lot, 2)
            state["capital"] = round(state["capital"] + pnl_rs, 2)

            closed = dict(trade)
            closed.update({
                "exit_price":  round(exit_price, 2),
                "exit_reason": exit_reason,
                "exit_time":   datetime.now(IST).isoformat(),
                "pnl_pts":     round(pnl_pts, 2),
                "pnl_rs":      pnl_rs,
                "won":         pnl_pts > 0,
                # Fields expected by app.py _all_trades()
                "Symbol":      trade["symbol"],
                "Segment":     trade.get("segment", "Futures" if trade["symbol"] in INDEX_SCAN else "Equity"),
                "TF":          trade["tf"].lower(),
                "Strategy":    trade["strategy"],
                "Direction":   direction,
                "Entry ₹":   round(entry, 2),
                "Exit ₹":    round(exit_price, 2),
                "Net PnL ₹": pnl_rs,
                "PnL %":       round(pnl_pts / max(abs(entry), 1) * 100, 2),
                "Charges ₹": 0,
                "Date":        trade.get("entry_time", "")[:10],
                "Exit Reason": exit_reason,
                "Qty":         trade.get("qty", 1),
            })
            state["closed_trades"].append(closed)
            log.info("CLOSED %s %s %s %s → ₹%.0f", trade["symbol"], direction, trade["strategy"], exit_reason, pnl_rs)
        else:
            remaining.append(trade)

    state["open_trades"] = remaining

# ── Signal scanner ────────────────────────────────────────────────────────────

def scan_and_open(state: dict):
    from india_all_strategies_engine import MultiTFScanner

    open_keys = {
        (t["symbol"], t["strategy"], t["tf"])
        for t in state["open_trades"]
    }
    now_str = datetime.now(IST).isoformat()
    new_count = 0

    # --- Indices ---
    for symbol, yf_ticker in INDEX_SCAN.items():
        for tf in TFs_INTRADAY:
            try:
                signals = MultiTFScanner.scan_symbol(yf_ticker, symbol, tf, days=12)
            except Exception as e:
                log.warning("Scan failed %s %s: %s", symbol, tf, e)
                continue
            for sig in signals:
                if sig.direction == "FLAT":
                    continue
                key = (sig.symbol, sig.strategy, sig.tf)
                if key in open_keys:
                    continue
                trade = {
                    "id":           str(uuid.uuid4())[:8],
                    "symbol":       sig.symbol,
                    "yf_ticker":    yf_ticker,
                    "segment":      "Futures",
                    "tf":           sig.tf,
                    "strategy":     sig.strategy,
                    "direction":    sig.direction,
                    "entry":        sig.entry,
                    "target":       sig.target,
                    "stop":         sig.stop,
                    "rr":           sig.rr,
                    "score":        sig.score,
                    "reason":       sig.reason,
                    "entry_time":   now_str,
                    "lot_size":     75 if symbol == "NIFTY" else (35 if symbol == "BANKNIFTY" else 20),
                    "qty":          1,
                    "current_price": sig.entry,
                }
                state["open_trades"].append(trade)
                open_keys.add(key)
                new_count += 1
                log.info("OPENED %s %s %s %s @ %.2f → T:%.2f S:%.2f",
                         symbol, sig.direction, sig.strategy, sig.tf,
                         sig.entry, sig.target, sig.stop)

    # --- Stocks ---
    for symbol, info in STOCK_SCAN.items():
        for tf in TFs_INTRADAY:
            try:
                signals = MultiTFScanner.scan_symbol(info["yf"], symbol, tf, days=12)
            except Exception as e:
                log.warning("Scan failed %s %s: %s", symbol, tf, e)
                continue
            for sig in signals:
                if sig.direction == "FLAT":
                    continue
                key = (sig.symbol, sig.strategy, sig.tf)
                if key in open_keys:
                    continue
                trade = {
                    "id":           str(uuid.uuid4())[:8],
                    "symbol":       sig.symbol,
                    "yf_ticker":    info["yf"],
                    "segment":      info["seg"],
                    "tf":           sig.tf,
                    "strategy":     sig.strategy,
                    "direction":    sig.direction,
                    "entry":        sig.entry,
                    "target":       sig.target,
                    "stop":         sig.stop,
                    "rr":           sig.rr,
                    "score":        sig.score,
                    "reason":       sig.reason,
                    "entry_time":   now_str,
                    "lot_size":     info["lot"],
                    "qty":          1,
                    "current_price": sig.entry,
                }
                state["open_trades"].append(trade)
                open_keys.add(key)
                new_count += 1
                log.info("OPENED %s %s %s %s @ %.2f",
                         symbol, sig.direction, sig.strategy, sig.tf, sig.entry)

    log.info("New signals opened: %d", new_count)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now_ist = datetime.now(IST)
    log.info("Paper trade run at %s", now_ist.strftime("%Y-%m-%d %H:%M:%S IST"))

    if not is_market_open():
        log.info("Market closed — nothing to do")
        return

    state = load_state()
    log.info("Loaded state: %d open, %d closed, capital=₹%.0f",
             len(state["open_trades"]), len(state["closed_trades"]), state["capital"])

    update_open_trades(state)
    scan_and_open(state)
    save_state(state)

    log.info("Done — %d open, %d closed, capital=₹%.0f",
             len(state["open_trades"]), len(state["closed_trades"]), state["capital"])

if __name__ == "__main__":
    main()
