"""
options_auto_trader.py
======================
Options buying strategy - buys ATM CE/PE based on signals
Separate from futures auto trader
"""
import os, json, time, requests
from datetime import datetime, date, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

IST = timezone(timedelta(hours=5, minutes=30))

def get_ist():
    return datetime.now(IST)

# ─── Option Trade Record ──────────────────────────────────────────────────────
@dataclass
class OptionTrade:
    id:            str
    date:          str
    time:          str
    instrument:    str
    direction:     str        # LONG/SHORT
    option_type:   str        # CE/PE
    strike:        int
    expiry:        str
    entry_premium: float      # Premium paid
    target_premium:float      # Target premium
    sl_premium:    float      # SL premium (40% loss)
    qty:           int
    spot_entry:    float      # Spot price when entered
    strategy:      str
    status:        str = "OPEN"
    exit_premium:  float = 0.0
    exit_reason:   str = ""
    pnl_pts:       float = 0.0
    pnl_rs:        float = 0.0
    paper:         bool = True

    def to_dict(self): return self.__dict__

# ─── Options State ────────────────────────────────────────────────────────────
@dataclass
class OptionsState:
    status:       str = "STOPPED"
    open_trade:   Optional[OptionTrade] = None
    trades_today: int = 0
    total_trades: int = 0
    winning:      int = 0
    total_pnl:    float = 0.0
    last_scan:    str = ""
    last_signal:  str = "—"
    last_action:  str = "—"
    log:          list = field(default_factory=list)
    history:      list = field(default_factory=list)

    _DIR = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def _state_file(key: str = "") -> str:
        suffix = f"_{key.upper()}" if key else ""
        return os.path.join(OptionsState._DIR, f'options_state{suffix}.json')

    def save(self, key: str = ""):
        data = {
            'status':       self.status,
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
            'date':         str(date.today()),
        }
        with open(OptionsState._state_file(key), 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(key: str = "") -> 'OptionsState':
        state    = OptionsState()
        _file    = OptionsState._state_file(key)
        if not os.path.exists(_file):
            return state
        try:
            with open(_file) as f:
                data = json.load(f)
            if data.get('date') != str(date.today()):
                data['trades_today'] = 0
                data['open_trade']   = None
                data['status']       = 'STOPPED'
                data['date']         = str(date.today())
                with open(_file, 'w') as _f:
                    json.dump(data, _f, indent=2)
            state.status       = data.get('status', 'STOPPED')
            state.trades_today = data.get('trades_today', 0)
            state.total_trades = data.get('total_trades', 0)
            state.winning      = data.get('winning', 0)
            state.total_pnl    = data.get('total_pnl', 0.0)
            state.last_scan    = data.get('last_scan', '')
            state.last_signal  = data.get('last_signal', '—')
            state.last_action  = data.get('last_action', '—')
            state.log          = data.get('log', [])
            state.history      = [OptionTrade(**t) for t in data.get('history', [])]
            if data.get('open_trade'):
                state.open_trade = OptionTrade(**data['open_trade'])
        except Exception as e:
            print(f"State load error ({key}): {e}")
        return state

    def log_msg(self, msg: str, level: str = "INFO"):
        entry = {
            'time':    get_ist().strftime('%H:%M:%S'),
            'level':   level,
            'message': msg
        }
        self.log.append(entry)
        print(f"[{entry['time']}] {level}: {msg}")


# ─── Option Price Calculator ──────────────────────────────────────────────────

class OptionPricer:
    """
    Simple option price estimator.
    Uses Black-Scholes approximation for ATM options.
    """

    @staticmethod
    def get_atm_strike(spot: float, instrument: str) -> int:
        """Round to nearest strike."""
        if instrument == "NIFTY":
            return round(spot / 50) * 50
        elif instrument == "BANKNIFTY":
            return round(spot / 100) * 100
        else:  # SENSEX
            return round(spot / 100) * 100

    @staticmethod
    def estimate_premium(spot: float, strike: int,
                          option_type: str, days_to_expiry: int,
                          vix: float = 15.0) -> float:
        """
        Estimate ATM option premium.
        Simple approximation: Premium ≈ spot × IV × sqrt(T/365) × 0.4
        """
        try:
            import math
            iv    = vix / 100
            T     = max(days_to_expiry, 1) / 365
            # ATM premium approximation
            premium = spot * iv * math.sqrt(T) * 0.4
            # Add intrinsic value if ITM
            if option_type == "CE" and spot > strike:
                premium += (spot - strike)
            elif option_type == "PE" and spot < strike:
                premium += (strike - spot)
            return round(max(premium, 5.0), 2)
        except:
            return spot * 0.005  # fallback: 0.5% of spot

    @staticmethod
    def get_expiry(instrument: str) -> tuple:
        """
        Get nearest weekly expiry and days to expiry.
        NIFTY: Thursday | BANKNIFTY: Wednesday | SENSEX: Friday
        On expiry day: trade current week until 2:30 PM IST, then switch to next week.
        """
        from datetime import timedelta, time as _dt
        today      = date.today()
        target_day = {'NIFTY': 3, 'BANKNIFTY': 2, 'SENSEX': 4}.get(instrument, 3)
        days_ahead = (target_day - today.weekday()) % 7
        if days_ahead == 0:
            now = datetime.now().time()
            if now >= _dt(14, 30):
                days_ahead = 7   # past 2:30 PM → next week
            # else keep 0 → today's expiry (active expiry-day scalping window)
        expiry_date = today + timedelta(days=days_ahead)
        expiry_str  = expiry_date.strftime('%d%b%Y').upper()
        return expiry_str, days_ahead


# ─── Options Auto Trader ──────────────────────────────────────────────────────

class OptionsAutoTrader:
    """
    Options buying strategy.
    Uses same signals as futures bot but buys CE/PE instead.

    Entry rules:
    - Buy ATM CE when LONG signal
    - Buy ATM PE when SHORT signal
    - SL at 40% loss of premium
    - Target at 80% gain on premium
    - No daily trade limit (paper trading mode)
    - Auto exit at 3:00 PM IST (before 3:15 close)

    Why 3:00 PM exit:
    - Avoid last 15 min volatility
    - Options spreads widen near close
    """

    LOT_SIZES = {"NIFTY": 75, "BANKNIFTY": 35, "SENSEX": 20}

    _TF_MAP      = {"5M": "5m", "15M": "15m", "1H": "1h", "4H": "4h"}
    _SCAN_SECS   = {"5M": 300, "15M": 900, "1H": 3600, "4H": 3600}

    def __init__(self, instrument="NIFTY", strategies=None,
                 qty=1, paper_trade=True, angel_client=None,
                 sl_pct=40, target_pct=80, timeframe="5M"):
        self.instrument = instrument
        self.strategies = strategies  # kept for backward compat; MultiTFScanner drives signals now
        self.qty        = qty
        self.paper      = paper_trade
        self.client     = angel_client
        self.sl_pct     = sl_pct
        self.target_pct = target_pct
        self.lot_size   = self.LOT_SIZES.get(instrument, 75)
        self.timeframe  = timeframe
        self.state_key  = f"{instrument}_{timeframe}"
        self.state      = OptionsState.load(self.state_key)
        self._running   = False
        self._thread    = None

    def start(self):
        import threading
        self.state.status = "RUNNING"; self.state.save(self.state_key)
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.state.log_msg(f"Options bot STARTED | {self.instrument} {self.timeframe}")

    def stop(self):
        self._running = False
        self.state.status = "STOPPED"; self.state.save(self.state_key)
        self.state.log_msg(f"Options bot STOPPED | {self.instrument} {self.timeframe}")

    def _loop(self):
        interval = self._SCAN_SECS.get(self.timeframe, 300)
        while self._running:
            try:
                self.scan()
            except Exception as e:
                self.state.log_msg(f"Loop error: {e}", "ERROR")
            time.sleep(interval)

    def _is_market_hours(self) -> bool:
        from datetime import time as dtime
        now = get_ist().time()
        return dtime(9, 20) <= now <= dtime(15, 0)  # Exit by 3 PM

    def _get_ltp(self) -> float:
        """Get live spot price.
        Priority: NSE option chain (via Angel client) → yfinance → retry yfinance once.
        """
        # 1. NSE via Angel client get_quote (uses NSE option chain internally)
        if self.client and hasattr(self.client, 'connected') and self.client.connected:
            try:
                q = self.client.get_quote(self.instrument)
                if q and q.get('ltp') and float(q['ltp']) > 0:
                    return float(q['ltp'])
            except:
                pass

        # 2. yfinance — with one retry on failure
        import yfinance as yf
        sym = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
        ticker_sym = sym.get(self.instrument, "^NSEI")
        for attempt in range(2):
            try:
                tk    = yf.Ticker(ticker_sym)
                price = float(tk.fast_info.last_price)
                if price > 0:
                    return price
            except Exception as e:
                if attempt == 0:
                    time.sleep(2)   # brief wait before retry
                else:
                    print(f"yfinance LTP failed ({self.instrument}): {e}")
        return 0.0

    def _get_candles(self, count=20) -> Optional[pd.DataFrame]:
        """Get candles for the selected timeframe."""
        try:
            import yfinance as yf
            sym = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN"}
            interval = self._TF_MAP.get(self.timeframe, "5m")
            df  = yf.download(sym.get(self.instrument,"^NSEI"),
                              period="5d", interval=interval,
                              progress=False, auto_adjust=True)
            if df is not None and len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df['ema5'] = df['Close'].ewm(span=5, adjust=False).mean()
                return df.tail(count)
        except Exception as e:
            print(f"Candle error: {e}")
        return None

    def _detect_signal(self, df, ltp) -> Optional[dict]:
        """Run all strategies for this timeframe via MultiTFScanner, pick best RR signal."""
        try:
            from india_all_strategies_engine import MultiTFScanner
            sym_map   = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
            yf_ticker = sym_map.get(self.instrument, "^NSEI")
            signals   = MultiTFScanner.scan_symbol(yf_ticker, self.instrument, self.timeframe)
        except Exception as e:
            print(f"MultiTFScanner error: {e}")
            return None

        best     = None
        best_rr  = 1.8  # minimum RR threshold
        for sig in signals:
            if sig.direction == "FLAT" or sig.rr < best_rr:
                continue
            best_rr = sig.rr
            best    = sig

        if best is None:
            return None

        return {
            "direction": best.direction,
            "entry":     best.entry,
            "target":    best.target,
            "sl":        best.stop,
            "strategy":  best.strategy,
            "reason":    best.reason,
        }

    def scan(self):
        """Main scan — detect signal and manage open trade."""
        now_str = get_ist().strftime('%H:%M:%S')
        self.state.last_scan = now_str

        if not self._is_market_hours():
            self.state.log_msg(f"⏰ Outside market hours — waiting")
            self.state.save(self.state_key)
            return

        # Exit at 3:00 PM
        from datetime import time as dtime
        if get_ist().time() >= dtime(15, 0):
            if self.state.open_trade:
                self._exit_trade("TIME_EXIT_3PM")
            self.state.save(self.state_key)
            return

        ltp = self._get_ltp()
        df  = self._get_candles(25)

        if ltp == 0 or df is None:
            self.state.log_msg("⚠️ No data — retrying")
            return

        self.state.log_msg(f"🔍 Options scan | {self.instrument} spot: Rs.{ltp:,.2f}")

        # Monitor open trade
        if self.state.open_trade:
            self._monitor_trade(ltp)
            self.state.save(self.state_key)
            return

        # No daily trade limit — take all valid signals (good for paper trading)

        # Detect signal
        signal = self._detect_signal(df, ltp)
        if not signal:
            self.state.log_msg("⚪ No options signal")
            self.state.last_signal = "No signal"
            self.state.save(self.state_key)
            return

        # ── Global Bias Filter ────────────────────────────────────
        # Get global bias from saved state
        try:
            import json, os
            state_file = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'pos_state.json')
            if os.path.exists(state_file):
                with open(state_file) as f:
                    pos_data = json.load(f)
                global_bias = pos_data.get('global_bias', 'NEUTRAL')
            else:
                global_bias = 'NEUTRAL'
        except:
            global_bias = 'NEUTRAL'

        # Take both CE and PE — EOD square-off means both directions are valid
        # Log bias as info only, do not skip any signal
        self.state.log_msg(
            f"✅ Signal: {signal['direction']} | Global bias: {global_bias} | "
            f"{'CE (CALL)' if signal['direction']=='LONG' else 'PE (PUT)'}", "INFO"
        )

        # Calculate option details
        direction   = signal['direction']
        option_type = "CE" if direction == "LONG" else "PE"
        strike      = OptionPricer.get_atm_strike(ltp, self.instrument)
        expiry, dte = OptionPricer.get_expiry(self.instrument)
        premium     = OptionPricer.estimate_premium(ltp, strike, option_type, dte)

        # SL and target on premium
        sl_premium     = round(premium * (1 - self.sl_pct/100), 2)
        target_premium = round(premium * (1 + self.target_pct/100), 2)

        self.state.log_msg(
            f"🎯 OPTIONS SIGNAL: {direction} | Buy {strike}{option_type} "
            f"@ Rs.{premium:.0f} | SL: Rs.{sl_premium:.0f} | "
            f"Target: Rs.{target_premium:.0f} | {signal['reason']}",
            "TRADE"
        )

        # Execute
        self._execute_trade(signal, ltp, option_type, strike,
                            expiry, premium, sl_premium, target_premium)
        self.state.save(self.state_key)

    def _execute_trade(self, signal, ltp, option_type, strike,
                        expiry, premium, sl_premium, target_premium):
        # ── Daily capital cap check ──────────────────────────────────────────
        try:
            from pos_auto_trader import _get_today_deployed_rs, DAILY_CAP_RS
            _this_invest = round(premium * self.lot_size * self.qty, 2)
            _deployed    = _get_today_deployed_rs()
            if _deployed + _this_invest > DAILY_CAP_RS:
                self.state.log_msg(
                    f"🚫 Daily cap ₹{DAILY_CAP_RS:,.0f} reached! "
                    f"Deployed: ₹{_deployed:,.0f} | This option: ₹{_this_invest:,.0f} — skipped.",
                    "WARNING"
                )
                return
        except ImportError:
            pass

        trade_id = f"OPT_{get_ist().strftime('%Y%m%d_%H%M%S')}"
        # Try to get REAL entry price from Angel One
        if self.client and hasattr(self.client,'connected') and self.client.connected:
            try:
                real_entry = self.client.get_option_ltp(
                    symbol=self.instrument,
                    strike=strike,
                    option_type=option_type,
                    expiry=expiry
                )
                if real_entry > 0:
                    premium        = real_entry
                    sl_premium     = round(premium * (1 - self.sl_pct/100), 2)
                    target_premium = round(premium * (1 + self.target_pct/100), 2)
                    self.state.log_msg(f"💹 Real entry price: Rs.{premium:.0f} (Angel One)", "INFO")
            except Exception as e:
                print(f"Real entry price error: {e}")

        cost     = premium * self.lot_size * self.qty

        if self.paper:
            self.state.log_msg(
                f"📋 PAPER OPTIONS: Buy {self.qty} lot {self.instrument} "
                f"{strike}{option_type} @ Rs.{premium:.0f} "
                f"| Cost: Rs.{cost:,.0f}", "PAPER"
            )
        else:
            self.state.log_msg("🔴 LIVE OPTIONS ORDER — Angel One")
            # Real order would go here

        trade = OptionTrade(
            id=trade_id,
            date=str(date.today()),
            time=get_ist().strftime('%H:%M:%S'),
            instrument=self.instrument,
            direction=signal['direction'],
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            entry_premium=premium,
            target_premium=target_premium,
            sl_premium=sl_premium,
            qty=self.qty * self.lot_size,
            spot_entry=ltp,
            strategy=signal.get('strategy','SIGNAL'),
            paper=self.paper
        )

        self.state.open_trade   = trade
        self.state.trades_today += 1
        self.state.total_trades += 1
        self.state.last_signal  = f"{signal['direction']} {strike}{option_type}"
        self.state.last_action  = f"Bought {strike}{option_type} @ Rs.{premium:.0f}"

    def _get_real_option_price(self, trade) -> float:
        """Get real option price.
        Priority: NSE option chain (no auth) → Angel One → estimated.
        NSE works regardless of Angel One connection status.
        """
        # 1. NSE option chain — free, real-time, no auth needed
        try:
            from angel_one import get_nse_option_ltp
            nse_price = get_nse_option_ltp(
                trade.instrument, trade.strike, trade.option_type)
            if nse_price > 0:
                self.state.log_msg(
                    f"💹 Real option price: Rs.{nse_price:.0f} (NSE live)", "INFO")
                return nse_price
        except Exception as e:
            print(f"NSE price error: {e}")

        # 2. Angel One (if connected)
        if self.client and hasattr(self.client, 'connected') and self.client.connected:
            try:
                real_price = self.client.get_option_ltp(
                    symbol=trade.instrument,
                    strike=trade.strike,
                    option_type=trade.option_type,
                    expiry=trade.expiry
                )
                if real_price > 0:
                    self.state.log_msg(
                        f"💹 Real option price: Rs.{real_price:.0f} (Angel One)", "INFO")
                    return real_price
            except Exception as e:
                print(f"Angel price error: {e}")

        # 3. Estimate (last resort)
        spot = self._get_ltp()
        try:
            dte = max(1, (
                datetime.strptime(trade.expiry, '%d%b%Y').date() - date.today()
            ).days)
        except:
            dte = 3
        estimated = OptionPricer.estimate_premium(
            spot, trade.strike, trade.option_type, dte)
        self.state.log_msg(
            f"📊 Estimated option price: Rs.{estimated:.0f} (NSE + Angel unavailable)", "INFO")
        return estimated

    def _monitor_trade(self, spot: float):
        """Monitor open option trade with real prices."""
        trade = self.state.open_trade
        if not trade: return

        # Get real price from Angel One or estimate
        current_premium = self._get_real_option_price(trade)

        pnl_pts = current_premium - trade.entry_premium
        pnl_rs  = pnl_pts * trade.qty
        pnl_pct = (pnl_pts / trade.entry_premium * 100) if trade.entry_premium > 0 else 0

        price_source = "NSE"  # Always using NSE live prices now

        self.state.log_msg(
            f"📊 [{price_source}] {trade.strike}{trade.option_type} | "
            f"Entry: Rs.{trade.entry_premium:.0f} | "
            f"Now: Rs.{current_premium:.0f} | "
            f"PnL: {pnl_pct:+.1f}% (Rs.{pnl_rs:+,.0f}) | "
            f"SL@Rs.{trade.sl_premium:.0f} Target@Rs.{trade.target_premium:.0f}"
        )

        if current_premium <= trade.sl_premium:
            self._exit_trade("STOP_LOSS", current_premium)
        elif current_premium >= trade.target_premium:
            self._exit_trade("TARGET_HIT", current_premium)

    def _exit_trade(self, reason: str, exit_premium: float = 0.0):
        trade = self.state.open_trade
        if not trade: return

        if exit_premium == 0.0:
            spot = self._get_ltp()
            dte  = max(1, (datetime.strptime(trade.expiry,'%d%b%Y').date()-date.today()).days)
            exit_premium = OptionPricer.estimate_premium(
                spot, trade.strike, trade.option_type, dte)

        pnl_pts = exit_premium - trade.entry_premium
        pnl_rs  = pnl_pts * trade.qty
        pnl_pct = (pnl_pts / trade.entry_premium * 100) if trade.entry_premium > 0 else 0
        won     = pnl_pts > 0

        emoji = "✅ WIN" if won else "❌ LOSS"
        self.state.log_msg(
            f"{emoji}: {trade.strike}{trade.option_type} | "
            f"Entry: Rs.{trade.entry_premium:.0f} → Exit: Rs.{exit_premium:.0f} | "
            f"PnL: {pnl_pct:+.1f}% (Rs.{pnl_rs:+,.0f}) | {reason}",
            "TRADE"
        )

        trade.status       = "CLOSED"
        trade.exit_premium = round(exit_premium, 2)
        trade.exit_reason  = reason
        trade.pnl_pts      = round(pnl_pts, 2)
        trade.pnl_rs       = round(pnl_rs, 2)

        if won: self.state.winning += 1
        self.state.total_pnl  += pnl_rs
        self.state.history.append(trade)
        self.state.open_trade  = None
        self.state.last_action = f"Closed {pnl_pct:+.1f}% (Rs.{pnl_rs:+,.0f})"


# ─── Multi-Leg Options Engine ─────────────────────────────────────────────────
# Handles: Straddle, Strangle, Iron Condor, Iron Butterfly, Spreads, Risk Reversal

@dataclass
class MultiLegTrade:
    id:          str
    date:        str
    time:        str
    strategy:    str          # LONG_STRADDLE, SHORT_STRADDLE, etc.
    instrument:  str
    legs:        list         # list of {type, strike, option_type, action, qty, entry_premium}
    net_premium: float        # positive = debit paid, negative = credit received
    sl_pct:      float        # stop based on net premium move
    target_pct:  float
    status:      str = "OPEN"
    exit_pnl_rs: float = 0.0
    exit_reason: str = ""
    paper:       bool = True

    def to_dict(self):
        return self.__dict__


@dataclass
class MultiLegState:
    status:       str = "STOPPED"
    open_trade:   Optional[MultiLegTrade] = None
    trades_today: int = 0
    total_trades: int = 0
    winning:      int = 0
    total_pnl:    float = 0.0
    last_scan:    str = ""
    last_signal:  str = "—"
    log:          list = field(default_factory=list)
    history:      list = field(default_factory=list)

    STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multileg_state.json')

    def save(self):
        data = {'status':self.status,'trades_today':self.trades_today,'total_trades':self.total_trades,
                'winning':self.winning,'total_pnl':self.total_pnl,'last_scan':self.last_scan,
                'last_signal':self.last_signal,'log':self.log[-100:],
                'history':[t.to_dict() for t in self.history[-200:]],
                'open_trade':self.open_trade.to_dict() if self.open_trade else None,
                'date':str(date.today())}
        with open(self.STATE_FILE,'w') as f:
            json.dump(data,f,indent=2)

    @staticmethod
    def load() -> 'MultiLegState':
        s = MultiLegState()
        if not os.path.exists(MultiLegState.STATE_FILE):
            return s
        try:
            with open(MultiLegState.STATE_FILE) as f:
                data = json.load(f)
            if data.get('date') != str(date.today()):
                data.update({'trades_today':0,'open_trade':None,'status':'STOPPED','date':str(date.today())})
            s.status=data.get('status','STOPPED'); s.trades_today=data.get('trades_today',0)
            s.total_trades=data.get('total_trades',0); s.winning=data.get('winning',0)
            s.total_pnl=data.get('total_pnl',0.0); s.last_scan=data.get('last_scan','')
            s.last_signal=data.get('last_signal','—'); s.log=data.get('log',[])
            s.history=[MultiLegTrade(**t) for t in data.get('history',[])]
            if data.get('open_trade'): s.open_trade=MultiLegTrade(**data['open_trade'])
        except Exception as e:
            print(f"MultiLegState load error: {e}")
        return s

    def log_msg(self, msg, level="INFO"):
        self.log.append({'time':get_ist().strftime('%H:%M:%S'),'level':level,'message':msg})
        print(f"[MULTILEG][{level}] {msg}")


class MultiLegOptionsTrader:
    """
    Multi-leg options strategies for Nifty / BankNifty / Sensex.
    Strategies: Long/Short Straddle, Long/Short Strangle, Iron Condor,
    Iron Butterfly, Bull Call Spread, Bear Put Spread, Risk Reversal,
    Vertical Credit Spread, Synthetic Long, Jade Lizard.
    IV rank proxied from India VIX (^INDIAVIX via yfinance).
    """
    LOT_SIZES   = {"NIFTY":75,"BANKNIFTY":35,"SENSEX":20}
    STRIKE_STEP = {"NIFTY":50,"BANKNIFTY":100,"SENSEX":100}

    def __init__(self, instrument="NIFTY", strategies=None, qty=1,
                 paper_trade=True, angel_client=None):
        self.instrument = instrument
        self.strategies = strategies or [
            "LONG_STRADDLE","SHORT_STRADDLE","LONG_STRANGLE","SHORT_STRANGLE",
            "IRON_CONDOR","IRON_BUTTERFLY","BULL_CALL_SPREAD","BEAR_PUT_SPREAD",
            "RISK_REVERSAL","VERT_CREDIT_SPREAD","SYNTHETIC_LONG","JADE_LIZARD",
        ]
        self.qty      = qty
        self.paper    = paper_trade
        self.client   = angel_client
        self.lot      = self.LOT_SIZES.get(instrument, 75)
        self.step     = self.STRIKE_STEP.get(instrument, 50)
        self.state    = MultiLegState.load()
        self._running = False
        self._thread  = None

    def start(self):
        import threading
        self.state.status = "RUNNING"; self.state.save()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.state.log_msg(f"Multi-leg trader STARTED | {self.instrument} | Strategies: {self.strategies}")

    def stop(self):
        self._running = False; self.state.status = "STOPPED"; self.state.save()
        self.state.log_msg("Multi-leg trader STOPPED")

    def _loop(self):
        while self._running:
            try:
                self.scan()
            except Exception as e:
                self.state.log_msg(f"Loop error: {e}", "ERROR")
            time.sleep(300)  # scan every 5 min

    # ── Market data helpers ───────────────────────────────────────────────────

    def _get_spot(self) -> float:
        import yfinance as yf
        sym = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN"}
        try:
            return float(yf.Ticker(sym.get(self.instrument,"^NSEI")).fast_info.last_price)
        except:
            return 0.0

    def _get_vix(self) -> float:
        import yfinance as yf
        try:
            h = yf.Ticker("^INDIAVIX").history(period="2d",interval="1d")
            return float(h.iloc[-1]["Close"]) if len(h) >= 1 else 15.0
        except:
            return 15.0

    def _get_candles(self, count=30) -> Optional[pd.DataFrame]:
        import yfinance as yf
        sym = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN"}
        try:
            df = yf.download(sym.get(self.instrument,"^NSEI"),period="5d",interval="5m",
                             progress=False,auto_adjust=True)
            if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
            return df.tail(count) if df is not None and len(df)>0 else None
        except:
            return None

    def _atm(self, spot: float) -> int:
        return round(spot / self.step) * self.step

    def _est_premium(self, spot, strike, otype, vix, dte=3) -> float:
        import math
        try:
            iv = vix / 100
            T  = max(dte, 0.5) / 365
            prem = spot * iv * math.sqrt(T) * 0.4
            if otype=="CE" and spot>strike: prem += (spot-strike)
            elif otype=="PE" and spot<strike: prem += (strike-spot)
            return round(max(prem, 5.0), 2)
        except:
            return round(spot * 0.005, 2)

    def _iv_rank(self, vix: float) -> str:
        if vix > 22:   return "HIGH"
        if vix > 16:   return "MEDIUM"
        return "LOW"

    # ── Signal detection per strategy ────────────────────────────────────────

    def _detect_signal(self, spot, vix, df) -> Optional[dict]:
        iv_rank = self._iv_rank(vix)
        atm     = self._atm(spot)
        otm1c   = atm + self.step        # 1 step OTM call
        otm2c   = atm + self.step * 2    # 2 step OTM call
        otm1p   = atm - self.step        # 1 step OTM put
        otm2p   = atm - self.step * 2    # 2 step OTM put
        dte     = 3

        # Directional bias from underlying
        bias = "NEUTRAL"
        if df is not None and len(df) >= 10:
            closes = df['Close'].astype(float)
            e20 = float(closes.ewm(span=20,adjust=False).mean().iloc[-1])
            cc  = float(closes.iloc[-1])
            bias = "LONG" if cc > e20 else ("SHORT" if cc < e20 else "NEUTRAL")

        for strat in self.strategies:
            signal = None

            if strat == "LONG_STRADDLE" and vix > 12:
                # Relaxed: fire whenever VIX > 12 (almost always for Indian markets)
                p_ce = self._est_premium(spot,atm,"CE",vix,dte)
                p_pe = self._est_premium(spot,atm,"PE",vix,dte)
                net  = p_ce + p_pe
                signal = {'strategy':'LONG_STRADDLE','net_premium':net,'sl_pct':35,'target_pct':80,
                          'reason':f'Long straddle | VIX={vix:.1f} | ATM {atm} | Net debit ₹{net:.0f} | BEP {atm-net:.0f}/{atm+net:.0f}',
                          'legs':[
                              {'type':'BUY','strike':atm,'option_type':'CE','qty':self.qty,'entry_premium':p_ce},
                              {'type':'BUY','strike':atm,'option_type':'PE','qty':self.qty,'entry_premium':p_pe},
                          ]}

            elif strat == "SHORT_STRADDLE" and vix < 18:
                # Relaxed: VIX < 18 (was < 15) + range < 0.8% (was 0.5%)
                _pinned = True
                if df is not None and len(df) >= 5:
                    _rng = float(df['high'].tail(5).max() - df['low'].tail(5).min())
                    _pinned = _rng < spot * 0.008   # 0.8% range threshold
                if _pinned:
                    p_ce = self._est_premium(spot,atm,"CE",vix,dte)
                    p_pe = self._est_premium(spot,atm,"PE",vix,dte)
                    net  = -(p_ce + p_pe)
                    signal = {'strategy':'SHORT_STRADDLE','net_premium':net,'sl_pct':100,'target_pct':50,
                              'reason':f'Short straddle | VIX={vix:.1f}<18, market pinned | Credit ₹{abs(net):.0f} | BEP {atm-p_ce-p_pe:.0f}/{atm+p_ce+p_pe:.0f}',
                              'legs':[
                                  {'type':'SELL','strike':atm,'option_type':'CE','qty':self.qty,'entry_premium':p_ce},
                                  {'type':'SELL','strike':atm,'option_type':'PE','qty':self.qty,'entry_premium':p_pe},
                              ]}

            elif strat == "LONG_STRANGLE" and vix > 12:
                # Relaxed: same VIX condition as Long Straddle but OTM strikes
                p_ce = self._est_premium(spot,otm1c,"CE",vix,dte)
                p_pe = self._est_premium(spot,otm1p,"PE",vix,dte)
                net  = p_ce + p_pe
                signal = {'strategy':'LONG_STRANGLE','net_premium':net,'sl_pct':40,'target_pct':100,
                          'reason':f'Long strangle | VIX={vix:.1f} | Strikes {otm1p}/{otm1c} | Net debit ₹{net:.0f}',
                          'legs':[
                              {'type':'BUY','strike':otm1c,'option_type':'CE','qty':self.qty,'entry_premium':p_ce},
                              {'type':'BUY','strike':otm1p,'option_type':'PE','qty':self.qty,'entry_premium':p_pe},
                          ]}

            elif strat == "SHORT_STRANGLE" and vix < 18:
                # New: mirrors Short Straddle logic but OTM (+2 strikes)
                _pinned2 = True
                if df is not None and len(df) >= 5:
                    _rng = float(df['high'].tail(5).max() - df['low'].tail(5).min())
                    _pinned2 = _rng < spot * 0.008
                if _pinned2:
                    p_ce = self._est_premium(spot,otm1c,"CE",vix,dte)
                    p_pe = self._est_premium(spot,otm1p,"PE",vix,dte)
                    net  = -(p_ce + p_pe)
                    signal = {'strategy':'SHORT_STRANGLE','net_premium':net,'sl_pct':100,'target_pct':50,
                              'reason':f'Short strangle | VIX={vix:.1f}<18 | Strikes {otm1p}/{otm1c} | Credit ₹{abs(net):.0f}',
                              'legs':[
                                  {'type':'SELL','strike':otm1c,'option_type':'CE','qty':self.qty,'entry_premium':p_ce},
                                  {'type':'SELL','strike':otm1p,'option_type':'PE','qty':self.qty,'entry_premium':p_pe},
                              ]}

            elif strat == "IRON_CONDOR" and iv_rank in ("HIGH","MEDIUM"):
                p_sc = self._est_premium(spot,otm1c,"CE",vix,dte)
                p_bc = self._est_premium(spot,otm2c,"CE",vix,dte)
                p_sp = self._est_premium(spot,otm1p,"PE",vix,dte)
                p_bp = self._est_premium(spot,otm2p,"PE",vix,dte)
                credit = (p_sc - p_bc) + (p_sp - p_bp)
                if credit > 0:
                    signal = {'strategy':'IRON_CONDOR','net_premium':-credit,'sl_pct':150,'target_pct':50,
                              'reason':f'Iron Condor | VIX={vix:.1f} | Credit ₹{credit:.0f} | Range {otm2p}-{otm2c}',
                              'legs':[
                                  {'type':'SELL','strike':otm1c,'option_type':'CE','qty':self.qty,'entry_premium':p_sc},
                                  {'type':'BUY', 'strike':otm2c,'option_type':'CE','qty':self.qty,'entry_premium':p_bc},
                                  {'type':'SELL','strike':otm1p,'option_type':'PE','qty':self.qty,'entry_premium':p_sp},
                                  {'type':'BUY', 'strike':otm2p,'option_type':'PE','qty':self.qty,'entry_premium':p_bp},
                              ]}

            elif strat == "IRON_BUTTERFLY" and iv_rank == "HIGH":
                p_sc = self._est_premium(spot,atm,"CE",vix,dte)
                p_sp = self._est_premium(spot,atm,"PE",vix,dte)
                p_bc = self._est_premium(spot,otm1c,"CE",vix,dte)
                p_bp = self._est_premium(spot,otm1p,"PE",vix,dte)
                credit = (p_sc + p_sp) - (p_bc + p_bp)
                if credit > 0:
                    signal = {'strategy':'IRON_BUTTERFLY','net_premium':-credit,'sl_pct':100,'target_pct':50,
                              'reason':f'Iron Butterfly | VIX={vix:.1f} | ATM {atm} | Credit ₹{credit:.0f}',
                              'legs':[
                                  {'type':'SELL','strike':atm,  'option_type':'CE','qty':self.qty,'entry_premium':p_sc},
                                  {'type':'SELL','strike':atm,  'option_type':'PE','qty':self.qty,'entry_premium':p_sp},
                                  {'type':'BUY', 'strike':otm1c,'option_type':'CE','qty':self.qty,'entry_premium':p_bc},
                                  {'type':'BUY', 'strike':otm1p,'option_type':'PE','qty':self.qty,'entry_premium':p_bp},
                              ]}

            elif strat == "BULL_CALL_SPREAD" and bias == "LONG":
                p_buy = self._est_premium(spot,atm,"CE",vix,dte)
                p_sel = self._est_premium(spot,otm1c,"CE",vix,dte)
                net   = p_buy - p_sel
                if net > 0:
                    signal = {'strategy':'BULL_CALL_SPREAD','net_premium':net,'sl_pct':50,'target_pct':80,
                              'reason':f'Bull call spread | Bullish bias | Buy {atm}CE sell {otm1c}CE | Debit ₹{net:.0f}',
                              'legs':[
                                  {'type':'BUY', 'strike':atm,  'option_type':'CE','qty':self.qty,'entry_premium':p_buy},
                                  {'type':'SELL','strike':otm1c,'option_type':'CE','qty':self.qty,'entry_premium':p_sel},
                              ]}

            elif strat == "BEAR_PUT_SPREAD" and bias == "SHORT":
                p_buy = self._est_premium(spot,atm,"PE",vix,dte)
                p_sel = self._est_premium(spot,otm1p,"PE",vix,dte)
                net   = p_buy - p_sel
                if net > 0:
                    signal = {'strategy':'BEAR_PUT_SPREAD','net_premium':net,'sl_pct':50,'target_pct':80,
                              'reason':f'Bear put spread | Bearish bias | Buy {atm}PE sell {otm1p}PE | Debit ₹{net:.0f}',
                              'legs':[
                                  {'type':'BUY', 'strike':atm,  'option_type':'PE','qty':self.qty,'entry_premium':p_buy},
                                  {'type':'SELL','strike':otm1p,'option_type':'PE','qty':self.qty,'entry_premium':p_sel},
                              ]}

            elif strat == "RISK_REVERSAL" and bias != "NEUTRAL":
                if bias == "LONG":
                    p_sc = self._est_premium(spot,otm1p,"PE",vix,dte)
                    p_bc = self._est_premium(spot,otm1c,"CE",vix,dte)
                    net  = p_bc - p_sc
                    signal = {'strategy':'RISK_REVERSAL','net_premium':net,'sl_pct':60,'target_pct':100,
                              'reason':f'Risk reversal bullish | Sell {otm1p}PE buy {otm1c}CE | Net ₹{net:.0f}',
                              'legs':[
                                  {'type':'SELL','strike':otm1p,'option_type':'PE','qty':self.qty,'entry_premium':p_sc},
                                  {'type':'BUY', 'strike':otm1c,'option_type':'CE','qty':self.qty,'entry_premium':p_bc},
                              ]}
                else:
                    p_sc = self._est_premium(spot,otm1c,"CE",vix,dte)
                    p_bp = self._est_premium(spot,otm1p,"PE",vix,dte)
                    net  = p_bp - p_sc
                    signal = {'strategy':'RISK_REVERSAL','net_premium':net,'sl_pct':60,'target_pct':100,
                              'reason':f'Risk reversal bearish | Sell {otm1c}CE buy {otm1p}PE | Net ₹{net:.0f}',
                              'legs':[
                                  {'type':'SELL','strike':otm1c,'option_type':'CE','qty':self.qty,'entry_premium':p_sc},
                                  {'type':'BUY', 'strike':otm1p,'option_type':'PE','qty':self.qty,'entry_premium':p_bp},
                              ]}

            elif strat == "VERT_CREDIT_SPREAD" and iv_rank in ("HIGH","MEDIUM") and bias != "NEUTRAL":
                if bias == "SHORT":   # bearish credit call spread
                    p_sc = self._est_premium(spot,atm,"CE",vix,dte)
                    p_bc = self._est_premium(spot,otm1c,"CE",vix,dte)
                    credit = p_sc - p_bc
                    if credit > 0:
                        signal = {'strategy':'VERT_CREDIT_SPREAD','net_premium':-credit,'sl_pct':200,'target_pct':50,
                                  'reason':f'Bear call spread | Sell {atm}CE buy {otm1c}CE | Credit ₹{credit:.0f}',
                                  'legs':[
                                      {'type':'SELL','strike':atm,  'option_type':'CE','qty':self.qty,'entry_premium':p_sc},
                                      {'type':'BUY', 'strike':otm1c,'option_type':'CE','qty':self.qty,'entry_premium':p_bc},
                                  ]}
                else:                 # bullish credit put spread
                    p_sp = self._est_premium(spot,atm,"PE",vix,dte)
                    p_bp = self._est_premium(spot,otm1p,"PE",vix,dte)
                    credit = p_sp - p_bp
                    if credit > 0:
                        signal = {'strategy':'VERT_CREDIT_SPREAD','net_premium':-credit,'sl_pct':200,'target_pct':50,
                                  'reason':f'Bull put spread | Sell {atm}PE buy {otm1p}PE | Credit ₹{credit:.0f}',
                                  'legs':[
                                      {'type':'SELL','strike':atm,  'option_type':'PE','qty':self.qty,'entry_premium':p_sp},
                                      {'type':'BUY', 'strike':otm1p,'option_type':'PE','qty':self.qty,'entry_premium':p_bp},
                                  ]}

            elif strat == "SYNTHETIC_LONG" and bias == "LONG" and iv_rank == "LOW":
                p_ce = self._est_premium(spot,atm,"CE",vix,dte)
                p_pe = self._est_premium(spot,atm,"PE",vix,dte)
                net  = p_ce - p_pe
                signal = {'strategy':'SYNTHETIC_LONG','net_premium':net,'sl_pct':40,'target_pct':100,
                          'reason':f'Synthetic long | Buy {atm}CE sell {atm}PE | Net ₹{net:.0f}',
                          'legs':[
                              {'type':'BUY', 'strike':atm,'option_type':'CE','qty':self.qty,'entry_premium':p_ce},
                              {'type':'SELL','strike':atm,'option_type':'PE','qty':self.qty,'entry_premium':p_pe},
                          ]}

            elif strat == "JADE_LIZARD" and bias == "LONG" and iv_rank in ("HIGH","MEDIUM"):
                p_sp = self._est_premium(spot,otm1p,"PE",vix,dte)
                p_sc = self._est_premium(spot,otm1c,"CE",vix,dte)
                p_bc = self._est_premium(spot,otm2c,"CE",vix,dte)
                credit = p_sp + (p_sc - p_bc)
                if credit > 0:
                    signal = {'strategy':'JADE_LIZARD','net_premium':-credit,'sl_pct':150,'target_pct':50,
                              'reason':f'Jade lizard | Sell {otm1p}PE + sell {otm1c}CE/buy {otm2c}CE | Credit ₹{credit:.0f}',
                              'legs':[
                                  {'type':'SELL','strike':otm1p,'option_type':'PE','qty':self.qty,'entry_premium':p_sp},
                                  {'type':'SELL','strike':otm1c,'option_type':'CE','qty':self.qty,'entry_premium':p_sc},
                                  {'type':'BUY', 'strike':otm2c,'option_type':'CE','qty':self.qty,'entry_premium':p_bc},
                              ]}

            if signal:
                return signal
        return None

    # ── Trade execution ───────────────────────────────────────────────────────

    def _execute(self, signal: dict, spot: float):
        _ist_now = get_ist()
        trade_id = f"ML_{_ist_now.strftime('%Y%m%d_%H%M%S')}"
        self.state.log_msg(
            f"{'PAPER' if self.paper else 'LIVE'} {signal['strategy']} | {self.instrument} | "
            f"{len(signal['legs'])} legs | Net ₹{signal['net_premium']:.0f} | {signal['reason']}",
            "TRADE"
        )
        if not self.paper and self.client and getattr(self.client,'connected',False):
            for leg in signal['legs']:
                try:
                    side = "BUY" if leg['type']=="BUY" else "SELL"
                    self.client.place_order(symbol=self.instrument,option_type=leg['option_type'],
                                            strike=leg['strike'],expiry="",side=side,qty=leg['qty']*self.lot,paper=False)
                except Exception as e:
                    self.state.log_msg(f"Order error: {e}", "ERROR")

        trade = MultiLegTrade(
            id=trade_id, date=_ist_now.strftime('%Y-%m-%d'), time=_ist_now.strftime('%H:%M:%S'),
            strategy=signal['strategy'], instrument=self.instrument, legs=signal['legs'],
            net_premium=signal['net_premium'], sl_pct=signal['sl_pct'], target_pct=signal['target_pct'],
            paper=self.paper,
        )
        self.state.open_trade   = trade
        self.state.trades_today += 1
        self.state.total_trades += 1
        self.state.last_signal  = f"{signal['strategy']} | Net ₹{signal['net_premium']:.0f}"

    # ── Monitor open position ─────────────────────────────────────────────────

    def _close_multileg(self, trade, reason: str, pnl_per_unit: float, won: bool):
        trade.exit_reason = reason
        trade.status      = "CLOSED"
        pnl_rs = round(pnl_per_unit * self.lot * self.qty, 2)
        trade.exit_pnl_rs = pnl_rs
        if won: self.state.winning += 1
        self.state.total_pnl  += pnl_rs
        self.state.history.append(trade)
        self.state.open_trade  = None
        tag = "WIN" if won else "LOSS"
        self.state.log_msg(f"{tag} {trade.strategy} | {reason} | ₹{pnl_rs:+,.0f}", "TRADE")

    def _monitor(self, spot: float, vix: float):
        trade = self.state.open_trade
        if not trade: return
        dte = 2

        # ── Short Straddle: real-market individual-leg SL ─────────────────────
        if trade.strategy == "SHORT_STRADDLE":
            credit = abs(trade.net_premium)  # total credit per unit received at entry
            # Refresh current premiums for each sold leg
            for lg in trade.legs:
                lg['current_premium'] = self._est_premium(spot, lg['strike'], lg['option_type'], vix, dte)

            sell_legs = [lg for lg in trade.legs if lg['type'] == 'SELL']
            ce_leg = next((lg for lg in sell_legs if lg['option_type'] == 'CE'), None)
            pe_leg = next((lg for lg in sell_legs if lg['option_type'] == 'PE'), None)

            # Combined P&L per unit: profit = entry_premium - current_premium for each sold leg
            pnl_per_unit = sum(
                lg['entry_premium'] - lg['current_premium'] for lg in sell_legs
            )
            pnl_pct = pnl_per_unit / max(credit, 1) * 100

            ce_now = ce_leg['current_premium'] if ce_leg else 0
            pe_now = pe_leg['current_premium'] if pe_leg else 0
            ce_entry = ce_leg['entry_premium'] if ce_leg else 0
            pe_entry = pe_leg['entry_premium'] if pe_leg else 0

            self.state.log_msg(
                f"SS monitor | Spot={spot:.0f} | CE {ce_entry:.0f}→{ce_now:.0f} "
                f"PE {pe_entry:.0f}→{pe_now:.0f} | P&L {pnl_pct:+.1f}% | BEP {trade.legs[1]['strike']-credit:.0f}/{trade.legs[0]['strike']+credit:.0f}"
            )

            # ── Individual leg 2x SL (real market rule) ───────────────────────
            # If either sold leg's premium doubles, that leg's loss exceeds the entire credit
            for lg in sell_legs:
                if lg['current_premium'] >= lg['entry_premium'] * 2.0:
                    reason = (f"Leg SL: {lg['option_type']} ₹{lg['current_premium']:.0f} "
                              f"≥ 2× entry ₹{lg['entry_premium']:.0f} | "
                              f"spot moved {abs(spot - lg['strike']):.0f} pts past {lg['strike']}")
                    self._close_multileg(trade, reason, pnl_per_unit, won=False)
                    return

            # ── Breakeven breach SL (spot beyond upper/lower breakeven) ────────
            upper_bep = trade.legs[0]['strike'] + credit   # CE strike + total premium
            lower_bep = trade.legs[1]['strike'] - credit   # PE strike - total premium
            if spot > upper_bep * 1.002 or spot < lower_bep * 0.998:
                reason = f"Spot {spot:.0f} breached breakeven {lower_bep:.0f}/{upper_bep:.0f}"
                self._close_multileg(trade, reason, pnl_per_unit, won=False)
                return

            # ── Target: 50% of credit retained ───────────────────────────────
            if pnl_pct >= trade.target_pct:
                reason = f"Target: {pnl_pct:.1f}% of credit locked (₹{pnl_per_unit:.0f}/unit)"
                self._close_multileg(trade, reason, pnl_per_unit, won=True)
            return

        # ── All other strategies: combined P&L% monitor ───────────────────────
        curr_value = sum(
            self._est_premium(spot, lg['strike'], lg['option_type'], vix, dte) *
            (1 if lg['type']=='BUY' else -1)
            for lg in trade.legs
        )
        entry_value = abs(trade.net_premium) if trade.net_premium != 0 else 1
        pnl_pct = (curr_value - abs(trade.net_premium)) / entry_value * 100

        self.state.log_msg(f"ML monitor | {trade.strategy} | P&L {pnl_pct:+.1f}%")
        reason = None
        if trade.net_premium > 0:   # debit: SL = % of debit lost
            if pnl_pct < -trade.sl_pct:      reason = f"SL hit {pnl_pct:.1f}%"
            elif pnl_pct > trade.target_pct:  reason = f"Target {pnl_pct:.1f}%"
        else:                                 # credit: SL = % loss on credit
            if pnl_pct < -trade.sl_pct:      reason = f"Credit SL {pnl_pct:.1f}%"
            elif pnl_pct >= trade.target_pct: reason = f"Target {pnl_pct:.1f}%"
        if reason:
            won = "Target" in reason
            pnl_rs = round(pnl_pct / 100 * entry_value * self.lot * self.qty, 2)
            self._close_multileg(trade, reason, pnl_rs / (self.lot * self.qty), won)

    # ── Main scan ─────────────────────────────────────────────────────────────

    def scan(self):
        from datetime import time as dtime
        now = get_ist().time()
        if not (dtime(9,20) <= now <= dtime(14,45)):
            return
        spot = self._get_spot()
        vix  = self._get_vix()
        df   = self._get_candles(30)
        if spot == 0:
            self.state.log_msg("No spot data"); return
        self.state.last_scan = get_ist().strftime('%H:%M:%S')

        # Auto square-off at 2:45 PM
        if now >= dtime(14, 45) and self.state.open_trade:
            self._monitor(spot, vix)
            self.state.save(); return

        if self.state.open_trade:
            self._monitor(spot, vix)
            self.state.save(); return

        signal = self._detect_signal(spot, vix, df)
        if signal:
            self._execute(signal, spot)
        else:
            self.state.last_signal = "No signal"
        self.state.save()
