# 📈 India Intraday Trading System
## OI + Fibonacci + Global Cues Strategy | Zerodha Kite Integration

---

## STRATEGY OVERVIEW

**Target Accuracy: 60-70% | 1 Trade Per Day | Intraday Only**

### Core Logic
1. **Pre-market (8:30-9:15 AM)** — Check global cues: GIFT Nifty, Dow, Nasdaq, Nikkei, Hang Seng, DAX
2. **Calculate bias** — Weighted score determines BULLISH / BEARISH / NEUTRAL
3. **OI Analysis** — PCR, Max Pain, key support/resistance strikes
4. **Fibonacci Levels** — From previous day's High and Low
5. **9:20 AM signal** — After first 5-min candle, confluence check
6. **Entry** — At 61.8% (deep pullback) or 38.2% (strong trend) Fibonacci level
7. **Exit** — Target = next Fib level | Stop = beyond 78.6% Fib | Hard stop = OI key level

### Confluence Rules (minimum 6/10 required)
| Factor | Points |
|---|---|
| Global bias aligned with trade direction | 1-2 |
| OI signal aligned (PCR + OI change direction) | 1-2 |
| First candle strong and directional | 1-2 |
| Entry at 61.8% or 38.2% Fibonacci | 1-2 |
| Risk:Reward ≥ 2:1 | 1 |
| Volume above 500K | 1 |

### When NOT to trade
- India VIX < 10 (dead market) or > 25 (too volatile)
- Expiry day (erratic OI shifts)
- After 2:30 PM
- Confluence score < 6
- RBI rate decision / FOMC / major earnings day

---

## SETUP

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Zerodha
```bash
# Copy template and fill in your credentials
cp .env.template .env
# Edit .env with your API key and secret
# Get from: https://developers.kite.trade/
# Zerodha Kite API costs: Free for basic | ₹2000/month for historical data
```

### 3. Run the dashboard
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

---

## ZERODHA KITE API SETUP

1. Go to https://developers.kite.trade/
2. Create an app → get API Key + Secret
3. Add your Zerodha login credentials
4. **Token refresh**: Zerodha tokens expire daily at midnight
   - Dashboard handles this with the "Connect Zerodha" button
   - Must re-login each morning before 9:15 AM

### Kite API Subscriptions needed:
- **Basic (Free)**: Live quotes, order placement
- **Historical (₹2000/month)**: OHLCV data for Fibonacci calculation
- **Option chain OI**: Via `kite.quote()` — included in basic plan

---

## LIVE TRADING WORKFLOW

### Daily routine:
```
8:30 AM  → Open dashboard → Fetch Global Cues → check overnight moves
8:45 AM  → Check GIFT Nifty (most important pre-market cue)
9:00 AM  → Load OI data → note max pain, key support/resistance
9:10 AM  → Plot Fibonacci from yesterday's H/L
9:15 AM  → Market opens → DO NOT trade the first candle
9:20 AM  → Click "Generate Signal" → review confluence score
9:20-9:30 AM → If signal ≥ 6/10: execute trade
3:15 PM  → All positions auto-exit (MIS product = auto square-off)
```

### Paper Trade First (recommended 2 weeks minimum)
- Toggle "Paper Trade Mode" ON (default)
- All signals and entries are logged but no real orders placed
- Build confidence in the system before going live

---

## BACKTESTING NOTES

- **OHLCV data**: From yfinance (free, 60-day intraday limit)
- **OI data**: Simulated in backtest (NSE doesn't provide free historical OI API)
  - For real OI backtesting: subscribe to Opstra or Sensibull data feeds
- **Win rate target**: 60-70% achievable with strict confluence filtering
  - Fewer trades = higher quality = better win rate
  - A day with no signal is a GOOD day (protecting capital)

---

## RISK MANAGEMENT

- **Position size**: Max 2% of capital per trade
- **Option buying**: Buy ATM or 1-strike OTM only
- **Stop**: Fixed at Fibonacci level (not trailing initially)
- **Daily loss limit**: Stop trading if down 1% of capital for the day
- **Weekly review**: Check win rate, adjust confluence threshold if needed

---

## FILES

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit dashboard |
| `strategy_engine.py` | Core strategy: Global cues, OI analysis, Fibonacci, Signal generation |
| `backtester.py` | Walk-forward backtesting engine |
| `zerodha_live.py` | Kite API integration: auth, live data, order execution |
| `requirements.txt` | Python dependencies |
| `.env.template` | Credentials template |

---

## DISCLAIMER

This system is for educational purposes. Options trading involves substantial risk.
No strategy guarantees 60-70% accuracy in live markets. Always paper trade first.
Past backtest performance does not predict future results. OI data in backtest is simulated.
Consult a SEBI-registered advisor before trading with real capital.
