"""
angel_one.py - Angel One Smart API (Direct REST)
Fixed based on official SmartAPI documentation
"""
import os, json, time, requests, pyotp
from datetime import datetime, date
import pandas as pd

def load_env():
    _dir = os.path.dirname(os.path.abspath(__file__))
    for _name in ['.env', '.env.txt']:
        env_path = os.path.join(_dir, _name)
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        k, v = line.split('=', 1)
                        os.environ[k.strip()] = v.strip()
            break

load_env()

# On Streamlit Cloud, inject secrets into environment so the rest of the code
# can use os.getenv() without any other changes
try:
    import streamlit as st
    for _k in ["ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_MPIN", "ANGEL_TOTP_KEY"]:
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'angel_token.json')
BASE       = "https://apiconnect.angelone.in"

SYMBOL_TOKENS = {
    "NIFTY":     "26000",
    "BANKNIFTY": "26009",
    "SENSEX":    "1",
}
# ─── NSE Option Chain Fetcher (real-time, no auth needed) ────────────────────

NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept':          'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer':         'https://www.nseindia.com/option-chain',
}

_nse_session   = None
_nse_cache     = {}   # {symbol: (timestamp, data)}
NSE_CACHE_SECS = 15   # refresh every 15 seconds max

# ─── BSE Option Chain Fetcher (for SENSEX options) ───────────────────────────

BSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept':          'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer':         'https://www.bseindia.com/markets/derivatives/dervreports/sensex_opt.aspx',
}

_bse_session   = None
_bse_cache     = {}   # {symbol: (timestamp, data)}
BSE_CACHE_SECS = 15


def _get_bse_session():
    global _bse_session
    if _bse_session is None:
        import requests as _req
        _bse_session = _req.Session()
        _bse_session.headers.update(BSE_HEADERS)
        try:
            _bse_session.get('https://www.bseindia.com', timeout=10)
        except Exception:
            pass
    return _bse_session


def get_bse_option_ltp(strike: int, option_type: str) -> float:
    """
    Fetch live SENSEX option LTP from BSE option chain API.
    strike: e.g. 77300
    option_type: CE / PE
    Returns live premium or 0.0 on failure.
    """
    import time as _time
    global _bse_cache

    cache_key = 'SENSEX'
    cached = _bse_cache.get(cache_key)
    if cached and (_time.time() - cached[0]) < BSE_CACHE_SECS:
        chain_data = cached[1]
    else:
        try:
            sess = _get_bse_session()
            # BSE option chain endpoint for SENSEX (scrip code 1)
            url  = 'https://api.bseindia.com/BseIndiaAPI/api/getoptionchain/w?scripcode=1&expirydate=&optiontype=both'
            resp = sess.get(url, timeout=10)
            if not resp.text.strip():
                return 0.0
            chain_data = resp.json()
            _bse_cache[cache_key] = (_time.time(), chain_data)
        except Exception as e:
            print(f'BSE option chain error: {e}')
            return 0.0

    # Parse BSE chain for specific strike
    try:
        # BSE returns a list under 'Table' key
        records = chain_data.get('Table', [])
        opt_key = 'CE' if option_type == 'CE' else 'PE'
        for row in records:
            try:
                row_strike = int(float(row.get('StrikePrice', 0)))
                row_type   = row.get('OptionType', '').upper().strip()
                if row_strike == int(strike) and row_type == opt_key:
                    ltp = float(row.get('LTP', 0) or 0)
                    if ltp > 0:
                        return ltp
            except (ValueError, TypeError):
                continue
    except Exception as e:
        print(f'BSE parse error: {e}')
    return 0.0


def get_bse_spot() -> float:
    """Get SENSEX spot price from BSE option chain cache."""
    import time as _time
    cached = _bse_cache.get('SENSEX')
    if cached and (_time.time() - cached[0]) < BSE_CACHE_SECS:
        try:
            records = cached[1].get('Table', [])
            if records:
                return float(records[0].get('UnderlyingValue', 0) or 0)
        except Exception:
            pass
    # Trigger a fresh fetch
    get_bse_option_ltp(0, 'CE')
    cached = _bse_cache.get('SENSEX')
    if cached:
        try:
            records = cached[1].get('Table', [])
            if records:
                return float(records[0].get('UnderlyingValue', 0) or 0)
        except Exception:
            pass
    return 0.0


def _get_nse_session():
    global _nse_session
    if _nse_session is None:
        import requests as _req
        _nse_session = _req.Session()
        _nse_session.headers.update(NSE_HEADERS)
        # Hit homepage first to get cookies
        try:
            _nse_session.get('https://www.nseindia.com', timeout=10)
        except Exception:
            pass
    return _nse_session


def get_nse_option_ltp(symbol: str, strike: int, option_type: str) -> float:
    """
    Fetch live option LTP from NSE option chain API.
    symbol: NIFTY / BANKNIFTY  (NOT SENSEX — that is a BSE product, use get_bse_option_ltp)
    strike: 24100
    option_type: CE / PE
    Returns live premium or 0.0 on failure.
    """
    import time as _time
    global _nse_cache

    # SENSEX options are listed on BSE, not NSE — never query NSE for them
    if symbol.upper() == 'SENSEX':
        return 0.0

    nse_sym = {'NIFTY': 'NIFTY', 'BANKNIFTY': 'BANKNIFTY'}.get(symbol, symbol)

    # Use cache if fresh
    cached = _nse_cache.get(nse_sym)
    if cached and (_time.time() - cached[0]) < NSE_CACHE_SECS:
        chain_data = cached[1]
    else:
        try:
            sess = _get_nse_session()
            url  = f'https://www.nseindia.com/api/option-chain-indices?symbol={nse_sym}'
            resp = sess.get(url, timeout=10)
            if resp.status_code == 401 or resp.status_code == 403:
                # Session expired — reinit
                global _nse_session
                _nse_session = None
                sess = _get_nse_session()
                resp = sess.get(url, timeout=10)
            if not resp.text.strip():
                return 0.0
            chain_data = resp.json()
            _nse_cache[nse_sym] = (_time.time(), chain_data)
        except Exception as e:
            print(f'NSE option chain error: {e}')
            return 0.0

    # Parse the chain for the specific strike
    try:
        records = chain_data.get('records', {}).get('data', [])
        for row in records:
            if int(row.get('strikePrice', 0)) == int(strike):
                opt = row.get(option_type, {})
                ltp = float(opt.get('lastPrice', 0))
                if ltp > 0:
                    return ltp
    except Exception as e:
        print(f'NSE parse error: {e}')
    return 0.0


def get_nse_spot(symbol: str) -> float:
    """Get spot price from NSE option chain (most accurate).
    For SENSEX, delegates to get_bse_spot() — SENSEX is a BSE product."""
    import time as _time
    # SENSEX is BSE — NSE has no data for it
    if symbol.upper() == 'SENSEX':
        return get_bse_spot()
    nse_sym = {'NIFTY': 'NIFTY', 'BANKNIFTY': 'BANKNIFTY'}.get(symbol, symbol)
    cached  = _nse_cache.get(nse_sym)
    if cached and (_time.time() - cached[0]) < NSE_CACHE_SECS:
        try:
            return float(cached[1].get('records', {}).get('underlyingValue', 0))
        except: pass
    # Trigger a fresh fetch
    get_nse_option_ltp(symbol, 0, 'CE')
    cached = _nse_cache.get(nse_sym)
    if cached:
        try:
            return float(cached[1].get('records', {}).get('underlyingValue', 0))
        except: pass
    return 0.0



class AngelOneClient:
    def __init__(self, api_key="", client_id="", mpin="", totp_key=""):
        self.api_key   = api_key   or os.getenv('ANGEL_API_KEY','')
        self.client_id = client_id or os.getenv('ANGEL_CLIENT_ID','')
        self.mpin      = mpin      or os.getenv('ANGEL_MPIN','')
        self.totp_key  = totp_key  or os.getenv('ANGEL_TOTP_KEY','')
        self.jwt_token  = None
        self.connected  = False
        self._public_ip = None   # cached on first call

    def _get_public_ip(self) -> str:
        """Fetch real public IP (cached)."""
        if hasattr(self, '_public_ip') and self._public_ip:
            return self._public_ip
        for url in ["https://api.ipify.org", "https://checkip.amazonaws.com"]:
            try:
                r = requests.get(url, timeout=5)
                ip = r.text.strip()
                if ip:
                    self._public_ip = ip
                    return ip
            except:
                pass
        return "106.0.0.1"  # fallback — replace with your actual public IP if needed

    def _headers(self, auth=False):
        import socket, uuid
        try:    local_ip = socket.gethostbyname(socket.gethostname())
        except: local_ip = "127.0.0.1"
        try:
            mac = uuid.getnode()
            mac_str = ':'.join(f'{(mac>>(i*8))&0xff:02x}' for i in range(5,-1,-1))
        except: mac_str = "00:00:00:00:00:00"

        public_ip = self._get_public_ip()

        h = {
            "Content-Type":     "application/json",
            "Accept":           "application/json",
            "User-Agent":       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/124.0.0.0 Safari/537.36",
            "X-UserType":       "USER",
            "X-SourceID":       "WEB",
            "X-ClientLocalIP":  local_ip,
            "X-ClientPublicIP": public_ip,   # ← real public IP (WAF check)
            "X-MACAddress":     mac_str,
            "X-PrivateKey":     self.api_key,
        }
        if auth and self.jwt_token:
            h["Authorization"] = f"Bearer {self.jwt_token}"
        return h

    def connect(self):
        # Try saved token first
        if self._load_token():
            self.connected = True
            return True, f"Connected (saved session) as {self.client_id}"

        # Generate fresh TOTP
        try:
            totp_code = pyotp.TOTP(self.totp_key).now()
        except Exception as e:
            return False, f"TOTP error: {e} — check TOTP secret key"

        # Login — try up to 3 times for TOTP timing
        url = f"{BASE}/rest/auth/angelbroking/user/v1/loginByPassword"
        last_err = ""

        for attempt in range(3):
            try:
                # Exact format from SmartAPI docs
                body = {
                    "clientcode": str(self.client_id).strip(),
                    "userId":     str(self.client_id).strip(),
                    "password":   str(self.mpin).strip(),
                    "totp":       str(pyotp.TOTP(self.totp_key).now())
                }
                resp = requests.post(
                    url, json=body,
                    headers=self._headers(),
                    timeout=15
                )
                raw = resp.text
                try:
                    data = resp.json()
                except:
                    return False, f"Non-JSON response: {raw[:200]}"

                if data.get('status') is True:
                    d = data.get('data', {})
                    self.jwt_token = d.get('jwtToken')
                    self.connected = True
                    self._save_token(d)
                    return True, f"✓ Connected as {self.client_id}"
                else:
                    msg  = data.get('message', 'Unknown error')
                    code = data.get('errorcode', '')
                    last_err = f"{msg} (Code: {code})"

                    # TOTP error — wait and retry
                    if code in ['AB1050'] or 'totp' in msg.lower():
                        time.sleep(5)
                        continue
                    else:
                        return False, f"Login failed: {last_err}"

            except requests.exceptions.ConnectionError:
                return False, "Connection error — check internet"
            except Exception as e:
                last_err = str(e)
                time.sleep(2)

        return False, f"Failed after 3 attempts: {last_err}"

    def _ist_date(self):
        from datetime import timezone, timedelta
        return datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d')

    def _save_token(self, data):
        data['date'] = self._ist_date()
        try:
            with open(TOKEN_FILE, 'w') as f:
                json.dump(data, f)
        except: pass

    def _load_token(self):
        if not os.path.exists(TOKEN_FILE):
            return False
        try:
            with open(TOKEN_FILE) as f:
                data = json.load(f)
            if data.get('date') != self._ist_date():
                return False
            self.jwt_token = data.get('jwtToken')
            return bool(self.jwt_token)
        except:
            return False

    def get_quote(self, symbol="NIFTY"):
        """Get live spot price — BSE for SENSEX, NSE for NIFTY/BANKNIFTY, yfinance fallback."""
        # Try option chain spot (BSE for SENSEX, NSE for others)
        try:
            spot = get_nse_spot(symbol)   # already routes SENSEX → get_bse_spot()
            if spot > 0:
                return {'ltp': spot}
        except Exception as e:
            exchange_label = "BSE" if symbol.upper() == "SENSEX" else "NSE"
            print(f"{exchange_label} spot error: {e}")
        # Fallback: yfinance
        try:
            import yfinance as yf
            sym = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
            tk  = yf.Ticker(sym.get(symbol, "^NSEI"))
            price = float(tk.fast_info.last_price)
            if price > 0:
                return {'ltp': price}
        except Exception as e:
            print(f"yfinance spot error: {e}")
        return {}

    def get_funds(self):
        if not self.connected: return {}
        try:
            url = f"{BASE}/rest/secure/angelbroking/user/v1/getRMS"
            r   = requests.get(url, headers=self._headers(auth=True), timeout=10)
            d   = r.json()
            if d.get('status') and d.get('data'):
                return {
                    'available': float(d['data'].get('availablecash', 0)),
                    'total':     float(d['data'].get('net', 0)),
                }
        except Exception as e:
            print(f"Funds error: {e}")
        return {}

    def get_option_chain(self, symbol="NIFTY", expiry=None):
        if not self.connected: return {}
        try:
            if not expiry:
                from datetime import timedelta
                today  = date.today()
                target = 3 if symbol == "NIFTY" else 2
                days   = (target - today.weekday()) % 7 or 7
                expiry = (datetime.today() + timedelta(days=days)).strftime('%d%b%Y').upper()

            url  = f"{BASE}/rest/secure/angelbroking/marketData/v1/getOptionChain"
            body = {"name": symbol, "expirydate": expiry}
            r    = requests.post(url, json=body,
                                  headers=self._headers(auth=True), timeout=15)
            d    = r.json()
            if d.get('status') and d.get('data'):
                return self._parse_oi(d['data'])
        except Exception as e:
            print(f"OI error: {e}")
        return {}

    def _parse_oi(self, data):
        rows = []; spot = 0.0
        for item in data:
            s   = float(item.get('strikePrice', 0))
            co  = float(item.get('CE', {}).get('openInterest', 0))
            po  = float(item.get('PE', {}).get('openInterest', 0))
            coc = float(item.get('CE', {}).get('changeinOpenInterest', 0))
            poc = float(item.get('PE', {}).get('changeinOpenInterest', 0))
            if item.get('CE', {}).get('underlyingValue'):
                spot = float(item['CE']['underlyingValue'])
            rows.append({'strike':s,'call_oi':co,'put_oi':po,
                         'call_oi_change':coc,'put_oi_change':poc})
        if not rows: return {}
        df  = pd.DataFrame(rows).sort_values('strike')
        df['call_oi'] = df['call_oi'].astype(float)
        df['put_oi']  = df['put_oi'].astype(float)
        tco = df['call_oi'].sum(); tpo = df['put_oi'].sum()
        pcr = round(tpo/tco, 2) if tco > 0 else 1.0
        res = int(df.loc[df['call_oi'].idxmax(), 'strike'])
        sup = int(df.loc[df['put_oi'].idxmax(), 'strike'])
        sig = "BULLISH" if pcr>1.2 else "BEARISH" if pcr<0.8 else "NEUTRAL"
        return {'spot':spot,'pcr':pcr,'key_resistance':res,
                'key_support':sup,'signal':sig,'chain_df':df}


    def get_option_ltp(self, symbol: str, strike: int,
                        option_type: str, expiry: str) -> float:
        """
        Get real live option premium.
        Priority: BSE/NSE option chain → Angel One API → 0.0
        symbol: NIFTY/BANKNIFTY  → NSE chain
                SENSEX           → BSE chain (BSE product, not NSE)
        strike: 24100
        option_type: CE/PE
        expiry: 07MAY2026 (used for Angel One fallback only)
        """
        # 1a. SENSEX options live on BSE — never query NSE for them
        if symbol.upper() == 'SENSEX':
            try:
                ltp = get_bse_option_ltp(strike, option_type)
                if ltp > 0:
                    print(f"Option LTP [BSE] {strike}{option_type}: ₹{ltp:.2f}")
                    return ltp
            except Exception as e:
                print(f"BSE option LTP error: {e}")
        else:
            # 1b. NIFTY / BANKNIFTY → NSE option chain
            try:
                ltp = get_nse_option_ltp(symbol, strike, option_type)
                if ltp > 0:
                    print(f"Option LTP [NSE] {strike}{option_type}: ₹{ltp:.2f}")
                    return ltp
            except Exception as e:
                print(f"NSE option LTP error: {e}")

        # 2. Angel One fallback (only if connected and not WAF blocked)
        if self.connected:
            try:
                url      = f"{BASE}/rest/secure/angelbroking/market/v1/getLTPData"
                opt_char = "C" if option_type == "CE" else "P"
                exp_up   = expiry.upper()
                # SENSEX options trade on BSE's derivatives segment (BFO), not NFO
                exchange = "BFO" if symbol.upper() == "SENSEX" else "NFO"
                candidates = [
                    f"{symbol}{exp_up}{opt_char}{strike}",
                    f"{symbol}{exp_up}{option_type}{strike}",
                ]
                for ts in candidates:
                    body = {"exchange": exchange, "tradingsymbol": ts, "symboltoken": "0"}
                    r = requests.post(url, json=body,
                                      headers=self._headers(auth=True), timeout=8)
                    if r.status_code == 200 and r.text.strip():
                        d = r.json()
                        if d.get('status') and d.get('data'):
                            ltp = float(d['data'].get('ltp', 0))
                            if ltp > 0:
                                print(f"Option LTP [Angel] {strike}{option_type}: ₹{ltp:.2f}")
                                return ltp
            except Exception as e:
                print(f"Angel option LTP error: {e}")

        return 0.0

    def search_option_token(self, symbol: str, strike: int,
                             option_type: str, expiry: str) -> str:
        """Token lookup skipped — WAF blocks searchScrip endpoint.
        NSE option chain is used instead for live prices."""
        return "0"

    def place_order(self, symbol, option_type, strike, expiry, side, qty=1, paper=True):
        if paper:
            return f"PAPER_{int(time.time())}"
        if not self.connected: return None
        try:
            url  = f"{BASE}/rest/secure/angelbroking/order/v1/placeOrder"
            body = {
                "variety":"NORMAL","tradingsymbol":f"{symbol}{expiry}{strike}{option_type}",
                "symboltoken":"0","transactiontype":side,"exchange":"NFO",
                "ordertype":"MARKET","producttype":"INTRADAY",
                "duration":"DAY","quantity":str(qty),"price":"0",
                "squareoff":"0","stoploss":"0"
            }
            r = requests.post(url, json=body,
                               headers=self._headers(auth=True), timeout=10)
            d = r.json()
            return d['data']['orderid'] if d.get('status') else None
        except Exception as e:
            print(f"Order error: {e}")
        return None
