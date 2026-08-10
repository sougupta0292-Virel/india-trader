"""Replace Options Backtest sub-tab content with a redirect notice."""
APP = r"C:\Users\soura\Desktop\India trader\app.py"

with open(APP, encoding='utf-8') as f:
    src = f.read()

START = '''            # ── SUB-TAB C: OPTIONS BACKTEST ────────────────────────────────────────
            with _opt_subtabs[2]:
                st.markdown('<div style="font-family:JetBrains Mono;font-size:14px;font-weight:700;color:#60a5fa;padding:6px 0">🧪 Options Strategy Backtest</div>', unsafe_allow_html=True)
                _bt_c1, _bt_c2, _bt_c3 = st.columns(3)'''

END_MARKER = '''            # ── SUB-TAB D: STRATEGY LIBRARY ───────────────────────────────────────'''

REPLACEMENT = '''            # ── SUB-TAB C: OPTIONS BACKTEST (moved) ──────────────────────────────
            with _opt_subtabs[2]:
                st.info("📈 Options Backtest moved to the **🧪 Backtest** tab → **📈 Options Backtest** sub-tab.")

            # ── SUB-TAB D: STRATEGY LIBRARY ───────────────────────────────────────'''

i_start = src.find(START)
i_end   = src.find(END_MARKER, i_start)

if i_start == -1:
    print("START not found")
elif i_end == -1:
    print("END not found")
else:
    src = src[:i_start] + REPLACEMENT + src[i_end + len(END_MARKER):]
    print(f"Replaced Options Backtest tab content ({i_end - i_start} chars) with redirect")
    with open(APP, 'w', encoding='utf-8') as f:
        f.write(src)
    print("Saved.")

import ast
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
