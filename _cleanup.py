"""Remove the leftover old Power of Stocks backtest code from _st_tabs[3]."""
APP = r"C:\Users\soura\Desktop\India trader\app.py"

with open(APP, encoding='utf-8') as f:
    src = f.read()

# Find and remove everything from the bad if False: block to the old caption
START = '''
        if False:  # old code removed — backtest is now in _bt_tabs[3]
            if st.button("⚡ Run Power of Stocks Backtest", type="primary", width='stretch'):
            strat_key = strat_map[pos_strat]'''

END_MARKER = '''        st.caption("⚠️ Daily candle backtest. Real intraday execution may differ. Always paper trade first!")'''

i_start = src.find(START)
i_end   = src.find(END_MARKER, i_start)

if i_start == -1:
    print("START not found")
elif i_end == -1:
    print("END not found")
else:
    # Remove from START to just after the caption line (including the caption)
    after_end = src.find('\n', i_end) + 1
    src = src[:i_start] + '\n' + src[after_end:]
    print(f"Removed {i_end - i_start} chars of old code")

    with open(APP, 'w', encoding='utf-8') as f:
        f.write(src)
    print("Saved.")

import ast
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
