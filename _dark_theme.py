"""Bulk-replace all inline HTML style colors to dark theme."""
import re

APP = r"C:\Users\soura\Desktop\India trader\app.py"

with open(APP, encoding='utf-8') as f:
    src = f.read()

# Color map: light → dark
# Order matters — more specific first where needed
COLORS = [
    # Backgrounds
    ("background:#ffffff",     "background:#0f1729"),
    ("background: #ffffff",    "background:#0f1729"),
    ("background:#fff",        "background:#0f1729"),
    ("background:#f8f9fa",     "background:#151d2e"),
    ("background:#f0fdf4",     "background:#0d2018"),
    ("background:#fef2f2",     "background:#280d0d"),
    ("background:#fffde7",     "background:#241f0a"),
    ("background:#fff8e1",     "background:#231e0a"),
    ("background:#f1f5f9",     "background:#1a2235"),
    ("background:#f0f4f8",     "background:#0d1526"),
    ("background:#e8f5e9",     "background:#0a2212"),
    ("background:#dcfce7",     "background:#0a2212"),
    ("background:#fee2e2",     "background:#280d0d"),
    ("background:#fffbeb",     "background:#241f0a"),
    # Borders
    ("border:#dee2e6",         "border:#2d4a6e"),
    ("border: #dee2e6",        "border:#2d4a6e"),
    ("solid #dee2e6",          "solid #2d4a6e"),
    ("solid #e2e8f0",          "solid #2d4a6e"),
    # Text colors
    ("color:#212529",          "color:#e2e8f0"),
    ("color: #212529",         "color:#e2e8f0"),
    ("color:#6c757d",          "color:#94a3b8"),
    ("color: #6c757d",         "color:#94a3b8"),
    ("color:#64748b",          "color:#94a3b8"),
    ("color:#1565c0",          "color:#60a5fa"),
    ("color: #1565c0",         "color:#60a5fa"),
    ("color:#00e5ff",          "color:#38bdf8"),
    ("color:#00e676",          "color:#4ade80"),
    # Green P&L  (slightly brighter for dark bg)
    ("color:#1b8a4a",          "color:#16a34a"),
    ("color: #1b8a4a",         "color:#16a34a"),
    ("border-left-color:#1b8a4a", "border-left-color:#16a34a"),
    ("border:2px solid #1b8a4a",  "border:2px solid #16a34a"),
    ("border:3px solid #1b8a4a",  "border:3px solid #16a34a"),
    # Red P&L
    ("color:#c0392b",          "color:#ef4444"),
    ("color: #c0392b",         "color:#ef4444"),
    ("border:2px solid #c0392b",  "border:2px solid #ef4444"),
    ("border:3px solid #c0392b",  "border:3px solid #ef4444"),
    # Gold
    ("color:#b8860b",          "color:#f59e0b"),
    ("border:3px solid #b8860b",  "border:3px solid #f59e0b"),
    ("border-left-color:#b8860b", "border-left-color:#f59e0b"),
    # Table headers
    ('<tr style="background:#f1f5f9"',  '<tr style="background:#1a2235"'),
    ('<tr style="background:#f8f9fa"',  '<tr style="background:#151d2e"'),
    # Plot backgrounds
    ('plot_bgcolor="#ffffff"',   'plot_bgcolor="#111827"'),
    ('paper_bgcolor="#ffffff"',  'paper_bgcolor="#111827"'),
    ('plot_bgcolor="#f8f9fa"',   'plot_bgcolor="#111827"'),
    ('paper_bgcolor="#f8f9fa"',  'paper_bgcolor="#111827"'),
    # Sidebar header inline
    ('color:#1565c0;font-size:20px', 'color:#60a5fa;font-size:20px'),
    # Zeroline colors
    ('zerolinecolor="#94a3b8"',  'zerolinecolor="#2d4a6e"'),
    ('zerolinecolor="#dee2e6"',  'zerolinecolor="#2d4a6e"'),
    # Tick fonts in plotly
    ('"#6c757d"', '"#94a3b8"'),
]

original = src
for old, new in COLORS:
    src = src.replace(old, new)

changed = sum(1 for a, b in COLORS if a in original and a != b)
print(f"Applied {changed} color replacement rules")
print(f"Total chars changed: {len(original) - src.count(original[:100])}")

with open(APP, 'w', encoding='utf-8') as f:
    f.write(src)

print("Done — dark theme applied.")

# Syntax check
import ast
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
