@echo off
cd /d "C:\Users\soura\Desktop\India trader"
echo.
echo  Starting India Trader — please wait...
echo.
start "India Trader" python -m streamlit run app.py --server.port 8501 --server.headless true
timeout /t 10 /nobreak >nul
start "" "http://localhost:8501"
