@echo off
chcp 65001 >nul
echo ============================================================================
echo   AGPZAI: Запуск новой версии ИИ-системы предиктивной диагностики (МИТУ)
echo ============================================================================
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
cd /d "%~dp0"
python -m streamlit run app.py --server.port 8501
pause
