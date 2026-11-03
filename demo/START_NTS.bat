@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
title AI-PromPlatform - NTS-1
python server.py
if errorlevel 1 py server.py
pause
