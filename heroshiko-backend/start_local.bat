@echo off
title Heroshiko AI Studio Server (Local Only)
cd /d %~dp0
echo ========================================================
echo Starting Heroshiko Server (Local Only)...
echo ========================================================
uv run python run_server.py
pause

