@echo off
title Heroshiko AI Studio Server (Ngrok Tunnel)
cd /d %~dp0
echo ========================================================
echo Starting Heroshiko Server with Public Ngrok Tunnel...
echo ========================================================
uv run python run_server.py --tunnel
pause

