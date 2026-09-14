@echo off
title CrowdGuard Launcher

cd /d "D:\FINAL YEAR PROJECT\Explainable-Crowd-Risk-AI"

echo Starting CrowdGuard Backend...
start "CrowdGuard Backend" cmd /k ".venv\Scripts\activate && python -m uvicorn backend.main:app --reload"

timeout /t 3 /nobreak >nul

echo Starting CrowdGuard Frontend...
start "CrowdGuard Frontend" cmd /k "cd frontend && npm run dev"

timeout /t 4 /nobreak >nul

echo Opening CrowdGuard...
start http://localhost:5173

exit