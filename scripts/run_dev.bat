@echo off
title BhuVistaar V2 - Full-Stack Monorepo
echo ===================================================
echo   Starting BhuVistaar: AI Super Resolution System
echo ===================================================
echo.
echo Running unified single command: npm run dev
echo (FastAPI on http://127.0.0.1:8000 + Next.js on http://localhost:3000)
echo.

cd /d "%~dp0\.."
where npm >nul 2>nul
if %ERRORLEVEL% equ 0 (
    npm run dev
) else (
    echo npm not found in PATH, falling back to python scripts/dev.py...
    python scripts\dev.py
)
pause
