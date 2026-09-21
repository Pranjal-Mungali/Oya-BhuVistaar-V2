@echo off
echo ===================================================
echo   Starting BhuVistaar: AI Super Resolution System
echo ===================================================
echo.
echo Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "BhuVistaar Backend" cmd /k "python server.py"

echo Starting Next.js shadcn/ui Dashboard on http://localhost:3000 ...
start "BhuVistaar Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers launched!
echo Open your browser at: http://localhost:3000
