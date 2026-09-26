"""
BhuVistaar - Unified Development Runner
Runs both FastAPI backend (port 8000) and Next.js frontend (port 3000)
concurrently with unified colored console logging and graceful shutdown.

Usage:
    python dev.py
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def stream_output(process, prefix, color):
    """Streams process stdout line-by-line with colored prefix."""
    try:
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            print(f"{color}{BOLD}[{prefix}]{RESET} {line.rstrip()}", flush=True)
    except Exception:
        pass


def main():
    print(f"\n{BOLD}{GREEN}======================================================={RESET}")
    print(f"{BOLD}{GREEN}  Starting BhuVistaar Monorepo (Full-Stack Dev Server)  {RESET}")
    print(f"{BOLD}{GREEN}======================================================={RESET}\n")

    print(f"  {CYAN}* FastAPI Backend:{RESET}   http://127.0.0.1:8000 (Swagger docs: http://127.0.0.1:8000/docs)")
    print(f"  {MAGENTA}* Next.js Frontend:{RESET}  http://localhost:3000")
    print(f"  {YELLOW}Press Ctrl+C at any time to stop both servers.{RESET}\n")

    # Determine command executables
    python_cmd = sys.executable
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

    # Start backend
    backend_proc = subprocess.Popen(
        [python_cmd, "-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Start frontend
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    t_backend = threading.Thread(
        target=stream_output,
        args=(backend_proc, "Backend", CYAN),
        daemon=True
    )
    t_frontend = threading.Thread(
        target=stream_output,
        args=(frontend_proc, "Frontend", MAGENTA),
        daemon=True
    )

    t_backend.start()
    t_frontend.start()

    try:
        while True:
            # Check if either process exited unexpectedly
            b_ret = backend_proc.poll()
            f_ret = frontend_proc.poll()

            if b_ret is not None:
                print(f"\n{RED}[Backend] Process exited with code {b_ret}{RESET}")
                break
            if f_ret is not None:
                print(f"\n{RED}[Frontend] Process exited with code {f_ret}{RESET}")
                break

            time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}[BhuVistaar] Stopping all services...{RESET}")

    finally:
        for name, proc in [("Backend", backend_proc), ("Frontend", frontend_proc)]:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        print(f"{GREEN}[BhuVistaar] All services shut down successfully.{RESET}")


if __name__ == "__main__":
    main()
