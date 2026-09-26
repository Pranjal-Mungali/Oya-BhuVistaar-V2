"""
Background Job Manager for BhuVistaar Pipeline
Tracks asynchronous inference, processing steps, progress, and results.
"""

import uuid
import time
import threading
from typing import Dict, Any, Optional, Callable

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def create_job(self, job_type: str = "super_resolution", params: Optional[Dict[str, Any]] = None) -> str:
        job_id = str(uuid.uuid4())
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id,
                "type": job_type,
                "status": "QUEUED",
                "current_step": "QUEUED",
                "progress": 0,
                "created_at": time.time(),
                "updated_at": time.time(),
                "params": params or {},
                "logs": ["Job queued."],
                "result": None,
                "error": None
            }
        return job_id

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        step: Optional[str] = None,
        progress: Optional[int] = None,
        log: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        with self.lock:
            if job_id not in self.jobs:
                return
            job = self.jobs[job_id]
            job["updated_at"] = time.time()
            if status:
                job["status"] = status
            if step:
                job["current_step"] = step
            if progress is not None:
                job["progress"] = progress
            if log:
                job["logs"].append(f"[{time.strftime('%H:%M:%S')}] {log}")
            if result is not None:
                job["result"] = result
            if error is not None:
                job["error"] = error
                job["status"] = "FAILED"

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.jobs.get(job_id)

    def run_in_background(self, target: Callable, *args, **kwargs):
        thread = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

job_manager = JobManager()
