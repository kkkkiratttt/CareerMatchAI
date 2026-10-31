from __future__ import annotations

import io
import os
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from core import run_pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB per resume

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)
LAST_REQUEST_BY_IP: dict[str, float] = {}
MAX_ACTIVE_JOBS = 1
REQUEST_COOLDOWN_SECONDS = 45
JOB_TTL_SECONDS = 45 * 60
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def cleanup_old_jobs():
    now = time.time()
    with LOCK:
        expired = [key for key, value in JOBS.items() if now - value.get("created", now) > JOB_TTL_SECONDS]
        for key in expired:
            JOBS.pop(key, None)
        for ip, last in list(LAST_REQUEST_BY_IP.items()):
            if now - last > REQUEST_COOLDOWN_SECONDS * 4:
                LAST_REQUEST_BY_IP.pop(ip, None)


def execute_job(job_id: str, filename: str, payload: bytes):
    def progress(value: float, stage: str):
        with LOCK:
            job = JOBS.get(job_id)
            if job:
                job["status"] = "running"
                job["progress"] = max(0, min(100, int(value * 100)))
                job["stage"] = stage

    try:
        result = run_pipeline(filename, payload, progress)
        with LOCK:
            if job_id in JOBS:
                JOBS[job_id].update({
                    "status": "completed", "progress": 100, "stage": "Complete",
                    "result": {k: v for k, v in result.items() if k != "zip_bytes"},
                    "zip_bytes": result["zip_bytes"],
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
    except Exception as exc:
        with LOCK:
            if job_id in JOBS:
                JOBS[job_id].update({
                    "status": "failed", "stage": "Could not complete this run",
                    "error": f"{type(exc).__name__}: {str(exc)[:450]}",
                })
    finally:
        # The uploaded resume is only passed to the worker for this run; it is not written to disk.
        payload = b""


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok", "service": "CareerMatch AI"})


@app.post("/api/jobs")
def start_job():
    cleanup_old_jobs()
    upload = request.files.get("resume")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a resume file first."}), 400
    filename = os.path.basename(upload.filename)
    extension = os.path.splitext(filename.lower())[1]
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Upload a PDF, DOCX, or TXT resume."}), 400
    payload = upload.read()
    if not payload:
        return jsonify({"error": "The uploaded file is empty."}), 400
    if len(payload) > app.config["MAX_CONTENT_LENGTH"]:
        return jsonify({"error": "The resume is larger than 10 MB."}), 413

    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    now = time.time()
    with LOCK:
        last = LAST_REQUEST_BY_IP.get(ip, 0)
        if now - last < REQUEST_COOLDOWN_SECONDS:
            remaining = int(REQUEST_COOLDOWN_SECONDS - (now - last))
            return jsonify({"error": f"Please wait {remaining} seconds before starting another search."}), 429
        active = sum(1 for item in JOBS.values() if item.get("status") in {"queued", "running"})
        if active >= MAX_ACTIVE_JOBS:
            return jsonify({"error": "CareerMatch AI is processing another resume. Please try again shortly."}), 429
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {
            "status": "queued", "progress": 0, "stage": "Preparing your search",
            "created": now, "filename": filename,
        }
        LAST_REQUEST_BY_IP[ip] = now
    EXECUTOR.submit(execute_job, job_id, filename, payload)
    return jsonify({"job_id": job_id}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "This search has expired. Please upload your resume again."}), 404
        response = {k: v for k, v in job.items() if k not in {"zip_bytes", "created"}}
    return jsonify(response)


@app.get("/api/jobs/<job_id>/download")
def download_report(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job or job.get("status") != "completed" or not job.get("zip_bytes"):
            return jsonify({"error": "The report is not ready or has expired."}), 404
        content = job["zip_bytes"]
        filename = "CareerMatch_AI_Reports.zip"
    return send_file(io.BytesIO(content), mimetype="application/zip", as_attachment=True, download_name=filename)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify({"error": "The resume is larger than 10 MB."}), 413


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"error": "An unexpected server error occurred. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
