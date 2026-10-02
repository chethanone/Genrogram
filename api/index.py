import os
import sys

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()
_backend_app = None
_startup_error = None

@app.on_event("startup")
async def load_backend():
    global _backend_app, _startup_error
    try:
        from backend.main import app as backend_app
        _backend_app = backend_app
    except Exception as exc:
        _startup_error = f"{type(exc).__name__}: {exc}"

@app.get("/health")
def health():
    if _startup_error:
        return JSONResponse(
            status_code=500,
            content={
                "status": "startup_error",
                "error": _startup_error,
                "python_version": sys.version,
            },
        )
    if _backend_app is None:
        return JSONResponse(status_code=503, content={"status": "backend_not_loaded"})
    return {"status": "ok"}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def fallback(path: str):
    if _startup_error:
        return JSONResponse(
            status_code=500,
            content={"status": "startup_error", "error": _startup_error},
        )
    return JSONResponse(status_code=503, content={"status": "backend_not_loaded"})
