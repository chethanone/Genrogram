import os

from fastapi import FastAPI

try:
    from backend.main import app
except Exception as exc:
    # Temporary diagnostic fallback: if the backend fails during module import,
    # keep the Vercel function alive long enough to expose the startup error.
    error_text = f"{type(exc).__name__}: {exc}"
    app = FastAPI()

    @app.get("/health")
    def startup_health():
        return {
            "status": "startup_error",
            "error": error_text,
            "python_version": os.sys.version,
        }

# Vercel Python runtime entrypoint.
# FastAPI is exposed as the ASGI application used by the function.
