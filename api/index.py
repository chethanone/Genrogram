import sys

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
def health():
    return {
        "status": "runtime_ok",
        "python_version": sys.version,
    }

@app.get("/")
def root():
    return {"service": "genrogram-backend", "status": "runtime_ok"}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def backend_placeholder(path: str):
    return JSONResponse(
        status_code=503,
        content={
            "status": "backend_not_loaded",
            "message": "Vercel runtime is healthy; ML backend loading is the next diagnostic step.",
        },
    )
