"""
MEMANTO FastAPI Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from memanto.app import __version__
from memanto.app.config import settings
from memanto.app.routes import context, health, memory, namespaces, sessions
from memanto.app.ui.routes.ui_router import mount_ui_static
from memanto.app.ui.routes.ui_router import router as ui_router

# Create FastAPI app
app = FastAPI(
    title="Mnemosyne - Memory that AI Agents Love!",
    description="Mnemosyne — a memory layer service for agentic AI systems. Autonomous local backend (SQLite + embeddings) or Moorcheh cloud.",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])

# Session-Based API (Primary)
app.include_router(sessions.router, prefix="/api/v2", tags=["Sessions & Agents"])


# Internal/Advanced APIs
app.include_router(
    namespaces.router, prefix="/api/v1/namespaces", tags=["Namespaces (Internal)"]
)
app.include_router(memory.router, prefix="/api/v1/memory", tags=["Memory (Internal)"])
app.include_router(context.router, prefix="/api/v2/context", tags=["Context"])

# Web UI Dashboard
app.include_router(ui_router, tags=["Web UI"])
mount_ui_static(app)


@app.get("/")
async def root():
    return {
        "service": "MEMANTO",
        "description": "Memory that AI Agents Love!",
        "version": __version__,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
