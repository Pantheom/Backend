from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends

from app.auth import router as auth_router
from app.chat import router as chat_router
from app.ai_services.router import router as ai_router
from app.ai_services.client import close_cache_client
from app.dependencies import get_current_user


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — nothing to initialise (clients are lazy-created on first use)
    yield
    # Shutdown — cleanly close the httpx connection pool
    await close_cache_client()

app = FastAPI(
    title="Token Optimization Backend",
    version="0.1.0",
    lifespan=lifespan,
)


# =========================================
# ROUTERS
# =========================================

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(ai_router)


# =========================================
# HEALTH CHECK
# =========================================

@app.get("/")
async def root():
    return {
        "message": "Backend is running",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
    }



@app.get("/me")
async def get_me(
    current_user=Depends(get_current_user),
):
    return current_user