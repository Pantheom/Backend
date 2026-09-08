from fastapi import FastAPI

from app.auth import router as auth_router
from app.chat import router as chat_router

from fastapi import Depends
from app.dependencies import get_current_user

app = FastAPI(
    title="Token Optimization Backend",
    version="0.1.0",
)


# =========================================
# ROUTERS
# =========================================

app.include_router(auth_router)
app.include_router(chat_router)


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