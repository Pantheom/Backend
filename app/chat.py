from fastapi import APIRouter, Depends

from .database import supabase
from app.dependencies import get_current_user
from app.schemas import ChatMessageRequest


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


# =========================================
# SAVE USER MESSAGE
# =========================================

@router.post("/")
async def save_chat_message(
    data: ChatMessageRequest,
    current_user=Depends(get_current_user),
):

    uid = current_user["uid"]

    result = (
        supabase
        .table("chat_history")
        .insert(
            {
                "uid": uid,
                "session_id": str(data.session_id),
                "role": "user",
                "message": data.message,
            }
        )
        .execute()
    )

    return {
        "success": True,
        "data": result.data,
    }


# =========================================
# GET ALL USER CHAT HISTORY
# =========================================

@router.get("/")
async def get_user_chat_history(
    session_id: str | None = None,
    current_user=Depends(get_current_user),
):

    uid = current_user["uid"]

    query = (
        supabase
        .table("chat_history")
        .select("*")
        .eq("uid", uid)
    )

    if session_id:
        query = query.eq("session_id", session_id)

    result = query.order("created_at").execute()

    return {
        "success": True,
        "data": result.data,
    }


# =========================================
# GET CHAT HISTORY BY SESSION
# =========================================

@router.get("/{session_id}")
async def get_chat_history(
    session_id: str,
    current_user=Depends(get_current_user),
):

    uid = current_user["uid"]

    result = (
        supabase
        .table("chat_history")
        .select("*")
        .eq("uid", uid)
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )

    return {
        "success": True,
        "data": result.data,
    }