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
# GET CHAT HISTORY
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