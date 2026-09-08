from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# =========================================
# AUTH
# =========================================

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    uid: UUID
    email: EmailStr


# =========================================
# CHAT
# =========================================

class ChatMessageRequest(BaseModel):
    session_id: UUID
    message: str = Field(min_length=1)


class ChatMessageResponse(BaseModel):
    id: int
    uid: UUID
    session_id: UUID
    role: str
    message: str
    created_at: str