from fastapi import APIRouter, HTTPException, status

from .database import supabase
from app.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.security import (
    create_access_token,
    hash_password,
    verify_password,
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# =========================================
# REGISTER
# =========================================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(data: RegisterRequest):

    # Check if email already exists
    existing = (
        supabase
        .table("users")
        .select("uid")
        .eq("email", str(data.email))
        .execute()
    )

    if existing.data:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Hash password
    hashed_password = hash_password(data.password)

    # Insert user
    result = (
        supabase
        .table("users")
        .insert(
            {
                "email": str(data.email),
                "password_hash": hashed_password,
            }
        )
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=500,
            detail="Failed to create user",
        )

    user = result.data[0]

    return {
        "uid": user["uid"],
        "email": user["email"],
    }


# =========================================
# LOGIN
# =========================================

@router.post(
    "/login",
    response_model=TokenResponse,
)
async def login(data: LoginRequest):

    result = (
        supabase
        .table("users")
        .select("uid, email, password_hash")
        .eq("email", str(data.email))
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    user = result.data[0]

    password_valid = verify_password(
        data.password,
        user["password_hash"],
    )

    if not password_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token = create_access_token(
        uid=str(user["uid"]),
        email=user["email"],
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }