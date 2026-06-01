"""
backend/routers/auth.py
-----------------------
Minimal auth router. In a real deployment, swap the in-memory user
table for Supabase Auth (preferred) or your IdP of choice -- the
endpoints stay the same.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends

from backend.schemas.models import TokenResponse
from backend.utils.auth import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])

# Demo user table (replace with Supabase / your DB).
_DEMO_USERS = {
    os.getenv("DEMO_USER", "analyst"): hash_password(os.getenv("DEMO_PASSWORD", "analyst")),
}


@router.post("/login", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    hashed = _DEMO_USERS.get(form.username)
    if not hashed or not verify_password(form.password, hashed):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(access_token=create_access_token(form.username))
