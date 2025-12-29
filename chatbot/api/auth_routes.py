"""
Better-Auth Server API Endpoints for User Authentication and Profiling
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import os
import json
from datetime import datetime

router = APIRouter()

# In-memory storage for demo (replace with Neon Postgres in production)
users_db = {}
user_profiles_db = {}
sessions_db = {}


# Pydantic models
class SignupRequest(BaseModel):
    email: str
    password: str
    profile: dict


class SigninRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    profile: Optional[dict] = None


class AuthResponse(BaseModel):
    user: UserResponse
    session_token: str


@router.post("/api/auth/signup")
async def signup(request: SignupRequest):
    """
    Create a new user account with profile information.

    Stores user credentials and learning profile for personalization.
    """
    try:
        # Check if user exists
        if request.email in users_db:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create user
        user_id = f"user_{len(users_db) + 1}"
        users_db[request.email] = {
            "id": user_id,
            "email": request.email,
            "password": request.password,  # In production, hash this!
            "name": request.email.split("@")[0],
            "created_at": datetime.now().isoformat(),
        }

        # Store profile
        user_profiles_db[user_id] = {
            **request.profile,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Create session
        session_token = f"session_{user_id}_{datetime.now().timestamp()}"
        sessions_db[session_token] = {
            "user_id": user_id,
            "email": request.email,
            "created_at": datetime.now().isoformat(),
        }

        return AuthResponse(
            user=UserResponse(
                id=user_id,
                email=request.email,
                name=request.email.split("@")[0],
                profile=request.profile,
            ),
            session_token=session_token,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/auth/signin")
async def signin(request: SigninRequest):
    """
    Authenticate user and create session.
    """
    try:
        # Find user
        if request.email not in users_db:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        user = users_db[request.email]

        # Verify password (in production, use proper password hashing)
        if user["password"] != request.password:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Create session
        session_token = f"session_{user['id']}_{datetime.now().timestamp()}"
        sessions_db[session_token] = {
            "user_id": user["id"],
            "email": request.email,
            "created_at": datetime.now().isoformat(),
        }

        # Get profile
        profile = user_profiles_db.get(user["id"], {})

        return AuthResponse(
            user=UserResponse(
                id=user["id"],
                email=user["email"],
                name=user["name"],
                profile=profile,
            ),
            session_token=session_token,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/auth/signout")
async def signout(session_token: str):
    """
    End user session.
    """
    try:
        if session_token in sessions_db:
            del sessions_db[session_token]
        return {"status": "success", "message": "Signed out successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/auth/me")
async def get_current_user(session_token: str):
    """
    Get current authenticated user with profile.
    """
    try:
        if session_token not in sessions_db:
            raise HTTPException(status_code=401, detail="Not authenticated")

        session = sessions_db[session_token]
        user_id = session["user_id"]

        if user_id not in user_profiles_db:
            raise HTTPException(status_code=404, detail="Profile not found")

        return {
            "user": {
                "id": user_id,
                "email": session["email"],
                "name": session["email"].split("@")[0],
            },
            "profile": user_profiles_db[user_id],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/auth/profile")
async def update_profile(session_token: str, profile: dict):
    """
    Update user learning profile.
    """
    try:
        if session_token not in sessions_db:
            raise HTTPException(status_code=401, detail="Not authenticated")

        session = sessions_db[session_token]
        user_id = session["user_id"]

        # Update profile
        user_profiles_db[user_id] = {
            **user_profiles_db.get(user_id, {}),
            **profile,
            "updated_at": datetime.now().isoformat(),
        }

        return {
            "status": "success",
            "profile": user_profiles_db[user_id],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/auth/profiling-questions")
async def get_profiling_questions():
    """
    Return profiling questions for signup flow.
    """
    return [
        {
            "key": "programmingLevel",
            "question": "What is your programming experience level?",
            "options": [
                {"value": "beginner", "label": "Beginner (0-1 years)"},
                {"value": "intermediate", "label": "Intermediate (1-3 years)"},
                {"value": "advanced", "label": "Advanced (3+ years)"},
            ],
        },
        {
            "key": "roboticsExperience",
            "question": "What is your prior robotics experience?",
            "options": [
                {"value": "none", "label": "No experience"},
                {"value": "academic", "label": "Academic (courses, research)"},
                {"value": "professional", "label": "Professional (work experience)"},
            ],
        },
        {
            "key": "ros2Familiarity",
            "question": "How familiar are you with ROS 2?",
            "options": [
                {"value": "none", "label": "Never used it"},
                {"value": "basic", "label": "Basic (tried some tutorials)"},
                {"value": "intermediate", "label": "Intermediate (built some nodes)"},
                {"value": "expert", "label": "Expert (production experience)"},
            ],
        },
        {
            "key": "learningGoal",
            "question": "What is your primary learning goal?",
            "options": [
                {"value": "understanding", "label": "Understanding concepts"},
                {"value": "hands-on", "label": "Building projects"},
                {"value": "career", "label": "Career transition"},
            ],
        },
        {
            "key": "hardwareAccess",
            "question": "What hardware do you have access to?",
            "options": [
                {"value": "simulation", "label": "Simulation only (no hardware)"},
                {"value": "jetson", "label": "NVIDIA Jetson Orin"},
                {"value": "full-lab", "label": "Full robotics lab"},
            ],
        },
    ]
