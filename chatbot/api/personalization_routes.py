"""
Content Personalization API Endpoints

Adapts textbook content based on user profile using GPT-4.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import json

router = APIRouter()

# In-memory cache for personalized content
personalization_cache = {}


class PersonalizationRequest(BaseModel):
    chapter_id: str
    user_id: str
    chapter_content: str
    profile: dict


class PersonalizationResponse(BaseModel):
    personalized_content: str
    cached: bool


def generate_personalization_prompt(content: str, profile: dict) -> str:
    """Generate a GPT prompt for content personalization."""
    programming_level = profile.get('programmingLevel', 'intermediate')
    robotics_experience = profile.get('roboticsExperience', 'none')
    ros2_familiarity = profile.get('ros2Familiarity', 'none')
    learning_goal = profile.get('learningGoal', 'understanding')
    hardware_access = profile.get('hardwareAccess', 'simulation')

    prompt = f"""
You are an expert technical educator specializing in robotics and AI.

Personalize this textbook chapter for a student with the following background:

- Programming Experience: {programming_level}
- Robotics Experience: {robotics_experience}
- ROS 2 Familiarity: {ros2_familiarity}
- Learning Goal: {learning_goal}
- Hardware Access: {hardware_access}

ADAPTATION RULES:

1. **Explanation Detail Level:**
   - Beginner: Add analogies, step-by-step explanations, simple examples
   - Intermediate: Standard technical explanations with practical tips
   - Advanced: Concise explanations, focus on advanced concepts and optimizations

2. **Code Examples:**
   - Beginner: Extensive comments, explain each line, provide more scaffolding
   - Intermediate: Standard comments, focus on practical implementation
   - Advanced: Minimal comments, focus on best practices and performance

3. **Content to Include/Exclude:**
   - If no ROS 2 experience: Include extra ROS 2 setup guidance and basics
   - If simulation only: Skip hardware deployment sections
   - If has Jetson: Include edge deployment tips
   - If career goal: Include industry context and job-relevant skills

4. **Tone:**
   - Understanding goal: Focus on conceptual clarity
   - Hands-on goal: Focus on practical implementation
   - Career goal: Focus on industry relevance and best practices

IMPORTANT:
- Keep all technical terms (ROS 2, URDF, Gazebo, Isaac Sim, Nav2, etc.) in English
- Keep all code blocks unchanged
- Preserve all markdown formatting
- Only adapt the explanatory text
- Output ONLY the personalized markdown content (no explanations)

Original chapter content:
{content}

Personalized chapter content:
"""

    return prompt


@router.post("/api/personalize/chapter")
async def personalize_chapter(request: PersonalizationRequest):
    """
    Personalize chapter content based on user profile.

    Uses GPT-4 to adapt content and caches the result.
    """
    try:
        cache_key = f"{request.user_id}_{request.chapter_id}"

        # Check cache
        if cache_key in personalization_cache:
            return PersonalizationResponse(
                personalized_content=personalization_cache[cache_key],
                cached=True
            )

        # Get profile from user_profiles_db
        from api.auth_routes import user_profiles_db
        user_profile = user_profiles_db.get(request.user_id, {})

        # Merge provided profile with stored profile
        profile = {**user_profile, **request.profile}

        # Generate personalization prompt
        prompt = generate_personalization_prompt(request.chapter_content, profile)

        # Call AI service for personalization
        AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()

        if AI_PROVIDER == "openai":
            from api.services.openai_service import OpenAIService
            ai_service = OpenAIService()
            personalized = ai_service.chat(
                prompt=prompt,
                max_tokens=8000,
                temperature=0.5
            )
        else:
            from api.services.gemini_service import GeminiService
            ai_service = GeminiService()
            personalized = ai_service.chat(
                prompt=prompt,
                max_output_tokens=8000,
                temperature=0.5
            )

        # Cache the result
        personalization_cache[cache_key] = personalized

        return PersonalizationResponse(
            personalized_content=personalized,
            cached=False
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/personalize/invalidate-cache")
async def invalidate_personalization_cache(user_id: str, chapter_id: str = None):
    """
    Invalidate personalization cache for a user.
    Call this when user profile is updated.
    """
    try:
        if chapter_id:
            cache_key = f"{user_id}_{chapter_id}"
            if cache_key in personalization_cache:
                del personalization_cache[cache_key]
        else:
            # Invalidate all cached content for user
            keys_to_delete = [k for k in personalization_cache if k.startswith(f"{user_id}_")]
            for key in keys_to_delete:
                del personalization_cache[key]

        return {"status": "success", "message": "Cache invalidated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/personalize/status")
async def get_personalization_status(user_id: str, chapter_id: str):
    """
    Check if chapter content is already personalized for user.
    """
    cache_key = f"{user_id}_{chapter_id}"
    return {
        "cached": cache_key in personalization_cache,
        "chapter_id": chapter_id,
        "user_id": user_id
    }
