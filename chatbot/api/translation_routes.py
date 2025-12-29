"""
Urdu Translation API Endpoints

Translates textbook content to Urdu while preserving technical terms.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

router = APIRouter()

# In-memory cache for translations
translation_cache = {}


class TranslationRequest(BaseModel):
    chapter_id: str
    chapter_content: str
    target_language: str = "ur"


class TranslationResponse(BaseModel):
    translated_content: str
    cached: bool


def generate_translation_prompt(content: str, target_language: str = "ur") -> str:
    """Generate a GPT prompt for technical translation."""
    language_names = {
        "ur": "Urdu",
        "hi": "Hindi",
        "ar": "Arabic",
        "zh": "Chinese",
    }

    lang = language_names.get(target_language, "Urdu")

    prompt = f"""
You are an expert technical translator specializing in robotics, AI, and computer science.

Translate this textbook chapter from English to {lang}.

IMPORTANT RULES:
1. Keep technical terms in English: ROS 2, URDF, Gazebo, Isaac Sim, Nav2, Python, C++, SLAM, LIDAR, IMU, VLA, VSLAM, etc.
2. Keep all code blocks completely unchanged
3. Keep mathematical formulas and equations as-is
4. Keep all variable names, function names, and code syntax in English
5. Only translate code comments
6. Use Romanized technical terms naturally mixed with {lang} (e.g., "ROS 2 nodes topics کے ذریعے communicate کرتے ہیں")
7. Preserve all markdown formatting exactly
8. Maintain educational clarity and technical accuracy
9. Use natural {lang} flow while keeping technical terms accessible

Example translation style:
"ROS 2 nodes communicate via topics" → "ROS 2 nodes topics کے ذریعے communicate کرتے ہیں"
"The publisher sends messages to a topic" → "Publisher ایک topic پر messages بھیجتا ہے"

Output ONLY the translated markdown content (no explanations, no backticks, no code blocks around it):

English chapter:
{content}

{language_names.get(target_language, 'Urdu')} translation:
"""

    return prompt


@router.post("/api/translate/chapter")
async def translate_chapter(request: TranslationRequest):
    """
    Translate chapter content to target language.

    Preserves technical terms and code blocks.
    """
    try:
        cache_key = f"{request.chapter_id}_{request.target_language}"

        # Check cache
        if cache_key in translation_cache:
            return TranslationResponse(
                translated_content=translation_cache[cache_key],
                cached=True
            )

        # Generate translation prompt
        prompt = generate_translation_prompt(request.chapter_content, request.target_language)

        # Call AI service for translation
        AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()

        if AI_PROVIDER == "openai":
            from api.services.openai_service import OpenAIService
            ai_service = OpenAIService()
            translated = ai_service.chat(
                prompt=prompt,
                max_tokens=8000,
                temperature=0.3  # Lower temperature for consistent translation
            )
        else:
            from api.services.gemini_service import GeminiService
            ai_service = GeminiService()
            translated = ai_service.chat(
                prompt=prompt,
                max_output_tokens=8000,
                temperature=0.3
            )

        # Cache the result
        translation_cache[cache_key] = translated

        return TranslationResponse(
            translated_content=translated,
            cached=False
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/translate/status")
async def get_translation_status(chapter_id: str, target_language: str = "ur"):
    """
    Check if chapter is already translated.
    """
    cache_key = f"{chapter_id}_{target_language}"
    return {
        "cached": cache_key in translation_cache,
        "chapter_id": chapter_id,
        "target_language": target_language
    }


@router.post("/api/translate/invalidate-cache")
async def invalidate_translation_cache(chapter_id: str = None, target_language: str = "ur"):
    """
    Invalidate translation cache.
    """
    try:
        if chapter_id:
            cache_key = f"{chapter_id}_{target_language}"
            if cache_key in translation_cache:
                del translation_cache[cache_key]
        else:
            # Clear all translations for the target language
            keys_to_delete = [k for k in translation_cache if k.endswith(f"_{target_language}")]
            for key in keys_to_delete:
                del translation_cache[key]

        return {"status": "success", "message": "Translation cache invalidated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
