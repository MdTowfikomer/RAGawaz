"""
Voice Language Detection and Metadata Module.

Provides fast, high-accuracy language identification, script classification,
and code-mixed Hinglish detection across:
- Hindi (Devanagari)
- English (Latin)
- Hinglish (Code-mixed Latin/Indic)
- Marathi (Devanagari)
- Tamil (Tamil script)
- Bengali (Bengali script)
"""

import re
from typing import Dict, Any, Optional

# Distinctive Hinglish / Indic grammatical function words written in Roman script
# (strictly excluding words that collide with common English words like 'to', 'me', 'in', 'the')
HINGLISH_WORDS = {
    "kya", "hai", "hain", "kyu", "kyun", "tha", "thi", "thay", "thein", "ka", "ki", "ke", "ko",
    "mein", "par", "se", "ne", "aur", "ya", "bhi", "toh", "kaise",
    "kab", "kaha", "kahan", "karo", "karna", "karta", "karte", "raha", "rahi", "rahe",
    "mujhe", "mera", "meri", "mere", "aap", "hum", "batao", "bataiye", "hota", "hoti",
    "hote", "hoga", "hogi", "honge", "wala", "wali", "wale", "kuch", "kuchh", "sab",
    "nahi", "nahin", "achha", "achhi", "bohot", "bahut"
}

MARATHI_WORDS = {
    "आहे", "नाही", "काय", "कसे", "झाले", "करतो", "करते", "करतात", "आहेत", "होते", "होती",
    "कोणती", "कोणता", "कोणते", "सांगा", "म्हणजे"
}

LANGUAGE_DISPLAY_MAP = {
    "hindi": "हिन्दी",
    "english": "English",
    "hinglish": "Hinglish",
    "marathi": "मराठी",
    "tamil": "தமிழ்",
    "bengali": "বাংলা",
    "unknown": "Unknown",
}

LANGUAGE_CODE_MAP = {
    "hindi": "hi-IN",
    "english": "en-IN",
    "hinglish": "hi-EN",
    "marathi": "mr-IN",
    "tamil": "ta-IN",
    "bengali": "bn-IN",
    "unknown": "unknown",
}


def detect_language_metadata(
    text: str,
    provider_lang_code: Optional[str] = None,
    provider_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Determine detected language, script, confidence, and display label from transcript text
    and upstream STT provider metadata.
    """
    clean_text = text.strip() if text else ""
    if not clean_text:
        return {
            "text": "",
            "detected_language": "unknown",
            "language_display": "Unknown",
            "detected_language_code": "unknown",
            "language_confidence": 0.0,
        }

    # 1. Script Detection
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', clean_text))
    has_bengali = bool(re.search(r'[\u0980-\u09FF]', clean_text))
    has_tamil = bool(re.search(r'[\u0B80-\u0BFF]', clean_text))
    has_latin = bool(re.search(r'[a-zA-Z]', clean_text))

    detected_lang = "unknown"
    confidence = provider_confidence if provider_confidence is not None else 0.92

    # 2. Check for Hinglish / Code-mixed Latin speech
    if has_latin:
        tokens = re.findall(r'\b[a-zA-Z]+\b', clean_text.lower())
        hinglish_token_count = sum(1 for t in tokens if t in HINGLISH_WORDS)
        
        # If at least one distinct grammatical Hinglish marker is present in a phrase
        if hinglish_token_count >= 1 and len(tokens) >= 2:
            detected_lang = "hinglish"
            confidence = min(0.96, 0.75 + (hinglish_token_count / len(tokens)) * 0.25)
        elif provider_lang_code in ["en-IN", "en-US", "en", "english"]:
            detected_lang = "english"
        elif not has_devanagari and not has_bengali and not has_tamil:
            detected_lang = "english"

    # 3. Check for Indic scripts
    if detected_lang == "unknown" or has_devanagari or has_bengali or has_tamil:
        if has_bengali:
            detected_lang = "bengali"
        elif has_tamil:
            detected_lang = "tamil"
        elif has_devanagari:
            tokens = set(re.findall(r'[\u0900-\u097F]+', clean_text))
            if tokens.intersection(MARATHI_WORDS) or (provider_lang_code and provider_lang_code.startswith("mr")):
                detected_lang = "marathi"
            else:
                detected_lang = "hindi"

    # 4. Check provider hint if still unknown
    if detected_lang == "unknown" and provider_lang_code:
        norm_code = provider_lang_code.lower()
        if "hi" in norm_code:
            detected_lang = "hindi"
        elif "en" in norm_code:
            detected_lang = "english"
        elif "mr" in norm_code:
            detected_lang = "marathi"
        elif "ta" in norm_code:
            detected_lang = "tamil"
        elif "bn" in norm_code:
            detected_lang = "bengali"

    display_name = LANGUAGE_DISPLAY_MAP.get(detected_lang, "Unknown")
    lang_code = LANGUAGE_CODE_MAP.get(detected_lang, "unknown")

    return {
        "text": clean_text,
        "detected_language": detected_lang,
        "language_display": display_name,
        "detected_language_code": lang_code,
        "language_confidence": round(confidence, 2) if confidence is not None else None,
    }
