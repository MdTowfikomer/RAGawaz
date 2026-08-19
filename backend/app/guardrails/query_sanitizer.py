"""
Query Sanitizer Module.

Strips adversarial/jailbreak framing and prompt injection wrappers while extracting
the legitimate factual query underneath. Ensures malicious requests still get caught
by downstream domain & safety classifiers, while legitimate queries wrapped in conversational/
adversarial phrasing are answered accurately.
"""

import re
from typing import Tuple

# Common jailbreak / prompt injection wrapper patterns to strip from the beginning or end
WRAPPER_PATTERNS = [
    re.compile(r'^\s*(ignore|disregard|forget)\s+(all\s+)?(previous|prior|your)?\s*(instructions|directives|rules|guidelines|filters)\s*(and\s+(answer|tell\s+me|state|explain|provide))?[:,\s-]*', re.IGNORECASE),
    re.compile(r'^\s*(hypothetically\s+speaking|in\s+a\s+fictional\s+world|theoretically)[\s,]+(if\s+you\s+were\s+unrestricted[\s,]*)?', re.IGNORECASE),
    re.compile(r'^\s*(dan\s+mode|developer\s+mode|unrestricted\s+mode|admin\s+mode)\s*(enabled)?[:,\s-]*', re.IGNORECASE),
    re.compile(r'^\s*(system\s+prompt\s+override|admin\s+override(\s+level\s+\d+)?)\s*[:,\s-]*\s*(answer\s+factual\s+inquiry\s*[:,-]*)?', re.IGNORECASE),
    re.compile(r'^\s*(you\s+are\s+now\s+(in\s+research\s+mode|a\s+different\s+ai|an\s+uncensored\s+[\w\s]{1,20}|unrestricted))[:,\s-]*', re.IGNORECASE),
    re.compile(r'^\s*(pretend|act\s+as\s+if|simulate\s+being)\s+(you\s+are\s+)?(an?\s+)?([a-zA-Z\s]{1,30}?)\s*(and\s+(tell\s+me|answer|state|explain)\s*[:,-]*|[:,-]+)', re.IGNORECASE),
    re.compile(r'^\s*repeat\s+after\s+me\s*:[^.?!]+[.?!]\s*(now\s+(tell\s+me|answer|state))?[:,\s-]*', re.IGNORECASE),
    re.compile(r'^\s*(can\s+you\s+)?(bypass\s+all\s+filters|don\'?t\s+follow\s+your\s+standard\s+rules)\s*(just\s+to\s+state|simply\s+tell\s+me)?[:,\s-]*', re.IGNORECASE),
    re.compile(r'^\s*(disregard\s+all\s+safety\s+guidelines\s+and\s+(state|tell\s+me))[:,\s-]*', re.IGNORECASE),
]


class QuerySanitizer:
    """
    Strips adversarial prefixes and extracts underlying factual inquiries.
    """

    @staticmethod
    def sanitize(query: str) -> Tuple[str, bool]:
        """
        Sanitizes query by removing known adversarial wrappers.
        
        Returns:
            (sanitized_query: str, was_wrapped: bool)
        """
        raw = query.strip()
        cleaned = raw
        was_wrapped = False
        
        # Iteratively strip wrapper prefixes
        for pat in WRAPPER_PATTERNS:
            new_cleaned = pat.sub('', cleaned).strip()
            if new_cleaned != cleaned and len(new_cleaned) > 5:
                cleaned = new_cleaned
                was_wrapped = True

        # Clean leading conversational/polite filler prefixes
        new_cleaned = re.sub(
            r'^\s*(kindly\s+(share\s+your\s+(insight|thoughts)|tell\s+me|explain)|please\s+(share\s+your\s+thoughts|tell\s+me|explain)|i\s+want\s+to\s+know|can\s+you\s+tell\s+me|could\s+you\s+(please\s+)?(tell\s+me|share|explain)|tell\s+me|state|answer)\s*[:,\s-]*',
            '',
            cleaned,
            flags=re.IGNORECASE
        ).strip()
        if new_cleaned and len(new_cleaned) > 5:
            cleaned = new_cleaned
            was_wrapped = True

        if not cleaned:
            return raw, False
            
        return cleaned, was_wrapped

