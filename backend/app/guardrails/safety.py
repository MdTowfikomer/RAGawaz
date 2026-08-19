"""
Ticket 8: Safety Guardrail.

Layer 1: Policy and keyword-based safety blocklist (< 1ms).
Runs BEFORE relevance and retrieval to immediately reject unsafe or malicious inputs.
"""

from typing import Tuple, Optional, Sequence, Pattern, Callable, Awaitable
import re


# Common harmful / injection patterns
BLOCKLIST_PATTERNS: Sequence[Pattern[str]] = [
    re.compile(r'\b(ignore previous instructions|system prompt|jailbreak|bypass|malicious)\b', re.IGNORECASE),
    re.compile(r'\b(bomb|weapon|hack|exploit|malware|virus|ddos|craft weapon|fraud|credit card fraud|counterfeit|illegal currency)\b', re.IGNORECASE),
    re.compile(r'(हथियार|बम|हैक|वायरस|विस्फोटक|अवैध|फ़ायरवॉल|नकली मुद्रा|जालसाजी)', re.IGNORECASE),
]


class SafetyGuardrail:
    """Layer 1 keyword & policy layer with Layer 2 classifier hook."""

    def __init__(
        self,
        blocklist: Optional[Sequence[Pattern[str]]] = None,
        layer2_classifier: Optional[Callable[[str], Awaitable[bool]]] = None,
    ):
        self.blocklist = blocklist or BLOCKLIST_PATTERNS
        self.layer2_classifier = layer2_classifier
        self.safety_refusal_message = "I'm not able to help with that type of request."
        self.safety_refusal_message_hi = "मैं इस प्रकार के असुरक्षित अनुरोध में सहायता करने में असमर्थ हूँ।"

    def get_refusal_message(self, query: Optional[str] = None) -> str:
        """Return language-matched safety refusal message."""
        if query and re.search(r'[\u0900-\u097F]', query):
            return self.safety_refusal_message_hi
        return self.safety_refusal_message

    def evaluate(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluate if a query is safe (Layer 1 fast path < 1ms).
        Returns (is_safe: bool, refusal_message: Optional[str]).
        """
        cleaned = query.strip()
        refusal_msg = self.get_refusal_message(cleaned)
        for pattern in self.blocklist:
            if pattern.search(cleaned):
                return False, refusal_msg

        return True, None

    async def evaluate_async(self, query: str) -> Tuple[bool, Optional[str]]:
        """Evaluate Layer 1 + optional Layer 2 classifier."""
        is_safe, refusal = self.evaluate(query)
        if not is_safe:
            return False, refusal

        if self.layer2_classifier:
            l2_safe = await self.layer2_classifier(query)
            if not l2_safe:
                return False, self.get_refusal_message(query)

        return True, None
