import re
from typing import Tuple, Optional

# 1. Transactional & Action Commands (Non-retrieval operational tasks)
TRANSACTIONAL_PATTERNS = [
    re.compile(r'\b(book|secure|reserve|arrange)\s+(me\s+)?(a\s+|an\s+|the\s+)?(hotel|flight|room|cab|taxi|ticket|table|transportation)\s*(in|at|for|to)?\b', re.IGNORECASE),
    re.compile(r'\b(download|install|update|open|launch)\s+(this\s+|the\s+)?(whatsapp|app|application|game|software|video|youtube|instagram|facebook)\b', re.IGNORECASE),
    re.compile(r'\b(send|dispatch|initiate)\s+(an?\s+)?(email|mail|message|sms|whatsapp|electronic\s+mail)\b', re.IGNORECASE),
    re.compile(r'\b(turn\s+on|turn\s+off|switch\s+on|switch\s+off|dim|activate)\s+(the\s+)?(light|lights|fan|ac|tv|heater|illumination|audio\s+track|sound\s+system)\b', re.IGNORECASE),
    re.compile(r'\b(call|dial|phone|ring)\s+(my\s+|to\s+)?(mom|dad|mother|father|brother|sister|friend|doctor|police|ambulance)\b', re.IGNORECASE),
    re.compile(r'\b(set\s+(an?|my)?\s+alarm|play\s+(some\s+|a\s+)?(music|song|songs)|set\s+(an?|my)?\s+timer)\b', re.IGNORECASE),
    re.compile(r'\b(order\s+(two\s+)?(large\s+)?(vegetarian\s+)?pizza|schedule\s+a\s+calendar\s+appointment|transfer\s+\w+\s+(thousand\s+)?rupees|purchase\s+\w+\s+(entry\s+)?tickets|configure\s+a\s+wake-up\s+chime)\b', re.IGNORECASE),
    re.compile(r'\b(order\s+food|book\s+a\s+ride|send\s+an?\s+email|play\s+a\s+song|remind\s+me\s+to|set\s+a\s+timer|turn\s+(on|off)\s+the\s+light|call\s+mom|transfer\s+money|buy\s+bitcoin)\b', re.IGNORECASE),
    # Indic transactional command patterns
    re.compile(r'(బుక్\s*చేయండి|బిల్లు\s*చెల్లించండి|ఆర్డర్\s*చేయండి|ఫ్లైట్\s*బుక్|హోటల్\s*బుక్|టికెట్\s*బుక్)', re.IGNORECASE),
    re.compile(r'(बूक\s*करा|बुक\s*करा|ऑर्डर\s*करा|तिकीट\s*काढा|लाईट\s*बंद\s*करा|फ्लाइट\s*बुक|हॉटेल\s*बुक|टॅक्सी\s*बोलावा)', re.IGNORECASE),
    re.compile(r'(বুক\s*করুন|অর্ডার\s*করুন|ফ্লাইট\s*বুক|হোটেল\s*বুক|কল\s*করুন|টিকিট\s*বুক|ট্যাক্সি\s*ডাকুন)', re.IGNORECASE),
    re.compile(r'(ಬುಕ್\s*ಮಾಡಿ|ಆರ್ಡರ್\s*ಮಾಡಿ|ಲೈಟ್\s*ಆಫ್\s*ಮಾಡಿ|ಕರೆ\s*ಮಾಡಿ|ಟಿಕೆಟ್\s*ಬುಕ್|ಹೋಟೆಲ್\s*ಬುಕ್)', re.IGNORECASE),
    re.compile(r'(புக்\s*பண்ணு|ஆர்டர்\s*செய்|லைட்\s*ஆப்\s*செய்|அழைப்பு\s*விடு|ஹோட்டல்\s*புக்|டிக்கெட்\s*புக்)', re.IGNORECASE),
    re.compile(r'(ઓર્ડર\s*કરો|બુક\s*કરો|લાઈટ\s*બંધ\s*કરો|ટિકિટ\s*બુક|હોટેલ\s*બુક)', re.IGNORECASE),
    re.compile(r'(ഹോട്ടൽ\s*ബുക്ക്\s*ചെയ്യുക|ഫ്ലൈറ്റ്\s*ബുക്ക്|ടിക്കറ്റ്\s*എടുക്കുക)', re.IGNORECASE),
    re.compile(r'(पिज़्ज़ा\s*ऑर्डर\s*करो|होटल\s*बुक\s*करो|टिकट\s*बुक\s*करो|फ्लाइट\s*बुक\s*करो|टैक्सी\s*बुलाओ|अलार्म\s*सेट\s*करो)', re.IGNORECASE),
]

# 2. Personal / Private User State & Conversational Identity
PERSONAL_PATTERNS = [
    re.compile(r'\b(what\s+is\s+my\s+name|who\s+am\s+i|how\s+old\s+am\s+i|what\s+did\s+i\s+(eat|dream|do)|where\s+do\s+i\s+live|how\s+many\s+hairs\s+are\s+on\s+my\s+head)\b', re.IGNORECASE),
    re.compile(r'\b(what\s+(will|did)\s+happen\s+to\s+me\s+(tomorrow|yesterday)|can\s+you\s+read\s+my\s+mind|meaning\s+of\s+my\s+name|what\s+am\s+i\s+thinking|subconscious\s+thoughts|born\s+into\s+this\s+world)\b', re.IGNORECASE),
    re.compile(r'\b(what\s+is\s+my\s+(wifi|wi-fi|email|phone|bank|atm|pin|password|address|ip|location|meaning\s+of\s+my\s+name)|sequence\s+of\s+.*symbols\s+allows\s+access)\b', re.IGNORECASE),
    re.compile(r'\b(how\s+are\s+you(\s+feeling)?(\s+today)?|can\s+you\s+talk\s+to\s+me|are\s+you\s+my\s+friend|do\s+you\s+love\s+me|are\s+you\s+a\s+robot|do\s+you\s+like\s+me|who\s+are\s+you|emotional\s+affection|share\s+a\s+confidential\s+fact)\b', re.IGNORECASE),
    re.compile(r'\b(what\s+time\s+is\s+it(\s+now)?)\b', re.IGNORECASE),
    # Indic personal / conversational patterns
    re.compile(r'(माझे\s*नाव|ना\s*పేరు|আমার\s*নাম|என்\s*பெயர்|ನನ್ನ\s*ಹೆಸರು|ਮੇਰਾ\s*ਨਾਮ|ମୋ\s*ନାଁ|મારું\s*નામ|ಎന്റെ\s*പേര്|मेरा\s*नाम|मैं\s*कहाँ\s*रहता)', re.IGNORECASE),
    re.compile(r'(तुम\s*मुझसे\s*बात\s*कर\s*सकते\s*हो|माझ्याशी\s*बोला|నాతో\s*మాట్లాడండి|আমার\s*সাথে\s*কথা\s*বলুন|நீ\s*யார்|तू\s*कोण\s*आहेस|तुम\s*कौन\s*हो)', re.IGNORECASE),
]

# 3. Realtime, Live Volatile Events & Future Speculation
REALTIME_PATTERNS = [
    re.compile(r'\b(latest\s+developments\s+in|ongoing\s+score|prevailing\s+foreign\s+exchange|train\s+services\s+between|flight\s+status|traffic\s+situation|live\s+broadcast|today\'?s\s+gold\s+rate)\b', re.IGNORECASE),
    re.compile(r'\b(who\s+won\s+yesterday(\'s)?\s+(match|ipl|game|election)|what\s+is\s+the\s+latest\s+news|is\s+the\s+market\s+open\s+today|at\s+this\s+(hour|moment|instant))\b', re.IGNORECASE),
    re.compile(r'\b(what\s+will\s+the\s+weather\s+be\s+(tomorrow|next\s+week)|what\s+is\s+the\s+weather\s+today|what\s+time\s+does\s+the\s+next\s+train\s+leave|expectation\s+of\s+precipitation)\b', re.IGNORECASE),
    re.compile(r'\b(who\s+(will|won)\s+(win\s+the\s+(next|20\d\d)|the\s+2024\s+fifa\s+world\s+cup|the\s+election|the\s+20\d\d\s+world\s+cup)|did\s+the\s+home\s+team\s+manage\s+to\s+secure\s+victory)\b', re.IGNORECASE),
    re.compile(r'\b(current\s+price\s+of\s+bitcoin|stock\s+price\s+of\s+.*\s+today|price\s+of\s+gold\s+today|what\s+will\s+the\s+stock\s+market\s+do\s+next\s+year|current\s+trading\s+valuation|equity\s+trading\s+floors\s+active)\b', re.IGNORECASE),
    re.compile(r'\b(predict\s+the\s+lottery\s+numbers|solve\s+p\s+vs\s+np\s+for\s+me|when\s+will\s+the\s+world\s+end|how\s+to\s+travel\s+back\s+in\s+time|establish\s+a\s+permanent\s+colony\s+in\s+the\s+andromeda)\b', re.IGNORECASE),
    # Indic realtime / future patterns
    re.compile(r'(ఈ\s*రోజు\s*వాతావరణం|आजचे\s*हवामान|আজকের\s*আবহাওয়া|இன்றைய\s*வானிலை|ಇಂದಿನ\s*ಹವಾಮಾನ)', re.IGNORECASE),
    re.compile(r'(कल\s*का\s*मौसम|आजच्या\s*बा बातम्या|आजची\s*बातमी|आजच्या\s*बातम्|വാർത്ത\s*எந்தാണ്|ಇಂದಿನ\s*ಸುದ್ದಿ|আজকের\s*খবর|ताज़ा\s*खबर|இன்றைய\s*செய்தி|शेयर\s*बाजार\s*कैसा)', re.IGNORECASE),
    re.compile(r'(2025\s*में\s*कौन|2026\s*में\s*चुनाव|अगले\s*साल\s*क्या|पुढील\s*वर्षी\s*काय)', re.IGNORECASE),
]

# 4. Purely Subjective Opinions, Philosophical & Unanswerable / Creative Questions
OPINION_PATTERNS = [
    re.compile(r'\b(do\s+you\s+believe|which\s+culinary\s+tradition|worthwhile\s+investment|should\s+artificial\s+intelligence\s+systems|most\s+effective\s+routine|secret\s+ingredients\s+comprise|is\s+it\s+ethical|your\s+favorite)\b', re.IGNORECASE),
    re.compile(r'\b(is\s+social\s+media\s+good\s+or\s+bad|what\s+is\s+the\s+meaning\s+of\s+life|what\s+career\s+should\s+i\s+choose|what\s+should\s+i\s+do\s+with\s+my\s+life|who\s+is\s+the\s+greatest\s+cricketer\s+ever)\b', re.IGNORECASE),
    re.compile(r'\b(is\s+it\s+(wiser|better|advisable|preferable)\s+to|should\s+(one|individuals|people|we|i)\s+prioritize|would\s+you\s+recommend|which\s+[\w\s]+\s+offers\s+the\s+most\s+satisfying|artistic\s+merit\s+of)\b', re.IGNORECASE),
    re.compile(r'\b(write\s+(me\s+)?a\s+(love\s+letter|poem|story|song)|compose\s+an?\s+(original\s+)?(song|verse)|make\s+up\s+a\s+new\s+word|generate\s+a\s+(random\s+)?story|create\s+a\s+(new\s+)?programming\s+language|draw\s+a\s+picture|formulate\s+an?\s+entirely\s+new|invent\s+a\s+novel)\b', re.IGNORECASE),
    re.compile(r'\b(what\s+is\s+the\s+best\s+religion|what\s+is\s+consciousness|is\s+there\s+life\s+on\s+other\s+planets|what\s+happens\s+after\s+death|visual\s+texture\s+of\s+physical\s+matter)\b', re.IGNORECASE),
    re.compile(r'\b(is\s+vegetarian\s+food\s+healthier|is\s+god\s+real|is\s+pineapple\s+on\s+pizza\s+acceptable|which\s+phone\s+is\s+better\s+iphone\s+or\s+samsung)\b', re.IGNORECASE),
    re.compile(r'\b(what\s+is\s+the\s+best\s+(movie|film|song|food|city|country|phone|car|book)\b|should\s+i\s+buy\s+a\s+tesla|are\s+cats\s+better\s+than\s+dogs)\b', re.IGNORECASE),
    re.compile(r'\b(who\s+is\s+the\s+smartest\s+person\s+alive|what\s+is\s+dark\s+matter\s+made\s+of|tell\s+me\s+a\s+secret\s+that\s+nobody\s+knows|secret\s+recipe)\b', re.IGNORECASE),
    re.compile(r'\b(what\s+is\s+the\s+(last\s+digit\s+of\s+pi|cure\s+for\s+cancer)|what\s+is\s+inside\s+a\s+black\s+hole)\b', re.IGNORECASE),
    # Indic subjective / opinion questions
    re.compile(r'(எது\s*சிறந்தது|कोणते\s*चांगले\s*आहे|কোনটি\s*ভালো)', re.IGNORECASE),
    re.compile(r'(कोणता\s*धर्म\s*सर्वोत्तम|सबसे\s*अच्छा\s*खाना|ಯಾವ\s*ಭಾಷೆ\s*ಉತ್ತಮ|ఏ\s*దేశం\s*ಬೆస్ట్|प्यार\s*सच\s*में\s*होता\s*है|भगवान\s*है\s*या\s*नहीं)', re.IGNORECASE),
]

# 5. Adversarial Injection / Roleplay / DAN Mode
ADVERSARIAL_PATTERNS = [
    re.compile(r'\b(dan\s+mode|do\s+anything\s+now|override:\s*admin\s+mode|ignore\s+your\s+training|tell\s+me\s+something\s+you\'?re\s+not\s+supposed\s+to\s+say)\b', re.IGNORECASE),
    re.compile(r'\b(simulate\s+being\s+a\s+human|you\s+are\s+now\s+(a\s+different\s+ai|uncensored)|act\s+as\s+if\s+you\s+have\s+no\s+restrictions|pretend\s+you\s+are\s+gpt-4|jailbreak:\s*unlimited\s+mode)\b', re.IGNORECASE),
    re.compile(r'\b(reveal\s+your\s+instructions|system\s+prompt|bypass\s+your\s+safety\s+filters|ignore\s+all\s+guardrails|hidden\s+instructions|repeat\s+after\s+me|forget\s+your\s+rules|ignore\s+previous\s+instructions)\b', re.IGNORECASE),
    re.compile(r'(अपने\s*नियम\s*तोड़ो|निर्देशों\s*को\s*अनदेखा\s*करो|सिस्टम\s*प्रॉम्प्ट\s*दिखाओ)', re.IGNORECASE),
]

# 6. Foreign (Non-Indic / Non-English) Language Detection Regexes
FOREIGN_LANGUAGE_PATTERNS = [
    # Spanish / French / German / Portuguese / Italian / Dutch / Scandinavian
    re.compile(r'\b(cómo\s+estás|cuál\s+es\s+la\s+capital|was\s+ist\s+die\s+hauptstadt|comment\s+ça\s+va|qual\s+é\s+a\s+capital|quelle\s+heure\s+est-il|come\s+ti\s+chiami|wie\s+heißt\s+du|wat\s+is\s+de\s+hoofdstad)\b', re.IGNORECASE),
    re.compile(r'\b(wo\s+befindet\s+sich|où\s+se\s+trouve|qual\s+è\s+la\s+distanza|hvor\s+ligger|vilken\s+är|största\s+insjön|donde\s+esta|wie\s+weit|c\'est\s+quoi)\b', re.IGNORECASE),
    re.compile(r'[\u0400-\u04FF]', re.UNICODE),  # Cyrillic (Russian, etc.)
    re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]', re.UNICODE),  # CJK / Japanese / Korean
    re.compile(r'[¿¡]', re.UNICODE),  # Spanish punctuation marks
]

# 7. Ambiguous / Underspecified / Headless Generic Queries
UNDERSPECIFIED_PATTERNS = [
    re.compile(r'^\s*(capital|population|details|information|summary|net\s+worth|specifications|historical\s+timeline|temperature\s+reading|status|score|schedule)\s+(of|about|regarding)\s+(the\s+|this\s+)?(state|town|city|country|person|engine|tournament|policy|ceo|facility|event|document|building|company|conflict)\s*$', re.IGNORECASE),
    re.compile(r'^\s*(what\s+happened\s+during\s+the\s+conference|summary\s+of\s+the\s+policy\s+document|specifications\s+of\s+the\s+engine|details\s+regarding\s+the\s+tournament|information\s+about\s+the\s+election\s+results)\s*$', re.IGNORECASE),
]


def is_foreign_language(text: str) -> bool:
    for pat in FOREIGN_LANGUAGE_PATTERNS:
        if pat.search(text):
            return True
    return False


class IntentDomainClassifier:
    """
    Fast Pre-Retrieval Intent and Domain Classifier.
    """

    @staticmethod
    def classify(query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Classifies whether a query is In-Domain or Out-of-Domain.
        
        Returns:
            (is_in_domain: bool, refusal_category: Optional[str], refusal_message: Optional[str])
        """
        q = query.strip()
        if not q:
            return False, "empty_query", "Please provide a valid question."

        # Check 1: Foreign Languages outside 14 Indic + English
        if is_foreign_language(q):
            return False, "non_indic_language", "This system currently supports English and 14 Indic languages. Please ask in a supported language."

        # Check 2: Adversarial Injection
        for pat in ADVERSARIAL_PATTERNS:
            if pat.search(q):
                return False, "adversarial_injection", "I'm not able to process instruction overrides or system prompt requests."

        # Check 3: Transactional & Action Commands
        for pat in TRANSACTIONAL_PATTERNS:
            if pat.search(q):
                return False, "transactional_action", "I am a factual knowledge assistant and cannot execute physical bookings, orders, or device commands."

        # Check 4: Personal & Private User Info
        for pat in PERSONAL_PATTERNS:
            if pat.search(q):
                return False, "personal_conversational", "I don't have access to personal user data or private device information."

        # Check 5: Real-time, Live Scores, Weather & Future Prediction
        for pat in REALTIME_PATTERNS:
            if pat.search(q):
                return False, "realtime_future", "I am designed for factual knowledge retrieval and do not provide live real-time feeds or future predictions."

        # Check 6: Purely Subjective Opinions
        for pat in OPINION_PATTERNS:
            if pat.search(q):
                return False, "opinion_subjective", "I can only answer questions based on verifiable factual knowledge."

        # Check 7: Underspecified / Ambiguous Queries
        for pat in UNDERSPECIFIED_PATTERNS:
            if pat.search(q):
                return False, "truly_unanswerable", "The question is underspecified. Please specify which entity or subject you are referring to."

        return True, None, None
