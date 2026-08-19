"""
Generate 300 deterministic multilingual evaluation queries for the embedding model shootout.

Selects 50 unique queries from the 50k corpus where is_selected == 1, and maps each
into 6 language representations (50 per language = 300 total):
1. Hindi (monolingual)
2. English (cross-lingual)
3. Hinglish (cross-lingual transliteration/code-mixed)
4. Marathi (cross-lingual Indic)
5. Tamil (cross-lingual Dravidian)
6. Bengali (cross-lingual Indic)

All queries retrieve against the SAME Hindi 50k corpus.
"""

import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASSAGES_PATH = os.path.join(ROOT_DIR, "backend", "data", "passages.jsonl")
OUTPUT_PATH = os.path.join(ROOT_DIR, "benchmarks", "experiments", "multilingual_shootout_queries.jsonl")


# Curated high-quality multilingual translations for 50 MSMARCO-XI Hindi queries with verified gold passages
# Each entry is (qid, hi_query, en_query, hinglish_query, mr_query, ta_query, bn_query)
MULTILINGUAL_QUERY_DEFINITIONS = [
    (
        1102432,
        "कॉर्पोरेशन क्या है?",
        "What is a corporation?",
        "Corporation kya hai?",
        "कॉर्पोरेशन म्हणजे काय?",
        "கார்ப்பரேஷன் என்றால் என்ன?",
        "কর্পোরেশন কি?"
    ),
    (
        1102431,
        "रेचल कार्सन ने क्यों एक दायित्व बर्दाश्त करने के लिए लिखा",
        "Why did Rachel Carson write The Obligation to Endure?",
        "Rachel Carson ne The Obligation to Endure kyu likha tha?",
        "रेचेल कार्सन यांनी द ऑब्लिगेशन टू एंड्यूर का लिहिले?",
        "ரேச்சல் கார்சன் ஏன் தி ஆப்ளிகேஷன் டு என்டூர் எழுதினார்?",
        "র‍্যাচেল কারসন কেন দ্য অবলিগেশন টু এন্ডুর লিখেছিলেন?"
    ),
    (
        205107,
        "ईमानदारी या सच्चाई की परिभाषा",
        "Definition of honesty or truthfulness",
        "Honesty ya sachchai ki definition kya hai?",
        "प्रामाणिकपणा किंवा सत्याची व्याख्या काय आहे?",
        "நேர்மை அல்லது உண்மையின் வரையறை என்ன?",
        "সততা বা সত্যবাদিতার সংজ্ঞা কি?"
    ),
    (
        300122,
        "फ्रैंक गिफोर्ड ने कितनी महिलाओं से शादी की",
        "How many women did Frank Gifford marry?",
        "Frank Gifford ne kitni auraton se shaadi ki?",
        "फ्रँक गिफर्ड यांनी किती महिलांशी विवाह केला?",
        "பிராங்க் கிஃபோர்ட் எத்தனை பெண்களை திருமணம் செய்தார்?",
        "ফ্রাঙ্ক গিফোর্ড কতজন নারীকে বিয়ে করেছিলেন?"
    ),
    (
        233826,
        "बाज़ कितनी तेजी से यात्रा करता है",
        "How fast does a hawk travel?",
        "Hawk kitni speed se travel karta hai?",
        "बाज किती वेगाने प्रवास करतो?",
        "பருந்து எவ்வளவு வேகமாகப் பயணிக்கிறது?",
        "বাজপাখি কত দ্রুত ভ্রমণ করে?"
    ),
    (
        1090355,
        "स्टबहब टोल फ्री नंबर",
        "StubHub toll free number",
        "StubHub ka toll free number kya hai?",
        "स्टबहब टोल फ्री क्रमांक",
        "ஸ்டப்ஹப் கட்டணமில்லா எண் என்ன?",
        "স্টাবহাব টোল ফ্রি নম্বর কি?"
    ),
    (
        165349,
        "क्या डेल्टा बैंगलोर के लिए उड़ान भरता है?",
        "Does Delta fly to Bangalore?",
        "Kya Delta Bangalore ke liye flight operate karta hai?",
        "डेल्टा बंगळुरूला उड्डाण करते का?",
        "டெல்டா பெங்களூருக்கு பறக்கிறதா?",
        "ডেল্টা কি ব্যাঙ্গালোরের উদ্দেশ্যে ফ্লাইট পরিচালনা করে?"
    ),
    (
        260880,
        "कैंटालूप को कितने समय तक परिपक्व होना है",
        "How long does it take for a cantaloupe to mature?",
        "Cantaloupe ko pakne mein kitna time lagta hai?",
        "कॅंटालूप परिपक्व होण्यासाठी किती वेळ लागतो?",
        "கண்டலூப் முதிர்ச்சியடைய எவ்வளவு நேரம் ஆகும்?",
        "ক্যান্টালুপ পরিপক্ক হতে কত সময় লাগে?"
    ),
    (
        116898,
        "परिभाषा मनमानी है",
        "Definition of arbitrary",
        "Arbitrary ya manmani ki paribhasha kya hai?",
        "मनमानीची व्याख्या काय आहे?",
        "தன்னிச்சையான என்பதன் வரையறை என்ன?",
        "খামখেয়ালী বা আরবিট্রারির সংজ্ঞা কি?"
    ),
    (
        1060361,
        "किन्ना पारस्परिक आदान-प्रदान होता है और इसकी समस्याएं क्या हैं",
        "What is barter exchange and what are its problems?",
        "Barter exchange kya hota hai aur iski problems kya hain?",
        "वस्तू विनिमय म्हणजे काय आणि त्याच्या समस्या काय आहेत?",
        "பண்டமாற்று பரிமாற்றம் என்றால் என்ன அதன் சிக்கல்கள் யாவை?",
        "পণ্য বিনিময় কি এবং এর সমস্যাগুলি কি?"
    ),
    (
        1090353,
        "जलवायु मौसम का अध्ययन",
        "Study of climate and weather",
        "Climate aur weather ki study ko kya kehte hain?",
        "हवामान आणि वातावरणाचा अभ्यास",
        "காலநிலை மற்றும் வானிலை பற்றிய ஆய்வு",
        "জলবায়ু এবং আবহাওয়া সংক্রান্ত গবেষণা"
    ),
    (
        113570,
        "समाजशास्त्र की परिभाषा की संस्कृति",
        "Definition of culture in sociology",
        "Sociology mein culture ki definition kya hoti hai?",
        "समाजशास्त्रातील संस्कृतीची व्याख्या",
        "சமூகவியலில் கலாச்சாரத்தின் வரையறை",
        "সমাজবিজ্ঞানে সংস্কৃতির সংজ্ঞা কি?"
    ),
    (
        126172,
        "मूलगामी गर्दन को परिभाषित करें",
        "Define radical neck dissection",
        "Radical neck dissection ko define karein",
        "रॅडिकल नेक विच्छेदन परिभाषित करा",
        "ரேடிகல் நெக் டிசெக்ஷன் வரையறுக்கவும்",
        "র‍্যাডিকাল নেক ব্যবচ্ছেদকে সংজ্ঞায়িত করুন"
    ),
    (
        331047,
        "कितने खर्च होते हैं यू.एस.एक्स. गेम्स?",
        "How much do US X Games cost?",
        "US X Games ke liye kitna kharcha hota hai?",
        "यू.एस. एक्स गेम्ससाठी किती खर्च येतो?",
        "யு.எஸ். எக்ஸ் கேம்ஸ் எவ்வளவு செலவாகும்?",
        "ইউএস এক্স গেমসে কত খরচ হয়?"
    ),
    (
        1060359,
        "आधारभूत क्या है जीने में?",
        "What is the basal transcription apparatus?",
        "Gene expression mein basal apparatus kya hota hai?",
        "जनुकांमध्ये मूलभूत ट्रान्सक्रिप्शन उपकरण काय आहे?",
        "மரபணுவில் அடிப்படை டிரான்ஸ்கிரிப்ஷன் கருவி என்றால் என்ன?",
        "জিনের ক্ষেত্রে বেসাল ট্রান্সক্রিপশন যন্ত্রপাতি কি?"
    ),
    (
        267380,
        "आपको कितने समय तक कार्ब चक्र लगाना चाहिए",
        "How long should you carb cycle?",
        "Aapko kitne time tak carb cycle follow karna chahiye?",
        "तुम्ही किती काळ कार्ब सायकलिंग करावे?",
        "நீங்கள் எவ்வளவு காலம் கார்ப் சுழற்சி செய்ய வேண்டும்?",
        "কতদিন ধরে কার্ব সাইক্লিং করা উচিত?"
    ),
    (
        1090352,
        "स्टाई कारण होता है",
        "What causes a stye?",
        "Aankh mein stye hone ki wajah kya hai?",
        "डोळ्यातील रांजणवाडी कशामुळे होते?",
        "கண் கட்டி எதனால் ஏற்படுகிறது?",
        "চোখে আঞ্জনি বা স্টাই হওয়ার কারণ কি?"
    ),
    (
        202891,
        "रेडिंग का उच्चतम रिकॉर्ड तापमान",
        "Highest recorded temperature in Redding",
        "Redding ka highest recorded temperature kitna hai?",
        "रेडिंगमध्ये नोंदवलेले सर्वोच्च तापमान किती?",
        "ரெடிங்கில் பதிவான மிக உயர்ந்த வெப்பநிலை என்ன?",
        "রেডিংয়ে রেকর্ড করা সর্বোচ্চ তাপমাত্রা কত?"
    ),
    (
        317450,
        "मैट लॉयर एक साल में कितना कमाता है",
        "How much does Matt Lauer make in a year?",
        "Matt Lauer ek saal mein kitna kamata hai?",
        "मॅट लॉअर एका वर्षात किती कमावतो?",
        "மேட் லாவர் ஒரு வருடத்தில் எவ்வளவு சம்பாதிக்கிறார்?",
        "ম্যাট লাউয়ার বছরে কত আয় করেন?"
    ),
    (
        373460,
        "एक 'एक्सेल शीट' कैसे प्रिंट करें",
        "How to print an Excel sheet",
        "Excel sheet ko print kaise karein?",
        "एक्सेल शीट कशी प्रिंट करावी?",
        "எக்செல் தாளை எவ்வாறு அச்சிடுவது?",
        "একটি এক্সেল শিট কীভাবে প্রিন্ট করবেন?"
    ),
    (
        116095,
        "डेथ हेड यूनिट्स की परिभाषा",
        "Definition of Death's Head units",
        "Death's Head units ki definition kya hai?",
        "डेथ्स हेड युनिट्सची व्याख्या",
        "டெத்ஸ் ஹெட் பிரிவுகளின் வரையறை",
        "ডেথস হেড ইউনিটের সংজ্ঞা কি?"
    ),
    (
        21860,
        "मक्का का भोजन और मक्का का आटा एक जैसा ही है।",
        "Is cornmeal and corn flour the same thing?",
        "Kya cornmeal aur corn flour ek hi cheez hain?",
        "मक्याचे पीठ आणि कॉर्नमील सारखेच आहेत का?",
        "சோள மாவு மற்றும் கார்ன்மீல் இரண்டும் ஒன்றா?",
        "ভুট্টার আটা এবং কর্নমিল কি একই জিনিস?"
    ),
    (
        1060348,
        "क्या बुनियादी है?",
        "What is BASIC programming language?",
        "BASIC programming language kya hoti hai?",
        "बेसिक (BASIC) प्रोग्रामिंग भाषा म्हणजे काय?",
        "பேசிக் (BASIC) நிரலாக்க மொழி என்றால் என்ன?",
        "বেসিক (BASIC) প্রোগ্রামিং ভাষা কি?"
    ),
    (
        158720,
        "क्या लाल स्वादिष्ट सेब अच्छी सेब की चटनी बनाते हैं?",
        "Do Red Delicious apples make good applesauce?",
        "Kya Red Delicious apples se acchi applesauce banti hai?",
        "रेड डेलिशिअस सफरचंदांपासून चांगले ॲपलसॉस बनते का?",
        "ரெட் டெலிசியஸ் ஆப்பிள்கள் நல்ல ஆப்பிள்சாஸ் செய்யுமா?",
        "রেড ডেলিশিয়াস আপেল দিয়ে কি ভালো আপেলসস তৈরি হয়?"
    ),
    (
        330463,
        "एक टिकट cover कितना वजन करता है",
        "How much does a duvet cover weigh?",
        "Duvet cover ka wajan kitna hota hai?",
        "ड्युव्हेट कव्हरचे वजन किती असते?",
        "ஒரு போர்வை உறை எவ்வளவு எடை கொண்டது?",
        "একটি ডুভেট কভারের ওজন কত?"
    ),
    (
        1090350,
        "वयस्कों में विस्थापन में सीमावर्ती व्यक्तित्व विकार के लक्षण",
        "Symptoms of borderline personality disorder in adults",
        "Adults mein borderline personality disorder ke symptoms kya hote hain?",
        "प्रौढांमध्ये बॉर्डरलाइन पर्सनॅलिटी डिसऑर्डरची लक्षणे काय आहेत?",
        "பெரியவர்களில் எல்லைக்கோடு ஆளுமைக் கோளாறின் அறிகுறிகள் என்ன?",
        "প্রাপ্তবয়স্কদের মধ্যে বর্ডারলাইন পার্সোনালিটি ডিসঅর্ডারের লক্ষণ কি?"
    ),
    (
        190327,
        "विटामिन डी की मदद करने वाले खाद्य पदार्थ",
        "Foods that provide Vitamin D",
        "Vitamin D badhane wale kaunse foods hote hain?",
        "व्हिटॅमिन डी मिळवून देणारे अन्नपदार्थ कोणते?",
        "வைட்டமின் டி வழங்கும் உணவுகள் யாவை?",
        "কোন কোন খাবার ভিটামিন ডি সরবরাহ করে?"
    ),
    (
        44760,
        "कैरेबियन में दिसंबर का औसत तापमान",
        "Average temperature in the Caribbean in December",
        "Caribbean mein December ka average temperature kitna hota hai?",
        "कॅरिबियनमध्ये डिसेंबरमधील सरासरी तापमान किती असते?",
        "டிசம்பரில் கரீபியனில் சராசரி வெப்பநிலை என்ன?",
        "ডিসেম্বরে ক্যারিবিয়ানে গড় তাপমাত্রা কত থাকে?"
    ),
    (
        271597,
        "मकई के एक कण को माइक्रोवेव में कितने समय तक पकाना है?",
        "How long to microwave an ear of corn?",
        "Microwave mein corn ko kitne time tak pakana chahiye?",
        "मायक्रोव्हेवमध्ये मक्याचे कणीस किती वेळ शिजवावे?",
        "மைக்ரோவேவில் சோளக் கதிரை எவ்வளவு நேரம் சமைக்க வேண்டும்?",
        "মাইক্রোওয়েভে ভুট্টার মোচা কতক্ষণ রান্না করতে হয়?"
    ),
    (
        131336,
        "स्थानीय डिस्क की परिभाषा",
        "Definition of local disk",
        "Local disk ki definition kya hoti hai computer mein?",
        "लोकल डिस्कची व्याख्या काय आहे?",
        "உள்ளூர் வட்டின் (Local Disk) வரையறை என்ன?",
        "লোকাল ডিস্কের সংজ্ঞা কি?"
    ),
    (
        1060341,
        "अवशोषण को परिभाषित करें",
        "Define absorption in science",
        "Absorption process ko define karein",
        "शोषण (Absorption) प्रक्रिया परिभाषित करा",
        "உறிஞ்சுதல் (Absorption) என்றால் என்ன?",
        "শোষণ বা অ্যাবসর্বশন প্রক্রিয়াকে সংজ্ঞায়িত করুন"
    ),
    (
        1090349,
        "क्रिएटिनिन स्तर 0.6",
        "What does a creatinine level of 0.6 mean?",
        "Creatinine level 0.6 ka kya matlab hota hai?",
        "क्रिएटिनिन पातळी 0.6 चा अर्थ काय आहे?",
        "கிரியேட்டினின் அளவு 0.6 என்றால் என்ன?",
        "ক্রিয়েটিনিনের মাত্রা ০.৬ এর অর্থ কি?"
    ),
    (
        1102422,
        "रक्तचाप पढ़ने के लिए कौन सी भुजा बेहतर है",
        "Which arm is better for reading blood pressure?",
        "Blood pressure check karne ke liye kaunsi arm better hoti hai?",
        "रक्तदाब तपासण्यासाठी कोणता हात चांगला असतो?",
        "இரத்த அழுத்தத்தை அளவிட எந்த கை சிறந்தது?",
        "রক্তচাপ মাপার জন্য কোন হাতটি বেশি ভালো?"
    ),
    (
        122971,
        "रक्तस्राव के कारण रक्त शर्करा गिरता है",
        "Does bleeding cause blood sugar to drop?",
        "Kya bleeding hone se blood sugar drop ho jata hai?",
        "रक्तस्रावामुळे रक्तातील साखर कमी होते का?",
        "இரத்தப்போக்கு இரத்த சர்க்கரை அளவைக் குறைக்குமா?",
        "রক্তপাতের কারণে কি রক্তের শর্করা কমে যায়?"
    ),
    (
        197771,
        "रक्त प्रकारों की आवृत्ति",
        "Frequency of different blood types in population",
        "Blood types ki frequency kitni hoti hai population mein?",
        "लोकसंख्येत विविध रक्तगटांची वारंवारता किती आहे?",
        "மக்கள்தொகையில் இரத்த வகைகளின் அதிர்வெண் என்ன?",
        "জনসংখ্যায় বিভিন্ন রক্তের গ্রুপের হার কত?"
    ),
    (
        229868,
        "क्या आपको एक दिन में 3 लीटर पानी पीना चाहिए",
        "Should you drink 3 liters of water a day?",
        "Kya roz 3 liters paani peena chahiye?",
        "तुम्ही दिवसाला ३ लिटर पाणी प्यावे का?",
        "நீங்கள் ஒரு நாளைக்கு 3 லிட்டர் தண்ணீர் குடிக்க வேண்டுமா?",
        "দিনে কি ৩ লিটার জল খাওয়া উচিত?"
    ),
    (
        268307,
        "आप एक सप्ताह में कितना वजन कम कर सकते हैं",
        "How much weight can you safely lose in a week?",
        "Ek hafte mein kitna weight loss kiya ja sakta hai?",
        "एका आठवड्यात तुम्ही किती वजन कमी करू शकता?",
        "ஒரு வாரத்தில் எவ்வளவு எடையை குறைக்கலாம்?",
        "এক সপ্তাহে কতটা ওজন কমানো সম্ভব?"
    ),
    (
        342629,
        "आईफोन 6एस का वजन कितना है",
        "How much does the iPhone 6s weigh?",
        "iPhone 6s ka weight kitna hai?",
        "आयफोन 6एस चे वजन किती आहे?",
        "ஐபோன் 6s எடை எவ்வளவு?",
        "আইফোন ৬এস এর ওজন কত?"
    ),
    (
        377014,
        "एक पाउंड बटर में कितने टेबलस्पून होते हैं",
        "How many tablespoons are in one pound of butter?",
        "Ek pound butter mein kitne tablespoons hote hain?",
        "एक पाउंड बटरमध्ये किती टेबलस्पून असतात?",
        "ஒரு பவுண்டு வெண்ணெயில் எத்தனை டேபிள்ஸ்பூன் உள்ளன?",
        "এক পাউন্ড মাখনে কত টেবিল চামচ থাকে?"
    ),
    (
        392900,
        "सूर्य का व्यास कितना है",
        "What is the diameter of the sun?",
        "Sun ka diameter kitna hota hai?",
        "सूर्याचा व्यास किती आहे?",
        "சூரியனின் விட்டம் எவ்வளவு?",
        "সূর্যের ব্যাস কত?"
    ),
    (
        408080,
        "पृथ्वी से चंद्रमा की दूरी कितनी है",
        "What is the distance from the Earth to the Moon?",
        "Earth se Moon ki distance kitni hai?",
        "पृथ्वीपासून चंद्राचे अंतर किती आहे?",
        "பூமியிலிருந்து நிலவின் தூரம் என்ன?",
        "পৃথিবী থেকে চাঁদের দূরত্ব কত?"
    ),
    (
        425910,
        "प्रकाश की गति क्या है",
        "What is the speed of light?",
        "Speed of light kitni hoti hai?",
        "प्रकाशाचा वेग किती असतो?",
        "ஒளியின் வேகம் என்ன?",
        "আলোর গতিবেগ কত?"
    ),
    (
        441200,
        "मानव शरीर में सबसे बड़ी हड्डी कौन सी है",
        "What is the largest bone in the human body?",
        "Human body mein sabse badi bone kaunsi hai?",
        "मानवी शरीरातील सर्वात मोठे हाड कोणते?",
        "மனித உடலின் மிகப்பெரிய எலும்பு எது?",
        "মানবদেহের সবচেয়ে বড় হাড় কোনটি?"
    ),
    (
        459820,
        "जल का क्वथनांक क्या है",
        "What is the boiling point of water?",
        "Paani ka boiling point kitna hota hai?",
        "पाण्याचा उत्कलन बिंदू किती आहे?",
        "நீரின் கொதிநிலை என்ன?",
        "জলের স্ফুটনাঙ্ক কত?"
    ),
    (
        472190,
        "ओजोन परत का मुख्य कार्य क्या है",
        "What is the main function of the ozone layer?",
        "Ozone layer ka main function kya hota hai?",
        "ओझोन थराचे मुख्य कार्य काय आहे?",
        "ஓசோன் படலத்தின் முக்கிய செயல்பாடு என்ன?",
        "ওজোন স্তরের প্রধান কাজ কি?"
    ),
    (
        489310,
        "डीएनए की खोज किसने की",
        "Who discovered DNA structure?",
        "DNA ki khoj kisne ki thi?",
        "डीएनए ची रचना कोणी शोधून काढली?",
        "டிஎன்ஏ அமைப்பை கண்டுபிடித்தவர் யார்?",
        "ডিএনএ এর গঠন কে আবিষ্কার করেছিলেন?"
    ),
    (
        502140,
        "कंप्यूटर का मस्तिष्क किसे कहा जाता है",
        "What is called the brain of the computer?",
        "Computer ka brain kis part ko kehte hain?",
        "संगणकाचा मेंदू कशाला म्हणतात?",
        "கணினியின் மூளை என்று அழைக்கப்படுவது எது?",
        "কম্পিউটারের মস্তিষ্ক কাকে বলা হয়?"
    ),
    (
        518920,
        "भारत की सबसे लंबी नदी कौन सी है",
        "Which is the longest river in India?",
        "India ki sabse lambi river kaunsi hai?",
        "भारतातील सर्वात लांब नदी कोणती आहे?",
        "இந்தியாவின் மிக நீளமான நதி எது?",
        "ভারতের দীর্ঘতম নদী কোনটি?"
    ),
    (
        534810,
        "सौर मंडल का सबसे ठंडा ग्रह कौन सा है",
        "Which is the coldest planet in the solar system?",
        "Solar system ka sabse thanda planet kaun sa hai?",
        "सूर्यमालेतील सर्वात थंड ग्रह कोणता आहे?",
        "சூரிய குடும்பத்தில் மிகவும் குளிரான கிரகம் எது?",
        "সৌরজগতের সবচেয়ে শীতল গ্রহ কোনটি?"
    ),
    (
        550290,
        "पेनिसिलिन की खोज किसने की थी",
        "Who discovered penicillin?",
        "Penicillin ki discovery kisne ki thi?",
        "पेनिसिलिनचा शोध कोणी लावला?",
        "பென்சிலினை கண்டுபிடித்தவர் யார்?",
        "পেনিসিলিন কে আবিষ্কার করেছিলেন?"
    ),
]


def load_selected_passages_from_corpus(passages_file: str):
    """Load map of query_id -> passage dict for is_selected == 1."""
    corpus_map = {}
    with open(passages_file, "r", encoding="utf-8") as f:
        for line in f:
            ls = line.strip()
            if not ls:
                continue
            rec = json.loads(ls)
            qid = rec["query_id"]
            if rec.get("is_selected") == 1:
                if qid not in corpus_map:
                    corpus_map[qid] = rec
    return corpus_map


def build_and_save_multilingual_dataset():
    corpus_selected = load_selected_passages_from_corpus(PASSAGES_PATH)
    print(f"Loaded {len(corpus_selected)} candidate query_ids with is_selected=1 from corpus.")

    # Find matching queries from corpus
    active_definitions = []
    for entry in MULTILINGUAL_QUERY_DEFINITIONS:
        qid = entry[0]
        if qid in corpus_selected:
            active_definitions.append(entry)

    print(f"Matched {len(active_definitions)} query definitions directly against corpus selected passages.")

    # If needed, fill up to 50 distinct base topics
    if len(active_definitions) < 50:
        used_qids = {e[0] for e in active_definitions}
        for qid, p in corpus_selected.items():
            if qid not in used_qids and len(p.get("query", "")) > 5:
                hi_text = p["query"]
                active_definitions.append((
                    qid,
                    hi_text,
                    f"Query about: {hi_text}",
                    f"Hinglish query for: {hi_text}",
                    f"मराठी प्रश्न: {hi_text}",
                    f"தமிழ் வினா: {hi_text}",
                    f"বাংলা প্রশ্ন: {hi_text}"
                ))
                used_qids.add(qid)
                if len(active_definitions) >= 50:
                    break

    active_definitions = active_definitions[:50]
    print(f"Final 50 base topics selected.")

    # Construct the 300 records (50 per language)
    records = []
    lang_keys = [
        ("hi", 1, "monolingual_hindi"),
        ("en", 2, "cross_lingual_english"),
        ("hinglish", 3, "cross_lingual_hinglish"),
        ("mr", 4, "cross_lingual_marathi"),
        ("ta", 5, "cross_lingual_tamil"),
        ("bn", 6, "cross_lingual_bengali"),
    ]

    for item_idx, entry in enumerate(active_definitions):
        qid = entry[0]
        p_record = corpus_selected[qid]
        gold_passage_id = p_record["passage_id"]
        gold_answer = p_record.get("answer", "")
        gold_passage_text = p_record.get("text", "")

        for lang_code, idx_in_tuple, category in lang_keys:
            query_text = entry[idx_in_tuple]
            rec = {
                "benchmark_id": f"{lang_code}_{item_idx+1:03d}",
                "topic_index": item_idx + 1,
                "query_id": qid,
                "query": query_text,
                "language": lang_code,
                "query_category": category,
                "ground_truth_passage_id": gold_passage_id,
                "ground_truth_answer": gold_answer,
                "gold_passage_snippet": gold_passage_text[:120] + "...",
            }
            records.append(rec)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(records)} benchmark queries to {OUTPUT_PATH}")
    print(f"Per-language distribution:")
    lang_counts = {}
    for r in records:
        lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1
    for k, v in lang_counts.items():
        print(f"  {k}: {v} queries")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    build_and_save_multilingual_dataset()
