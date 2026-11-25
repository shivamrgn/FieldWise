"""
100% Offline Multilingual Localization Bundles for FieldWise PS1.

Supports:
  - `en`: English
  - `hi`: Hindi (हिन्दी - Devanagari)
  - `gu`: Gujarati (ગુજરાતી)
  - `mr`: Marathi (मराठी - Devanagari)
  - `te`: Telugu (తెలుగు)
  - `ta`: Tamil (தமிழ்)

Zero cloud translation APIs. Provides strict key parity across UI section
headers, severity badges, status banners, KVK / Kisan Call Center (1800-180-1551)
escalation notices, and crop / disease / pest names aligned with `data/agri_kb.json`.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Tuple

LOGGER = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: Tuple[str, ...] = ("en", "hi", "gu", "mr", "te", "ta")

_TRANSLATIONS: Dict[str, Dict[str, Any]] = {
    "en": {
        "language_code": "en",
        "language_name": "English",
        "fallback_warning": "Requested language is unsupported; safely defaulted to English (en).",
        "headers": {
            "crop": "Crop",
            "detected_condition": "Detected Condition",
            "severity_level": "Severity Level",
            "infected_area": "Infected Leaf Area",
            "confidence": "Confidence",
            "status": "Status",
            "symptoms": "Observed Symptoms",
            "regional_advisory": "Region-Specific Notes",
            "crop_info": "Crop Agronomic Profile",
            "explanation": "Assessment Summary",
            "immediate_actions": "Immediate Actions",
            "organic_cultural_control": "Organic & Cultural Control",
            "approved_chemical_guidance": "Approved Chemical Guidance",
            "preventive_measures": "Preventive Measures",
            "kvk_escalation": "KVK & Helpline Escalation",
            "disclaimer": "Safety Disclaimer",
            "language": "Language",
        },
        "severity_badges": {
            "low": "Low (<15%)",
            "moderate": "Moderate (15%–40%)",
            "severe": "Severe (>40%)",
            "undetermined": "Undetermined",
        },
        "status_banners": {
            "SUCCESS": "Advisory Ready — Grounded in ICAR/KVK Package of Practices",
            "UNCERTAIN_IMAGE": "Unclear Photo — Please Retake a Sharp, Well-Lit Image",
            "LOW_CONFIDENCE": "Low Confidence — Expert Verification Required at Local KVK",
            "SEVERE_ESCALATION": "Severe Infestation Alert — Escalate Immediately to Local KVK",
        },
        "kvk_escalation_messages": {
            "helpline_number": "1800-180-1551",
            "recommended_escalation": (
                "Severe condition detected. Visit your nearest Krishi Vigyan Kendra (KVK) "
                "or call the free Kisan Call Center helpline (1800-180-1551) promptly before spraying."
            ),
            "low_confidence_hold": (
                "Visual screening confidence is insufficient for automated guidance. Consult your "
                "local Krishi Vigyan Kendra (KVK) or Kisan Call Center (1800-180-1551) with a fresh plant sample."
            ),
            "routine_note": (
                "For localized dose confirmation, contact your district Krishi Vigyan Kendra (KVK) "
                "or Kisan Call Center (1800-180-1551)."
            ),
        },
        "crops": {
            "rice": "Rice (Paddy)",
            "wheat": "Wheat",
            "maize": "Maize (Corn)",
            "cotton": "Cotton",
            "sugarcane": "Sugarcane",
            "soybean": "Soybean",
            "groundnut": "Groundnut",
            "potato": "Potato",
            "chickpea": "Chickpea (Chana)",
            "lentil": "Lentil (Masoor)",
            "mustard": "Mustard",
            "tomato": "Tomato",
            "onion": "Onion",
            "banana": "Banana",
            "mango": "Mango",
            "jowar": "Jowar (Sorghum)",
            "apple": "Apple",
            "cherry": "Cherry",
            "grape": "Grape",
            "peach": "Peach",
            "orange": "Orange",
            "bell pepper": "Bell Pepper",
            "squash": "Squash",
            "strawberry": "Strawberry",
            "blueberry": "Blueberry",
            "raspberry": "Raspberry",
            "general": "Crop Foliage",
            "unknown": "Unknown Crop",
        },
        "conditions": {
            "rice_blast": "Rice Blast",
            "rice_brown_spot": "Brown Spot",
            "rice_bacterial_leaf_blight": "Bacterial Leaf Blight",
            "rice_brown_planthopper": "Brown Planthopper (BPH)",
            "wheat_leaf_rust": "Wheat Leaf Rust (Brown Rust)",
            "wheat_yellow_rust": "Wheat Yellow Rust (Stripe Rust)",
            "wheat_aphids": "Wheat Aphids",
            "cotton_bollworm": "Cotton Bollworm",
            "cotton_aphids": "Cotton Aphids",
            "cotton_whitefly": "Cotton Whitefly",
            "potato_late_blight": "Potato Late Blight",
            "potato_early_blight": "Potato Early Blight",
            "tomato_early_blight": "Tomato Early Blight",
            "tomato_late_blight": "Tomato Late Blight",
            "tomato_yellow_leaf_curl_virus": "Tomato Yellow Leaf Curl Virus",
            "maize_fall_armyworm": "Maize Fall Armyworm",
            "maize_northern_leaf_blight": "Maize Northern Corn Leaf Blight",
            "healthy": "Healthy Foliage",
            "unclear_image": "Unclear Image",
        },
        "tabs": {
            "disease_scan": "Disease & Pest Scan",
            "crop_recommendation": "Crop Recommendation (V2)",
            "fertilizer_advisory": "Fertilizer & Irrigation",
        },
        "untrained_vision_warning": (
            "Untrained smoke weights are active (~5.6% random confidence). "
            "Run download_pretrained.py once while online and restart with --vision-backend hf-local for real PlantVillage diagnosis."
        ),
        "analyzing_status": "Running on-device quality check, PlantVillage vision & offline KB advisory...",
        "advisory_fallback_notice": (
            "Advisory Runtime Notice: Real offline 38-class MobileNetV2 Vision + V2 Crop Engine + ICAR KB are active. "
            "Llama 3.2 .pte / ExecuTorch binaries are not installed on this host, so advisory prose uses the deterministic grounded KB fallback (test_stub_ci_only)."
        ),
        "crop_rec_ui": {
            "title": "Offline Crop Recommendation (FieldWise V2 Engine)",
            "state": "State (Mandatory)",
            "sowing_month": "Planned Sowing Month (1–12, Mandatory)",
            "irrigation_access": "Irrigation Access",
            "soil_texture": "Soil Texture",
            "soil_ph": "Soil pH",
            "nitrogen": "Available Nitrogen (N, kg/ha)",
            "phosphorus": "Available Phosphorus (P₂O₅, kg/ha)",
            "potassium": "Available Potassium (K₂O, kg/ha)",
            "seasonal_rainfall": "Seasonal Rainfall (mm)",
            "temperature": "Mean Temperature (°C)",
            "humidity": "Relative Humidity (%)",
            "has_soil_test": "I have a certified Soil Health Card (SHC) test",
            "submit_btn": "Run Offline V2 Crop Recommendation",
            "suitability": "Suitability",
            "yield_range": "Expected Yield",
            "water_status": "Water Status",
            "limiting_factor": "Limiting Factor",
            "fertilizer_plan": "Stoichiometric Fertilizer Plan",
            "excluded_crops": "Excluded Crops & Reasons",
        },
        "fert_ui": {
            "title": "Fertilizer & Irrigation Advisory (Offline KB)",
            "select_crop": "Select Crop",
            "farm_area": "Farm Area",
            "unit_ha": "Hectares (ha)",
            "unit_acres": "Acres (auto-converted to ha)",
            "submit_btn": "Get Scaled Fertilizer & Irrigation Plan",
            "fertilizer": "Fertilizer",
            "per_ha_rate": "Standard Rate",
            "scaled_qty": "Scaled for Farm Area",
            "nutrients": "Nutrients",
            "timing": "Application Stage & Timing",
            "climate_soil_guide": "Crop Irrigation, Soil & Climate Guide",
            "safety_note": "Verify Soil Health Card (SHC) readings and local KVK recommendations before procuring fertilizer bags.",
        },
    },
    "hi": {
        "language_code": "hi",
        "language_name": "हिन्दी",
        "fallback_warning": "अनुरोधित भाषा समर्थित नहीं है; सुरक्षित रूप से अंग्रेज़ी (en) का उपयोग किया गया है।",
        "headers": {
            "crop": "फसल",
            "detected_condition": "संभावित रोग / कीट",
            "severity_level": "गंभीरता स्तर",
            "infected_area": "प्रभावित पत्ती क्षेत्र",
            "confidence": "विश्वास स्तर",
            "status": "स्थिति",
            "symptoms": "प्रमुख लक्षण",
            "regional_advisory": "क्षेत्र-विशेष कृषि सलाह",
            "crop_info": "फसल जलवायु व मिट्टी विवरण",
            "explanation": "जांच सारांश",
            "immediate_actions": "तुरंत करने योग्य कार्य",
            "organic_cultural_control": "जैविक एवं सस्य नियंत्रण",
            "approved_chemical_guidance": "अनुमोदित रासायनिक मार्गदर्शन",
            "preventive_measures": "रोकथाम के उपाय",
            "kvk_escalation": "कृषि विज्ञान केंद्र (KVK) एवं हेल्पलाइन संपर्क",
            "disclaimer": "सुरक्षा चेतावनी",
            "language": "भाषा",
        },
        "severity_badges": {
            "low": "कम (<15%)",
            "moderate": "मध्यम (15%–40%)",
            "severe": "गंभीर (>40%)",
            "undetermined": "अनिर्धारित",
        },
        "status_banners": {
            "SUCCESS": "सलाह तैयार — ICAR/KVK अनुशंसित कृषि पद्धति पर आधारित",
            "UNCERTAIN_IMAGE": "फोटो अस्पष्ट है — कृपया साफ रोशनी में स्पष्ट फोटो दोबारा लें",
            "LOW_CONFIDENCE": "कम विश्वास स्तर — नजदीकी KVK विशेषज्ञ से पुष्टि आवश्यक",
            "SEVERE_ESCALATION": "गंभीर प्रकोप चेतावनी — तुरंत नजदीकी KVK से संपर्क करें",
        },
        "kvk_escalation_messages": {
            "helpline_number": "1800-180-1551",
            "recommended_escalation": (
                "गंभीर रोग/कीट स्तर पाया गया है। छिड़काव से पहले तुरंत नजदीकी कृषि विज्ञान केंद्र (KVK) "
                "या निःशुल्क किसान कॉल सेंटर (1800-180-1551) से सलाह लें।"
            ),
            "low_confidence_hold": (
                "फोटो से पहचान पक्की नहीं है। ताजा प्रभावित पत्ती के साथ नजदीकी कृषि विज्ञान केंद्र (KVK) "
                "या किसान कॉल सेंटर (1800-180-1551) से जांच कराएं।"
            ),
            "routine_note": (
                "स्थानीय सही मात्रा और सलाह के लिए अपने जिले के कृषि विज्ञान केंद्र (KVK) "
                "या किसान कॉल सेंटर (1800-180-1551) से संपर्क करें।"
            ),
        },
        "crops": {
            "rice": "धान (चावल)",
            "wheat": "गेहूं",
            "maize": "मक्का",
            "cotton": "कपास",
            "sugarcane": "गन्ना",
            "soybean": "सोयाबीन",
            "groundnut": "मूंगफली",
            "potato": "आलू",
            "chickpea": "चना",
            "lentil": "मसूर",
            "mustard": "सरसों",
            "tomato": "टमाटर",
            "onion": "प्याज",
            "banana": "केला",
            "mango": "आम",
            "jowar": "ज्वार",
            "apple": "सेब",
            "cherry": "चेरी",
            "grape": "अंगूर",
            "peach": "आड़ू",
            "orange": "संतरा",
            "bell pepper": "शिमला मिर्च",
            "squash": "कद्दू / स्क्वैश",
            "strawberry": "स्ट्रॉबेरी",
            "blueberry": "ब्लूबेरी",
            "raspberry": "रसभरी",
            "general": "फसल पत्ती",
            "unknown": "अज्ञात फसल",
        },
        "conditions": {
            "rice_blast": "धान का झोंका रोग (ब्लास्ट)",
            "rice_brown_spot": "धान का भूरा धब्बा रोग",
            "rice_bacterial_leaf_blight": "धान का जीवाणु झुलसा रोग",
            "rice_brown_planthopper": "भूरा फुदका (माहू / BPH)",
            "wheat_leaf_rust": "गेहूं का भूरा रतुआ (पत्ती रतुआ)",
            "wheat_yellow_rust": "गेहूं का पीला रतुआ (धारीदार रतुआ)",
            "wheat_aphids": "गेहूं का माहू (चेपा / एफिड)",
            "cotton_bollworm": "कपास की सुंडी (बॉलवर्म)",
            "cotton_aphids": "कपास का माहू (एफिड)",
            "cotton_whitefly": "कपास की सफेद मक्खी",
            "potato_late_blight": "आलू का पछेती झुलसा रोग",
            "potato_early_blight": "आलू का अगेती झुलसा रोग",
            "tomato_early_blight": "टमाटर का अगेती झुलसा रोग",
            "tomato_late_blight": "टमाटर का पछेती झुलसा रोग",
            "tomato_yellow_leaf_curl_virus": "टमाटर का पीला पत्ता मरोड़ विषाणु रोग",
            "maize_fall_armyworm": "मक्का का फॉल आर्मीवर्म",
            "maize_northern_leaf_blight": "मक्का का उत्तरी पत्ती झुलसा रोग",
            "healthy": "स्वस्थ फसल",
            "unclear_image": "अस्पष्ट फोटो",
        },
        "tabs": {
            "disease_scan": "रोग एवं कीट जांच",
            "crop_recommendation": "फसल चयन (V2 इंजन)",
            "fertilizer_advisory": "उर्वरक एवं सिंचाई सलाह",
        },
        "untrained_vision_warning": (
            "अभी अप्रशिक्षित स्मोक मॉडल सक्रिय है (~5.6% यादृच्छिक विश्वास)। "
            "वास्तविक PlantVillage पहचान के लिए ऑनलाइन रहते हुए एक बार download_pretrained.py चलाएं और --vision-backend hf-local से शुरू करें।"
        ),
        "analyzing_status": "ऑफ़लाइन गुणवत्ता जांच, PlantVillage विज़न और स्थानीय KB सलाह चल रही है...",
        "advisory_fallback_notice": (
            "सलाह रनटाइम सूचना: वास्तविक ऑफ़लाइन PlantVillage विज़न + V2 फसल इंजन + ICAR ज्ञान आधार सक्रिय हैं। "
            "इस डिवाइस पर Llama 3.2 .pte / ExecuTorch उपलब्ध नहीं होने के कारण प्रमाणित स्थानीय KB फ़ॉलबैक (test_stub_ci_only) उपयोग हो रहा है।"
        ),
        "crop_rec_ui": {
            "title": "ऑफ़लाइन फसल चयन (FieldWise V2 इंजन)",
            "state": "राज्य (अनिवार्य)",
            "sowing_month": "बुवाई का महीना (1–12, अनिवार्य)",
            "irrigation_access": "सिंचाई सुविधा",
            "soil_texture": "मिट्टी का प्रकार",
            "soil_ph": "मिट्टी का pH",
            "nitrogen": "उपलब्ध नाइट्रोजन (N, किग्रा/हे.)",
            "phosphorus": "उपलब्ध फॉस्फोरस (P₂O₅, किग्रा/हे.)",
            "potassium": "उपलब्ध पोटाश (K₂O, किग्रा/हे.)",
            "seasonal_rainfall": "मौसमी वर्षा (मिमी)",
            "temperature": "औसत तापमान (°C)",
            "humidity": "सापेक्ष आर्द्रता (%)",
            "has_soil_test": "मेरे पास प्रमाणित मृदा स्वास्थ्य कार्ड (SHC) रिपोर्ट है",
            "submit_btn": "ऑफ़लाइन V2 फसल सलाह देखें",
            "suitability": "उपयुक्तता अंक",
            "yield_range": "अपेक्षित उपज",
            "water_status": "जल स्थिति",
            "limiting_factor": "मुख्य सीमित घटक",
            "fertilizer_plan": "संतुलित उर्वरक योजना (DAP/यूरिया/MOP)",
            "excluded_crops": "अयोग्य फसलें व कारण",
        },
        "fert_ui": {
            "title": "उर्वरक एवं सिंचाई सलाह (ऑफ़लाइन KB)",
            "select_crop": "फसल चुनें",
            "farm_area": "खेत का क्षेत्रफल",
            "unit_ha": "हेक्टेयर (ha)",
            "unit_acres": "एकड़ (स्वतः हेक्टेयर में बदलेगा)",
            "submit_btn": "क्षेत्रफल अनुसार उर्वरक व सिंचाई योजना देखें",
            "fertilizer": "उर्वरक",
            "per_ha_rate": "मानक दर (प्रति हेक्टेयर)",
            "scaled_qty": "खेत हेतु कुल मात्रा",
            "nutrients": "पोषक तत्व",
            "timing": "प्रयोग समय व अवस्था",
            "climate_soil_guide": "फसल सिंचाई, मिट्टी एवं जलवायु मार्गदर्शन",
            "safety_note": "उर्वरक खरीदने से पहले मृदा स्वास्थ्य कार्ड (SHC) और नजदीकी KVK सलाह की पुष्टि करें।",
        },
    },
    "gu": {
        "language_code": "gu",
        "language_name": "ગુજરાતી",
        "fallback_warning": "વિનંતી કરેલી ભાષા સમર્થિત નથી; સુરક્ષિત રીતે અંગ્રેજી (en) પસંદ કરવામાં આવ્યું છે.",
        "headers": {
            "crop": "પાક",
            "detected_condition": "ઓળખાયેલ રોગ / જીવાત",
            "severity_level": "તીવ્રતા સ્તર",
            "infected_area": "અસરગ્રસ્ત પાન વિસ્તાર",
            "confidence": "વિશ્વાસ સ્તર",
            "status": "સ્થિતિ",
            "symptoms": "મુખ્ય લક્ષણો",
            "regional_advisory": "પ્રદેશ-વિશિષ્ટ કૃષિ નોંધ",
            "crop_info": "પાકની જમીન અને આબોહવા માહિતી",
            "explanation": "તપાસ સારાંશ",
            "immediate_actions": "તાત્કાલિક પગલાં",
            "organic_cultural_control": "જૈવિક અને ખેતી પદ્ધતિ નિયંત્રણ",
            "approved_chemical_guidance": "મંજૂર રાસાયણિક માર્ગદર્શન",
            "preventive_measures": "નિવારક પગલાં",
            "kvk_escalation": "KVK અને હેલ્પલાઇન સંપર્ક",
            "disclaimer": "સલામતી સૂચના",
            "language": "ભાષા",
        },
        "severity_badges": {
            "low": "ઓછી (<15%)",
            "moderate": "મધ્યમ (15%–40%)",
            "severe": "ગંભીર (>40%)",
            "undetermined": "અનિશ્ચિત",
        },
        "status_banners": {
            "SUCCESS": "સલાહ તૈયાર — ICAR/KVK ભલામણ આધારિત માર્ગદર્શન",
            "UNCERTAIN_IMAGE": "ફોટો અસ્પષ્ટ છે — કૃપા કરીને ચોખ્ખા પ્રકાશમાં સ્પષ્ટ ફોટો લો",
            "LOW_CONFIDENCE": "ઓછો વિશ્વાસ — સ્થાનિક KVK નિષ્ણાતની ચકાસણી જરૂરી",
            "SEVERE_ESCALATION": "ગંભીર ચેતવણી — તાત્કાલિક સ્થાનિક KVK નો સંપર્ક કરો",
        },
        "kvk_escalation_messages": {
            "helpline_number": "1800-180-1551",
            "recommended_escalation": (
                "ગંભીર રોગ/જીવાતની સ્થિતિ જણાઈ છે. છંટકાવ પહેલાં તાત્કાલિક નજીકના કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) "
                "અથવા મફત કિસાન કોલ સેન્ટર (1800-180-1551) નો સંપર્ક કરો."
            ),
            "low_confidence_hold": (
                "ફોટા પરથી ચોક્કસ ઓળખ થઈ શકી નથી. તાજા અસરગ્રસ્ત પાન સાથે નજીકના કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) "
                "અથવા કિસાન કોલ સેન્ટર (1800-180-1551) ની સલાહ લો."
            ),
            "routine_note": (
                "સ્થાનિક યોગ્ય માત્રાની ખાતરી માટે તમારા જિલ્લાના કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) "
                "અથવા કિસાન કોલ સેન્ટર (1800-180-1551) નો સંપર્ક કરો."
            ),
        },
        "crops": {
            "rice": "ડાંગર (ચોખા)",
            "wheat": "ઘઉં",
            "maize": "મકાઈ",
            "cotton": "કપાસ",
            "sugarcane": "શેરડી",
            "soybean": "સોયાબીન",
            "groundnut": "મગફળી",
            "potato": "બટાકા",
            "chickpea": "ચણા",
            "lentil": "મસૂર",
            "mustard": "રાયડો",
            "tomato": "ટામેટા",
            "onion": "ડુંગળી",
            "banana": "કેળા",
            "mango": "કેરી",
            "jowar": "જુવાર",
            "apple": "સફરજન",
            "cherry": "ચેરી",
            "grape": "દ્રાક્ષ",
            "peach": "પીચ",
            "orange": "સંતરા",
            "bell pepper": "શિમલા મરચા",
            "squash": "કોળું / સ્ક્વોશ",
            "strawberry": "સ્ટ્રોબેરી",
            "blueberry": "બ્લુબેરી",
            "raspberry": "રાસ્પબેરી",
            "general": "પાકનું પાન",
            "unknown": "અજ્ઞાત પાક",
        },
        "conditions": {
            "rice_blast": "ડાંગરનો કરમોડી (બ્લાસ્ટ) રોગ",
            "rice_brown_spot": "ડાંગરનો ભૂખરો ટપકાંનો રોગ",
            "rice_bacterial_leaf_blight": "ડાંગરનો જીવાણુ પાનનો સુકારો",
            "rice_brown_planthopper": "ડાંગરના ભૂખરા ચૂસિયા (BPH)",
            "wheat_leaf_rust": "ઘઉંનો ભૂખરો ગેરુ રોગ",
            "wheat_yellow_rust": "ઘઉંનો પીળો ગેરુ રોગ",
            "wheat_aphids": "ઘઉંની મોલોમશી (એફિડ)",
            "cotton_bollworm": "કપાસની ઈયળ (બોલવોર્મ)",
            "cotton_aphids": "કપાસની મોલોમશી (એફિડ)",
            "cotton_whitefly": "કપાસની સફેદ માખી",
            "potato_late_blight": "બટાકાનો પાછોતરો સુકારો",
            "potato_early_blight": "બટાકાનો આગોતરો સુકારો",
            "tomato_early_blight": "ટામેટાનો આગોતરો સુકારો",
            "tomato_late_blight": "ટામેટાનો પાછોતરો સુકારો",
            "tomato_yellow_leaf_curl_virus": "ટામેટાનો પાન કોકડવા વિષાણુ રોગ",
            "maize_fall_armyworm": "મકાઈની લશ્કરી ઈયળ (ફોલ આર્મીવોર્મ)",
            "maize_northern_leaf_blight": "મકાઈનો ઉત્તરીય પાનનો સુકારો",
            "healthy": "તંદુરસ્ત પાક",
            "unclear_image": "અસ્પષ્ટ ફોટો",
        },
        "tabs": {
            "disease_scan": "રોગ અને જીવાત તપાસ",
            "crop_recommendation": "પાક ભલામણ (V2 એન્જિન)",
            "fertilizer_advisory": "ખાતર અને પિયત સલાહ",
        },
        "untrained_vision_warning": (
            "અત્યારે બિનતાલીમબદ્ધ સ્મોક મોડેલ સક્રિય છે (~5.6% રેન્ડમ વિશ્વાસ). "
            "સાચા PlantVillage નિદાન માટે ઓનલાઇન હો ત્યારે એકવાર download_pretrained.py ચલાવો અને --vision-backend hf-local થી શરૂ કરો."
        ),
        "analyzing_status": "ઓફલાઇન ગુણવત્તા તપાસ, PlantVillage વિઝન અને સ્થાનિક KB સલાહ ચાલુ છે...",
        "advisory_fallback_notice": (
            "સલાહ રનટાઇમ સૂચના: સાચું ઓફલાઇન PlantVillage વિઝન + V2 પાક એન્જિન + ICAR જ્ઞાન ભંડાર સક્રિય છે. "
            "આ ડિવાઇસ પર Llama 3.2 .pte / ExecuTorch ઉપલબ્ધ ન હોવાથી પ્રમાણિત સ્થાનિક KB ફોલબેક (test_stub_ci_only) ચાલી રહ્યો છે."
        ),
        "crop_rec_ui": {
            "title": "ઓફલાઇન પાક ભલામણ (FieldWise V2 એન્જિન)",
            "state": "રાજ્ય (ફરજિયાત)",
            "sowing_month": "વાવણીનો મહિનો (1–12, ફરજિયાત)",
            "irrigation_access": "પિયતની સુવિધા",
            "soil_texture": "જમીનનો પ્રકાર",
            "soil_ph": "જમીનનો pH",
            "nitrogen": "ઉપલબ્ધ નાઇટ્રોજન (N, કિગ્રા/હે.)",
            "phosphorus": "ઉપલબ્ધ ફોસ્ફરસ (P₂O₅, કિગ્રા/હે.)",
            "potassium": "ઉપલબ્ધ પોટાશ (K₂O, કિગ્રા/હે.)",
            "seasonal_rainfall": "મોસમી વરસાદ (મિમી)",
            "temperature": "સરેરાશ તાપમાન (°C)",
            "humidity": "ભેજનું પ્રમાણ (%)",
            "has_soil_test": "મારી પાસે પ્રમાણિત સોઇલ હેલ્થ કાર્ડ (SHC) રિપોર્ટ છે",
            "submit_btn": "ઓફલાઇન V2 પાક ભલામણ મેળવો",
            "suitability": "યોગ્યતા ગુણ",
            "yield_range": "અંદાજિત ઉત્પાદન",
            "water_status": "પાણીની સ્થિતિ",
            "limiting_factor": "મુખ્ય મર્યાદિત પરિબળ",
            "fertilizer_plan": "સંતુલિત ખાતર યોજના (DAP/યુરિયા/MOP)",
            "excluded_crops": "અયોગ્ય પાકો અને કારણો",
        },
        "fert_ui": {
            "title": "ખાતર અને પિયત માર્ગદર્શન (ઓફલાઇન KB)",
            "select_crop": "પાક પસંદ કરો",
            "farm_area": "ખેતરનું ક્ષેત્રફળ",
            "unit_ha": "હેક્ટર (ha)",
            "unit_acres": "એકર (આપોઆપ હેક્ટરમાં બદલાશે)",
            "submit_btn": "ક્ષેત્રફળ મુજબ ખાતર અને પિયત યોજના મેળવો",
            "fertilizer": "ખાતર",
            "per_ha_rate": "માનક દર (પ્રતિ હેક્ટર)",
            "scaled_qty": "ખેતર મુજબ કુલ જથ્થો",
            "nutrients": "પોષક તત્વો",
            "timing": "આપવાનો સમય અને અવસ્થા",
            "climate_soil_guide": "પાક પિયત, જમીન અને આબોહવા માર્ગદર્શન",
            "safety_note": "ખાતર ખરીદતા પહેલાં સોઇલ હેલ્થ કાર્ડ (SHC) અને સ્થાનિક KVK ની સલાહ ચકાસો.",
        },
    },
    "mr": {
        "language_code": "mr",
        "language_name": "मराठी",
        "fallback_warning": "विनंती केलेली भाषा समर्थित नाही; सुरक्षितपणे इंग्रजी (en) निवडली आहे.",
        "headers": {
            "crop": "पीक",
            "detected_condition": "ओळखलेला रोग / कीड",
            "severity_level": "तीव्रता पातळी",
            "infected_area": "प्रादुर्भाव झालेले क्षेत्र",
            "confidence": "विश्वास पातळी",
            "status": "स्थिती",
            "symptoms": "प्रमुख लक्षणे",
            "regional_advisory": "प्रदेशनिहाय कृषी नोंद",
            "crop_info": "पीक जमीन व हवामान माहिती",
            "explanation": "तपासणी सारांश",
            "immediate_actions": "त्वरित करावयाची कामे",
            "organic_cultural_control": "सेंद्रिय व मशागती नियंत्रण",
            "approved_chemical_guidance": "मान्यताप्राप्त रासायनिक मार्गदर्शन",
            "preventive_measures": "प्रतिबंधात्मक उपाय",
            "kvk_escalation": "KVK व हेल्पलाइन संपर्क",
            "disclaimer": "सुरक्षा सूचना",
            "language": "भाषा",
        },
        "severity_badges": {
            "low": "कमी (<15%)",
            "moderate": "मध्यम (15%–40%)",
            "severe": "तीव्र (>40%)",
            "undetermined": "अनिश्चित",
        },
        "status_banners": {
            "SUCCESS": "सल्ला तयार — ICAR/KVK शिफारस आधारित मार्गदर्शन",
            "UNCERTAIN_IMAGE": "फोटो अस्पष्ट आहे — कृपया स्वच्छ प्रकाशात स्पष्ट फोटो काढा",
            "LOW_CONFIDENCE": "कमी विश्वास पातळी — जवळच्या KVK तज्ज्ञांची खात्री आवश्यक",
            "SEVERE_ESCALATION": "तीव्र प्रादुर्भाव इशारा — त्वरित जवळच्या KVK शी संपर्क साधा",
        },
        "kvk_escalation_messages": {
            "helpline_number": "1800-180-1551",
            "recommended_escalation": (
                "तीव्र रोग/कीड पातळी आढळली आहे. फवारणीपूर्वी त्वरित जवळच्या कृषी विज्ञान केंद्र (KVK) "
                "किंवा मोफत किसान कॉल सेंटर (1800-180-1551) वर संपर्क साधा."
            ),
            "low_confidence_hold": (
                "फोटोवरून अचूक ओळख होऊ शकली नाही. ताज्या प्रभावित पानासह जवळच्या कृषी विज्ञान केंद्राशी (KVK) "
                "किंवा किसान कॉल सेंटर (1800-180-1551) शी संपर्क साधा."
            ),
            "routine_note": (
                "स्थानिक अचूक प्रमाणासाठी आपल्या जिल्ह्यातील कृषी विज्ञान केंद्र (KVK) "
                "किंवा किसान कॉल सेंटर (1800-180-1551) यांचा सल्ला घ्या."
            ),
        },
        "crops": {
            "rice": "भात (तांदूळ)",
            "wheat": "गहू",
            "maize": "मका",
            "cotton": "कापूस",
            "sugarcane": "ऊस",
            "soybean": "सोयाबीन",
            "groundnut": "भुईमूग",
            "potato": "बटाटा",
            "chickpea": "हरभरा",
            "lentil": "मसूर",
            "mustard": "मोहरी",
            "tomato": "टोमॅटो",
            "onion": "कांदा",
            "banana": "केळी",
            "mango": "आंबा",
            "jowar": "ज्वारी",
            "apple": "सफरचंद",
            "cherry": "चेरी",
            "grape": "द्राक्षे",
            "peach": "पीच",
            "orange": "संत्रे",
            "bell pepper": "ढोबळी मिरची",
            "squash": "भोपळा / स्क्वॅश",
            "strawberry": "स्ट्रॉबेरी",
            "blueberry": "ब्लूबेरी",
            "raspberry": "रास्पबेरी",
            "general": "पीक पान",
            "unknown": "अज्ञात पीक",
        },
        "conditions": {
            "rice_blast": "भातावरील करपा रोग (ब्लास्ट)",
            "rice_brown_spot": "भातावरील तपकिरी ठिपके रोग",
            "rice_bacterial_leaf_blight": "भातावरील जिवाणू करपा रोग",
            "rice_brown_planthopper": "भातावरील तपकिरी तुडतुडे (BPH)",
            "wheat_leaf_rust": "गव्हावरील तपकिरी तांबेरा",
            "wheat_yellow_rust": "गव्हावरील पिवळा तांबेरा",
            "wheat_aphids": "गव्हावरील मावा (एफिड्स)",
            "cotton_bollworm": "कापसावरील बोंडअळी",
            "cotton_aphids": "कापसावरील मावा",
            "cotton_whitefly": "कापसावरील पांढरी माशी",
            "potato_late_blight": "बटाट्यावरील उशिरा येणारा करपा",
            "potato_early_blight": "बटाट्यावरील लवकर येणारा करपा",
            "tomato_early_blight": "टोमॅटोवरील लवकर येणारा करपा",
            "tomato_late_blight": "टोमॅटोवरील उशिरा येणारा करपा",
            "tomato_yellow_leaf_curl_virus": "टोमॅटोवरील पर्णगुच्छ विषाणू रोग",
            "maize_fall_armyworm": "मक्यावरील लष्करी अळी (फॉल आर्मीवर्म)",
            "maize_northern_leaf_blight": "मक्यावरील पर्णकरपा रोग",
            "healthy": "निरोगी पीक",
            "unclear_image": "अस्पष्ट फोटो",
        },
        "tabs": {
            "disease_scan": "रोग व कीड तपासणी",
            "crop_recommendation": "पीक शिफारस (V2 इंजिन)",
            "fertilizer_advisory": "खत व पाणी व्यवस्थापन",
        },
        "untrained_vision_warning": (
            "सध्या अप्रशिक्षित स्मोक मॉडेल सुरू आहे (~5.6% यादृच्छिक विश्वास). "
            "खऱ्या PlantVillage तपासणीसाठी ऑनलाइन असताना एकदा download_pretrained.py चालवा आणि --vision-backend hf-local सह सुरू करा."
        ),
        "analyzing_status": "ऑफलाइन गुणवत्ता तपासणी, PlantVillage व्हिजन आणि स्थानिक KB सल्ला सुरू आहे...",
        "advisory_fallback_notice": (
            "सल्ला रनटाइम सूचना: खरे ऑफलाइन PlantVillage व्हिजन + V2 पीक इंजिन + ICAR ज्ञान भांडार सक्रिय आहेत. "
            "या उपकरणावर Llama 3.2 .pte / ExecuTorch नसल्याने प्रमाणित स्थानिक KB फॉलबॅक (test_stub_ci_only) वापरला जात आहे."
        ),
        "crop_rec_ui": {
            "title": "ऑफलाइन पीक शिफारस (FieldWise V2 इंजिन)",
            "state": "राज्य (आवश्यक)",
            "sowing_month": "पेरणीचा महिना (1–12, आवश्यक)",
            "irrigation_access": "सिंचन सुविधा",
            "soil_texture": "जमिनीचा प्रकार",
            "soil_ph": "जमिनीचा सामू (pH)",
            "nitrogen": "उपलब्ध नत्र (N, किलो/हे.)",
            "phosphorus": "उपलब्ध स्फुरद (P₂O₅, किलो/हे.)",
            "potassium": "उपलब्ध पालाश (K₂O, किलो/हे.)",
            "seasonal_rainfall": "हंगामी पाऊस (मिमी)",
            "temperature": "सरासरी तापमान (°C)",
            "humidity": "सापेक्ष आर्द्रता (%)",
            "has_soil_test": "माझ्याकडे प्रमाणित मृदा आरोग्य पत्रिका (SHC) आहे",
            "submit_btn": "ऑफलाइन V2 पीक शिफारस पहा",
            "suitability": "योग्यता गुण",
            "yield_range": "अपेक्षित उत्पादन",
            "water_status": "पाणी स्थिती",
            "limiting_factor": "मुख्य मर्यादित घटक",
            "fertilizer_plan": "संतुलित खत योजना (DAP/युरिया/MOP)",
            "excluded_crops": "अयोग्य पिके व कारणे",
        },
        "fert_ui": {
            "title": "खत व सिंचन मार्गदर्शन (ऑफलाइन KB)",
            "select_crop": "पीक निवडा",
            "farm_area": "शेताचे क्षेत्रफळ",
            "unit_ha": "हेक्टर (ha)",
            "unit_acres": "एकर (आपोआप हेक्टरमध्ये रूपांतरित होईल)",
            "submit_btn": "क्षेत्रफळानुसार खत व सिंचन योजना पहा",
            "fertilizer": "खत",
            "per_ha_rate": "प्रमाणित दर (प्रति हेक्टर)",
            "scaled_qty": "शेतासाठी एकूण मात्रा",
            "nutrients": "अन्नद्रव्ये",
            "timing": "वापराची वेळ व अवस्था",
            "climate_soil_guide": "पीक सिंचन, जमीन व हवामान माहिती",
            "safety_note": "खत खरेदीपूर्वी मृदा आरोग्य पत्रिका (SHC) आणि स्थानिक KVK सल्ल्याची खात्री करा.",
        },
    },
    "te": {
        "language_code": "te",
        "language_name": "తెలుగు",
        "fallback_warning": "అభ్యర్థించిన భాషకు మద్దతు లేదు; సురక్షితంగా ఆంగ్లం (en) ఉపయోగించబడింది.",
        "headers": {
            "crop": "పంట",
            "detected_condition": "గుర్తించిన వ్యాధి / తెగులు",
            "severity_level": "తీవ్రత స్థాయి",
            "infected_area": "ప్రభావిత ఆకు ప్రాంతం",
            "confidence": "నమ్మక స్థాయి",
            "status": "స్థితి",
            "symptoms": "ప్రధాన లక్షణాలు",
            "regional_advisory": "ప్రాంతీయ వ్యవసాయ సూచనలు",
            "crop_info": "పంట నేల & వాతావరణ వివరాలు",
            "explanation": "పరీక్ష సారాంశం",
            "immediate_actions": "తక్షణ చర్యలు",
            "organic_cultural_control": "సేంద్రీయ & సాగు నియంత్రణ",
            "approved_chemical_guidance": "ఆమోదించబడిన రసాయన మార్గదర్శకం",
            "preventive_measures": "నివారణ చర్యలు",
            "kvk_escalation": "KVK & హెల్ప్‌లైన్ సంప్రదింపు",
            "disclaimer": "భద్రతా గమనిక",
            "language": "భాష",
        },
        "severity_badges": {
            "low": "తక్కువ (<15%)",
            "moderate": "మధ్యస్థం (15%–40%)",
            "severe": "తీవ్రం (>40%)",
            "undetermined": "నిర్ధారించబడలేదు",
        },
        "status_banners": {
            "SUCCESS": "సలహా సిద్ధం — ICAR/KVK సిఫార్సులపై ఆధారితం",
            "UNCERTAIN_IMAGE": "ఫోటో స్పష్టంగా లేదు — దయచేసి వెలుతురులో స్పష్టమైన ఫోటో తీయండి",
            "LOW_CONFIDENCE": "తక్కువ నమ్మకం — సమీప KVK నిపుణుల నిర్ధారణ అవసరం",
            "SEVERE_ESCALATION": "తీవ్ర వ్యాధి హెచ్చరిక — వెంటనే సమీప KVK ను సంప్రదించండి",
        },
        "kvk_escalation_messages": {
            "helpline_number": "1800-180-1551",
            "recommended_escalation": (
                "తీవ్రమైన వ్యాధి/తెగులు స్థాయి గుర్తించబడింది. పిచికారీకి ముందు వెంటనే సమీప కృషి విజ్ఞాన కేంద్రం (KVK) "
                "లేదా ఉచిత కిసాన్ కాల్ సెంటర్ (1800-180-1551) ను సంప్రదించండి."
            ),
            "low_confidence_hold": (
                "ఫోటో ఆధారంగా ఖచ్చితమైన నిర్ధారణ సాధ్యం కాలేదు. తాజా ఆకు నమూనాతో సమీప KVK "
                "లేదా కిసాన్ కాల్ సెంటర్ (1800-180-1551) ను సంప్రదించండి."
            ),
            "routine_note": (
                "స్థానిక మోతాదు నిర్ధారణ కోసం మీ జిల్లా కృషి విజ్ఞాన కేంద్రం (KVK) "
                "లేదా కిసాన్ కాల్ సెంటర్ (1800-180-1551) తో మాట్లాడండి."
            ),
        },
        "crops": {
            "rice": "వరి",
            "wheat": "గోధుమ",
            "maize": "మొక్కజొన్న",
            "cotton": "పత్తి",
            "sugarcane": "చెరకు",
            "soybean": "సోయాబీన్",
            "groundnut": "వేరుశనగ",
            "potato": "బంగాళాదుంప",
            "chickpea": "శనగలు",
            "lentil": "మసూర్ పప్పు",
            "mustard": "ఆవాలు",
            "tomato": "టమాటా",
            "onion": "ఉల్లిపాయ",
            "banana": "అరటి",
            "mango": "మామిడి",
            "jowar": "జొన్న",
            "apple": "ఆపిల్",
            "cherry": "చెర్రీ",
            "grape": "ద్రాక్ష",
            "peach": "పీచ్",
            "orange": "నారింజ",
            "bell pepper": "క్యాప్సికమ్",
            "squash": "గుమ్మడి",
            "strawberry": "స్ట్రాబెర్రీ",
            "blueberry": "బ్లూబెర్రీ",
            "raspberry": "రాస్ప్బెర్రీ",
            "general": "పంట ఆకు",
            "unknown": "తెలియని పంట",
        },
        "conditions": {
            "rice_blast": "వరి అగ్గి తెగులు (బ్లాస్ట్)",
            "rice_brown_spot": "వరి గోధుమ రంగు మచ్చ తెగులు",
            "rice_bacterial_leaf_blight": "వరి బాక్టీరియా ఆకు ఎండు తెగులు",
            "rice_brown_planthopper": "గోధుమ దోమ (BPH)",
            "wheat_leaf_rust": "గోధుమ ఆకు తుప్పు తెగులు",
            "wheat_yellow_rust": "గోధుమ పసుపు తుప్పు తెగులు",
            "wheat_aphids": "గోధుమ పేనుబంక",
            "cotton_bollworm": "పత్తి కాయతొలుచు పురుగు",
            "cotton_aphids": "పత్తి పేనుబంక",
            "cotton_whitefly": "పత్తి తెల్లదోమ",
            "potato_late_blight": "బంగాళాదుంప ఆలస్య మాడు తెగులు",
            "potato_early_blight": "బంగాళాదుంప ముందస్తు మాడు తెగులు",
            "tomato_early_blight": "టమాటా ముందస్తు మాడు తెగులు",
            "tomato_late_blight": "టమాటా ఆలస్య మాడు తెగులు",
            "tomato_yellow_leaf_curl_virus": "టమాటా పసుపు ఆకు ముడత వైరస్",
            "maize_fall_armyworm": "మొక్కజొన్న కత్తెర పురుగు (ఫాల్ ఆర్మీవార్మ్)",
            "maize_northern_leaf_blight": "మొక్కజొన్న ఉత్తర ఆకు మాడు తెగులు",
            "healthy": "ఆరోగ్యకరమైన పంట",
            "unclear_image": "అస్పష్టమైన ఫోటో",
        },
        "tabs": {
            "disease_scan": "వ్యాధి & తెగులు పరీక్ష",
            "crop_recommendation": "పంట సిఫార్సు (V2)",
            "fertilizer_advisory": "ఎరువులు & నీటిపారుదల",
        },
        "untrained_vision_warning": (
            "శిక్షణ లేని స్మోక్ నమూనా యాక్టివ్‌గా ఉంది. అసలైన PlantVillage గుర్తింపు కోసం "
            "--vision-backend hf-local తో ప్రారంభించండి."
        ),
        "analyzing_status": "ఆఫ్‌లైన్ నాణ్యత తనిఖీ, PlantVillage విజన్ మరియు స్థానిక KB సలహా నడుస్తోంది...",
        "advisory_fallback_notice": (
            "సలహా రన్‌టైమ్ గమనిక: ఆఫ్‌లైన్ PlantVillage విజన్ + V2 పంట ఇంజిన్ + ICAR జ్ఞాన భాండాగారం యాక్టివ్‌గా ఉన్నాయి."
        ),
        "crop_rec_ui": {
            "title": "ఆఫ్‌లైన్ పంట సిఫార్సు (FieldWise V2 ఇంజిన్)",
            "state": "రాష్ట్రం (తప్పనిసరి)",
            "sowing_month": "విత్తే నెల (1–12, తప్పనిసరి)",
            "irrigation_access": "నీటిపారుదల సౌకర్యం",
            "soil_texture": "నేల రకం",
            "soil_ph": "నేల pH",
            "nitrogen": "లభ్యమయ్యే నత్రజని (N, కిలో/హె.)",
            "phosphorus": "లభ్యమయ్యే భాస్వరం (P₂O₅, కిలో/హె.)",
            "potassium": "లభ్యమయ్యే పొటాష్ (K₂O, కిలో/హె.)",
            "seasonal_rainfall": "సీజన్ వర్షపాతం (మి.మీ)",
            "temperature": "సగటు ఉష్ణోగ్రత (°C)",
            "humidity": "సాపేక్ష ఆర్ద్రత (%)",
            "has_soil_test": "నా వద్ద ధృవీకరించబడిన సాయిల్ హెల్త్ కార్డ్ (SHC) ఉంది",
            "submit_btn": "పంట సిఫార్సులు పొందండి",
            "suitability": "అనుకూలత స్కోరు",
            "yield_range": "అంచనా దిగుబడి",
            "water_status": "నీటి స్థితి",
            "limiting_factor": "ప్రధాన పరిమితి అంశం",
            "fertilizer_plan": "సమతుల్య ఎరువుల ప్రణాళిక (DAP/యూరియా/MOP)",
            "excluded_crops": "అనర్హమైన పంటలు & కారణాలు",
        },
        "fert_ui": {
            "title": "ఎరువులు & నీటిపారుదల సలహా (ఆఫ్‌లైన్ KB)",
            "select_crop": "పంటను ఎంచుకోండి",
            "farm_area": "పొలం విస్తీర్ణం",
            "unit_ha": "హెక్టార్లు (ha)",
            "unit_acres": "ఎకరాలు (స్వయంచాలకంగా హెక్టార్లలోకి మారుతుంది)",
            "submit_btn": "ఎరువుల ప్రణాళిక పొందండి",
            "fertilizer": "ఎరువు",
            "per_ha_rate": "ప్రామాణిక మోతాదు (ప్రతి హెక్టారుకు)",
            "scaled_qty": "పొలానికి మొత్తం మోతాదు",
            "nutrients": "పోషకాలు",
            "timing": "వేసే సమయం & దశ",
            "climate_soil_guide": "పంట నీటిపారుదల, నేల & వాతావరణ మార్గదర్శి",
            "safety_note": "ఎరువులు కొనుగోలు చేసే ముందు సాయిల్ హెల్త్ కార్డ్ (SHC) మరియు స్థానిక KVK సలహాను సరిచూసుకోండి.",
        },
    },
    "ta": {
        "language_code": "ta",
        "language_name": "தமிழ்",
        "fallback_warning": "கோரப்பட்ட மொழி ஆதரிக்கப்படவில்லை; பாதுகாப்பாக ஆங்கிலம் (en) தேர்ந்தெடுக்கப்பட்டது.",
        "headers": {
            "crop": "பயிர்",
            "detected_condition": "கண்டறியப்பட்ட நோய் / பூச்சி",
            "severity_level": "தீவிர நிலை",
            "infected_area": "பாதிக்கப்பட்ட இலைப் பரப்பு",
            "confidence": "நம்பிக்கை அளவு",
            "status": "நிலை",
            "symptoms": "முக்கிய அறிகுறிகள்",
            "regional_advisory": "பகுதி சார்ந்த வேளாண் குறிப்புகள்",
            "crop_info": "பயிர் மண் மற்றும் காலநிலை விவரம்",
            "explanation": "ஆய்வுச் சுருக்கம்",
            "immediate_actions": "உடனடி நடவடிக்கைகள்",
            "organic_cultural_control": "இயற்கை & சாகுபடி கட்டுப்பாடு",
            "approved_chemical_guidance": "அங்கீகரிக்கப்பட்ட இரசாயன வழிகாட்டல்",
            "preventive_measures": "தடுப்பு நடவடிக்கைகள்",
            "kvk_escalation": "KVK & உதவி எண் தொடர்பு",
            "disclaimer": "பாதுகாப்பு எச்சரிக்கை",
            "language": "மொழி",
        },
        "severity_badges": {
            "low": "குறைவு (<15%)",
            "moderate": "மிதமானது (15%–40%)",
            "severe": "தீவிரம் (>40%)",
            "undetermined": "தீர்மானிக்கப்படவில்லை",
        },
        "status_banners": {
            "SUCCESS": "ஆலோசனை தயார் — ICAR/KVK பரிந்துரைகளின் அடிப்படையில்",
            "UNCERTAIN_IMAGE": "படம் தெளிவாக இல்லை — வெளிச்சத்தில் தெளிவான படம் எடுக்கவும்",
            "LOW_CONFIDENCE": "குறைந்த நம்பிக்கை — உள்ளூர் KVK நிபுணர் சரிபார்ப்பு தேவை",
            "SEVERE_ESCALATION": "தீவிர பாதிப்பு எச்சரிக்கை — உடனடியாக உள்ளூர் KVK-ஐ அணுகவும்",
        },
        "kvk_escalation_messages": {
            "helpline_number": "1800-180-1551",
            "recommended_escalation": (
                "தீவிர நோய்/பூச்சி நிலை கண்டறியப்பட்டுள்ளது. தெளிப்பதற்கு முன் உடனடியாக அருகிலுள்ள வேளாண் அறிவியல் மையம் (KVK) "
                "அல்லது இலவச கிசான் கால் சென்டரை (1800-180-1551) தொடர்பு கொள்ளவும்."
            ),
            "low_confidence_hold": (
                "படத்திலிருந்து உறுதியான அடையாளம் காண முடியவில்லை. புதிய இலை மாதிரியுடன் உள்ளூர் KVK "
                "அல்லது கிசான் கால் சென்டரை (1800-180-1551) அணுகவும்."
            ),
            "routine_note": (
                "உள்ளூர் சரியான அளவை உறுதிப்படுத்த உங்கள் மாவட்ட வேளாண் அறிவியல் மையம் (KVK) "
                "அல்லது கிசான் கால் சென்டரை (1800-180-1551) தொடர்பு கொள்ளவும்."
            ),
        },
        "crops": {
            "rice": "நெல்",
            "wheat": "கோதுமை",
            "maize": "மக்காச்சோளம்",
            "cotton": "பருத்தி",
            "sugarcane": "கரும்பு",
            "soybean": "சோயாபீன்",
            "groundnut": "நிலக்கடலை",
            "potato": "உருளைக்கிழங்கு",
            "chickpea": "கொண்டைக்கடலை",
            "lentil": "மசூர் பருப்பு",
            "mustard": "கடுகு",
            "tomato": "தக்காளி",
            "onion": "வெங்காயம்",
            "banana": "வாழை",
            "mango": "மாம்பழம்",
            "jowar": "சோளம்",
            "apple": "ஆப்பிள்",
            "cherry": "செர்ரி",
            "grape": "திராட்சை",
            "peach": "பீச்",
            "orange": "ஆரஞ்சு",
            "bell pepper": "குடைமிளகாய்",
            "squash": "பூசணி",
            "strawberry": "ஸ்ட்ராபெரி",
            "blueberry": "ப்ளூபெரி",
            "raspberry": "ராஸ்பெரி",
            "general": "பயிர் இலை",
            "unknown": "அறியப்படாத பயிர்",
        },
        "conditions": {
            "rice_blast": "நெல் குலைநோய் (பிளாஸ்ட்)",
            "rice_brown_spot": "நெல் பழுப்புப் புள்ளி நோய்",
            "rice_bacterial_leaf_blight": "நெல் பாக்டீரியா இலை கருகல் நோய்",
            "rice_brown_planthopper": "பழுப்புத் தத்துப்பூச்சி (BPH)",
            "wheat_leaf_rust": "கோதுமை இலை துரு நோய்",
            "wheat_yellow_rust": "கோதுமை மஞ்சள் துரு நோய்",
            "wheat_aphids": "கோதுமை அசுவினி",
            "cotton_bollworm": "பருத்தி காய்ப்புழு",
            "cotton_aphids": "பருத்தி அசுவினி",
            "cotton_whitefly": "பருத்தி வெள்ளை ஈ",
            "potato_late_blight": "உருளைக்கிழங்கு பின்பருவ கருகல் நோய்",
            "potato_early_blight": "உருளைக்கிழங்கு முன்பருவ கருகல் நோய்",
            "tomato_early_blight": "தக்காளி முன்பருவ கருகல் நோய்",
            "tomato_late_blight": "தக்காளி பின்பருவ கருகல் நோய்",
            "tomato_yellow_leaf_curl_virus": "தக்காளி மஞ்சள் இலை சுருட்டு வைரஸ்",
            "maize_fall_armyworm": "மக்காச்சோளம் படைப்புழு (பால் ஆர்மிவோர்ம்)",
            "maize_northern_leaf_blight": "மக்காச்சோளம் வடக்கு இலை கருகல் நோய்",
            "healthy": "ஆரோக்கியமான பயிர்",
            "unclear_image": "தெளிவற்ற படம்",
        },
        "tabs": {
            "disease_scan": "நோய் & பூச்சி ஆய்வு",
            "crop_recommendation": "பயிர் பரிந்துரை (V2)",
            "fertilizer_advisory": "உரம் & நீர்ப்பாசனம்",
        },
        "untrained_vision_warning": (
            "பயிற்சி பெறாத மாதிரி இயக்கத்தில் உள்ளது. உண்மையான PlantVillage கணிப்பிற்கு --vision-backend hf-local உடன் தொடங்கவும்."
        ),
        "analyzing_status": "ஆஃப்லைன் தரச் சோதனை, PlantVillage பார்வை & உள்ளூர் KB ஆலோசனை இயங்குகிறது...",
        "advisory_fallback_notice": (
            "ஆலோசனை இயக்கக் குறிப்பு: ஆஃப்லைன் PlantVillage பார்வை + V2 பயிர் இயந்திரம் + ICAR அறிவுத் தொகுப்பு செயல்பாட்டில் உள்ளன."
        ),
        "crop_rec_ui": {
            "title": "ஆஃப்லைன் பயிர் பரிந்துரை (FieldWise V2 இயந்திரம்)",
            "state": "மாநிலம் (கட்டாயம்)",
            "sowing_month": "விதைப்பு மாதம் (1–12, கட்டாயம்)",
            "irrigation_access": "நீர்ப்பாசன வசதி",
            "soil_texture": "மண் வகை",
            "soil_ph": "மண் pH",
            "nitrogen": "கிடைக்கும் தழைச்சத்து (N, கிலோ/எக்.)",
            "phosphorus": "கிடைக்கும் மணிச்சத்து (P₂O₅, கிலோ/எக்.)",
            "potassium": "கிடைக்கும் சாம்பல்சத்து (K₂O, கிலோ/எக்.)",
            "seasonal_rainfall": "பருவ மழை அளவு (மி.மீ)",
            "temperature": "சராசரி வெப்பநிலை (°C)",
            "humidity": "ஈரப்பதம் (%)",
            "has_soil_test": "என்னிடம் மண் வள அட்டை (SHC) அறிக்கை உள்ளது",
            "submit_btn": "பயிர் பரிந்துரைகளைப் பெறுக",
            "suitability": "பொருத்த மதிப்பெண்",
            "yield_range": "எதிர்பார்க்கப்படும் மகசூல்",
            "water_status": "நீர் நிலை",
            "limiting_factor": "முக்கிய கட்டுப்படுத்தும் காரணி",
            "fertilizer_plan": "சமச்சீர் உரத் திட்டம் (DAP/யூரியா/MOP)",
            "excluded_crops": "பொருந்தாத பயிர்கள் & காரணங்கள்",
        },
        "fert_ui": {
            "title": "உரம் & நீர்ப்பாசன ஆலோசனை (ஆஃப்லைன் KB)",
            "select_crop": "பயிரைத் தேர்ந்தெடுக்கவும்",
            "farm_area": "வயல் பரப்பளவு",
            "unit_ha": "ஹெக்டேர் (ha)",
            "unit_acres": "ஏக்கர் (தானாக ஹெக்டேராக மாற்றப்படும்)",
            "submit_btn": "உர அட்டவணையைப் பெறுக",
            "fertilizer": "உரம்",
            "per_ha_rate": "நிலையான அளவு (ஹெக்டேருக்கு)",
            "scaled_qty": "வயலுக்கான மொத்த அளவு",
            "nutrients": "ஊட்டச்சத்துக்கள்",
            "timing": "இடும் நேரம் & நிலை",
            "climate_soil_guide": "பயிர் நீர்ப்பாசனம், மண் & காலநிலை வழிகாட்டி",
            "safety_note": "உரங்கள் வாங்குவதற்கு முன் மண் வள அட்டை (SHC) மற்றும் உள்ளூர் KVK ஆலோசனையைச் சரிபார்க்கவும்.",
        },
    },
}


def get_ui_bundle(lang: str) -> Dict[str, Any]:
    """
    Returns an offline localized UI dictionary for `lang` (`en`, `hi`, `gu`, `mr`, `te`, `ta`).
    Safely defaults to `en` on unsupported codes and attaches a warning flag.
    """
    requested = str(lang or "").strip()
    norm = requested.lower()
    fallback_used = norm not in _TRANSLATIONS
    effective = "en" if fallback_used else norm

    if fallback_used:
        LOGGER.warning("Unsupported language '%s'; falling back to 'en'.", requested)

    bundle = copy.deepcopy(_TRANSLATIONS[effective])
    bundle["requested_language"] = requested or "en"
    bundle["language_fallback_used"] = fallback_used
    return bundle


def localize_status_message(status: str, lang: str = "en") -> str:
    """Returns the localized status banner message for an `AdvisoryStatus` value."""
    bundle = get_ui_bundle(lang)
    status_key = getattr(status, "value", str(status)).strip().upper()
    banners: Dict[str, str] = bundle["status_banners"]
    if status_key in banners:
        return banners[status_key]
    return status_key


def localize_crop_name(crop_id: str, lang: str = "en") -> str:
    """Returns localized crop display name if known, else the original string."""
    bundle = get_ui_bundle(lang)
    key = str(crop_id or "").strip().lower()
    return bundle["crops"].get(key, crop_id)


def localize_condition_name(condition_id_or_name: str, lang: str = "en") -> str:
    """Returns localized condition display name when mapped by ID or normalized name."""
    bundle = get_ui_bundle(lang)
    raw = str(condition_id_or_name or "").strip()
    key = raw.lower().replace(" ", "_")
    if key in bundle["conditions"]:
        return bundle["conditions"][key]

    # Match against English condition display names as reverse fallback
    en_conditions = _TRANSLATIONS["en"]["conditions"]
    for cid, en_label in en_conditions.items():
        if en_label.lower() == raw.lower():
            return bundle["conditions"].get(cid, raw)
    return raw
