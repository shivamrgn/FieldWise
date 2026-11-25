"""
Offline Advisory Engine & Safety/Hallucination Guardrails (Pipeline B).

Combines deterministic pre-LLM safety gates, grounded Llama 3.2 prompt
construction, resilient JSON extraction, post-LLM chemical grounding
verification, and severity-driven KVK escalation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ps1_offline_assistant.advisory.prompt_builder import build_llama3_prompt
from ps1_offline_assistant.advisory.runtime import LlamaAdvisoryBackend
from ps1_offline_assistant.kb.retriever import OfflineKBRetriever
from ps1_offline_assistant.schemas.contracts import (
    AdvisoryRequest,
    AdvisoryResponse,
    AdvisoryStatus,
    DiseasePestCard,
    SeverityLevel,
    SupportedLanguage,
    VisionCategory,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.65

# Common Indian agrochemical active ingredients (including restricted/banned molecules
# such as monocrotophos/endosulfan/phorate/methyl parathion) monitored to catch
# ungrounded LLM pesticide hallucinations.
MONITORED_CHEMICAL_MOLECULES: Set[str] = {
    "monocrotophos",
    "endosulfan",
    "phorate",
    "carbofuran",
    "methyl parathion",
    "parathion",
    "ddt",
    "chlorpyrifos",
    "carbendazim",
    "hexaconazole",
    "propineb",
    "fipronil",
    "paraquat",
    "glyphosate",
    "cypermethrin",
    "deltamethrin",
    "lambda-cyhalothrin",
    "profenofos",
    "triazophos",
    "acephate",
    "quinalphos",
    "dimethoate",
    "malathion",
    "imidacloprid",
    "thiamethoxam",
    "acetamiprid",
    "dinotefuran",
    "pymetrozine",
    "buprofezin",
    "flonicamid",
    "pyriproxyfen",
    "spiromesifen",
    "diafenthiuron",
    "tricyclazole",
    "isoprothiolane",
    "azoxystrobin",
    "difenoconazole",
    "propiconazole",
    "tebuconazole",
    "mancozeb",
    "cymoxanil",
    "dimethomorph",
    "metalaxyl",
    "chlorothalonil",
    "emamectin",
    "chlorantraniliprole",
    "spinetoram",
    "spinosad",
    "copper hydroxide",
    "copper oxychloride",
    "streptocycline",
    "bordeaux",
}

_FORMULATION_PATTERN = re.compile(
    r"\b([a-z][a-z0-9-]{3,})\s+\d+(?:\.\d+)?\s*%", re.IGNORECASE
)
_DOSAGE_RATE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ml|g|gm|kg|l)\s*/\s*(?:l|liter|litre|ha|acre)\b",
    re.IGNORECASE,
)


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Resiliently extracts a JSON object from LLM text, handling markdown code
    fences or stray text around the outermost `{ ... }` block.
    """
    if not raw_text or not raw_text.strip():
        return None
    text = raw_text.strip()

    # 1. Direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences (```json ... ```)
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Scan for outermost balanced JSON object
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None
    return None


def _approved_chemical_corpus(kb_entry: DiseasePestCard) -> Tuple[str, Set[str]]:
    """Builds normalized approved text and active-ingredient tokens from `kb_entry`."""
    raw_chunks = []
    approved_molecules: Set[str] = set()
    stop_tokens = {
        "or",
        "and",
        "with",
        "only",
        "wp",
        "ec",
        "sc",
        "sl",
        "sg",
        "wg",
        "df",
        "sp",
        "vector",
        "control",
        "whitefly",
    }

    for opt in kb_entry.approved_chemical_options:
        raw_chunks.append(opt.active_ingredient.lower())
        raw_chunks.append(opt.target_stage.lower())
        raw_chunks.append(opt.cibrc_aligned_note.lower())
        raw_chunks.append(opt.safety_warning.lower())

        for token in re.findall(r"[a-z][a-z0-9-]+", opt.active_ingredient.lower()):
            if len(token) >= 4 and token not in stop_tokens:
                approved_molecules.add(token)

    return " ".join(raw_chunks), approved_molecules


def _verify_chemical_grounding(
    parsed: Dict[str, Any],
    kb_entry: DiseasePestCard,
) -> Tuple[bool, str]:
    """
    Verifies that LLM output does not invent unapproved pesticides or dosages
    not present in `kb_entry`.
    """
    approved_text, approved_molecules = _approved_chemical_corpus(kb_entry)
    full_kb_text = " ".join(
        [
            approved_text,
            " ".join(kb_entry.cultural_organic_control).lower(),
        ]
    )

    chem_list = parsed.get("chemical_guidance_if_approved", [])
    if not isinstance(chem_list, list):
        return False, "chemical_guidance_if_approved is not a list."

    # 1. Every chemical guidance entry must reference at least one approved active molecule
    for item in chem_list:
        if not isinstance(item, str) or not item.strip():
            return False, "Empty or non-string item in chemical_guidance_if_approved."
        item_lower = item.lower()
        if approved_molecules and not any(mol in item_lower for mol in approved_molecules):
            return (
                False,
                f"Chemical guidance entry '{item}' does not reference any approved "
                f"active ingredient from kb_entry ({sorted(approved_molecules)}).",
            )

        # Check for invented explicit dosage rates not present in kb_entry
        for rate_match in _DOSAGE_RATE_PATTERN.findall(item_lower):
            if rate_match.replace(" ", "") not in approved_text.replace(" ", ""):
                return (
                    False,
                    f"LLM output invented ungrounded chemical dosage rate '{rate_match}'.",
                )

    # 2. Scan all advisory text fields for unapproved pesticide molecules or formulations
    all_text_parts: List[str] = [str(parsed.get("explanation", ""))]
    for list_key in (
        "immediate_actions",
        "organic_cultural_steps",
        "chemical_guidance_if_approved",
        "preventive_measures",
    ):
        val = parsed.get(list_key, [])
        if isinstance(val, list):
            all_text_parts.extend(str(x) for x in val)
    combined_lower = " ".join(all_text_parts).lower()

    for molecule in MONITORED_CHEMICAL_MOLECULES:
        if re.search(rf"\b{re.escape(molecule)}\b", combined_lower):
            if molecule not in full_kb_text:
                return (
                    False,
                    f"Hallucinated or unapproved chemical '{molecule}' detected outside kb_entry.",
                )

    for form_word in _FORMULATION_PATTERN.findall(combined_lower):
        fw_lower = form_word.lower()
        if fw_lower not in full_kb_text:
            return (
                False,
                f"Ungrounded pesticide formulation '{fw_lower}' detected outside kb_entry.",
            )

    return True, "ok"


def _clean_str_list(value: Any, fallback: List[str]) -> List[str]:
    if not isinstance(value, list):
        return list(fallback)
    cleaned = [str(x).strip() for x in value if isinstance(x, str) and x.strip()]
    return cleaned if cleaned else list(fallback)


def _build_uncertain_image_response(language: SupportedLanguage) -> AdvisoryResponse:
    templates = {
        SupportedLanguage.EN: {
            "explanation": "The photo is blurry, too dark, or overexposed. Reliable disease or pest screening cannot be performed from an unclear image.",
            "immediate": [
                "Take a new close-up photo in clear daylight holding the camera steady 15–25 cm from the leaf or pest.",
                "Ensure affected spots or insects are sharply in focus against a plain background.",
            ],
            "cultural": [
                "Inspect the lower and upper leaf surfaces visually while capturing a sharper photograph."
            ],
            "preventive": [
                "Avoid spraying any pesticide until the condition is clearly identified."
            ],
            "disclaimer": "No diagnosis or chemical guidance issued due to poor image quality. Retake photo or consult your local Krishi Vigyan Kendra (KVK).",
        },
        SupportedLanguage.HI: {
            "explanation": "फोटो धुंधली, बहुत अंधेरी या अधिक रोशनी वाली है। स्पष्ट फोटो के बिना रोग या कीट की सही पहचान नहीं की जा सकती।",
            "immediate": [
                "दिन की साफ रोशनी में पत्ती या कीट के पास (15–25 सेमी दूरी से) कैमरा स्थिर रखकर नई स्पष्ट फोटो लें।",
                "ध्यान रखें कि प्रभावित दाग या कीट फोटो में साफ दिखाई दें।",
            ],
            "cultural": [
                "दोबारा फोटो लेते समय पत्तियों के ऊपरी और निचले हिस्से को ध्यान से देखें।"
            ],
            "preventive": [
                "सही पहचान हुए बिना किसी भी रासायनिक दवा का छिड़काव न करें।"
            ],
            "disclaimer": "अस्पष्ट फोटो के कारण कोई रोग पहचान या रसायन सलाह नहीं दी गई है। स्पष्ट फोटो लें या नजदीकी कृषि विज्ञान केंद्र (KVK) से संपर्क करें।",
        },
        SupportedLanguage.GU: {
            "explanation": "ફોટો ઝાંખો, ખૂબ અંધારો અથવા વધુ પડતા પ્રકાશવાળો છે. અસ્પષ્ટ ફોટા પરથી રોગ કે જીવાતની વિશ્વસનીય તપાસ થઈ શકતી નથી.",
            "immediate": [
                "દિવસના ચોખ્ખા પ્રકાશમાં પાન અથવા જીવાતથી 15–25 સેમી દૂર કેમેરા સ્થિર રાખી નવો સ્પષ્ટ ફોટો લો.",
                "અસરગ્રસ્ત ભાગ ફોકસમાં સ્પષ્ટ દેખાય તેની ખાતરી કરો.",
            ],
            "cultural": [
                "ફરીથી ફોટો લેતી વખતે પાનની નીચેની અને ઉપરની સપાટીનું નિરીક્ષણ કરો."
            ],
            "preventive": [
                "યોગ્ય ઓળખ થયા વિના કોઈ પણ દવાનો છંટકાવ કરશો નહીં."
            ],
            "disclaimer": "અસ્પષ્ટ ફોટાને કારણે કોઈ નિદાન કે રાસાયણિક સલાહ આપવામાં આવી નથી. સ્પષ્ટ ફોટો લો અથવા સ્થાનિક કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) નો સંપર્ક કરો.",
        },
        SupportedLanguage.MR: {
            "explanation": "फोटो धूसर, खूप अंधुक किंवा अतिप्रकाशी आहे. अस्पष्ट फोटोवरून रोग किंवा किडीची खात्रीशीर तपासणी करता येत नाही.",
            "immediate": [
                "दिवसाच्या स्वच्छ प्रकाशात पानाजवळ (15–25 सेमी अंतरावरून) कॅमेरा स्थिर ठेवून नवीन स्पष्ट फोटो काढा.",
                "रोगाचे डाग किंवा कीड फोकसमध्ये स्पष्ट दिसतील याची खात्री करा.",
            ],
            "cultural": [
                "नवीन फोटो काढताना पानाची वरची आणि खालची बाजू नीट तपासा."
            ],
            "preventive": [
                "अचूक ओळख होईपर्यंत कोणतीही रासायनिक फवारणी करू नका."
            ],
            "disclaimer": "अस्पष्ट फोटोमुळे कोणतेही निदान किंवा रासायनिक सल्ला दिलेला नाही. स्पष्ट फोटो घ्या किंवा स्थानिक कृषी विज्ञान केंद्राशी (KVK) संपर्क साधा.",
        },
        SupportedLanguage.TE: {
            "explanation": "ఫోటో మసకగా, చాలా చీకటిగా లేదా ఎక్కువ వెలుతురుతో ఉంది. అస్పష్టమైన ఫోటో ఆధారంగా వ్యాధి లేదా తెగులు నిర్ధారణ సాధ్యం కాదు.",
            "immediate": [
                "పగటి స్పష్టమైన వెలుతురులో ఆకు లేదా తెగులుకు 15–25 సెం.మీ దూరంలో కెమెరా స్థిరంగా పట్టి కొత్త ఫోటో తీయండి.",
                "దెబ్బతిన్న మచ్చలు లేదా పురుగులు స్పష్టంగా ఫోకస్‌లో ఉండేలా చూసుకోండి.",
            ],
            "cultural": [
                "కొత్త ఫోటో తీసేటప్పుడు ఆకుల పైభాగం మరియు కింది భాగాన్ని జాగ్రత్తగా పరిశీలించండి."
            ],
            "preventive": [
                "వ్యాధి ఖచ్చితంగా నిర్ధారణ అయ్యే వరకు ఎలాంటి పురుగుమందులు పిచికారీ చేయవద్దు."
            ],
            "disclaimer": "ఫోటో స్పష్టంగా లేనందున ఎటువంటి నిర్ధారణ లేదా రసాయన సలహా ఇవ్వబడలేదు. స్పష్టమైన ఫోటో తీయండి లేదా సమీప KVK ను సంప్రదించండి.",
        },
        SupportedLanguage.TA: {
            "explanation": "புகைப்படம் மங்கலாக, மிகவும் இருட்டாக அல்லது அதிக வெளிச்சத்துடன் உள்ளது. தெளிவற்ற படத்திலிருந்து நோய் அல்லது பூச்சி ஆய்வு செய்ய முடியாது.",
            "immediate": [
                "பகல் வெளிச்சத்தில் இலை அல்லது பூச்சியிலிருந்து 15–25 செ.மீ தூரத்தில் கேமராவை அசையாமல் பிடித்து புதிய தெளிவான படம் எடுக்கவும்.",
                "பாதிக்கப்பட்ட புள்ளிகள் அல்லது பூச்சிகள் தெளிவாகத் தெரிவதை உறுதி செய்யவும்.",
            ],
            "cultural": [
                "மீண்டும் படம் எடுக்கும்போது இலையின் மேல் மற்றும் கீழ் பக்கங்களை கவனமாகப் பார்க்கவும்."
            ],
            "preventive": [
                "நோய் சரியாக அடையாளம் காணப்படும் வரை எந்த பூச்சிக்கொல்லியும் தெளிக்க வேண்டாம்."
            ],
            "disclaimer": "தெளிவற்ற படம் காரணமாக நோயறிதல் அல்லது இரசாயன ஆலோசனை வழங்கப்படவில்லை. தெளிவான படம் எடுக்கவும் அல்லது அருகிலுள்ள KVK-ஐ அணுகவும்.",
        },
    }[language]

    return AdvisoryResponse(
        status=AdvisoryStatus.UNCERTAIN_IMAGE,
        explanation=templates["explanation"],
        immediate_actions=templates["immediate"],
        organic_cultural_steps=templates["cultural"],
        chemical_guidance_if_approved=[],
        preventive_measures=templates["preventive"],
        escalate_to_kvk=False,
        disclaimer=templates["disclaimer"],
        language=language,
        ttft_ms=0.0,
        tokens_per_sec=0.0,
    )


def _build_low_confidence_response(
    language: SupportedLanguage,
    state_region: str,
    confidence: float,
    detected_crop: Optional[str] = None,
    detected_condition: Optional[str] = None,
    severity_level: Optional[str] = None,
    infected_area_pct: Optional[float] = None,
    confident_unmapped: bool = False,
) -> AdvisoryResponse:
    if confident_unmapped and detected_crop and detected_condition:
        sev = severity_level or "undetermined"
        area = f"{infected_area_pct:.1f}%" if infected_area_pct is not None else "N/A"
        unmapped_templates = {
            SupportedLanguage.EN: {
                "explanation": (
                    f"Visual screening detected {detected_condition} on {detected_crop} "
                    f"(confidence {confidence:.2f}, estimated {sev} severity, ~{area} affected area). "
                    f"However, our offline ICAR knowledge base does not include an approved chemical treatment card for this specific class."
                ),
                "immediate": [
                    f"Share this finding ({detected_crop}: {detected_condition}) with your nearest Krishi Vigyan Kendra (KVK) for state-approved treatment in {state_region}.",
                    "Call the free Kisan Call Center helpline at 1800-180-1551 for verified dosage guidance.",
                ],
                "cultural": [
                    "Isolate or prune heavily affected leaves, avoid overhead irrigation splash, and inspect neighboring plants.",
                ],
                "preventive": [
                    "Maintain field sanitation and do not spray unverified pesticides without local agronomist confirmation.",
                ],
                "disclaimer": (
                    "CONFIDENT SCREENING / UNMAPPED KB CARD: Chemical guidance is withheld because no local ICAR treatment card exists for this class. "
                    "Consult your local KVK or Kisan Call Center (1800-180-1551)."
                ),
            },
            SupportedLanguage.HI: {
                "explanation": (
                    f"कैमरा जांच में {detected_crop} पर {detected_condition} की पहचान हुई है "
                    f"(विश्वास {confidence:.2f}, अनुमानित गंभीरता {sev}, ~{area} प्रभावित क्षेत्र)। "
                    f"इस विशेष रोग के लिए स्थानीय ICAR उपचार कार्ड उपलब्ध नहीं है।"
                ),
                "immediate": [
                    f"{state_region} में अनुमोदित उपचार के लिए नजदीकी कृषि विज्ञान केंद्र (KVK) को यह परिणाम ({detected_condition}) दिखाएं।",
                    "निःशुल्क किसान कॉल सेंटर हेल्पलाइन 1800-180-1551 पर सलाह लें।",
                ],
                "cultural": [
                    "अधिक प्रभावित पत्तियों को अलग कर नष्ट करें, सिंचाई के छींटों से बचें और आसपास के पौधों की जांच करें।",
                ],
                "preventive": [
                    "खेत की स्वच्छता रखें और विशेषज्ञ की पुष्टि के बिना अनजान रासायनिक छिड़काव न करें।",
                ],
                "disclaimer": (
                    "रोग पहचान हुई पर स्थानीय KB उपचार कार्ड उपलब्ध नहीं है: रासायनिक मात्रा नहीं दी गई है। "
                    "कृपया नजदीकी KVK या किसान कॉल सेंटर (1800-180-1551) से सलाह लें।"
                ),
            },
            SupportedLanguage.GU: {
                "explanation": (
                    f"કેમેરા તપાસમાં {detected_crop} પર {detected_condition} ઓળખાયું છે "
                    f"(વિશ્વાસ {confidence:.2f}, અંદાજિત તીવ્રતા {sev}, ~{area} અસરગ્રસ્ત વિસ્તાર). "
                    f"આ રોગ માટે સ્થાનિક ICAR સારવાર કાર્ડ ઉપલબ્ધ નથી."
                ),
                "immediate": [
                    f"{state_region} માં માન્ય સારવાર માટે નજીકના કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) ને આ પરિણામ બતાવો.",
                    "મફત કિસાન કોલ સેન્ટર હેલ્પલાઇન 1800-180-1551 પર સંપર્ક કરો.",
                ],
                "cultural": [
                    "વધુ અસરગ્રસ્ત પાન દૂર કરો, પિયતના છાંટા ટાળો અને આસપાસના છોડ તપાસો.",
                ],
                "preventive": [
                    "ખેતર સ્વચ્છ રાખો અને નિષ્ણાતની સલાહ વિના કોઈ દવા છાંટશો નહીં.",
                ],
                "disclaimer": (
                    "ઓળખ થયેલ છે પરંતુ સ્થાનિક KB સારવાર કાર્ડ નથી: રાસાયણિક સલાહ આપવામાં આવી નથી. KVK અથવા 1800-180-1551 નો સંપર્ક કરો."
                ),
            },
            SupportedLanguage.MR: {
                "explanation": (
                    f"कॅमेरा तपासणीत {detected_crop} वर {detected_condition} आढळले आहे "
                    f"(विश्वास {confidence:.2f}, अंदाजे तीव्रता {sev}, ~{area} बाधित क्षेत्र). "
                    f"या रोगासाठी स्थानिक ICAR उपचार कार्ड उपलब्ध नाही."
                ),
                "immediate": [
                    f"{state_region} मध्ये शिफारस केलेल्या उपचारासाठी जवळच्या कृषी विज्ञान केंद्राशी (KVK) संपर्क साधा.",
                    "मोफत किसान कॉल सेंटर 1800-180-1551 वर सल्ला घ्या.",
                ],
                "cultural": [
                    "जास्त बाधित पाने वेगळी करा, फवारणी/पाण्याचे शिंतोडे टाळा आणि शेजारील झाडे तपासा.",
                ],
                "preventive": [
                    "शेतात स्वच्छता राखा आणि तज्ज्ञांच्या सल्ल्याशिवाय फवारणी करू नका.",
                ],
                "disclaimer": (
                    "रोग ओळखला गेला आहे पण स्थानिक KB उपचार कार्ड उपलब्ध नाही: रासायनिक सल्ला दिलेला नाही. KVK किंवा 1800-180-1551 वर संपर्क साधा."
                ),
            },
            SupportedLanguage.TE: {
                "explanation": (
                    f"కెమెరా పరీక్షలో {detected_crop} పంటపై {detected_condition} గుర్తించబడింది "
                    f"(నమ్మకం {confidence:.2f}, అంచనా తీవ్రత {sev}, ~{area} ప్రభావిత ప్రాంతం). "
                    f"దీనికి స్థానిక ICAR రసాయన చికిత్స కార్డు ఇంకా అందుబాటులో లేదు."
                ),
                "immediate": [
                    f"{state_region} లో ఆమోదించిన చికిత్స కోసం సమీప KVK ను సంప్రదించండి.",
                    "ఉచిత కిసాన్ కాల్ సెంటర్ 1800-180-1551 కు కాల్ చేయండి.",
                ],
                "cultural": [
                    "బాగా దెబ్బతిన్న ఆకులను తొలగించి, పక్కన ఉన్న మొక్కలను పరిశీలించండి.",
                ],
                "preventive": [
                    "పొలం పరిశుభ్రంగా ఉంచి నిపుణుల నిర్ధారణ లేకుండా రసాయనాలు పిచికారీ చేయవద్దు.",
                ],
                "disclaimer": (
                    "వ్యాధి గుర్తించబడింది కానీ స్థానిక KB చికిత్స కార్డు లేదు: రసాయన సలహా ఇవ్వబడలేదు. KVK లేదా 1800-180-1551 ను సంప్రదించండి."
                ),
            },
            SupportedLanguage.TA: {
                "explanation": (
                    f"கேமரா ஆய்வில் {detected_crop} பயிரில் {detected_condition} கண்டறியப்பட்டுள்ளது "
                    f"(நம்பிக்கை {confidence:.2f}, மதிப்பிடப்பட்ட தீவிரம் {sev}, ~{area} பாதிக்கப்பட்ட பரப்பு). "
                    f"இதற்கு உள்ளூர் ICAR இரசாயன சிகிச்சை அட்டை இல்லை."
                ),
                "immediate": [
                    f"{state_region} பகுதியில் அங்கீகரிக்கப்பட்ட சிகிச்சைக்கு அருகிலுள்ள KVK-ஐ அணுகவும்.",
                    "இலவச கிசான் கால் சென்டர் 1800-180-1551-ஐ அழைக்கவும்.",
                ],
                "cultural": [
                    "அதிகம் பாதிக்கப்பட்ட இலைகளை அகற்றி, அருகிலுள்ள செடிகளைப் பரிசோதிக்கவும்.",
                ],
                "preventive": [
                    "வயலை சுத்தமாக வைத்து நிபுணர் ஆலோசனை இன்றி மருந்து தெளிக்க வேண்டாம்.",
                ],
                "disclaimer": (
                    "நோய் கண்டறியப்பட்டது ஆனால் உள்ளூர் KB சிகிச்சை அட்டை இல்லை: இரசாயன ஆலோசனை வழங்கப்படவில்லை. KVK அல்லது 1800-180-1551-ஐ அணுகவும்."
                ),
            },
        }[language]
        return AdvisoryResponse(
            status=AdvisoryStatus.LOW_CONFIDENCE,
            explanation=unmapped_templates["explanation"],
            immediate_actions=unmapped_templates["immediate"],
            organic_cultural_steps=unmapped_templates["cultural"],
            chemical_guidance_if_approved=[],
            preventive_measures=unmapped_templates["preventive"],
            escalate_to_kvk=True,
            disclaimer=unmapped_templates["disclaimer"],
            language=language,
            ttft_ms=0.0,
            tokens_per_sec=0.0,
        )

    templates = {
        SupportedLanguage.EN: {
            "explanation": (
                f"Low confidence or unmapped condition (confidence {confidence:.2f}). "
                f"To keep your crop safe in {state_region}, the assistant cannot guess a treatment from an uncertain visual match."
            ),
            "immediate": [
                "Consult an agronomist at your nearest Krishi Vigyan Kendra (KVK) or State Agriculture Officer with fresh affected leaf/plant samples.",
                "Call the free Kisan Call Center helpline at 1800-180-1551 for verified local guidance.",
            ],
            "cultural": [
                "Isolate or tag affected plants and check neighboring rows, irrigation drainage, and soil moisture before taking action.",
            ],
            "preventive": [
                "Do not apply unverified pesticides or tank mixtures without expert confirmation.",
            ],
            "disclaimer": (
                "LOW CONFIDENCE SAFETY HOLD: No chemical guidance is provided. Please verify "
                "with your local Krishi Vigyan Kendra (KVK) or Kisan Call Center (1800-180-1551)."
            ),
        },
        SupportedLanguage.HI: {
            "explanation": (
                f"तस्वीर से रोग या कीट की पहचान पक्की नहीं हो सकी (विश्वास स्तर {confidence:.2f})। "
                f"{state_region} में आपकी फसल की सुरक्षा के लिए बिना पक्की पहचान के कोई अनुमानित उपचार नहीं दिया जा रहा है।"
            ),
            "immediate": [
                "प्रभावित पत्ती/पौधे का नमूना लेकर नजदीकी कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी से सलाह लें।",
                "निःशुल्क किसान कॉल सेंटर हेल्पलाइन 1800-180-1551 पर बात करें।",
            ],
            "cultural": [
                "प्रभावित पौधों को चिन्हित करें और आसपास की फसल, सिंचाई जल निकासी तथा नमी की जांच करें।",
            ],
            "preventive": [
                "कृषि विशेषज्ञ की पुष्टि के बिना किसी भी अनजान कीटनाशक या मिश्रण का छिड़काव न करें।",
            ],
            "disclaimer": (
                "कम विश्वास सुरक्षा चेतावनी: कोई रासायनिक छिड़काव सलाह नहीं दी गई है। कृपया "
                "नजदीकी कृषि विज्ञान केंद्र (KVK) या किसान कॉल सेंटर (1800-180-1551) से संपर्क करें।"
            ),
        },
        SupportedLanguage.GU: {
            "explanation": (
                f"ફોટા પરથી રોગ કે જીવાતની ચોક્કસ ઓળખ થઈ શकी નથી (વિશ્વાસ સ્તર {confidence:.2f}). "
                f"{state_region} માં પાકની સલામતી માટે અનિશ્ચિત ઓળખ પરથી કોઈ અંદાજિત દવા સૂચવવામાં આવતી નથી."
            ),
            "immediate": [
                "અસરગ્રસ્ત પાન/છોડના નમૂના સાથે નજીકના કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) અથવા ખેતીવાડી અધિકારીની સલાહ લો.",
                "મફત કિસાન કોલ સેન્ટર હેલ્પલાઇન 1800-180-1551 પર સંપર્ક કરો.",
            ],
            "cultural": [
                "અસરગ્રસ્ત છોડને અલગ તપાસો અને આસપાસની હરોળ તથા ખેતરમાં પાણીના નિકાલની સ્થિતિ જુઓ.",
            ],
            "preventive": [
                "નિષ્ણાતની ખાતરી વિના કોઈ પણ બિનચકાસાયેલ જંતુનાશક કે મિશ્રણનો છંટકાવ કરશો નહીં.",
            ],
            "disclaimer": (
                "ઓછા વિશ્વાસને કારણે સલામતી રોક: કોઈ રાસાયણિક સલાહ આપવામાં આવી નથી. કૃપા કરીને "
                "સ્થાનિક કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) અથવા કિસાન કોલ સેન્ટર (1800-180-1551) નો સંપર્ક કરો."
            ),
        },
        SupportedLanguage.MR: {
            "explanation": (
                f"फोटोवरून रोग किंवा किडीची खात्रीशीर ओळख होऊ शकली नाही (विश्वास पातळी {confidence:.2f}). "
                f"{state_region} मधील पिकाच्या सुरक्षिततेसाठी अंदाजाने कोणतेही औषध सुचवले जात नाही."
            ),
            "immediate": [
                "प्रभावित पानांचा नमुना घेऊन जवळच्या कृषी विज्ञान केंद्र (KVK) किंवा कृषी अधिकाऱ्यांचा सल्ला घ्या.",
                "मोफत किसान कॉल सेंटर हेल्पलाइन 1800-180-1551 वर संपर्क साधा.",
            ],
            "cultural": [
                "प्रभावित झाडे तपासा तसेच आजूबाजूच्या ओळी आणि पाण्याचा निचरा नीट तपासून पहा.",
            ],
            "preventive": [
                "तज्ज्ञांच्या खात्रीशिवाय कोणतीही रासायनिक फवारणी किंवा मिश्रण वापरू नका.",
            ],
            "disclaimer": (
                "कमी विश्वास पातळी सुरक्षा सूचना: कोणताही रासायनिक सल्ला दिलेला नाही. कृपया "
                "स्थानिक कृषी विज्ञान केंद्र (KVK) किंवा किसान कॉल सेंटर (1800-180-1551) यांच्याशी संपर्क साधा."
            ),
        },
        SupportedLanguage.TE: {
            "explanation": (
                f"ఫోటో నుండి వ్యాధి లేదా తెగులుపై నమ్మకమైన నిర్ధారణ రాలేదు (నమ్మకం {confidence:.2f}). "
                f"{state_region} ప్రాంతంలోని మీ పంట భద్రత కోసం అనిశ్చిత ఫలితంపై అంచనా మందులు సూచించబడవు."
            ),
            "immediate": [
                "తాజా దెబ్బతిన్న ఆకు/మొక్క నమూనాతో సమీప కృషి విజ్ఞాన కేంద్రం (KVK) లేదా వ్యవసాయ అధికారిని సంప్రదించండి.",
                "ఉచిత కిసాన్ కాల్ సెంటర్ హెల్ప్‌లైన్ 1800-180-1551 కు కాల్ చేయండి.",
            ],
            "cultural": [
                "ప్రభావిత మొక్కలను గుర్తించి పక్కన ఉన్న సాళ్లు, నీటి పారుదల మరియు నేల తేమను తనిఖీ చేయండి.",
            ],
            "preventive": [
                "నిపుణుల నిర్ధారణ లేకుండా తెలియని పురుగుమందులు లేదా మిశ్రమాలను పిచికారీ చేయవద్దు.",
            ],
            "disclaimer": (
                "తక్కువ నమ్మకం భద్రతా నిలుపుదల: ఎటువంటి రసాయన సలహా ఇవ్వబడలేదు. దయచేసి "
                "సమీప KVK లేదా కిసాన్ కాల్ సెంటర్ (1800-180-1551) తో నిర్ధారించుకోండి."
            ),
        },
        SupportedLanguage.TA: {
            "explanation": (
                f"படத்திலிருந்து நோய் அல்லது பூச்சியை உறுதியாக அடையாளம் காண முடியவில்லை (நம்பிக்கை {confidence:.2f}). "
                f"{state_region} பகுதியில் உங்கள் பயிர் பாதுகாப்பிற்காக உறுதியற்ற முடிவில் மருந்து பரிந்துரை வழங்கப்படவில்லை."
            ),
            "immediate": [
                "புதிய பாதிக்கப்பட்ட இலை/செடி மாதிரியுடன் அருகிலுள்ள வேளாண் அறிவியல் மையம் (KVK) அல்லது வேளாண் அதிகாரியை அணுகவும்.",
                "இலவச கிசான் கால் சென்டர் உதவி எண் 1800-180-1551-ஐ அழைக்கவும்.",
            ],
            "cultural": [
                "பாதிக்கப்பட்ட செடிகளைக் குறித்து வைத்து, அருகிலுள்ள வரிசைகள் மற்றும் வடிகால் நிலையை சரிபார்க்கவும்.",
            ],
            "preventive": [
                "நிபுணர் உறுதிப்படுத்தல் இன்றி எந்தவொரு பூச்சிக்கொல்லி கலவையையும் தெளிக்க வேண்டாம்.",
            ],
            "disclaimer": (
                "குறைந்த நம்பிக்கை பாதுகாப்பு நிறுத்தம்: இரசாயன ஆலோசனை வழங்கப்படவில்லை. தயவுசெய்து "
                "உங்கள் உள்ளூர் KVK அல்லது கிசான் கால் சென்டர் (1800-180-1551) மூலம் சரிபார்க்கவும்."
            ),
        },
    }[language]

    return AdvisoryResponse(
        status=AdvisoryStatus.LOW_CONFIDENCE,
        explanation=templates["explanation"],
        immediate_actions=templates["immediate"],
        organic_cultural_steps=templates["cultural"],
        chemical_guidance_if_approved=[],
        preventive_measures=templates["preventive"],
        escalate_to_kvk=True,
        disclaimer=templates["disclaimer"],
        language=language,
        ttft_ms=0.0,
        tokens_per_sec=0.0,
    )


def _build_healthy_response(
    language: SupportedLanguage,
    crop: str,
    confidence: float,
) -> AdvisoryResponse:
    templates = {
        SupportedLanguage.EN: {
            "explanation": f"The {crop} leaf appears healthy (confidence {confidence:.2f}) with no visible disease or pest damage.",
            "immediate": [
                "Continue routine weekly field scouting across multiple rows and leaf undersides."
            ],
            "cultural": [
                "Maintain balanced irrigation and Soil Health Card (SHC)-guided nutrient management."
            ],
            "preventive": [
                "Keep field bunds free of weeds and do not apply unnecessary preventive chemical sprays."
            ],
            "disclaimer": "Offline visual screening indicates healthy foliage. Contact local KVK if hidden root or stem symptoms appear.",
        },
        SupportedLanguage.HI: {
            "explanation": f"{crop} की पत्ती स्वस्थ दिखाई दे रही है (विश्वास स्तर {confidence:.2f}) और कोई रोग या कीट लक्षण नहीं दिख रहे हैं।",
            "immediate": [
                "हर सप्ताह खेत की नियमित निगरानी जारी रखें और पत्तियों की निचली सतह भी जांचते रहें।"
            ],
            "cultural": [
                "संतुलित सिंचाई और मृदा स्वास्थ्य कार्ड (SHC) के अनुसार पोषण प्रबंधन बनाए रखें।"
            ],
            "preventive": [
                "मेड़ों से खरपतवार साफ रखें और बिना जरूरत के रासायनिक छिड़काव न करें।"
            ],
            "disclaimer": "यह ऑफ़लाइन जांच केवल पत्तियों की दृश्य स्थिति बताती है। जड़ या तने में समस्या दिखने पर KVK से संपर्क करें।",
        },
        SupportedLanguage.GU: {
            "explanation": f"{crop} નું પાન તંદુરસ્ત જણાય છે (વિશ્વાસ સ્તર {confidence:.2f}) અને કોઈ રોગ કે જીવાતનું નુકસાન દેખાતું નથી.",
            "immediate": [
                "દર અઠવાડિયે ખેતરમાં પાનની નીચેની અને ઉપરની બાજુની નિયમિત તપાસ ચાલુ રાખો."
            ],
            "cultural": [
                "સંતુલિત પિયત અને સોઇલ હેલ્થ કાર્ડ (SHC) મુજબ પોષણ વ્યવસ્થાપન જાળવો."
            ],
            "preventive": [
                "શેઢા-પાળા પરથી નીંદણ દૂર રાખો અને બિનજરૂરી રાસાયણિક છંટકાવ ટાળો."
            ],
            "disclaimer": "ઓફલાઇન તપાસ તંદુરસ્ત પાન દર્શાવે છે. મૂળ કે થડમાં લક્ષણો જણાય તો KVK નો સંપર્ક કરો.",
        },
        SupportedLanguage.MR: {
            "explanation": f"{crop} चे पान निरोगी दिसत आहे (विश्वास पातळी {confidence:.2f}) आणि कोणताही रोग किंवा कीड आढळलेली नाही.",
            "immediate": [
                "दर आठवड्याला शेतातील पिकांची आणि पानांच्या खालच्या बाजूची नियमित पाहणी सुरू ठेवा."
            ],
            "cultural": [
                "संतुलित पाणी व्यवस्थापन आणि मृदा आरोग्य पत्रिका (SHC) नुसार खत व्यवस्थापन करा."
            ],
            "preventive": [
                "बांधावरील तण स्वच्छ ठेवा आणि गरज नसताना रासायनिक फवारणी टाळा."
            ],
            "disclaimer": "ऑफलाइन तपासणीत पान निरोगी आढळले आहे. खोड किंवा मुळांवर लक्षणे दिसल्यास KVK शी संपर्क साधा.",
        },
        SupportedLanguage.TE: {
            "explanation": f"{crop} ఆకు ఆరోగ్యంగా కనిపిస్తోంది (నమ్మకం {confidence:.2f}), వ్యాధి లేదా తెగులు నష్టం కనిపించలేదు.",
            "immediate": [
                "ప్రతి వారం పొలంలోని వరుసలు మరియు ఆకుల కింది భాగాలను నిరంతరం గమనిస్తూ ఉండండి."
            ],
            "cultural": [
                "సమతుల్య నీటిపారుదల మరియు సాయిల్ హెల్త్ కార్డ్ (SHC) ప్రకారం పోషక యాజమాన్యం పాటించండి."
            ],
            "preventive": [
                "పొలం గట్లపై కలుపు మొక్కలు లేకుండా ఉంచి అనవసర రసాయన పిచికారీలు నివారించండి."
            ],
            "disclaimer": "ఆఫ్‌లైన్ దృశ్య పరీక్ష ఆరోగ్యకరమైన ఆకులను చూపుతోంది. వేరు లేదా కాండంపై లక్షణాలు కనిపిస్తే సమీప KVK ను సంప్రదించండి.",
        },
        SupportedLanguage.TA: {
            "explanation": f"{crop} இலை ஆரோக்கியமாக உள்ளது (நம்பிக்கை {confidence:.2f}), நோய் அல்லது பூச்சி சேதம் ஏதும் தென்படவில்லை.",
            "immediate": [
                "வாரந்தோறும் வயல் வரிசைகள் மற்றும் இலைகளின் அடிப்பகுதியைத் தொடர்ந்து கண்காணிக்கவும்."
            ],
            "cultural": [
                "சீரான நீர்ப்பாசனம் மற்றும் மண் வள அட்டை (SHC) வழிகாட்டுதலின்படி உர மேலாண்மை செய்யவும்."
            ],
            "preventive": [
                "வரப்புகளில் களைகள் இன்றி வைத்து தேவையற்ற இரசாயனத் தெளிப்பைத் தவிர்க்கவும்."
            ],
            "disclaimer": "ஆஃப்லைன் காட்சி ஆய்வு ஆரோக்கியமான இலைகளைக் காட்டுகிறது. வேர் அல்லது தண்டில் அறிகுறிகள் தோன்றினால் உள்ளூர் KVK-ஐ அணுகவும்.",
        },
    }[language]

    return AdvisoryResponse(
        status=AdvisoryStatus.SUCCESS,
        explanation=templates["explanation"],
        immediate_actions=templates["immediate"],
        organic_cultural_steps=templates["cultural"],
        chemical_guidance_if_approved=[],
        preventive_measures=templates["preventive"],
        escalate_to_kvk=False,
        disclaimer=templates["disclaimer"],
        language=language,
        ttft_ms=0.0,
        tokens_per_sec=0.0,
    )


def _build_grounded_fallback_response(
    request: AdvisoryRequest,
    kb_entry: DiseasePestCard,
    status: AdvisoryStatus,
    escalate: bool,
    ttft_ms: float,
    tokens_per_sec: float,
    fallback_note: str,
) -> AdvisoryResponse:
    """Builds a 100%-grounded response directly from `kb_entry` if LLM output fails checks."""
    vf = request.vision_finding
    chem_guidance: List[str] = []
    if vf.severity_level != SeverityLevel.LOW:
        chem_guidance = [
            f"{opt.active_ingredient} ({opt.target_stage}): {opt.safety_warning}"
            for opt in kb_entry.approved_chemical_options
        ]

    regional_hint = kb_entry.region_notes.get(
        request.state_region,
        kb_entry.region_notes.get("general", ""),
    )

    return AdvisoryResponse(
        status=status,
        explanation=(
            f"Visual screening indicates {kb_entry.condition} on {kb_entry.crop} "
            f"(confidence {vf.confidence:.2f}, severity {vf.severity_level.value}, "
            f"infected area {vf.infected_area_pct:.1f}%). {regional_hint}".strip()
        ),
        immediate_actions=[
            kb_entry.cultural_organic_control[0],
            kb_entry.kvk_escalation_trigger,
        ],
        organic_cultural_steps=list(kb_entry.cultural_organic_control),
        chemical_guidance_if_approved=chem_guidance,
        preventive_measures=[
            f"Watch for key symptoms: {kb_entry.symptoms[0]}",
            f"Reference: {kb_entry.source_reference}",
        ],
        escalate_to_kvk=escalate,
        disclaimer=(
            f"Grounded ICAR/KVK KB fallback applied ({fallback_note}). "
            "Offline AI screening is not a laboratory diagnosis; confirm CIB&RC label "
            "dosage and Pre-Harvest Interval (PHI) with local KVK before spraying."
        ),
        language=request.language,
        ttft_ms=ttft_ms,
        tokens_per_sec=tokens_per_sec,
    )


class AdvisoryEngine:
    """
    Pipeline B Orchestrator combining deterministic pre-LLM safety gates,
    grounded Llama 3.2 prompt generation, chemical grounding validation,
    and severe-case KVK escalation.
    """

    def __init__(
        self,
        backend: LlamaAdvisoryBackend,
        kb_retriever: Optional[OfflineKBRetriever] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.backend = backend
        self.kb_retriever = kb_retriever if kb_retriever is not None else OfflineKBRetriever()
        self.confidence_threshold = float(confidence_threshold)

    def advise(self, request: AdvisoryRequest) -> AdvisoryResponse:
        """Processes an `AdvisoryRequest` and returns a validated `AdvisoryResponse`."""
        vf = request.vision_finding

        # Gate 1: Poor image quality short-circuit (no LLM call, no chemicals)
        if not vf.image_quality_ok:
            return _build_uncertain_image_response(request.language)

        # Gate 1b: Non-leaf / no plant foliage detected -> reject safely as UNCERTAIN_IMAGE
        if vf.condition_name == "non_leaf_or_no_plant":
            non_leaf_msg = {
                SupportedLanguage.EN: "No plant foliage was detected in this photo. Upload a clear close-up image of a crop leaf against a plain background.",
                SupportedLanguage.HI: "इस फोटो में फसल की पत्ती दिखाई नहीं दे रही है। कृपया साफ़ पृष्ठभूमि के साथ फसल की पत्ती का स्पष्ट फोटो अपलोड करें।",
                SupportedLanguage.GU: "આ ફોટામાં પાકના પાન જણાતા નથી. કૃપા કરીને સાદા બેકગ્રાઉન્ડ સાથે પાકના પાનનો સ્પષ્ટ ફોટો અપલોડ કરો.",
                SupportedLanguage.MR: "या फोटोमध्ये पिकाचे पान आढळले नाही. कृपया साध्या पार्श्वभूमीवर पिकाच्या पानाचा स्पष्ट फोटो अपलोड करा.",
                SupportedLanguage.TE: "ఈ ఫోటోలో పంట ఆకు కనిపించలేదు. దయచేసి సాదా నేపథ్యంతో పంట ఆకు యొక్క స్పష్టమైన ఫోటోను అప్‌లోడ్ చేయండి.",
                SupportedLanguage.TA: "இந்தப் படத்தில் பயிர் இலை கண்டறியப்படவில்லை. தயவுசெய்து தெளிவான பின்னணியுடன் பயிர் இலையின் நெருக்கமான படத்தைப் பதிவேற்றவும்.",
            }[request.language]
            return AdvisoryResponse(
                status=AdvisoryStatus.UNCERTAIN_IMAGE,
                explanation=non_leaf_msg,
                immediate_actions=["Frame a single affected leaf filling most of the camera view before tapping Analyze Crop."],
                organic_cultural_steps=["Do not apply any treatment based on a non-leaf photo."],
                chemical_guidance_if_approved=[],
                preventive_measures=["Capture photos in daylight 15–25 cm from the leaf surface."],
                escalate_to_kvk=False,
                disclaimer="Non-leaf or missing foliage guard triggered. No disease prediction or chemical advice issued.",
                language=request.language,
                ttft_ms=0.0,
                tokens_per_sec=0.0,
            )

        # Gate 2: Healthy crop short-circuit (no LLM call, preventive guidance only)
        if vf.category == VisionCategory.HEALTHY and vf.confidence >= self.confidence_threshold:
            return _build_healthy_response(request.language, vf.crop, vf.confidence)

        # Resolve KB card offline if caller omitted it
        kb_entry = request.kb_entry
        if kb_entry is None and vf.category in (VisionCategory.DISEASE, VisionCategory.PEST):
            kb_entry = self.kb_retriever.retrieve_for_finding(vf)

        # Gate 3: Low-confidence, unknown class, or unmapped KB entry -> KVK hold
        if (
            vf.category == VisionCategory.UNKNOWN
            or vf.confidence < self.confidence_threshold
            or kb_entry is None
        ):
            return _build_low_confidence_response(
                language=request.language,
                state_region=request.state_region,
                confidence=vf.confidence,
                detected_crop=vf.crop,
                detected_condition=vf.condition_name,
                severity_level=vf.severity_level.value,
                infected_area_pct=vf.infected_area_pct,
                confident_unmapped=(
                    vf.category in (VisionCategory.DISEASE, VisionCategory.PEST)
                    and vf.confidence >= self.confidence_threshold
                    and kb_entry is None
                ),
            )

        # Severity Escalation Rule (deterministic regardless of LLM prose)
        is_severe = vf.severity_level == SeverityLevel.SEVERE
        target_status = (
            AdvisoryStatus.SEVERE_ESCALATION if is_severe else AdvisoryStatus.SUCCESS
        )
        escalate_to_kvk = bool(is_severe)

        effective_request = AdvisoryRequest(
            vision_finding=vf,
            state_region=request.state_region,
            language=request.language,
            kb_entry=kb_entry,
        )

        prompt = build_llama3_prompt(effective_request)
        raw_text, ttft_ms, tokens_per_sec = self.backend.generate(prompt)
        ttft_ms = max(0.0, float(ttft_ms))
        tokens_per_sec = max(0.0, float(tokens_per_sec))

        parsed = _extract_json_object(raw_text)
        if parsed is None:
            LOGGER.warning("Llama output failed JSON parsing; using deterministic KB fallback.")
            return _build_grounded_fallback_response(
                request=effective_request,
                kb_entry=kb_entry,
                status=target_status,
                escalate=escalate_to_kvk,
                ttft_ms=ttft_ms,
                tokens_per_sec=tokens_per_sec,
                fallback_note="invalid JSON output from model",
            )

        is_grounded, ground_reason = _verify_chemical_grounding(parsed, kb_entry)
        if not is_grounded:
            LOGGER.warning("Post-LLM chemical grounding rejected output: %s", ground_reason)
            return _build_grounded_fallback_response(
                request=effective_request,
                kb_entry=kb_entry,
                status=target_status,
                escalate=escalate_to_kvk,
                ttft_ms=ttft_ms,
                tokens_per_sec=tokens_per_sec,
                fallback_note=ground_reason,
            )

        explanation = parsed.get("explanation")
        disclaimer = parsed.get("disclaimer")
        if not isinstance(explanation, str) or not explanation.strip():
            LOGGER.warning("Llama output missing explanation; using deterministic KB fallback.")
            return _build_grounded_fallback_response(
                request=effective_request,
                kb_entry=kb_entry,
                status=target_status,
                escalate=escalate_to_kvk,
                ttft_ms=ttft_ms,
                tokens_per_sec=tokens_per_sec,
                fallback_note="missing explanation in model JSON",
            )

        if not isinstance(disclaimer, str) or not disclaimer.strip():
            disclaimer = (
                "Offline AI screening only (not a lab diagnosis). Verify CIB&RC label "
                "dosage and Pre-Harvest Interval (PHI) with local KVK before spraying."
            )

        immediate_actions = _clean_str_list(
            parsed.get("immediate_actions"),
            [kb_entry.cultural_organic_control[0]],
        )
        organic_steps = _clean_str_list(
            parsed.get("organic_cultural_steps"),
            list(kb_entry.cultural_organic_control),
        )
        raw_chem = parsed.get("chemical_guidance_if_approved", [])
        chem_steps = (
            [str(x).strip() for x in raw_chem if isinstance(x, str) and x.strip()]
            if isinstance(raw_chem, list)
            else []
        )
        preventive_steps = _clean_str_list(
            parsed.get("preventive_measures"),
            [kb_entry.kvk_escalation_trigger],
        )

        try:
            return AdvisoryResponse(
                status=target_status,
                explanation=explanation.strip(),
                immediate_actions=immediate_actions,
                organic_cultural_steps=organic_steps,
                chemical_guidance_if_approved=chem_steps,
                preventive_measures=preventive_steps,
                escalate_to_kvk=escalate_to_kvk,
                disclaimer=disclaimer.strip(),
                language=request.language,
                ttft_ms=ttft_ms,
                tokens_per_sec=tokens_per_sec,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("AdvisoryResponse validation fallback triggered: %s", exc)
            return _build_grounded_fallback_response(
                request=effective_request,
                kb_entry=kb_entry,
                status=target_status,
                escalate=escalate_to_kvk,
                ttft_ms=ttft_ms,
                tokens_per_sec=tokens_per_sec,
                fallback_note=f"contract validation fallback ({exc})",
            )
