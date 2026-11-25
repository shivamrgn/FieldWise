"""
Deterministic Offline Knowledge Base Compiler for FieldWise PS1.

Reads existing FieldWise V1 JavaScript databases (`cropDatabase.js` and
`fertilizerDatabase.js`) in strictly read-only mode, merges them with
ICAR/KVK/CIB&RC-aligned disease and pest treatment cards, validates all
cards against `DiseasePestCard` contracts, and compiles a deterministic
`ps1_offline_assistant/data/agri_kb.json`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow running both as `python3 -m ps1_offline_assistant.scripts.build_offline_kb`
# and directly as `python3 ps1_offline_assistant/scripts/build_offline_kb.py`.
_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.schemas.contracts import DiseasePestCard  # noqa: E402


KB_SCHEMA_VERSION = "1.0.0"
DEFAULT_CROP_DB_JS = _MODEL_ROOT / "cropDatabase.js"
DEFAULT_FERT_DB_JS = _MODEL_ROOT / "fertilizerDatabase.js"
DEFAULT_OUTPUT_JSON = _PS1_ROOT / "data" / "agri_kb.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_js_exports_via_node(crop_js: Path, fert_js: Path) -> Dict[str, Any]:
    """Executes a read-only Node.js snippet to load CommonJS exports cleanly."""
    node_bin = shutil.which("node")
    if not node_bin:
        raise FileNotFoundError("node binary not found on PATH")

    js_loader = (
        "const { crops } = require(process.argv[1]);\n"
        "const { fertilizers } = require(process.argv[2]);\n"
        "process.stdout.write(JSON.stringify({ crops, fertilizers }));\n"
    )
    proc = subprocess.run(
        [node_bin, "-e", js_loader, str(crop_js.resolve()), str(fert_js.resolve())],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _extract_js_literal_python(js_text: str, var_name: str) -> Any:
    """
    Pure-Python fallback parser for simple CommonJS object/array exports
    (`const <var_name> = ...;`) without executing external binaries.
    """
    pattern = rf"const\s+{re.escape(var_name)}\s*=\s*"
    match = re.search(pattern, js_text)
    if not match:
        raise ValueError(f"Could not find 'const {var_name} =' in JS source.")
    rest = js_text[match.end() :]
    end_idx = rest.find("module.exports")
    if end_idx != -1:
        rest = rest[:end_idx]
    rest = rest.strip().rstrip(";").strip()

    # Convert unquoted JS property identifiers outside quotes into Python dict keys.
    out_chars: List[str] = []
    in_single = False
    in_double = False
    escape = False
    i = 0
    n = len(rest)
    while i < n:
        ch = rest[i]
        if escape:
            out_chars.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\":
            out_chars.append(ch)
            escape = True
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            out_chars.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out_chars.append(ch)
            i += 1
            continue
        if not in_single and not in_double:
            # Strip single-line // comments if present
            if ch == "/" and i + 1 < n and rest[i + 1] == "/":
                nl = rest.find("\n", i)
                i = n if nl == -1 else nl
                continue
            # Detect unquoted object keys: identifier followed by ':'
            if (ch.isalpha() or ch == "_") and (
                not out_chars or out_chars[-1] in "{,\n\r\t "
            ):
                j = i
                while j < n and (rest[j].isalnum() or rest[j] == "_"):
                    j += 1
                k = j
                while k < n and rest[k] in " \t":
                    k += 1
                if k < n and rest[k] == ":":
                    key_ident = rest[i:j]
                    out_chars.append(f"'{key_ident}'")
                    i = j
                    continue
        out_chars.append(ch)
        i += 1

    py_expr = "".join(out_chars)
    return ast.literal_eval(py_expr)


def load_fieldwise_js_databases(
    crop_js_path: Path = DEFAULT_CROP_DB_JS,
    fert_js_path: Path = DEFAULT_FERT_DB_JS,
) -> Dict[str, Any]:
    """Reads `cropDatabase.js` and `fertilizerDatabase.js` without mutating them."""
    if not crop_js_path.is_file():
        raise FileNotFoundError(f"Missing cropDatabase.js at {crop_js_path}")
    if not fert_js_path.is_file():
        raise FileNotFoundError(f"Missing fertilizerDatabase.js at {fert_js_path}")

    try:
        return _extract_js_exports_via_node(crop_js_path, fert_js_path)
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        crops = _extract_js_literal_python(
            crop_js_path.read_text(encoding="utf-8"), "crops"
        )
        fertilizers = _extract_js_literal_python(
            fert_js_path.read_text(encoding="utf-8"), "fertilizers"
        )
        return {"crops": crops, "fertilizers": fertilizers}


def _build_disease_pest_cards() -> List[Dict[str, Any]]:
    """
    Returns grounded ICAR/DPPQS/CIB&RC-aligned disease and pest treatment cards
    for key Indian crops (Rice, Wheat, Cotton, Potato, Tomato, Maize) covering
    major PlantVillage and IP102 classes.
    """
    raw_cards: List[Dict[str, Any]] = [
        {
            "id": "rice_blast",
            "crop": "rice",
            "condition": "Rice Blast",
            "type": "disease",
            "aliases": [
                "Rice___Blast",
                "Rice___Leaf_Blast",
                "Leaf blast",
                "Neck blast",
                "Magnaporthe oryzae",
            ],
            "symptoms": [
                "Spindle-shaped or diamond-shaped lesions on leaves with whitish-grey centers and dark brown/reddish margins.",
                "In advanced infection, dark necrotic lesions at the panicle neck (neck blast) causing chaffy grains or broken panicles.",
            ],
            "severity_thresholds": {
                "low": "< 5% leaf area affected with isolated leaf lesions and no neck/node involvement.",
                "moderate": "5%–25% leaf area affected or > 5% tillers showing active expanding lesions.",
                "severe": "> 25% leaf area affected, coalescing blight, or any panicle neck/node blast infection.",
            },
            "cultural_organic_control": [
                "Avoid excessive split top-dressing of nitrogenous fertilizer (urea), which promotes succulent blast-susceptible foliage.",
                "Maintain field sanitation by removing collateral weed hosts on bunds and avoiding severe water stress.",
                "Seed/root-dip bio-management with Pseudomonas fluorescens as recommended in local SAU/ICAR Package of Practices.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Tricyclazole 75% WP or Isoprothiolane 40% EC",
                    "target_stage": "Active foliar blast at moderate incidence or preventive boot/heading spray in endemic belts",
                    "cibrc_aligned_note": "CIB&RC registered fungicide for paddy blast management.",
                    "safety_warning": "Apply only if moderate/severe ETL is crossed. Verify exact CIB&RC label dilution and Pre-Harvest Interval (PHI) on the container or with local KVK/DAO before spraying; wear PPE and avoid drift to water bodies.",
                },
                {
                    "active_ingredient": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC",
                    "target_stage": "Panicle emergence / neck blast risk stage",
                    "cibrc_aligned_note": "Combination systemic fungicide registered for paddy blast/sheath diseases.",
                    "safety_warning": "Rotate FRAC fungicide groups to delay resistance. Confirm exact label dosage and PHI with local KVK before application.",
                },
            ],
            "region_notes": {
                "general": "Favored by high relative humidity (>85%), prolonged leaf wetness/dew, cloudy weather, and high nitrogen doses.",
                "Eastern India & Coastal": "Monitor closely during Kharif tillering and Boro booting in Assam, West Bengal, Odisha, and coastal Andhra.",
                "North-Western Plains": "High blast susceptibility reported in Basmati varieties across Punjab, Haryana, and Western UP.",
            },
            "kvk_escalation_trigger": "Escalate immediately to local KVK if neck/node blast appears at panicle emergence or >25% leaf area is blighted.",
            "source_reference": "ICAR-IIRR / DPPQS IPM Package for Rice",
        },
        {
            "id": "rice_brown_spot",
            "crop": "rice",
            "condition": "Brown Spot",
            "type": "disease",
            "aliases": [
                "Rice___Brown_Spot",
                "Rice Brown Spot",
                "Brown spot",
                "Bipolaris oryzae",
                "Helminthosporium leaf spot",
            ],
            "symptoms": [
                "Circular to oval dark brown spots on leaves with a yellow halo around the margin.",
                "Severe infection causes black discoloration on glumes/grains and poor grain filling.",
            ],
            "severity_thresholds": {
                "low": "< 5% leaf area covered by small discrete brown spots.",
                "moderate": "5%–20% leaf area affected with spots coalescing on lower/middle leaves.",
                "severe": "> 20% leaf area affected or widespread spikelet/grain discoloration at flowering.",
            },
            "cultural_organic_control": [
                "Correct underlying soil nutrient deficiency (especially potassium, silicon, and zinc) based on a Soil Health Card test.",
                "Use disease-free certified seed and perform hot-water or Pseudomonas bio-priming prior to nursery sowing.",
                "Avoid sudden drought stress by maintaining shallow uniform moisture during active tillering.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Mancozeb 75% WP or Propiconazole 25% EC",
                    "target_stage": "Moderate foliar spread or boot-leaf protection against grain spotting",
                    "cibrc_aligned_note": "CIB&RC-approved fungicide options for rice brown spot.",
                    "safety_warning": "Brown spot is strongly linked to nutrient/moisture stress; address soil fertility first. Confirm exact label dosage and PHI with local KVK/DAO and wear protective gear.",
                }
            ],
            "region_notes": {
                "general": "Historically severe in nutrient-poor, rainfed upland, or moisture-stressed paddy soils.",
                "Eastern & Central India": "Common in rainfed shallow lowlands of Bihar, Jharkhand, Chhattisgarh, Odisha, and West Bengal.",
            },
            "kvk_escalation_trigger": "Escalate to KVK for soil nutrient diagnostics if brown spot persists across consecutive seasons or affects >20% foliage.",
            "source_reference": "ICAR-IIRR / DPPQS IPM Package for Rice",
        },
        {
            "id": "rice_bacterial_leaf_blight",
            "crop": "rice",
            "condition": "Bacterial Leaf Blight",
            "type": "disease",
            "aliases": [
                "Rice___Bacterial_Leaf_Blight",
                "Bacterial leaf blight",
                "BLB",
                "Xanthomonas oryzae",
            ],
            "symptoms": [
                "Water-soaked yellow-to-straw-colored stripes starting from leaf tips and margins with wavy inner edges.",
                "Early morning bacterial ooze droplets on young lesions that dry into yellowish beads.",
            ],
            "severity_thresholds": {
                "low": "< 5% leaf area showing marginal blighting on older leaves.",
                "moderate": "5%–20% leaf area blighted across localized field patches.",
                "severe": "> 20% leaf area blighted or Kresek (systemic seedling/tiller wilting) observed.",
            },
            "cultural_organic_control": [
                "Drain standing water temporarily and withhold nitrogen top-dressing immediately upon first BLB detection.",
                "Avoid moving irrigation water from infected plots into healthy downstream paddy fields.",
                "Clip leaf tips only if healthy; avoid mechanical injury during weeding in wet conditions.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Copper Hydroxide 53.8% DF or Copper Oxychloride 50% WP",
                    "target_stage": "Early localized focus spread (fungicides like Tricyclazole/Carbendazim are ineffective against bacteria)",
                    "cibrc_aligned_note": "Bactericidal copper formulation; verify current state KVK advisory on bactericides.",
                    "safety_warning": "Standard fungal sprays do not control bacterial blight. Confirm locally approved bactericide protocol and label dosage with KVK/DAO; avoid routine antibiotic misuse.",
                }
            ],
            "region_notes": {
                "general": "Triggered by monsoon storms, leaf wounding, high nitrogen, and stagnant standing water.",
                "Indo-Gangetic & Coastal Deltas": "Frequent post-cyclone/heavy-rain outbreaks in Punjab, Haryana, UP, Andhra Pradesh, and Tamil Nadu.",
            },
            "kvk_escalation_trigger": "Escalate urgently to KVK if Kresek wilting appears or wavy blight spreads rapidly after storm damage.",
            "source_reference": "ICAR-IIRR / DPPQS IPM Package for Rice",
        },
        {
            "id": "rice_brown_planthopper",
            "crop": "rice",
            "condition": "Brown Planthopper",
            "type": "pest",
            "aliases": [
                "brown plant hopper",
                "Brown Planthopper (BPH)",
                "BPH",
                "Nilaparvata lugens",
            ],
            "symptoms": [
                "Nymphs and brownish adults clustering at the basal stem near water level sucking plant sap.",
                "Circular patches of yellowing, drying, and lodging plants known as 'hopper burn'.",
            ],
            "severity_thresholds": {
                "low": "< 5 hoppers per hill (monitor natural predators like spiders and mirid bugs).",
                "moderate": "5–10 hoppers per hill at vegetative stage or 10–20 at panicle stage.",
                "severe": "> 10–20 hoppers per hill or visible circular hopper-burn patches.",
            },
            "cultural_organic_control": [
                "Drain standing field water for 3–4 days to expose basal stems and disrupt BPH microclimate.",
                "Form alleyways (skipping 1 row every 8–10 rows) during planting/mid-season to improve aeration and light penetration.",
                "Conserve natural enemies (wolf spiders, mirid bugs); avoid synthetic pyrethroids which trigger BPH resurgence.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Pymetrozine 50% WG or Dinotefuran 20% SG or Buprofezin 25% SC",
                    "target_stage": "Direct spray towards stem base only when ETL (10+ hoppers/hill) is exceeded",
                    "cibrc_aligned_note": "CIB&RC-approved selective hoppers-targeted insecticides.",
                    "safety_warning": "Avoid broad-spectrum synthetic pyrethroids (cause pest resurgence). Direct nozzle to plant base; check label dosage, PHI, and bee/fish safety warnings with local KVK.",
                }
            ],
            "region_notes": {
                "general": "Thrives in dense, heavily fertilized, continuously flooded paddy canopies.",
                "Godavari/Krishna & IGP Belts": "Major late-Kharif and Rabi threat in Andhra Pradesh, Telangana, Odisha, Punjab, and Haryana.",
            },
            "kvk_escalation_trigger": "Escalate immediately to KVK if hopper-burn patches appear or hopper counts exceed 15–20 per hill despite drainage.",
            "source_reference": "ICAR-IIRR / DPPQS IPM Package for Rice",
        },
        {
            "id": "wheat_leaf_rust",
            "crop": "wheat",
            "condition": "Wheat Leaf Rust (Brown Rust)",
            "type": "disease",
            "aliases": [
                "Wheat___Leaf_Rust",
                "Wheat___Brown_Rust",
                "Wheat Rust",
                "Leaf rust",
                "Brown rust",
                "Puccinia triticina",
            ],
            "symptoms": [
                "Small, round to oval orange-brown pustules (uredia) scattered irregularly on the upper leaf surface.",
                "Rusty orange spore powder rubs off easily on fingers when touched.",
            ],
            "severity_thresholds": {
                "low": "Trace to < 5% leaf area covered on lower leaves.",
                "moderate": "5%–20% leaf area covered moving onto flag/upper leaves.",
                "severe": "> 20% leaf area covered on upper/flag leaves during booting or grain filling.",
            },
            "cultural_organic_control": [
                "Cultivate locally recommended rust-resistant wheat varieties (e.g., ICAR-IIWBR released varieties for your zone).",
                "Avoid late sowing and excessive nitrogen application which prolong succulent foliage.",
                "Scout fields weekly starting January–February for early foci.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Propiconazole 25% EC or Tebuconazole 25.9% EC",
                    "target_stage": "Flag-leaf protection when pustules appear on upper canopy above trace levels",
                    "cibrc_aligned_note": "CIB&RC-registered triazole fungicides recommended by ICAR-IIWBR for wheat rusts.",
                    "safety_warning": "Do not spray if crop is near physiological maturity. Verify CIB&RC label dosage, PHI, and protective equipment requirements with local KVK.",
                }
            ],
            "region_notes": {
                "general": "Favored by mild temperatures (15–25°C) and morning dew/humidity across wheat belts.",
                "Central & Peninsular India": "Early appearance possible in Madhya Pradesh, Maharashtra, and Gujarat.",
            },
            "kvk_escalation_trigger": "Escalate to KVK/ICAR-IIWBR alert network if rust appears on flag leaves or on previously resistant cultivars.",
            "source_reference": "ICAR-IIWBR Wheat Crop Protection Advisory",
        },
        {
            "id": "wheat_yellow_rust",
            "crop": "wheat",
            "condition": "Wheat Yellow Rust (Stripe Rust)",
            "type": "disease",
            "aliases": [
                "Wheat___Yellow_Rust",
                "Yellow rust",
                "Stripe rust",
                "Wheat Stripe Rust",
                "Puccinia striiformis",
            ],
            "symptoms": [
                "Bright yellow-to-lemon pustules arranged in narrow linear stripes parallel to leaf veins.",
                "Yellow dust/spores stain clothing or fingers when brushing past infected leaves.",
            ],
            "severity_thresholds": {
                "low": "Isolated early foci (< 5% plants) in shaded/sub-mountainous pockets.",
                "moderate": "5%–15% leaf severity spreading outward from initial foci.",
                "severe": "> 15% leaf severity or rapid multi-leaf epidemic spread before grain fill.",
            },
            "cultural_organic_control": [
                "Inspect field borders, poplar-shaded rows, and early-sown patches weekly during cool winter weather (10–18°C).",
                "Remove/avoid susceptible varieties in sub-mountainous zones and adopt recommended varietal diversification.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Propiconazole 25% EC or Tebuconazole 25% WG or Azoxystrobin + Tebuconazole",
                    "target_stage": "Immediate spot-spraying around initial yellow rust foci to halt downwind spore spread",
                    "cibrc_aligned_note": "ICAR-IIWBR recommended rust fungicides for NWPZ stripe rust containment.",
                    "safety_warning": "Spot-spray foci promptly rather than waiting for field-wide spread. Confirm exact label dilution and PHI with local KVK/Agriculture Department.",
                }
            ],
            "region_notes": {
                "general": "Cool-weather pathogen (10–20°C with fog/dew); spreads rapidly by wind.",
                "North-Western Plains & Hills": "High priority surveillance in Punjab, Haryana, Himachal foothills, Uttarakhand, J&K, and Western UP.",
            },
            "kvk_escalation_trigger": "Escalate immediately to KVK/State Agriculture Department on first confirmed yellow rust focus to trigger regional containment.",
            "source_reference": "ICAR-IIWBR Stripe Rust Surveillance & Management Guidelines",
        },
        {
            "id": "wheat_aphids",
            "crop": "wheat",
            "condition": "Wheat Aphids",
            "type": "pest",
            "aliases": [
                "aphids",
                "wheat aphids",
                "cereal aphid",
                "Sitobion avenae",
                "Rhopalosiphum maidis",
            ],
            "symptoms": [
                "Colonies of soft-bodied green/yellowish aphids clustering on earheads, flag leaves, and tender stems.",
                "Sticky honeydew on leaves attracting black sooty mold and yellowing/shriveled grains.",
            ],
            "severity_thresholds": {
                "low": "< 5 aphids per earhead/tiller with ladybird beetles present.",
                "moderate": "5–10 aphids per earhead/tiller on border rows.",
                "severe": "> 10 aphids per earhead/tiller across the main field during milky grain stage.",
            },
            "cultural_organic_control": [
                "Preserve natural predators (coccinellid ladybird beetles, syrphid flies, lacewings) — avoid unnecessary sprays below ETL.",
                "Install yellow sticky traps along field borders for early monitoring.",
                "Use neem-based formulation (Azadirachtin) on initial border colonies if predators are scarce.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Imidacloprid 17.8% SL or Thiamethoxam 25% WG",
                    "target_stage": "Spray only when aphid count crosses ETL (>10 aphids/earhead) with low predator activity",
                    "cibrc_aligned_note": "CIB&RC-registered systemic aphicides for wheat.",
                    "safety_warning": "Do not spray if ladybird beetles are actively controlling small colonies. Follow label PHI and avoid spraying during pollinator foraging hours; verify dose with KVK.",
                }
            ],
            "region_notes": {
                "general": "Typically builds up during February–March warming as wheat heads emerge.",
                "Punjab, Haryana, UP & Rajasthan": "Often starts from field margins; border-only spraying frequently suffices.",
            },
            "kvk_escalation_trigger": "Escalate to KVK if aphid densities exceed 15–20 per earhead at grain filling or fail to respond to border management.",
            "source_reference": "ICAR-IIWBR / DPPQS IPM Package for Wheat",
        },
        {
            "id": "cotton_bollworm",
            "crop": "cotton",
            "condition": "Cotton Bollworm",
            "type": "pest",
            "aliases": [
                "bollworm",
                "cotton bollworm",
                "American bollworm",
                "Pink bollworm",
                "Helicoverpa armigera",
                "Pectinophora gossypiella",
            ],
            "symptoms": [
                "Circular bore holes in squares, flowers, and green bolls with larval frass (excreta) or flared/rosetted flowers.",
                "Premature square/boll shedding and stained lint inside damaged bolls.",
            ],
            "severity_thresholds": {
                "low": "< 5% damaged squares/bolls or below pheromone trap threshold.",
                "moderate": "5%–10% damaged fruiting bodies or >= 8–10 moths/trap/night for 3 consecutive nights.",
                "severe": "> 10% damaged bolls/squares or widespread pink bollworm rosette flowers across field.",
            },
            "cultural_organic_control": [
                "Install pheromone traps (5 traps/ha) to monitor adult moth catches and time interventions accurately.",
                "Collect and destroy fallen infested squares/rosetted flowers and erect bird perches (10–15/ha).",
                "Spray Neem Seed Kernel Extract (NSKE 5%) or Bacillus thuringiensis (Bt) / HaNPV against early-instar larvae.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Emamectin benzoate 5% SG or Chlorantraniliprole 18.5% SC or Spinosad 45% SC",
                    "target_stage": "Early larval instar emergence coinciding with ETL crossing (5–10% damage)",
                    "cibrc_aligned_note": "CIB&RC-registered larvicides per ICAR-CICR resistance management guidelines.",
                    "safety_warning": "Never use banned insecticides or tank-mix arbitrary cocktails. Avoid repeated synthetic pyrethroid use early season; verify label dose, PHI, and IRAC rotation with local KVK.",
                }
            ],
            "region_notes": {
                "general": "Pink bollworm resistance to Bt cotton has increased importance of pheromone monitoring and timely crop termination.",
                "Maharashtra, Gujarat, Telangana & Punjab": "Follow state ICAR-CICR seasonal termination schedules and avoid extending ratoon/late cotton.",
            },
            "kvk_escalation_trigger": "Escalate to KVK immediately if bollworm infestation exceeds 10% green bolls or affects Bt-cotton hybrids.",
            "source_reference": "ICAR-CICR / DPPQS Cotton IPM & Pink Bollworm Advisory",
        },
        {
            "id": "cotton_aphids",
            "crop": "cotton",
            "condition": "Cotton Aphids",
            "type": "pest",
            "aliases": [
                "aphids",
                "cotton aphid",
                "cotton aphids",
                "Aphis gossypii",
            ],
            "symptoms": [
                "Dense clusters of yellow-green to dark green aphids on tender shoots and leaf undersides.",
                "Downward curling/crinkling of young leaves coated with shiny honeydew and black sooty mold.",
            ],
            "severity_thresholds": {
                "low": "< 10% plants infested with light shoot colonies and active predators.",
                "moderate": "10%–20% plants showing leaf curling and honeydew deposition.",
                "severe": "> 20% plants severely stunted with heavy honeydew/sooty mold threatening open bolls.",
            },
            "cultural_organic_control": [
                "Avoid excessive nitrogenous fertilizer which increases lush sap-feeding attractiveness.",
                "Conserve coccinellid predators and chrysopids; deploy yellow sticky traps for monitoring.",
                "Apply neem-based botanicals (Azadirachtin 1500 ppm) during initial build-up.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Flonicamid 50% WG or Imidacloprid 17.8% SL or Acetamiprid 20% SP",
                    "target_stage": "Only when sucking pest ETL (15–20 aphids/leaf or >15% infested plants) is reached",
                    "cibrc_aligned_note": "CIB&RC-approved selective sucking-pest insecticides.",
                    "safety_warning": "Wait until ETL is exceeded to protect beneficial predators and avoid whitefly flare-up. Spray during late afternoon to protect honeybees; confirm label dosage with KVK.",
                }
            ],
            "region_notes": {
                "general": "Frequently peaks during dry spells following early monsoon or mid-vegetative growth.",
                "Central & Southern Cotton Zones": "Monitor closely in rainfed black cotton soils of Vidarbha, Marathwada, Gujarat, and Telangana.",
            },
            "kvk_escalation_trigger": "Escalate to KVK if honeydew contaminates open lint or multiple sucking pests co-occur above ETL.",
            "source_reference": "ICAR-CICR / DPPQS IPM Package for Cotton",
        },
        {
            "id": "cotton_whitefly",
            "crop": "cotton",
            "condition": "Cotton Whitefly",
            "type": "pest",
            "aliases": [
                "whitefly",
                "cotton whitefly",
                "Bemisia tabaci",
            ],
            "symptoms": [
                "Tiny white moth-like insects rising in clouds from undersides of leaves when disturbed.",
                "Yellowing leaves, sticky honeydew, black sooty mold, and potential Cotton Leaf Curl Virus (CLCuV) transmission.",
            ],
            "severity_thresholds": {
                "low": "< 4 adults per leaf.",
                "moderate": "4–6 adults per leaf (approaching ETL of 6 adults/leaf).",
                "severe": "> 6–8 adults per leaf or heavy nymphal scales with sooty mold / leaf curl.",
            },
            "cultural_organic_control": [
                "Install yellow sticky traps (25–40/ha) and remove weed hosts (kanghi buti, peeli buti) on field bunds.",
                "Spray Neem/NSKE 5% or fish oil rosin soap targeting leaf undersides early in infestation.",
                "Avoid synthetic pyrethroids and tank mixes which trigger severe whitefly resurgence.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Pyriproxyfen 10% EC or Spiromesifen 22.9% SC or Diafenthiuron 50% WP",
                    "target_stage": "Egg/nymphal stages or adult ETL (>= 6 whiteflies/leaf) using thorough under-leaf coverage",
                    "cibrc_aligned_note": "ICAR-CICR recommended IGRs/selective whitefly molecules.",
                    "safety_warning": "Strictly avoid unapproved insecticide mixtures and pyrethroids. Rotate modes of action, verify CIB&RC label dose/PHI with KVK, and protect pollinators.",
                }
            ],
            "region_notes": {
                "general": "High humidity alternating with dry hot spells accelerates whitefly generations.",
                "North Cotton Zone": "Critical vector of Cotton Leaf Curl Disease (CLCuD) in Punjab, Haryana, and Rajasthan.",
            },
            "kvk_escalation_trigger": "Escalate urgently to KVK if whitefly exceeds 6–8 adults/leaf or leaf curl symptoms emerge.",
            "source_reference": "ICAR-CICR Whitefly Management Protocol",
        },
        {
            "id": "potato_late_blight",
            "crop": "potato",
            "condition": "Potato Late Blight",
            "type": "disease",
            "aliases": [
                "Potato___Late_blight",
                "Late blight",
                "Potato Late Blight",
                "Phytophthora infestans",
            ],
            "symptoms": [
                "Irregular water-soaked dark green to purplish-black lesions on leaf tips/margins expanding rapidly.",
                "White cottony fungal-like growth visible on the underside of lesions during humid mornings, accompanied by a characteristic rotting odor.",
            ],
            "severity_thresholds": {
                "low": "< 5% foliage affected in isolated plants under cool humid conditions.",
                "moderate": "5%–15% foliage infected (rapid epidemic potential within 3–5 days).",
                "severe": "> 15% foliage blighted or stem lesions visible; high risk of tuber rot.",
            },
            "cultural_organic_control": [
                "Use certified blight-free seed tubers and destroy volunteer plants and cull piles.",
                "Ensure proper earthing up (ridge height) to prevent zoospores washing onto tubers, and avoid over-irrigation during cloudy/foggy spells.",
                "Rogue and bury isolated early-infected plants carefully away from field.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Cymoxanil 8% + Mancozeb 64% WP or Dimethomorph 50% WP + Mancozeb or Metalaxyl-M 4% + Mancozeb 64% WP",
                    "target_stage": "Immediate curative/systemic + contact spray upon first symptom or local blight forecast alert",
                    "cibrc_aligned_note": "ICAR-CPRI recommended anti-oomycete fungicides for late blight.",
                    "safety_warning": "Late blight spreads explosively in fog/rain; never rely on single-site systemic fungicides alone (use co-formulations to prevent resistance). Verify label dosage & PHI with KVK.",
                }
            ],
            "region_notes": {
                "general": "Explodes rapidly at 10–20°C with >80% humidity, fog, or winter drizzle.",
                "Indo-Gangetic Plains & Hills": "Monitor ICAR-CPRI Indo-Blightcast alerts in UP, West Bengal, Bihar, Punjab, and hill regions.",
            },
            "kvk_escalation_trigger": "Escalate immediately to KVK/CPRI advisory on first blight detection during foggy/rainy weather to prevent total crop collapse.",
            "source_reference": "ICAR-CPRI Late Blight Management Advisory",
        },
        {
            "id": "potato_early_blight",
            "crop": "potato",
            "condition": "Potato Early Blight",
            "type": "disease",
            "aliases": [
                "Potato___Early_blight",
                "Early blight",
                "Potato Early Blight",
                "Alternaria solani",
            ],
            "symptoms": [
                "Dark brown to black circular/angular spots on older lower leaves displaying distinct concentric rings ('target-board' pattern).",
                "Yellow chlorotic halo around lesions leading to premature defoliation of lower canopy.",
            ],
            "severity_thresholds": {
                "low": "< 5% area on lower senescent leaves.",
                "moderate": "5%–20% leaf area affected progressing up to middle leaves.",
                "severe": "> 20% foliage blighted on middle/upper canopy prior to tuber bulking completion.",
            },
            "cultural_organic_control": [
                "Maintain balanced crop nutrition (adequate N and K) as nutrient-stressed plants suffer severe early blight.",
                "Remove and compost/destroy infected crop debris after harvest and rotate with non-solanaceous crops (cereals/pulses).",
                "Avoid overhead sprinkler irrigation late in the afternoon.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Mancozeb 75% WP or Chlorothalonil 75% WP or Azoxystrobin 23% SC",
                    "target_stage": "When concentric lesions spread upward onto middle leaves",
                    "cibrc_aligned_note": "CIB&RC-approved fungicides for Alternaria blight in potato.",
                    "safety_warning": "Ensure spray reaches lower and middle foliage. Verify exact CIB&RC label dilution, PHI, and PPE instructions with local KVK.",
                }
            ],
            "region_notes": {
                "general": "Favored by warm temperatures (24–30°C) alternating with wet dew/irrigation cycles.",
                "Plains & Plateau Potato Belts": "Common in mid-to-late crop stages across UP, Gujarat, Madhya Pradesh, and Karnataka.",
            },
            "kvk_escalation_trigger": "Escalate to KVK if upper canopy defoliation exceeds 20% during active tuber bulking.",
            "source_reference": "ICAR-CPRI / DPPQS IPM Package for Potato",
        },
        {
            "id": "tomato_early_blight",
            "crop": "tomato",
            "condition": "Tomato Early Blight",
            "type": "disease",
            "aliases": [
                "Tomato___Early_blight",
                "Early blight",
                "Tomato Early Blight",
                "Alternaria solani",
            ],
            "symptoms": [
                "Dark brown spots with concentric target-board rings surrounded by yellow halos starting on older lower leaves.",
                "Sunken dark leathery lesions may also develop near the stem-end calyx of fruits.",
            ],
            "severity_thresholds": {
                "low": "< 5% lower leaf area affected.",
                "moderate": "5%–20% foliage affected spreading toward mid-canopy and fruit trusses.",
                "severe": "> 20% foliage defoliated or stem/fruit calyx lesions causing fruit drop.",
            },
            "cultural_organic_control": [
                "Prune and destroy heavily infected lower leaves touching soil and stake plants for airflow.",
                "Apply organic mulch around base to prevent soil splash onto lower foliage during irrigation/rain.",
                "Follow 2–3 year crop rotation away from tomato, potato, and brinjal.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Mancozeb 75% WP or Azoxystrobin 23% SC or Difenoconazole 25% EC",
                    "target_stage": "At initial upward spread from lower leaves to protect fruiting trusses",
                    "cibrc_aligned_note": "CIB&RC-registered fungicides for tomato Alternaria early blight.",
                    "safety_warning": "Strictly observe Pre-Harvest Interval (PHI) on picking-stage tomato fruits! Verify label dilution and safe waiting days with KVK before application.",
                }
            ],
            "region_notes": {
                "general": "Thrives in warm humid conditions (24–29°C) with frequent dew or drip/furrow splash.",
                "Maharashtra, Karnataka, AP & North India": "Pervasive during post-monsoon and late-Rabi tomato harvests.",
            },
            "kvk_escalation_trigger": "Escalate to KVK if stem-collar cankers or >20% fruit/foliage loss threatens marketable harvest.",
            "source_reference": "ICAR-IIHR / DPPQS IPM Package for Tomato",
        },
        {
            "id": "tomato_late_blight",
            "crop": "tomato",
            "condition": "Tomato Late Blight",
            "type": "disease",
            "aliases": [
                "Tomato___Late_blight",
                "Late blight",
                "Tomato Late Blight",
                "Phytophthora infestans",
            ],
            "symptoms": [
                "Large, water-soaked greenish-black blotches on leaves and dark brown greasy streaks on stems.",
                "Firm, dark brown greasy marbled rot on green fruits, with white mold visible in humid mornings.",
            ],
            "severity_thresholds": {
                "low": "< 5% foliage showing isolated water-soaked patches.",
                "moderate": "5%–15% foliage or stem streaks detected under cool wet weather.",
                "severe": "> 15% foliage/stem blight or active fruit rot across plants.",
            },
            "cultural_organic_control": [
                "Stake tomato plants, remove excess suckers for canopy ventilation, and avoid late-evening irrigation.",
                "Promptly rogue and bury severely blighted stems/fruits away from the field.",
                "Avoid planting tomato immediately adjacent to late-blight-infected potato fields.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Cymoxanil 8% + Mancozeb 64% WP or Dimethomorph 50% WP + Mancozeb",
                    "target_stage": "Prompt intervention at first water-soaked lesion under cool humid forecasts",
                    "cibrc_aligned_note": "CIB&RC-aligned oomycete fungicide formulations for tomato.",
                    "safety_warning": "Observe strict Pre-Harvest Interval (PHI) before harvesting tomatoes for consumption. Confirm label dilution and safe waiting period with local KVK.",
                }
            ],
            "region_notes": {
                "general": "Cool wet weather (12–22°C, RH >85%) drives rapid stem and fruit collapse.",
                "Hills, North Plains & Western Ghats": "High risk during winter rains and monsoon tomato cultivation.",
            },
            "kvk_escalation_trigger": "Escalate immediately to KVK if stem blight or fruit rot appears in cool humid weather.",
            "source_reference": "ICAR-IIHR / DPPQS IPM Package for Tomato",
        },
        {
            "id": "tomato_yellow_leaf_curl_virus",
            "crop": "tomato",
            "condition": "Tomato Yellow Leaf Curl Virus",
            "type": "disease",
            "aliases": [
                "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                "Tomato Leaf Curl Virus",
                "TYLCV",
                "ToLCV",
                "Leaf curl",
            ],
            "symptoms": [
                "Severe upward curling and cupping of leaves, yellowing along leaf margins, and bushy stunted plant growth.",
                "Drastic flower drop and miniature or absent fruit set.",
            ],
            "severity_thresholds": {
                "low": "< 5% plants infected with isolated whitefly vector presence.",
                "moderate": "5%–15% plants showing curling symptoms.",
                "severe": "> 15% plants stunted with leaf curl prior to flowering.",
            },
            "cultural_organic_control": [
                "Rogue and destroy infected plants immediately (bag before pulling to avoid dispersing whiteflies), as viral diseases have NO chemical cure.",
                "Raise nursery seedlings under insect-proof 40–50 mesh nylon net and plant resistant hybrids.",
                "Install yellow sticky traps (20–30/ha) and spray Neem/NSKE 5% to deter the whitefly vector (Bemisia tabaci).",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Imidacloprid 17.8% SL or Thiamethoxam 25% WG or Spiromesifen 22.9% SC (Whitefly Vector Control Only)",
                    "target_stage": "Targeted ONLY against the whitefly vector on healthy/border plants when whiteflies are present; zero effect on virus itself",
                    "cibrc_aligned_note": "CIB&RC-registered vector insecticides; fungicides/bactericides cannot cure viruses.",
                    "safety_warning": "NO pesticide cures Leaf Curl Virus. Never waste money spraying fungicides for viral curling. Manage whitefly vectors safely per KVK label dosage and observe strict PHI on fruiting crops.",
                }
            ],
            "region_notes": {
                "general": "Transmitted exclusively by whitefly (Bemisia tabaci); worst in dry warm pre-monsoon/summer plantings.",
                "Karnataka, Maharashtra, AP & North Plains": "Use ToLCV-tolerant hybrids and barrier border crops (maize/sorghum) around nurseries.",
            },
            "kvk_escalation_trigger": "Escalate to KVK if >15% nursery or early transplanted crop develops viral leaf curl for varietal and vector management advice.",
            "source_reference": "ICAR-IIHR / DPPQS Tomato Leaf Curl IPM Guidelines",
        },
        {
            "id": "maize_fall_armyworm",
            "crop": "maize",
            "condition": "Fall Armyworm",
            "type": "pest",
            "aliases": [
                "fall armyworm",
                "army worm",
                "Maize Fall Armyworm",
                "Corn Fall Armyworm",
                "Spodoptera frugiperda",
            ],
            "symptoms": [
                "Ragged elongated 'window-pane' feeding holes in whorl leaves with abundant sawdust-like moist frass inside the funnel.",
                "Caterpillar has an inverted 'Y' marking on the head capsule and four square-arranged dark spots on the tail segment.",
            ],
            "severity_thresholds": {
                "low": "< 5% whorl-damaged plants with early pin-hole scraping.",
                "moderate": "5%–10% whorl damage during early seedling-to-knee-high stage (ETL ~5–10% whorl infestation).",
                "severe": "> 10%–20% damaged whorls or cob/tassel boring by late-instar larvae.",
            },
            "cultural_organic_control": [
                "Install FAW pheromone traps (4–5/acre) right after germination and hand-crush visible egg masses/neonates.",
                "Apply dry sand + lime/soil mixture or neem cake/ash into leaf whorls in small plots for mechanical/contact larval deterrence.",
                "Spray Azadirachtin 1500 ppm or Metarhizium anisopliae / Bacillus thuringiensis (Bt) directed straight into the whorl at egg hatch.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Chlorantraniliprole 18.5% SC or Spinetoram 11.7% SC or Emamectin benzoate 5% SG",
                    "target_stage": "Direct whorl-targeted spray when >5–10% plants show fresh whorl damage",
                    "cibrc_aligned_note": "ICAR-IIMR / DPPQS ad-hoc CIB&RC recommended molecules for Fall Armyworm in maize.",
                    "safety_warning": "Target spray nozzle directly into leaf whorls during morning/evening hours; avoid spraying at peak bee pollen shedding. Verify exact label dilution and PHI with local KVK.",
                }
            ],
            "region_notes": {
                "general": "Invasive polyphagous pest capable of defoliating maize within days if missed in the seedling-to-knee-high whorl stage.",
                "Karnataka, Maharashtra, Telangana, MP & Bihar": "High year-round pressure across Kharif, Rabi, and Spring maize belts.",
            },
            "kvk_escalation_trigger": "Escalate immediately to KVK if >15% whorls are heavily shredded or larvae begin boring into developing cobs.",
            "source_reference": "ICAR-IIMR / DPPQS Fall Armyworm Management Advisory",
        },
        {
            "id": "maize_northern_leaf_blight",
            "crop": "maize",
            "condition": "Maize Northern Corn Leaf Blight",
            "type": "disease",
            "aliases": [
                "Corn_(maize)___Northern_Leaf_Blight",
                "Northern Corn Leaf Blight",
                "Northern leaf blight",
                "Turcicum leaf blight",
                "Exserohilum turcicum",
            ],
            "symptoms": [
                "Long, cigar-shaped (elliptical) greyish-green to tan lesions (2.5–15 cm) running parallel to leaf veins.",
                "Lesions coalesce to scorch whole lower leaves, starving grain fill if upper ear-leaf is hit.",
            ],
            "severity_thresholds": {
                "low": "< 5% leaf area affected restricted to lowest 2–3 leaves.",
                "moderate": "5%–20% leaf area affected reaching middle leaves before tasseling.",
                "severe": "> 20% blight on ear leaf and upper foliage between tasseling and grain fill.",
            },
            "cultural_organic_control": [
                "Adopt resistant single-cross maize hybrids and practice deep ploughing/removal of infected maize stubble after harvest.",
                "Rotate maize with non-host legumes or oilseeds to break residue-borne inoculum.",
                "Maintain balanced potassium fertilization to improve foliar disease tolerance.",
            ],
            "approved_chemical_options": [
                {
                    "active_ingredient": "Mancozeb 75% WP or Azoxystrobin 18.2% + Difenoconazole 11.4% SC",
                    "target_stage": "Protective spray if cigar-shaped lesions climb toward the ear leaf around pre-tasseling/silking",
                    "cibrc_aligned_note": "CIB&RC-aligned fungicide options for Turcicum leaf blight in maize.",
                    "safety_warning": "Verify exact CIB&RC label dilution, PHI, and safety equipment guidance with local KVK/DAO before spraying.",
                }
            ],
            "region_notes": {
                "general": "Favored by moderate temperatures (18–27°C) and prolonged heavy dew or monsoon rains.",
                "Karnataka, Himachal, Bihar & J&K": "Endemic in humid hill regions and winter/Rabi maize pockets.",
            },
            "kvk_escalation_trigger": "Escalate to KVK if cigar-shaped lesions reach the ear leaf before silking/grain filling.",
            "source_reference": "ICAR-IIMR / DPPQS Maize Disease Management Package",
        },
    ]

    # Strictly validate every card against DiseasePestCard contract
    validated_cards: List[Dict[str, Any]] = []
    seen_ids = set()
    for raw in raw_cards:
        card = DiseasePestCard(**raw)
        if card.id in seen_ids:
            raise ValueError(f"Duplicate disease/pest card id: {card.id}")
        seen_ids.add(card.id)
        validated_cards.append(card.model_dump(mode="json"))

    return validated_cards


def build_offline_kb(
    crop_js_path: Path = DEFAULT_CROP_DB_JS,
    fert_js_path: Path = DEFAULT_FERT_DB_JS,
    output_json_path: Path = DEFAULT_OUTPUT_JSON,
) -> Dict[str, Any]:
    """
    Compiles `agri_kb.json` deterministically from FieldWise JS databases
    plus grounded ICAR/KVK disease and pest treatment cards.
    """
    extracted = load_fieldwise_js_databases(crop_js_path, fert_js_path)
    crops = extracted["crops"]
    fertilizers = extracted["fertilizers"]
    cards = _build_disease_pest_cards()

    kb_payload: Dict[str, Any] = {
        "kb_schema_version": KB_SCHEMA_VERSION,
        "provenance": {
            "crop_database_source": "model/cropDatabase.js",
            "crop_database_sha256": _sha256_file(crop_js_path),
            "fertilizer_database_source": "model/fertilizerDatabase.js",
            "fertilizer_database_sha256": _sha256_file(fert_js_path),
            "fertilizer_safety_policy": (
                "Baseline FieldWise Package of Practices reference only. Physical commercial "
                "fertilizer bag quantities require a verified laboratory Soil Health Card (SHC) test."
            ),
            "agronomic_alignment": "ICAR / DPPQS / CIB&RC IPM Offline Reference (Zero-Cloud)",
        },
        "crops": crops,
        "fertilizers": fertilizers,
        "disease_pest_cards": cards,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(kb_payload, indent=2, ensure_ascii=False) + "\n"
    output_json_path.write_text(serialized, encoding="utf-8")
    return kb_payload


def main() -> None:
    kb = build_offline_kb()
    print(
        f"Compiled offline KB -> {DEFAULT_OUTPUT_JSON} "
        f"(crops={len(kb['crops'])}, fertilizers={len(kb['fertilizers'])}, "
        f"disease_pest_cards={len(kb['disease_pest_cards'])})"
    )


if __name__ == "__main__":
    main()
