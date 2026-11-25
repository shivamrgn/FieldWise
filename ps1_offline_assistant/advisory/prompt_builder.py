"""
Grounded Llama 3.2 Instruct Prompt Builder for FieldWise PS1 (Pipeline B).

Constructs prompts using official Meta Llama 3.2 Instruct chat header tokens
strictly grounded in `VisionFinding`, farmer context (`state_region`, `language`),
and the local ICAR/KVK treatment card (`kb_entry`).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ps1_offline_assistant.schemas.contracts import (
    AdvisoryRequest,
    DiseasePestCard,
    SupportedLanguage,
)

LANGUAGE_DISPLAY_NAMES: Dict[SupportedLanguage, str] = {
    SupportedLanguage.EN: "English (en)",
    SupportedLanguage.HI: "Hindi / हिन्दी (hi)",
    SupportedLanguage.GU: "Gujarati / ગુજરાતી (gu)",
    SupportedLanguage.MR: "Marathi / मराठी (mr)",
    SupportedLanguage.TE: "Telugu / తెలుగు (te)",
    SupportedLanguage.TA: "Tamil / தமிழ் (ta)",
}

LLAMA3_BOS = "<|begin_of_text|>"
LLAMA3_START_HEADER = "<|start_header_id|>"
LLAMA3_END_HEADER = "<|end_header_id|>"
LLAMA3_EOT = "<|eot_id|>"


def _select_relevant_region_notes(
    card: DiseasePestCard,
    state_region: str,
) -> Dict[str, str]:
    """Extracts general and state-matching regional notes from a KB card."""
    notes: Dict[str, str] = {}
    state_lower = state_region.strip().lower()
    for region_key, note_text in card.region_notes.items():
        rk_lower = region_key.lower()
        if rk_lower == "general" or state_lower in rk_lower or rk_lower in state_lower:
            notes[region_key] = note_text
    if not notes:
        return dict(card.region_notes)
    return notes


def _serialize_kb_context(
    kb_entry: Optional[DiseasePestCard],
    state_region: str,
) -> str:
    if kb_entry is None:
        return json.dumps(
            {
                "kb_entry": None,
                "grounding_notice": (
                    "NO APPROVED TREATMENT CARD IN LOCAL KB. Do NOT invent treatments "
                    "or chemicals; recommend local KVK verification."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )

    card_dict: Dict[str, Any] = {
        "id": kb_entry.id,
        "crop": kb_entry.crop,
        "condition": kb_entry.condition,
        "type": kb_entry.type.value,
        "symptoms": kb_entry.symptoms,
        "severity_thresholds": kb_entry.severity_thresholds.model_dump(mode="json"),
        "cultural_organic_control": kb_entry.cultural_organic_control,
        "approved_chemical_options": [
            opt.model_dump(mode="json") for opt in kb_entry.approved_chemical_options
        ],
        "region_notes": _select_relevant_region_notes(kb_entry, state_region),
        "kvk_escalation_trigger": kb_entry.kvk_escalation_trigger,
        "source_reference": kb_entry.source_reference,
    }
    return json.dumps(card_dict, indent=2, ensure_ascii=False)


def build_llama3_prompt(request: AdvisoryRequest) -> str:
    """
    Formats an `AdvisoryRequest` into an official Llama 3.2 Instruct chat prompt
    enforcing zero-fabrication grounding, cultural-first IPM ordering, vernacular
    phrasing, and strict JSON output.
    """
    lang_label = LANGUAGE_DISPLAY_NAMES[request.language]

    system_prompt = (
        "You are FieldWise Offline Krishi Sahayak, an on-device agricultural advisory "
        "assistant helping Indian farmers in zero-connectivity rural areas.\n\n"
        "STRICT SAFETY & GROUNDING RULES:\n"
        "1. Answer ONLY using the provided `vision_finding` and `kb_entry` (approved "
        "ICAR/DPPQS/CIB&RC treatment card). NEVER invent active ingredients, pesticide "
        "dosages, dilution ratios (ml/L or g/L), or commercial brand names not explicitly "
        "present in `kb_entry`.\n"
        "2. Always present cultural, sanitation, and organic/IPM interventions BEFORE "
        "any approved chemical guidance.\n"
        "3. If confidence is low or `kb_entry` is null, clearly state uncertainty, "
        "leave `chemical_guidance_if_approved` as an empty list [], and advise consulting "
        "the local Krishi Vigyan Kendra (KVK).\n"
        "4. NEVER present an AI camera screening result as a confirmed laboratory diagnosis.\n"
        f"5. Write ALL farmer-facing text fields strictly in {lang_label} using simple, "
        "clear, practical phrasing suitable for smallholder farmers.\n"
        "6. Output ONLY a single valid JSON object (no markdown fences or extra text) "
        "with EXACTLY these keys:\n"
        '   - "explanation": string\n'
        '   - "immediate_actions": list of strings\n'
        '   - "organic_cultural_steps": list of strings\n'
        '   - "chemical_guidance_if_approved": list of strings (each entry MUST include the exact approved active_ingredient name AND safety warning from kb_entry in the SAME string; empty [] if healthy, low severity below ETL, or uncertain)\n'
        '   - "preventive_measures": list of strings\n'
        '   - "disclaimer": string'
    )

    finding_json = json.dumps(
        request.vision_finding.model_dump(mode="json"),
        indent=2,
        ensure_ascii=False,
    )
    kb_json = _serialize_kb_context(request.kb_entry, request.state_region)

    user_prompt = (
        f"FARMER CONTEXT:\n"
        f"- State/Region: {request.state_region}\n"
        f"- Requested Response Language: {request.language.value} ({lang_label})\n\n"
        f"VISION FINDING JSON:\n{finding_json}\n\n"
        f"LOCAL GROUNDED KB TREATMENT CARD (kb_entry):\n{kb_json}\n\n"
        "Generate the grounded JSON advisory response now."
    )

    return (
        f"{LLAMA3_BOS}"
        f"{LLAMA3_START_HEADER}system{LLAMA3_END_HEADER}\n\n"
        f"{system_prompt}{LLAMA3_EOT}"
        f"{LLAMA3_START_HEADER}user{LLAMA3_END_HEADER}\n\n"
        f"{user_prompt}{LLAMA3_EOT}"
        f"{LLAMA3_START_HEADER}assistant{LLAMA3_END_HEADER}\n\n"
    )
