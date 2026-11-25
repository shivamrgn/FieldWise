# FieldWise Offline Edge-AI Architecture (Bharat AI-SoC Challenge 2026–27 — PS1)

**Problem Statement**: PS1 — *Offline Crop-Disease/Pest Detection & Advisory for Farmers*  
**Target Platform**: Arm-based Edge Hardware (Raspberry Pi 5 + AI HAT+ / Hailo NPU, RPi 5 CPU-only, Android Arm64)  
**Connectivity Mode**: 100% Offline, On-Device Execution (Zero Cloud APIs, Zero Outbound Network Calls)

---

## 1. Architectural Principles & Preservation Boundaries

1. **Strict Isolation from Production FieldWise**:
   All PS1 offline edge-AI code lives exclusively under `ps1_offline_assistant/`. Existing FieldWise V1 (`predictor.js`, `cropDatabase.js`, `fertilizerDatabase.js`) and V2 (`ml_pipeline/recommendation_engine/*`) files remain strictly read-only and unmodified.
2. **Zero Cloud Dependencies**:
   No Supabase, Firebase, Gemini Vision API, Open-Meteo, remote vector stores, or external HTTP calls exist anywhere in the runtime path. All inference weights, OpenCV transforms, knowledge bases (`agri_kb.json`), and safety rules reside locally on flash/SD storage.
3. **Deterministic Grounding & Zero-Fabrication Safety**:
   Inspired by FieldWise V2 (`ml_pipeline/recommendation_engine/explanations.py` and `provenance_registry.py`), the on-device LLM **never** invents pesticide names, chemical dosages, or disease diagnoses from parametric memory alone. Every clinical/chemical recommendation is strictly grounded in the retrieved `agri_kb.json` treatment card (`kb_entry`), and uncertain or severe cases trigger deterministic safety guardrails or Krishi Vigyan Kendra (KVK) escalation.

---

## 2. Two-Pipeline Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE A: ON-DEVICE VISION                         │
│                                                                             │
│  [Leaf / Pest Image]                                                        │
│         │                                                                   │
│         ▼                                                                   │
│  [OpenCV Preprocessing & Quality Gate]                                      │
│    • Laplacian variance blur check (blur_score, image_quality_ok)           │
│    • Exposure / contrast check                                              │
│    • Resize + RGB/BGR normalization for model input                         │
│    • HSV / Lab lesion segmentation -> infected_area_pct (0–100%)            │
│         │                                                                   │
│         ▼                                                                   │
│  [On-Device Vision Classifier (PlantVillage + IP102)]                       │
│    • HailoRT NPU (.hef) on RPi 5 + AI HAT+                                  │
│    • OR ExecuTorch / ONNX / TFLite (XNNPACK) on Arm CPU/GPU                 │
│         │                                                                   │
│         ▼                                                                   │
│  [Strict Structured Contract: VisionFinding JSON]                           │
│    {crop, condition_name, category, confidence, severity_level,             │
│     infected_area_pct, image_quality_ok, blur_score,                        │
│     inference_latency_ms, runtime_backend}                                  │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 DETERMINISTIC SAFETY & GROUNDING GATEWAY                    │
│                                                                             │
│  1. Image Quality Gate: image_quality_ok == False -> status=UNCERTAIN_IMAGE │
│  2. Confidence / Unknown Gate: category=="unknown" or confidence < τ        │
│     or kb_entry is None -> status=LOW_CONFIDENCE (withhold chemicals)       │
│  3. Severity Escalation Gate: severity_level=="severe" or                   │
│     infected_area_pct >= escalation_threshold -> status=SEVERE_ESCALATION   │
│     (sets escalate_to_kvk=True)                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PIPELINE B: ON-DEVICE ADVISORY                        │
│                                                                             │
│  Inputs: VisionFinding + Farmer Context (state_region, language) +          │
│          Local Grounded KB Lookup (kb/retriever.py -> agri_kb.json)         │
│         │                                                                   │
│         ▼                                                                   │
│  [AdvisoryRequest JSON]                                                     │
│         │                                                                   │
│         ▼                                                                   │
│  [Llama 3.2 (1B / 3B-Instruct) via ExecuTorch on Arm CPU]                   │
│    • Quantized (.pte, 4-bit groupwise / INT8 dynamic via XNNPACK/KleidiAI)  │
│    • Grounded prompt strictly constrained to kb_entry + VisionFinding       │
│    • Target language generation: en | hi | gu | mr                          │
│    • Structured post-validation + deterministic chemical/KVK safety guard   │
│         │                                                                   │
│         ▼                                                                   │
│  [Strict Structured Contract: AdvisoryResponse JSON]                        │
│    {status, explanation, immediate_actions, organic_cultural_steps,         │
│     chemical_guidance_if_approved, preventive_measures, escalate_to_kvk,    │
│     disclaimer, language, ttft_ms, tokens_per_sec}                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Inter-Pipeline Contract Flow (`schemas/contracts.py`)
1. **Pipeline A emits `VisionFinding`**: Captures predicted `crop`, `condition_name`, `category` (`disease | pest | healthy | unknown`), `confidence` (`0.0–1.0`), `severity_level` (`low | moderate | severe | undetermined`), `infected_area_pct` (`0.0–100.0`), image quality telemetry (`image_quality_ok`, `blur_score`), and edge profiling metrics (`inference_latency_ms`, `runtime_backend`).
2. **KB Retriever (`kb/retriever.py`) resolves `kb_entry`**: Normalizes PlantVillage / IP102 labels offline against `data/agri_kb.json` (`disease_pest_cards`). Unknown or unmapped classes cleanly return `None` rather than forcing an approximate match.
3. **Pipeline B consumes `AdvisoryRequest` and emits `AdvisoryResponse`**: Enforces one of four deterministic statuses (`SUCCESS`, `UNCERTAIN_IMAGE`, `LOW_CONFIDENCE`, `SEVERE_ESCALATION`), structured treatment lists, mandatory KVK escalation flag (`escalate_to_kvk`), safety disclaimers, and LLM edge metrics (`ttft_ms`, `tokens_per_sec`).

---

## 3. Hardware Execution Split: HailoRT vs. ExecuTorch

The two-pipeline architecture cleanly decouples visual perception from language reasoning so each workload executes on the most suitable Arm-based edge accelerator:

| Hardware Profile | Pipeline A (Vision Classifier) | Pipeline B (Llama 3.2 1B/3B Advisory) | Orchestration & Memory Strategy |
| :--- | :--- | :--- | :--- |
| **Profile 1: Raspberry Pi 5 + AI HAT+ (Hailo-8 / Hailo-8L NPU)** | **Hailo NPU via HailoRT** (`.hef` compiled model via `hailo_platform` runtime; `runtime_backend="hailort_npu"`). Offloads CNN/ViT convolutions completely off the Arm Cortex-A76 cores. | **Arm Cortex-A76 CPU via ExecuTorch** (`.pte` 4-bit quantized model with XNNPACK / Arm KleidiAI NEON/dotprod kernels). | Vision NPU DMA buffers remain isolated from CPU LLM KV-cache & weights. Enables near-zero CPU contention between image capture/classification and token generation. |
| **Profile 2: Raspberry Pi 5 (CPU-Only, 4 GB / 8 GB)** | **Arm Cortex-A76 CPU via ExecuTorch / ONNX Runtime / TFLite XNNPACK** (`runtime_backend="executorch_xnnpack_cpu"` or `"onnxruntime_arm_cpu"`). | **Arm Cortex-A76 CPU via ExecuTorch** (Llama 3.2 1B-Instruct 4-bit preferred on 4 GB SKUs; 3B-Instruct 4-bit supported on 8 GB SKUs). | **Sequential Two-Stage Execution**: Vision model runs first, returns `VisionFinding`, and releases transient image tensors before LLM decoding begins to prevent OOM and thermal throttling. |
| **Profile 3: Android Arm64 Smartphone** | **Arm CPU (XNNPACK) or Mobile GPU/NPU Delegate** (ExecuTorch XNNPACK/Vulkan or TFLite GPU/NNAPI; `runtime_backend="executorch_vulkan_gpu"` / `"tflite_arm_cpu"`). | **Arm64 CPU via ExecuTorch** (XNNPACK + KleidiAI `i8mm` / `dotprod` acceleration on Cortex-A7xx/X-series cores). | Bounded context window (<= 1024–1536 tokens) using concise single-card retrieval from `agri_kb.json` to keep Time-To-First-Token (`ttft_ms`) low and preserve thermal headroom. |

### Why This Split Matters
- **No Cloud Round-Trip**: Farmers in zero-connectivity fields receive complete visual diagnosis + vernacular advisory (`en`, `hi`, `gu`, `mr`) locally.
- **Strict Grounding Replaces Large-Model Memorization**: By feeding Llama 3.2 1B/3B a single compact, verified JSON card (`kb_entry` from `agri_kb.json`), a 1B–3B parameter model achieves high factual precision without hallucinating unapproved pesticides or wrong dosages.
- **Accurate Profiling Telemetry**: Every `VisionFinding` records `inference_latency_ms` and `runtime_backend`, and every `AdvisoryResponse` records `ttft_ms` and `tokens_per_sec`, ensuring benchmark measurements reflect genuine hardware execution.

---

## 4. FieldWise Reuse Map

The table below traces every asset extracted or adapted from the existing FieldWise repository into `ps1_offline_assistant/`:

| Existing FieldWise Source (Read-Only) | Extracted / Adapted Artifact in `ps1_offline_assistant/` | Purpose & Adaptation Notes |
| :--- | :--- | :--- |
| `model/cropDatabase.js` | `data/agri_kb.json` (`"crops"` section via `scripts/build_offline_kb.py`) | Deterministic extraction of all 16 baseline crops (`id`, `name`, `season`, rainfall/temperature/humidity bounds, `soilTypes`, NPK ranges, `phRange`, `yieldPerHectare`, `description`). Provides host-crop agronomic baseline context alongside disease/pest cards. |
| `model/fertilizerDatabase.js` | `data/agri_kb.json` (`"fertilizers"` section via `scripts/build_offline_kb.py`) | Deterministic extraction of all 16 crop fertilizer schedules (`name`, `dosage`, `applicationTime`, `nutrients`) as baseline Package of Practices reference data, tagged with SHC soil-test safety guardrails. |
| `model/ml_pipeline/recommendation_engine/schemas.py` | `schemas/contracts.py` | Reused strict typed schema design, explicit execution status enums (`SUCCESS`, `UNCERTAIN_IMAGE`, `LOW_CONFIDENCE`, `SEVERE_ESCALATION`), and non-silently-coercing numeric/enum contracts. |
| `model/ml_pipeline/recommendation_engine/input_validator.py` | `schemas/contracts.py` & `kb/retriever.py` | Adapted `STATE_NORMALIZATION_MAP` concept and strict input boundary validation (`NaN`/infinity rejection, non-negative bounds, normalized string identifiers). |
| `model/ml_pipeline/recommendation_engine/dci_confidence.py` | Pipeline A/B Uncertainty Gating (`UNCERTAIN_IMAGE`, `LOW_CONFIDENCE`) | Adapted the explicit epistemic rejection philosophy (`DCI < 0.500 -> INSUFFICIENT_DATA`): poor blur quality or low classifier confidence explicitly halts standard prescriptions rather than emitting overconfident advice. |
| `model/ml_pipeline/recommendation_engine/explanations.py` | `data/agri_kb.json` (`disease_pest_cards` safety warnings & KVK escalation) | Adapted FieldWise's fertilizer/agronomic safety mandates: withhold unverified prescriptions, never invent unapproved chemicals or dosages, attach safety warnings to every chemical option, and enforce mandatory Krishi Vigyan Kendra (KVK) referral triggers. |
| `model/ml_pipeline/recommendation_engine/provenance_registry.py` & `stcr_registry.py` | Zero-Fabrication Governance in `docs/ARCHITECTURE.md` & `agri_kb.json` | Adopted the strict rule against fabricated authority claims, mock model accuracies, or invented numeric calibration constants. |
| `model/predictor.js` | *Not directly executed in PS1* (Read-only reference) | Tabular suitability scoring remains untouched for FieldWise V1 compatibility. |

---

## 5. Offline Knowledge Base (`data/agri_kb.json`) Design

`scripts/build_offline_kb.py` deterministically compiles `data/agri_kb.json` from:
1. **Live extraction** of `model/cropDatabase.js` (`crops` — 16 crops).
2. **Live extraction** of `model/fertilizerDatabase.js` (`fertilizers` — 16 crop schedules).
3. **Curated `disease_pest_cards`** covering major Indian crops (**Rice, Wheat, Cotton, Potato, Tomato, Maize**) and major **PlantVillage** and **IP102** classes (`17` grounded cards compiled in `data/agri_kb.json`, alongside `16` crop agronomic profiles and `16` fertilizer schedules).
4. **Vision Backends Supported**:
   - **Pretrained Offline PlantVillage Classifier**: `HuggingFaceLocalVisionBackend` (`38`-class `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`, loaded via local `model.safetensors` or zero-`transformers` `plantvillage_mobilenetv2.pt2`).
   - **Custom 18-Class Edge Exporter**: `vision/train_and_export.py` (`MobileNetV3` -> `.pt2` / `.onnx` / `.pte`), plus V2 Crop Recommendation integration directly inside `app/server.py` (`POST /api/recommend-crop`).

Each card in `disease_pest_cards` enforces a uniform schema:
- `id`: Stable canonical identifier (e.g., `"rice_blast"`).
- `crop`: Normalized host crop key matching `crops` (e.g., `"rice"`).
- `condition`: Human-readable condition name.
- `type`: `"disease"` or `"pest"`.
- `aliases`: Normalized PlantVillage/IP102 class names and synonyms for deterministic offline lookup.
- `symptoms`: Diagnostic visual symptoms for farmer confirmation.
- `severity_thresholds`: `low`, `moderate`, and `severe` agronomic thresholds.
- `cultural_organic_control`: Immediate non-chemical and IPM practices.
- `approved_chemical_options`: CIB&RC-aligned chemical molecules with target stage and mandatory `safety_warning` (ETL check, label verification with KVK/DAO, PPE, PHI, resistance rotation).
- `region_notes`: Zone/state-specific guidance across Indian agro-climatic regions.
- `kvk_escalation_trigger`: Explicit criteria triggering mandatory `escalate_to_kvk = True`.
