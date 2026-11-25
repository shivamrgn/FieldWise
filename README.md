# FieldWise — Unified Offline Agricultural Intelligence Project

A 100% offline agricultural AI suite built for the **Bharat AI-SoC Challenge 2026–27 (PS1)** combining:
1. **Disease & Pest Detection + Advisory (`ps1_offline_assistant/`)** — Real pretrained offline 38-class PlantVillage **MobileNetV2** (`huggingface_local_offline` / `plantvillage_mobilenetv2.pt2`), OpenCV blur/exposure QC + HSV/CIE Lab severity estimation, grounded ICAR/CIB&RC KB (`agri_kb.json`), ExecuTorch Llama 3.2 support, and 4-language UI (`en`, `hi`, `gu`, `mr`).
2. **Agronomic Crop Recommendation Engine V2 (`ml_pipeline/recommendation_engine/`)** — Deterministic two-tier biological survival gating + Liebig scoring across 19 canonical Indian crops + SHC soil-test STCR fertilizer stoichiometry.
3. **FieldWise V1 Baseline Databases (`cropDatabase.js`, `fertilizerDatabase.js`, `predictor.js`)** — Preserved read-only JS baseline and source data deterministically compiled into `ps1_offline_assistant/data/agri_kb.json`.

---

## Quickstart (100% Offline)

Run via the unified root launcher `app.py`:

```bash
# 1. Start the 3-tab Offline Web App (http://127.0.0.1:8080/)
# Automatically selects --vision-backend hf-local when pretrained_hf/config.json exists
python3 app.py serve --port 8080

# (Or using the project virtualenv)
ps1_offline_assistant/.venv/bin/python app.py serve --port 8080
```

Other unified commands:

```bash
# Run an offline single-image CLI scan with instant Gujarati translation
python3 app.py scan --image path/to/leaf.jpg --state Punjab --lang hi --switch-lang gu

# Run the V2 Crop Recommendation Engine directly (flags or piped JSON)
python3 app.py crop --state Punjab --sowing-month 6 --soil-texture Loam
echo '{"state":"Punjab","sowing_month":6,"soil_texture":"Loam"}' | python3 app.py crop

# Rebuild agri_kb.json from cropDatabase.js & fertilizerDatabase.js
python3 app.py rebuild-kb

# Run all automated tests across PS1, V2 Engine, and V1 NodeJS
python3 app.py test
```

If deploying on Raspberry Pi 5 / Arm with compiled Llama 3.2 `.pte` weights:

```bash
python3 app.py serve \
  --vision-backend hf-local \
  --advisory-backend executorch-cli \
  --llama-bin /path/to/llama_main \
  --llama-pte /path/to/llama3_2_1b_xnnpack_8da4w.pte \
  --llama-tokenizer /path/to/tokenizer.model \
  --port 8080
```

---

## Vision Model Details & Supported Crops

- **Active Model**: `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` (`MobileNetV2ForImageClassification`, **38 PlantVillage classes**).
- **Offline Artifacts (`ps1_offline_assistant/vision/artifacts/pretrained_hf/`)**:
  - `plantvillage_mobilenetv2.pt2` (`~10.3 MB` `torch.export` bundle, runs offline under standard Python + PyTorch without `transformers`)
  - `model.safetensors` (`~8.8 MB`, used when `transformers` is present with `local_files_only=True`)
  - `kb_label_mapping.json`, `config.json`, `preprocessor_config.json`
- **Preprocessing (`preprocessor_config.json`)**: `224×224` RGB center-cropped from `shortest_edge=256`, rescaled `1/255`, normalized with `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`.
- **Supported Crops (14 crops, 38 classes)**:
  Apple (4), Blueberry (1), Cherry (2), Corn/Maize (4), Grape (4), Orange (1), Peach (2), Bell Pepper (2), Potato (3), Raspberry (1), Soybean (1), Squash (1), Strawberry (2), Tomato (10 — including *Spider Mites* pest class).
- **Optional Custom Backbone**: `ps1_offline_assistant/vision/train_and_export.py` provides an 18-class `MobileNetV3` training + ONNX / ExecuTorch `.pte` export pipeline for custom datasets (`PlantVillage` / `Cassava` / `IP102`).

---

## Dual-Pipeline Architecture

- **Pipeline A (Vision & Severity)**: OpenCV Laplacian blur + brightness QC -> `MobileNetV2` 38-class offline inference (`VisionBackend`) -> deterministic OpenCV HSV + CIE Lab lesion segmentation -> strict `VisionFinding` JSON. Low-quality (`UNCERTAIN_IMAGE`) and low-confidence/unmapped (`LOW_CONFIDENCE`, `< 0.65`) inputs short-circuit safely without inventing advice.
- **Pipeline B (Grounded Advisory)**: `OfflineKBRetriever` matches `kb_card_id` in `data/agri_kb.json` -> `AdvisoryEngine` runs `LlamaAdvisoryBackend` (ExecuTorch Llama 3.2 on Arm target, or deterministic ICAR KB fallback on hosts without `.pte`) -> post-LLM active-ingredient grounding & CIB&RC/KVK `1800-180-1551` escalation checks -> strict `AdvisoryResponse` JSON + instant zero-vision-rerun i18n (`en`, `hi`, `gu`, `mr`).

---

## Clean Project Structure

```text
model/
├── app.py                               # Unified project CLI & offline server entrypoint
├── README.md                            # This unified project guide
├── package.json                         # V1 NodeJS CommonJS manifest
├── cropDatabase.js                      # V1 crop baseline (compiled into agri_kb.json)
├── fertilizerDatabase.js                # V1 fertilizer schedules (compiled into agri_kb.json)
├── predictor.js                         # V1 Gaussian baseline predictor (read-only)
├── ml_pipeline/
│   ├── recommendation_engine/           # Active V2 agronomic rule engine + test suites
│   └── reports/                         # Minimal audit CSVs for V2 tests & test execution output
└── ps1_offline_assistant/               # Offline Edge-AI Assistant (Vision + Advisory + 3-Tab UI)
    ├── advisory/                        # Pipeline B: Llama 3.2 ExecuTorch & safety guardrails
    ├── app/                             # ThreadingHTTPServer + zero-network index.html UI
    ├── benchmarks/                      # Reproducible latency/memory benchmark runner & results
    ├── configs/                         # Llama 3.2 XNNPACK 8da4w export configuration
    ├── data/agri_kb.json                # Compiled offline crop, fertilizer & IPM treatment KB
    ├── docs/                            # ARCHITECTURE.md, TECHNICAL_REPORT.md, DEMO_SCRIPT.md
    ├── i18n/                            # Offline translations (`en`, `hi`, `gu`, `mr`)
    ├── kb/                              # Offline KB retriever & normalization
    ├── orchestrator/                    # Dual-pipeline orchestrator & CLI
    ├── report.md                        # Consolidated PS1 master integration report
    ├── requirements.txt                 # Pretrained HF download dependencies
    ├── schemas/                         # Strict Pydantic runtime contracts
    ├── scripts/build_offline_kb.py      # Deterministic JS -> agri_kb.json compiler
    ├── tests/                           # Phases 1-7 unittest suites (45 tests)
    └── vision/                          # Pipeline A: OpenCV gates, severity, ONNX/ExecuTorch/HF/.pt2
        └── artifacts/pretrained_hf/     # Cached 38-class MobileNetV2 PlantVillage (.pt2 + .safetensors)
```

---

## Runtime Dependencies, Testing & Known Limitations

- **Offline Runtime Stack (No `.venv` Needed)**: Global Python requires `torch`, `torchvision`, `opencv-python`, `numpy`, `pydantic`, and `pyyaml`. The cached `plantvillage_mobilenetv2.pt2` model runs 100% offline directly under system `python3` with zero `transformers` or internet access.
- **Automated Verification (`python3 app.py test`)**: Runs all PS1 Phase 1–7 unit tests (`python3 -m unittest discover -s ps1_offline_assistant/tests -p "test_*.py" -v`), FieldWise V2 recommendation engine tests, and V1 NodeJS sanity checks. *(Note: Local HTTP server integration tests skip cleanly when executed inside a loopback-restricted sandbox.)*
- **Llama 3.2 + ExecuTorch Advisory Status**: **Integrated and verified live locally via `ExecuTorchWorkerAdvisoryBackend` (`executorch_llama_worker`)** using the isolated `executorch_env` (`executorch 1.6.0`, `pytorch_tokenizers 1.6.0`), `ps1_offline_assistant/advisory/artifacts/xnnpack/Llama-3.2-1B-Instruct-8da4w-gptq-2k.pte` (`916 MB`), and `tokenizer.json` (`8.7 MB`). Measured locally on Apple Silicon M2 (`Darwin-arm64`) at `4.1–6.2 s` TTFT, `19–24 tok/s` sustained decode (`64 tok/s` short prompt), and `~1.11 GB` worker peak RSS. Whenever quantized 1B output violates strict JSON or chemical grounding constraints, `AdvisoryEngine` automatically falls back to the verified `agri_kb.json` ICAR entry. If `.pte`/`executorch_env` artifacts are absent, `app.py` transparently falls back to `test_stub_ci_only`.
- **Known Limitations**: Out of 38 PlantVillage classes, 6 Indian crop disease classes map directly to full ICAR treatment cards in `agri_kb.json` and 12 healthy classes emit preventive monitoring guidance; remaining non-KB classes route safely to `LOW_CONFIDENCE` KVK verification (`1800-180-1551`) rather than fabricating chemical dosages. PlantVillage is also heavily focused on leaf diseases with only 1 pest class (*Tomato Spider Mites*).

---

## Clean Submission Packaging Checklist

To create a clean archive without virtual environments, caches, temporary files, or `.DS_Store` metadata while **preserving** required model weights (`ps1_offline_assistant/vision/artifacts/pretrained_hf/`) and V2 verification manifests (`ml_pipeline/reports/`):

```bash
find . \( -name "__pycache__" -o -name ".DS_Store" -o -name "*.pyc" \) -exec rm -rf {} +
zip -r ../FieldWise_PS1_Submission.zip . \
  -x "ps1_offline_assistant/.venv/*" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x "*.DS_Store" \
  -x "ps1_offline_assistant/data/field_scans.jsonl" \
  -x "*.tmp"
```
# FieldWise
