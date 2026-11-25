

**Target Program**: Bharat AI-SoC Challenge 2026–27 (PS1 — *Offline Crop-Disease/Pest Detection & Advisory for Farmers*)  
**Scope**: Pipeline A (OpenCV image quality gating, HSV/Lab severity estimation, MobileNetV3 training & edge export, pluggable offline inference backends, and strict `VisionFinding` emission).  
**Connectivity**: 100% Offline, Zero Cloud Dependencies.

---

## 1. Files Created in Phase 2

All Phase 2 artifacts reside strictly under `ps1_offline_assistant/`:

| Module / File | Purpose |
| :--- | :--- |
| `ps1_offline_assistant/vision/__init__.py` | Package exports for preprocessing, severity estimation, backends, and `VisionPipeline`. |
| `ps1_offline_assistant/vision/preprocess.py` | OpenCV Laplacian blur (`blur_score`) & brightness quality gate (`assess_image_quality`), plus BGR->RGB ImageNet `(1, 3, 224, 224)` float32 NCHW normalization (`preprocess_for_model`). |
| `ps1_offline_assistant/vision/severity.py` | Deterministic HSV + CIE Lab leaf/lesion segmentation (`estimate_severity`) mapping foliar disease percentage to `low` (`<15%`), `moderate` (`15%–40%`), `severe` (`>40%`), and supporting pest box/contour proxies as well as `healthy`/`unknown`/bad-quality gating. |
| `ps1_offline_assistant/vision/train_and_export.py` | Offline `MobileNetV3` (`mobilenet_v3_small` / `mobilenet_v3_large`) constructor & `ImageFolder` trainer aligned to `data/agri_kb.json` (`18` canonical classes: `17` disease/pest cards + `healthy`), PlantVillage/IP102 folder mapper, `class_labels.json` generator, and static `(1, 3, 224, 224)` edge exporters (`export_to_executorch`, `export_to_onnx`, `export_to_torch_exported`). |
| `ps1_offline_assistant/vision/inference.py` | `VisionPipeline` orchestrator + `VisionBackend` abstraction (`ExecuTorchVisionBackend`, `OnnxVisionBackend`, `HailoRTVisionBackend`, `PyTorchVisionBackend`), enforcing image-quality short-circuiting, `< 0.65` confidence routing to `category="unknown"`, and `time.perf_counter()` latency measurement. |
| `ps1_offline_assistant/tests/test_phase2_vision.py` | Synthetic-fixture test suite covering quality gates, HSV/Lab severity brackets, KB-aligned `MobileNetV3` graph export & inference, dependency-error guards, and `VisionFinding` schema compliance. |
| `ps1_offline_assistant/report.md` | This Phase 2 integration report. |

---

## 2. Pipeline A Architecture & Safety Gating

1. **Image Quality Gate (`preprocess.py`)**:
   - Computes Laplacian variance (`blur_score`) and mean grayscale intensity (`mean_brightness`).
   - Rejects severely blurred (`blur_score < 50.0`), underexposed (`< 35.0`), or overexposed (`> 225.0`) frames with `image_quality_ok = False`, short-circuiting before classification (`category="unknown"`, `condition_name="unclear_image"`, `severity_level="undetermined"`).
2. **Preprocessing (`preprocess.py`)**:
   - Converts `BGR -> RGB`, resizes to `(224, 224)`, normalizes with ImageNet mean/std, and emits contiguous `float32` NCHW `(1, 3, 224, 224)`.
3. **Classification & Confidence Gate (`inference.py`)**:
   - Runs pluggable `VisionBackend.predict_logits`, applies numerically stable softmax, and rejects top-1 predictions with `confidence < 0.65` by setting `category="unknown"` while preserving the raw `confidence` and setting `severity_level="undetermined"`.
4. **Severity Estimation (`severity.py`)**:
   - **Disease**: HSV + CIE Lab segmentation isolates leaf foreground from neutral backgrounds, excludes green chlorophyll tissue, fills enclosed dark necrotic lesion interiors, and computes exact `infected_area_pct` (`< 15% -> low`, `15%–40% -> moderate`, `> 40% -> severe`).
   - **Pest**: Supports explicit bounding-box counts/area coverage (`pest_boxes`) and falls back to an OpenCV contour/foliar-damage heuristic when running classification-only models.
5. **Phase 1 Contract & KB Integration**:
   - Emits strictly validated `VisionFinding` objects (`schemas/contracts.py`).
   - Class manifest (`class_labels.json`) maps 1:1 to `data/agri_kb.json` (`17` `disease_pest_cards` + `healthy` class), ensuring `kb/retriever.py` cleanly resolves every confident disease/pest detection and returns `None` whenever `category="unknown"` or `category="healthy"`.

---

## 3. Export & Runtime Backend Verification Status in Current Environment

Current host environment (`Python 3.13.5` on macOS):
- **Installed & Verified Live**:
  - `cv2 5.0.0`, `numpy 2.4.1`, `PIL 11.3.0`, `torch 2.14.0`, `torchvision 0.29.0`, `pydantic 2.13.4`
  - **`torch.export.export` + `.pt2` (`PyTorchVisionBackend`)**: Verified live on `mobilenet_v3_small` (`(1, 3, 224, 224)` -> `18` logits), including full `VisionPipeline.predict()` round-trip into validated `VisionFinding` JSON with measured `inference_latency_ms > 0.0`.
  - **`export_to_executorch` Stage 1 (`torch.export.export`)**: Verified live; raises explicit `EdgeExportDependencyError` at lowering because `executorch` is not installed locally.
- **Not Installed in Current Sandbox Host (Guarded & Tested for Clean Failure)**:
  - `onnx`, `onnxscript`, `onnxruntime`: Not installed locally. `export_to_onnx` raises `EdgeExportDependencyError` and `OnnxVisionBackend` raises `MissingBackendDependencyError` instead of creating stub artifacts.
  - `executorch`: Not installed locally. `ExecuTorchVisionBackend` raises `MissingBackendDependencyError`.
  - `hailo_platform`: Not installed locally (requires Raspberry Pi 5 + AI HAT+ Hailo-8/8L hardware and HailoRT SDK). `HailoRTVisionBackend` raises `MissingBackendDependencyError` cleanly on non-Hailo hardware.

### Deployment Note for Edge Targets
- **Raspberry Pi 5 + AI HAT+ (Hailo NPU)**: Run `train_and_export.py --export-onnx` on a Linux/x86 build host with `onnx` installed, compile the static `(1, 3, 224, 224)` `.onnx` into `.hef` via the Hailo Dataflow Compiler, and instantiate `VisionPipeline(HailoRTVisionBackend("model.hef"), "class_labels.json")`.
- **Arm64 CPU (RPi 5 CPU-only / Android)**: Run `train_and_export.py --export-executorch` on a Linux build host with `executorch` installed (`XnnpackPartitioner`) to emit `.pte`, and instantiate `VisionPipeline(ExecuTorchVisionBackend("model.pte"), "class_labels.json")`.

---

## 4. Synthetic Smoke-Test Measurements & Test Suite Verification

> **Honesty Boundary**: No external PlantVillage/IP102 training dataset or pretrained agricultural checkpoint was downloaded or run in this offline environment. All image measurements below come strictly from deterministic in-memory synthetic BGR leaf fixtures (`320x320` crisp leaf on neutral background).

Observed synthetic fixture metrics:
- Healthy leaf (`0%` target lesion): `blur_score = 228.55`, `mean_brightness = 191.04`, `image_quality_ok = True`, `infected_area_pct = 0.0%`, `severity_level = "low"`
- Low lesion leaf (`8%` target): `blur_score = 268.98`, `image_quality_ok = True`, `infected_area_pct = 8.0%`, `severity_level = "low"`
- Moderate lesion leaf (`25%` target): `blur_score = 292.67`, `image_quality_ok = True`, `infected_area_pct = 25.0%`, `severity_level = "moderate"`
- Severe lesion leaf (`55%` target): `blur_score = 312.71`, `image_quality_ok = True`, `infected_area_pct = 55.0%`, `severity_level = "severe"`
- Blurry leaf (`GaussianBlur (35, 35), sigma=12.0`): `blur_score = 0.61`, `image_quality_ok = False`

### Preservation Check
All existing FieldWise V1 (`predictor.js`, `cropDatabase.js`, `fertilizerDatabase.js`, `package.json`) and V2 (`ml_pipeline/recommendation_engine/*.py`) files remain unmodified with identical SHA-256 checksums.

---

## 5. Phase 3 Integration Summary (Pipeline B: Offline Llama 3.2 + ExecuTorch Advisory)

### Files Created under `ps1_offline_assistant/advisory/`, `configs/`, and `tests/`
- `ps1_offline_assistant/configs/llama3_2_1b_xnnpack_8da4w.yaml`: ExecuTorch `export_llm` config targeting Arm CPUs (`model_class: "llama3_2"`, `quantization.qmode: "torchao:8da4w"`, `group_size: 64`, `embedding_quantize: "4,32"`, `use_sdpa_with_kv_cache: true`, `max_seq_length: 1024`, `max_context_length: 1024`).
- `ps1_offline_assistant/advisory/__init__.py`: Package exports for `AdvisoryEngine`, `build_llama3_prompt`, and pluggable backends.
- `ps1_offline_assistant/advisory/export_llama_pte.py`: Validates YAML config and Meta checkpoint artifacts (`consolidated.00.pth`, `params.json`, `tokenizer.model`), supports `--dry-run` command emission, invokes `executorch.extension.llm.export.export_llm` when installed, and records `.pte` size & SHA-256 in a manifest JSON.
- `ps1_offline_assistant/advisory/prompt_builder.py`: Official Meta Llama 3.2 Instruct header-token prompt formatter (`build_llama3_prompt`) grounding output strictly in `VisionFinding` and `kb_entry`, cultural-first IPM ordering, vernacular output (`en`, `hi`, `gu`, `mr`), and strict JSON keys.
- `ps1_offline_assistant/advisory/runtime.py`: Abstract `LlamaAdvisoryBackend` + `ExecuTorchLlamaCliBackend` (streaming C++ `llama_main` subprocess with `shell=False` and wall-clock TTFT/tok/s telemetry), `ExecuTorchPythonRuntimeBackend` (`executorch.runtime.Runtime` + `pytorch_tokenizers`), and CI/test-only `TestStubAdvisoryBackend`.
- `ps1_offline_assistant/advisory/engine.py`: `AdvisoryEngine` implementing pre-LLM deterministic short-circuits (`UNCERTAIN_IMAGE`, `LOW_CONFIDENCE` with KVK/KCC `1800-180-1551` referral and zero chemicals, healthy crop), resilient JSON extraction, post-LLM chemical grounding verification (catching unapproved active molecules, invented dosage rates, or chemical leaks into cultural steps), and mandatory `SEVERE_ESCALATION` (`escalate_to_kvk=True`).
- `ps1_offline_assistant/tests/test_phase3_advisory.py`: Phase 3 test suite (`8` tests covering export dry-run, prompt formatting, pre-LLM safety gates, multilingual & severe escalation, post-LLM hallucination fallbacks, and runtime wrappers).

### Honesty & Verification Scope
- **Environment**: `PyYAML` is installed and validated live; `executorch`, `pytorch_tokenizers`, `pytest`, and Llama 3.2 `.pte` weights are not installed in this local macOS sandbox.
- `ExecuTorchLlamaCliBackend` and `ExecuTorchPythonRuntimeBackend` cleanly raise `MissingLlamaRuntimeError` when missing artifacts/runtime; CLI subprocess streaming telemetry is verified against an isolated mock executable in `tempfile.TemporaryDirectory()`, and `AdvisoryEngine` policy tests use the explicitly labeled `TestStubAdvisoryBackend`.
- **Full Test Suite (`python3 -m unittest discover -s ps1_offline_assistant/tests -p "test_*.py" -v`)**: **26 / 26 tests passed** across Phase 1 (`10` tests), Phase 2 (`8` tests), and Phase 3 (`8` tests) with zero modifications to existing FieldWise V1/V2 files.

---

## 6. Phase 4 Integration Summary (Offline Multilingual i18n & Dual-Pipeline Orchestrator)

### Files Created under `ps1_offline_assistant/i18n/`, `orchestrator/`, and `tests/`
- `ps1_offline_assistant/i18n/__init__.py`: Package exports for offline UI localization.
- `ps1_offline_assistant/i18n/translations.py`: Offline localization dictionaries for `en` (English), `hi` (Hindi), `gu` (Gujarati), and `mr` (Marathi) covering UI headers, severity badges, status banners, KVK / Kisan Call Center (`1800-180-1551`) notices, all `16` `agri_kb.json` crops, and all `17` disease/pest cards, with `get_ui_bundle()` and `localize_status_message()` safely defaulting to `en`.
- `ps1_offline_assistant/orchestrator/__init__.py`: Package exports for orchestrator and scan history reader.
- `ps1_offline_assistant/orchestrator/pipeline.py`: Defines `PipelineMetadata`, `AssistantResult`, `FieldAssistantOrchestrator` (`analyze_image`, instant zero-vision-rerun `switch_language`), and offline JSONL/SQLite `read_field_history()` logging.
- `ps1_offline_assistant/orchestrator/cli.py`: Offline CLI entrypoint (`python3 -m ps1_offline_assistant.orchestrator.cli`) supporting `--image`, `--state`, `--lang`, `--vision-backend`, `--advisory-backend`, `--switch-lang`, `--log-path`, and `--json`.
- `ps1_offline_assistant/tests/test_phase4_orchestrator_i18n.py`: Phase 4 `unittest.TestCase` test suite (`6` tests).

### Final Verification Summary (Phases 1–4)
- **Full Test Suite (`python3 -m unittest discover -s ps1_offline_assistant/tests -p "test_*.py" -v`)**: **32 / 32 tests passed** (`10` Phase 1 + `8` Phase 2 + `8` Phase 3 + `6` Phase 4) with zero errors and zero skipped assertions.
- **Preservation Confirmation**: 0 modifications to existing FieldWise V1/V2 files (verified via SHA-256 digests).

---

## 7. Phase 5 Integration Summary (100% Offline Web App Interface)

### Files Created under `ps1_offline_assistant/app/` and `tests/`
- `ps1_offline_assistant/app/__init__.py`: Lazy exports for `OfflineAssistantHTTPServer` and `create_offline_server`.
- `ps1_offline_assistant/app/server.py`: Built-in `ThreadingHTTPServer` application exposing `GET /`, `GET /api/health`, `POST /api/analyze`, `POST /api/switch-language`, and `GET /api/history`, plus CLI launcher.
- `ps1_offline_assistant/app/static/index.html`: Zero-network, mobile-first offline frontend (`OFFLINE MODE • ON-DEVICE ARM`, instant camera preview, instant language switcher bar, detection & severity card, KVK `1800-180-1551` callout, grounded advisory panels, telemetry strip, and local history table).
- `ps1_offline_assistant/tests/test_phase5_app.py`: Localhost `127.0.0.1:0` integration and zero-network HTML audit tests.

### Final Verification Summary (Phases 1–5)
- **Zero-Network Audit**: Verified `index.html` contains zero `http://`, `https://`, or protocol-relative CDN URLs, and backend AST audit confirms zero cloud SDKs or outbound HTTP client imports.
- **Phases 1–4 (`test_phase[1-4]*.py`)**: **32 / 32 tests passed**.
- **Phase 5 (`test_phase5_app.py`)**: **3 / 3 tests passed**.
- **Combined Total**: **35 / 35 tests passed** across Phases 1–5 with 0 modifications to existing FieldWise V1/V2 files.

---

## 8. Phase 6 & Project Closure Summary (On-Device Benchmarking & PS1 Submission Docs)

### Files Created in Phase 6
- `ps1_offline_assistant/benchmarks/__init__.py` & `ps1_offline_assistant/benchmarks/run_benchmarks.py`: Reproducible CLI benchmark runner measuring vision latency (`mean`/`p50`/`p95`), advisory TTFT & tok/s, zero-vision-rerun language switching (`en -> hi -> gu -> mr`), peak RSS (`resource.getrusage`), disk footprint, and optional Raspberry Pi 5 thermal/PMIC hooks (`sysfs` + `vcgencmd`, returning `null` on macOS).
- `ps1_offline_assistant/benchmarks/results/benchmark_results.json` & `benchmark_summary.md`: Measured `Darwin-arm64` smoke benchmark (`vision_latency_ms` mean `39.254 ms`, `language_switch` mean `1.472 ms`, `peak_rss_mb = 317.531 MB`, `agri_kb.json = 59.5 KB`, `is_stub_advisory = true`).
- `ps1_offline_assistant/docs/TECHNICAL_REPORT.md`: Submission-ready PS1 technical report covering architecture, HailoRT vs. ExecuTorch Arm CPU split, datasets, safety guardrails, multilingual design, host smoke numbers, `[PENDING_TARGET_ARM_MEASUREMENT]` tables, and reproducibility commands.
- `ps1_offline_assistant/docs/DEMO_SCRIPT.md`: Timed 3–4 minute offline video demo script + pre-recording checklist.
- `ps1_offline_assistant/tests/test_phase6_benchmarks_docs.py`: Phase 6 unit tests (`4` tests).

### Complete `ps1_offline_assistant/` Directory Tree
```text
ps1_offline_assistant/
├── __init__.py
├── report.md
├── advisory/
│   ├── __init__.py
│   ├── engine.py
│   ├── export_llama_pte.py
│   ├── prompt_builder.py
│   └── runtime.py
├── app/
│   ├── __init__.py
│   ├── server.py
│   └── static/
│       └── index.html
├── benchmarks/
│   ├── __init__.py
│   ├── run_benchmarks.py
│   └── results/
│       ├── benchmark_results.json
│       └── benchmark_summary.md
├── configs/
│   └── llama3_2_1b_xnnpack_8da4w.yaml
├── data/
│   └── agri_kb.json
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEMO_SCRIPT.md
│   └── TECHNICAL_REPORT.md
├── i18n/
│   ├── __init__.py
│   └── translations.py
├── kb/
│   ├── __init__.py
│   └── retriever.py
├── orchestrator/
│   ├── __init__.py
│   ├── cli.py
│   └── pipeline.py
├── schemas/
│   ├── __init__.py
│   └── contracts.py
├── scripts/
│   ├── __init__.py
│   └── build_offline_kb.py
├── tests/
│   ├── __init__.py
│   ├── test_phase1_contracts_and_kb.py
│   ├── test_phase2_vision.py
│   ├── test_phase3_advisory.py
│   ├── test_phase4_orchestrator_i18n.py
│   ├── test_phase5_app.py
│   └── test_phase6_benchmarks_docs.py
└── vision/
    ├── __init__.py
    ├── inference.py
    ├── preprocess.py
    ├── severity.py
    └── train_and_export.py
```

### Final Verification & Target Arm Deployment Commands
- **All Phases 1–6 Test Pass Count**: **39 / 39 tests passed** (`10` Phase 1 + `8` Phase 2 + `8` Phase 3 + `6` Phase 4 + `3` Phase 5 + `4` Phase 6), zero errors, zero skipped assertions.
- **Preservation Confirmation**: **0 existing FieldWise V1/V2 files modified** (identical SHA-256 digests across `predictor.js`, `cropDatabase.js`, `fertilizerDatabase.js`, `package.json`, and all 15 `ml_pipeline/recommendation_engine/*.py` modules).

#### Next-Step Deployment Commands on Target Arm Device (Raspberry Pi 5 / Android Arm64)
1. **Train & export MobileNetV3 (`.pte` for Arm CPU or `.onnx` -> `.hef` via Hailo Dataflow Compiler)**:
   ```bash
   python3 ps1_offline_assistant/vision/train_and_export.py \
       --train-dir /path/to/plantvillage_ip102/train --val-dir /path/to/plantvillage_ip102/val \
       --output-dir ps1_offline_assistant/vision/artifacts --epochs 15 --export-onnx --export-executorch
   ```
2. **Export Llama 3.2 1B-Instruct `.pte` (`torchao:8da4w`, XNNPACK + KleidiAI)**:
   ```bash
   python3 ps1_offline_assistant/advisory/export_llama_pte.py \
       --checkpoint-dir /path/to/Llama-3.2-1B-Instruct
   ```
3. **Run On-Device Arm Benchmarks & Fill `[PENDING_TARGET_ARM_MEASUREMENT]`**:
   ```bash
   python3 -m ps1_offline_assistant.benchmarks.run_benchmarks \
       --runs 10 --warmup 2 \
       --vision-backend hailort --vision-model /path/to/mobilenet_v3_small.hef \
       --advisory-backend executorch-cli --llama-bin /path/to/llama_main \
       --llama-pte ps1_offline_assistant/advisory/artifacts/llama3_2_1b_xnnpack_8da4w.pte \
       --llama-tokenizer /path/to/Llama-3.2-1B-Instruct/tokenizer.model \
       --output-dir ps1_offline_assistant/benchmarks/results
   ```
4. **Launch Offline Field App (`http://127.0.0.1:8080`)**:
   ```bash
   python3 -m ps1_offline_assistant.app.server \
       --host 0.0.0.0 --port 8080 \
       --vision-backend hailort --vision-model /path/to/mobilenet_v3_small.hef \
       --advisory-backend executorch-cli --llama-bin /path/to/llama_main \
       --llama-pte ps1_offline_assistant/advisory/artifacts/llama3_2_1b_xnnpack_8da4w.pte \
       --llama-tokenizer /path/to/Llama-3.2-1B-Instruct/tokenizer.model
   ```

---

## 9. Phase 7 Upgrade Summary (V2 Crop Recommendation, Fertilizer Scaling & Pretrained PlantVillage Loader)

### Files Added & Updated in Phase 7
- `ps1_offline_assistant/requirements.txt` & `ps1_offline_assistant/.venv`: Project-scoped Python virtual environment (`--system-site-packages`) pinning `transformers==5.24.1`, `huggingface_hub==2.4.0`, and `safetensors==0.8.0` cleanly outside global site-packages.
- `ps1_offline_assistant/vision/download_pretrained.py`: One-time setup script auto-delegating to `.venv/bin/python` when needed, handling legacy `MobileNetV2FeatureExtractor` configs in `transformers>=5`, downloading `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` into `ps1_offline_assistant/vision/artifacts/pretrained_hf/`, exporting zero-`transformers` `plantvillage_mobilenetv2.pt2`, and compiling `kb_label_mapping.json` (`38` classes, `6` matched `agri_kb.json` cards).
- `ps1_offline_assistant/vision/inference.py` & `ps1_offline_assistant/vision/__init__.py`: Added `HuggingFaceLocalVisionBackend` loading strictly with `local_files_only=True` (plus zero-`transformers` `plantvillage_mobilenetv2.pt2` fallback when invoked from base `python3`), zero runtime network calls, clean `MissingBackendDependencyError` if local weights are missing, and `is_untrained_smoke` tracking.
- `ps1_offline_assistant/orchestrator/cli.py`: Added `--vision-backend hf-local` support.
- `ps1_offline_assistant/i18n/translations.py`: Added 100%-parity multilingual UI keys across `en`, `hi`, `gu`, and `mr` for Tabs 1, 2, 3 and untrained-weights warning banners.
- `ps1_offline_assistant/app/server.py`: Wired read-only FieldWise V2 `CropRecommendationEngine` (`POST /api/recommend-crop`), `agri_kb.json` area-scaled fertilizer advisory (`GET /api/fertilizer-guide`), `GET /api/ui-bundle`, `vision_weights_trained` health status, and `--vision-backend hf-local`.
- `ps1_offline_assistant/app/static/index.html`: Upgraded to a 3-tab zero-network mobile-first UI (**Disease & Pest Scan**, **Crop Recommendation (V2)**, **Fertilizer & Irrigation Advisory**) with global 4-language switching and untrained-smoke warning banner.
- `ps1_offline_assistant/benchmarks/run_benchmarks.py` & `benchmarks/results/`: Updated to support `--vision-backend hf-local` and regenerated against the real cached 38-class MobileNetV2 PlantVillage `.pt2` model (`25.094 ms` mean vision latency, `1.402 ms` mean language switch, `19.174 MB` pretrained vision artifacts, `is_stub_advisory=true` on macOS).
- `ps1_offline_assistant/i18n/translations.py`, `orchestrator/pipeline.py`, `app/server.py`, & `app/static/index.html`: Added `symptoms`, `regional_advisory`, `crop_info`, PlantVillage crop localizations across `en`/`hi`/`gu`/`mr`, loading status indicator, and honest `advisoryModeBanner` + platform `/api/health` fields.

### Final Verification Summary (Unified Submission Check)
- **PS1 Offline Edge-AI Suite (`ps1_offline_assistant/tests`)**: **45 / 45 tests passed**.
- **FieldWise V2 Recommendation Engine (`ml_pipeline/recommendation_engine/tests`)**: **60 / 62** passed in `run_all_tests.py` + **16 / 17** passed in `test_adapter_cli.py` (`3` failures caused by missing historical `ml_pipeline/reports/*_checksums.txt` files removed from disk outside the session; all `15` `ml_pipeline/recommendation_engine/*.py` files remain unmodified with verified SHA-256 checksums).
- **Root Unified Entrypoint (`app.py` + `README.md`)**: Binds V1 JS sources, V2 Crop Engine, and PS1 Edge-AI assistant cleanly; zero modifications to frozen V1/V2 production files.
