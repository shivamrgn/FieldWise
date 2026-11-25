# FieldWise PS1 — Offline Demo Video Script & Recording Checklist (3–4 Minutes)

**Target Duration**: 3 min 30 sec (max 4 min 00 sec)  
**Goal**: Demonstrate 100% offline, end-to-end Arm edge execution (Pipeline A Vision -> Local KB -> Pipeline B Llama 3.2 Advisory) with instant multilingual switching and safety/KVK escalation guardrails.

---

## Pre-Recording Checklist

- [ ] **Airplane Mode / Network Disabled**: Turn Wi-Fi OFF, unplug Ethernet cable, and confirm OS network status icon shows disconnected / Airplane Mode.
- [ ] **Server Running Locally**: Start `python3 app.py serve --port 8080` (or `python3 -m ps1_offline_assistant.app.server --vision-backend hf-local --advisory-backend executorch-cli ...` on Arm with Llama `.pte`).
- [ ] **Prepare 3 Test Images** on device storage (or camera subjects):
  1. Sharp diseased crop leaf (e.g., Tomato Early Blight or Potato Late Blight, moderate ~20–30% lesion coverage).
  2. Severely blurred or poorly exposed leaf image.
  3. Severe diseased crop leaf (> 45% lesion coverage).
- [ ] **Verify Clean Field History Log**: Optionally clear or inspect local history before recording so the history table updates cleanly live.
- [ ] **Browser DevTools Network Tab (Optional Bonus)**: Open DevTools -> Network tab to prove zero external requests and 100% `127.0.0.1` traffic.

---

## Timed Demo Script

### Segment 1: Offline Proof & Edge Backend Verification (`0:00 – 0:40`)
- **Action**:
  1. Show system toolbar / terminal confirming **Wi-Fi and Ethernet are completely disabled** (or Airplane Mode ON).
  2. Open browser to `http://127.0.0.1:8080`.
  3. Point cursor to the green top bar badge **`OFFLINE MODE`** and live backend pills (`Vision: ...` | `Advisory: ...`), plus the honest advisory mode notice banner if running without local Llama `.pte` weights.
- **Voiceover**:
  > "Welcome to FieldWise PS1, our 100% offline crop disease and pest assistant built for the Bharat AI-SoC Challenge. Notice our device has Wi-Fi and Ethernet completely disabled. Every component—computer vision, severity measurement, agricultural knowledge base, and Llama 3.2 advisory—runs locally on Arm edge hardware at `127.0.0.1:8080`."

### Segment 2: Crop Image Capture, Disease Detection & Severity (`0:40 – 1:30`)
- **Action**:
  1. Ensure the **📷 Crop Disease Scan** tab is active (alongside **🌱 Crop Recommendation** and **💧 Fertilizer Guide** tabs), and select **State / Region** (e.g., `Punjab`) and **Language** (`English`).
  2. Click or drag-and-drop a photo into the **Tap to Upload or Capture Photo** zone to load the sharp moderate diseased leaf image -> show instant local preview.
  3. Click **🔍 Analyze Crop**.
  4. Highlight the diagnosis card: **Crop**, **Detected Condition**, **Category**, **Model Confidence** bar, **Estimated Infected Area %**, and color-coded **Moderate** severity badge alongside the analyzed photo.
- **Voiceover**:
  > "Pipeline A first checks image sharpness and lighting using OpenCV, runs our compact vision classifier (38-class MobileNetV2 PlantVillage locally / MobileNetV3 on custom edge exports), and computes exact infected leaf area percentage in HSV and CIE Lab color space. Here it detects Tomato Early Blight with real model confidence, severity badge, KB symptoms, and estimated infected area."

### Segment 3: Grounded ICAR/KVK Advisory & Telemetry (`1:30 – 2:15`)
- **Action**:
  1. Scroll through the grounded advisory panels: **Assessment Summary**, **Immediate Actions**, **Organic & Cultural Control** (shown before chemicals), **Approved Chemical Guidance** (with safety warning), **Preventive Measures**, and **Safety Disclaimer**.
  2. Highlight the **On-Device Telemetry Strip** (`Vision Latency`, `Advisory TTFT`, `Decode Speed tok/s`, `Total Latency`).
- **Voiceover**:
  > "Pipeline B retrieves the matched ICAR/DPPQS treatment card from our local `agri_kb.json` and grounds Llama 3.2 via ExecuTorch. Notice cultural and organic steps always come first, approved chemicals include CIB&RC safety warnings with zero invented dosages, and all inference latencies are tracked locally."

### Segment 4: Instant Zero-Vision-Rerun Multilingual Switching (`2:15 – 2:55`)
- **Action**:
  1. Click **`हिन्दी`** -> show immediate UI headers, severity badge, and advisory switch to Hindi.
  2. Click **`ગુજરાતી`** -> show Gujarati UI and text.
  3. Click **`मराठी`** -> show Marathi UI and text.
  4. Point to the low switching latency and confirm vision did not rerun.
- **Voiceover**:
  > "For farmers across India, our instant Language Switcher toggles between English, Hindi, Gujarati, and Marathi completely offline. Because our dual-pipeline orchestrator caches the verified `VisionFinding`, switching languages only re-runs Pipeline B and localization without repeating vision inference."

### Segment 5: Uncertainty Gating, Severe KVK Escalation & History Log (`2:55 – 3:40`)
- **Action**:
  1. Upload the **blurry image** and click **Analyze Crop** -> show `UNCERTAIN_IMAGE` yellow warning banner, retake-photo guidance, and **zero chemical advice**.
  2. Upload the **severe leaf image** and click **Analyze Crop** -> show red `SEVERE_ESCALATION` banner and prominent **Krishi Vigyan Kendra (KVK) / Kisan Call Center (`1800-180-1551`)** escalation callout.
  3. Expand **Local Field Scan History (Offline Log)** to show persisted offline scan records.
- **Voiceover**:
  > "Safety comes first: blurry photos trigger `UNCERTAIN_IMAGE` and withhold all pesticide advice, while severe infestations trigger `SEVERE_ESCALATION` directing the farmer to their local Krishi Vigyan Kendra or the `1800-180-1551` helpline. Every scan is saved locally for extension workers—100% offline from capture to guidance."
