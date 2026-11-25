# FieldWise PS1 On-Device Benchmark Summary

## Platform & Configuration

| Field | Value |
| :--- | :--- |
| Host Platform | `Darwin-arm64` (`Darwin` 25.6.0) |
| CPU Architecture | `arm64` |
| Python Version | `3.13.5` |
| Vision Backend | `huggingface_local_offline` (linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification (38-class MobileNetV2 PlantVillage @ ps1_offline_assistant/vision/artifacts/pretrained_hf)) |
| Advisory Backend | `test_stub_ci_only` |
| Is Stub Advisory | `True` |
| Warmup / Timed Runs | `1` / `5` |
| Input Image Source | `synthetic_320x320_leaf_25pct_lesion` |

> **Honesty Notice**: `is_stub_advisory=true` (`test_stub_ci_only`). Advisory TTFT and tokens/sec reflect local host Python/stub harness timings, NOT real Llama 3.2 `.pte` quantized inference on Arm CPU.

## Pipeline Latency & Throughput

| Metric | Mean | P50 | P95 |
| :--- | ---: | ---: | ---: |
| Vision Latency (ms) | 32.813 | 32.546 | 33.585 |
| Advisory TTFT (ms) | 0.000 | 0.000 | 0.000 |
| Advisory Speed (tokens/s) | 0.000 | 0.000 | 0.000 |
| End-to-End Latency (ms) | 32.963 | 32.690 | 33.763 |

## Instant Language Switch Latency (`en -> hi -> gu -> mr`)

- **Vision Re-Executed**: `False` (zero Pipeline A rerun)
- **Mean Switch Latency**: `1.916 ms` (P50 `1.515 ms`, P95 `2.654 ms`)

| Transition | Latency (ms) |
| :--- | ---: |
| `en->hi` | 2.780 |
| `hi->gu` | 1.515 |
| `gu->mr` | 1.452 |

## Memory, Disk Footprint & Arm Thermal Telemetry

| Resource | Measured Value |
| :--- | :--- |
| Peak Process RSS | `360.703 MB` |
| `agri_kb.json` Footprint | `59.5 KB` (60926 bytes) |
| Vision Model Size | `19.174 MB` |
| Llama `.pte` Size | `None MB` |
| Total Measured Artifact Footprint | `19.232 MB` |
| Linux Thermal Zone (`temp_c`) | `None` |
| RPi `vcgencmd` Temp (`temp_c`) | `None` |
