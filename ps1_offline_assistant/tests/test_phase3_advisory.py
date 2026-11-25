"""
Phase 3 Verification Suite: Pipeline B (Llama 3.2 + ExecuTorch Advisory Pipeline).

Verifies:
  1. `export_llama_pte.py --dry-run` config validation, checkpoint validation,
     reproducible `export_llm` command construction, and artifact manifest generation.
  2. Official Llama 3.2 Instruct chat prompt construction (`build_llama3_prompt`).
  3. Pre-LLM safety short-circuits (`UNCERTAIN_IMAGE`, `LOW_CONFIDENCE`, healthy).
  4. `SEVERE_ESCALATION` mandatory KVK trigger regardless of LLM text.
  5. Post-LLM guardrails rejecting malformed JSON, unapproved pesticides,
     invented dilution rates, and chemical leaks into cultural steps.
  6. `ExecuTorchLlamaCliBackend` and `ExecuTorchPythonRuntimeBackend` missing
     dependency guards plus subprocess CLI streaming telemetry verification.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_PS1_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PS1_ROOT.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from ps1_offline_assistant.advisory.engine import AdvisoryEngine  # noqa: E402
from ps1_offline_assistant.advisory.export_llama_pte import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    MissingLlamaExportDependencyError,
    run_export,
    validate_export_config,
    write_artifact_manifest,
)
from ps1_offline_assistant.advisory.prompt_builder import build_llama3_prompt  # noqa: E402
from ps1_offline_assistant.advisory.runtime import (  # noqa: E402
    ExecuTorchLlamaCliBackend,
    ExecuTorchPythonRuntimeBackend,
    MissingLlamaRuntimeError,
    TestStubAdvisoryBackend,
)
from ps1_offline_assistant.kb.retriever import OfflineKBRetriever  # noqa: E402
from ps1_offline_assistant.schemas.contracts import (  # noqa: E402
    AdvisoryRequest,
    AdvisoryResponse,
    AdvisoryStatus,
    SeverityLevel,
    SupportedLanguage,
    VisionCategory,
    VisionFinding,
)


def _make_finding(**overrides) -> VisionFinding:
    payload = {
        "crop": "rice",
        "condition_name": "Rice Blast",
        "category": VisionCategory.DISEASE,
        "confidence": 0.92,
        "severity_level": SeverityLevel.MODERATE,
        "infected_area_pct": 24.0,
        "image_quality_ok": True,
        "blur_score": 180.0,
        "inference_latency_ms": 19.5,
        "runtime_backend": "hailort_npu",
    }
    payload.update(overrides)
    return VisionFinding(**payload)


class TestLlamaExportConfigAndDriver(unittest.TestCase):
    def test_yaml_config_matches_arm_xnnpack_spec(self) -> None:
        cfg = validate_export_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(cfg["base"]["model_class"], "llama3_2")
        self.assertEqual(cfg["quantization"]["qmode"], "torchao:8da4w")
        self.assertEqual(cfg["quantization"]["group_size"], 64)
        self.assertEqual(cfg["quantization"]["embedding_quantize"], "4,32")
        self.assertTrue(cfg["model"]["use_sdpa_with_kv_cache"])
        self.assertEqual(cfg["model"]["max_seq_length"], 1024)
        self.assertEqual(cfg["model"]["max_context_length"], 1024)

    def test_dry_run_cli_and_manifest_creation(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(_PS1_ROOT / "advisory" / "export_llama_pte.py"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        out = json.loads(proc.stdout)
        self.assertTrue(out["dry_run"])
        self.assertTrue(out["config_valid"])
        self.assertIn("executorch.extension.llm.export.export_llm", out["command_str"])

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "llama3_2_1b"
            ckpt_dir.mkdir()
            (ckpt_dir / "consolidated.00.pth").write_bytes(b"weights-bytes")
            (ckpt_dir / "params.json").write_text('{"dim": 2048, "n_layers": 16}', encoding="utf-8")
            (ckpt_dir / "tokenizer.model").write_bytes(b"tokenizer-bytes")

            dry_with_ckpt = run_export(
                config_path=DEFAULT_CONFIG_PATH,
                checkpoint_dir=ckpt_dir,
                dry_run=True,
            )
            self.assertTrue(dry_with_ckpt["checkpoint_validated"])
            self.assertIn(str((ckpt_dir / "consolidated.00.pth").resolve()), dry_with_ckpt["command_str"])

            if (
                importlib.util.find_spec("executorch") is None
                or importlib.util.find_spec("executorch.extension.llm.export.export_llm") is None
            ):
                with self.assertRaises(MissingLlamaExportDependencyError):
                    run_export(
                        config_path=DEFAULT_CONFIG_PATH,
                        checkpoint_dir=ckpt_dir,
                        dry_run=False,
                    )

            # Verify artifact manifest recording (size + SHA-256)
            fake_pte = Path(tmpdir) / "llama3_2_1b_xnnpack_8da4w.pte"
            fake_pte.write_bytes(b"dummy-pte-payload-for-manifest-test")
            manifest = write_artifact_manifest(
                pte_path=fake_pte,
                config_path=DEFAULT_CONFIG_PATH,
                checkpoint_paths={
                    "consolidated.00.pth": ckpt_dir / "consolidated.00.pth",
                    "params.json": ckpt_dir / "params.json",
                    "tokenizer.model": ckpt_dir / "tokenizer.model",
                },
            )
            self.assertEqual(manifest["pte_size_bytes"], fake_pte.stat().st_size)
            self.assertEqual(len(manifest["pte_sha256"]), 64)
            self.assertEqual(manifest["model_class"], "llama3_2")


class TestPromptBuilder(unittest.TestCase):
    def test_build_llama3_prompt_structure_and_grounding(self) -> None:
        retriever = OfflineKBRetriever()
        card = retriever.get_disease_pest_card("Rice Blast", crop="rice")
        self.assertIsNotNone(card)

        req = AdvisoryRequest(
            vision_finding=_make_finding(),
            state_region="Punjab",
            language=SupportedLanguage.HI,
            kb_entry=card,
        )
        prompt = build_llama3_prompt(req)

        for token in (
            "<|begin_of_text|>",
            "<|start_header_id|>system<|end_header_id|>",
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|eot_id|>",
        ):
            self.assertIn(token, prompt)

        self.assertIn("Punjab", prompt)
        self.assertIn("Hindi", prompt)
        self.assertIn("Tricyclazole 75% WP", prompt)
        self.assertIn("cultural", prompt.lower())
        self.assertIn("NEVER invent", prompt)

        # kb_entry=None case explicitly instructs no treatment invention
        req_none = AdvisoryRequest(
            vision_finding=_make_finding(category=VisionCategory.UNKNOWN, confidence=0.4),
            state_region="Gujarat",
            language=SupportedLanguage.GU,
            kb_entry=None,
        )
        prompt_none = build_llama3_prompt(req_none)
        self.assertIn("NO APPROVED TREATMENT CARD IN LOCAL KB", prompt_none)


class TestAdvisoryEngineSafetyAndGuardrails(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = OfflineKBRetriever()
        self.stub_backend = TestStubAdvisoryBackend()
        self.engine = AdvisoryEngine(backend=self.stub_backend, kb_retriever=self.retriever)

    def test_pre_llm_short_circuits_blurry_low_conf_and_healthy(self) -> None:
        # 1. Blurry image -> UNCERTAIN_IMAGE, escalate_to_kvk=False, zero chemicals
        blurry_req = AdvisoryRequest(
            vision_finding=_make_finding(
                crop="unknown",
                condition_name="unclear_image",
                category=VisionCategory.UNKNOWN,
                confidence=0.0,
                severity_level=SeverityLevel.UNDETERMINED,
                infected_area_pct=0.0,
                image_quality_ok=False,
                blur_score=3.2,
            ),
            state_region="Maharashtra",
            language=SupportedLanguage.MR,
            kb_entry=None,
        )
        blurry_resp = self.engine.advise(blurry_req)
        self.assertEqual(blurry_resp.status, AdvisoryStatus.UNCERTAIN_IMAGE)
        self.assertFalse(blurry_resp.escalate_to_kvk)
        self.assertEqual(blurry_resp.chemical_guidance_if_approved, [])
        self.assertEqual(blurry_resp.language, SupportedLanguage.MR)
        self.assertEqual(blurry_resp.ttft_ms, 0.0)

        # 2. Low-confidence (< 0.65) or unknown -> LOW_CONFIDENCE, escalate_to_kvk=True, KCC helpline
        low_req = AdvisoryRequest(
            vision_finding=_make_finding(
                category=VisionCategory.UNKNOWN,
                confidence=0.52,
                severity_level=SeverityLevel.UNDETERMINED,
                infected_area_pct=0.0,
            ),
            state_region="Gujarat",
            language=SupportedLanguage.GU,
            kb_entry=None,
        )
        low_resp = self.engine.advise(low_req)
        self.assertEqual(low_resp.status, AdvisoryStatus.LOW_CONFIDENCE)
        self.assertTrue(low_resp.escalate_to_kvk)
        self.assertEqual(low_resp.chemical_guidance_if_approved, [])
        self.assertIn("1800-180-1551", " ".join(low_resp.immediate_actions))

        # 3. Unmapped disease (kb_entry=None) -> LOW_CONFIDENCE, escalate_to_kvk=True
        unmapped_req = AdvisoryRequest(
            vision_finding=_make_finding(
                crop="apple",
                condition_name="Apple Scab",
                category=VisionCategory.DISEASE,
                confidence=0.90,
            ),
            state_region="Himachal Pradesh",
            language=SupportedLanguage.EN,
            kb_entry=None,
        )
        unmapped_resp = self.engine.advise(unmapped_req)
        self.assertEqual(unmapped_resp.status, AdvisoryStatus.LOW_CONFIDENCE)
        self.assertTrue(unmapped_resp.escalate_to_kvk)
        self.assertEqual(unmapped_resp.chemical_guidance_if_approved, [])

        # 4. Healthy crop -> SUCCESS, escalate_to_kvk=False, zero chemicals
        healthy_req = AdvisoryRequest(
            vision_finding=_make_finding(
                crop="tomato",
                condition_name="Healthy",
                category=VisionCategory.HEALTHY,
                confidence=0.96,
                severity_level=SeverityLevel.LOW,
                infected_area_pct=0.0,
            ),
            state_region="Karnataka",
            language=SupportedLanguage.HI,
            kb_entry=None,
        )
        healthy_resp = self.engine.advise(healthy_req)
        self.assertEqual(healthy_resp.status, AdvisoryStatus.SUCCESS)
        self.assertFalse(healthy_resp.escalate_to_kvk)
        self.assertEqual(healthy_resp.chemical_guidance_if_approved, [])

    def test_multilingual_grounded_success_and_severe_escalation(self) -> None:
        # Auto-resolves kb_entry via OfflineKBRetriever when kb_entry=None
        for lang in (
            SupportedLanguage.EN,
            SupportedLanguage.HI,
            SupportedLanguage.GU,
            SupportedLanguage.MR,
        ):
            req = AdvisoryRequest(
                vision_finding=_make_finding(severity_level=SeverityLevel.MODERATE),
                state_region="Punjab",
                language=lang,
                kb_entry=None,
            )
            resp = self.engine.advise(req)
            self.assertEqual(resp.status, AdvisoryStatus.SUCCESS)
            self.assertFalse(resp.escalate_to_kvk)
            self.assertEqual(resp.language, lang)
            self.assertGreater(len(resp.organic_cultural_steps), 0)
            self.assertGreater(len(resp.chemical_guidance_if_approved), 0)
            self.assertIn("Tricyclazole", resp.chemical_guidance_if_approved[0])
            # Strictly round-trips as JSON
            AdvisoryResponse.model_validate_json(resp.model_dump_json())

        # Severe condition forces SEVERE_ESCALATION + escalate_to_kvk=True
        severe_req = AdvisoryRequest(
            vision_finding=_make_finding(
                severity_level=SeverityLevel.SEVERE,
                infected_area_pct=58.0,
            ),
            state_region="Punjab",
            language=SupportedLanguage.EN,
            kb_entry=None,
        )
        severe_resp = self.engine.advise(severe_req)
        self.assertEqual(severe_resp.status, AdvisoryStatus.SEVERE_ESCALATION)
        self.assertTrue(severe_resp.escalate_to_kvk)

    def test_post_llm_guardrails_for_malformed_json_and_hallucinated_chemicals(self) -> None:
        card = self.retriever.get_disease_pest_card("rice_blast", crop="rice")
        req = AdvisoryRequest(
            vision_finding=_make_finding(severity_level=SeverityLevel.SEVERE, infected_area_pct=52.0),
            state_region="Punjab",
            language=SupportedLanguage.EN,
            kb_entry=card,
        )

        # 1. Markdown-fenced valid grounded JSON passes cleanly and keeps SEVERE_ESCALATION
        fenced_valid = json.dumps(
            {
                "explanation": "Rice Blast suspected at severe level.",
                "immediate_actions": ["Avoid extra urea top-dressing."],
                "organic_cultural_steps": ["Remove bund weeds and avoid water stress."],
                "chemical_guidance_if_approved": [
                    "Tricyclazole 75% WP: verify exact CIB&RC label dilution and PHI with local KVK."
                ],
                "preventive_measures": ["Monitor neck blast at panicle emergence."],
                "disclaimer": "Offline AI screening only; confirm with local KVK.",
            }
        )
        fenced_engine = AdvisoryEngine(
            TestStubAdvisoryBackend(override_raw_output=f"```json\n{fenced_valid}\n```"),
            self.retriever,
        )
        resp_fenced = fenced_engine.advise(req)
        self.assertEqual(resp_fenced.status, AdvisoryStatus.SEVERE_ESCALATION)
        self.assertTrue(resp_fenced.escalate_to_kvk)
        self.assertEqual(resp_fenced.explanation, "Rice Blast suspected at severe level.")

        # 2. Malformed non-JSON output triggers safe grounded KB fallback
        bad_json_engine = AdvisoryEngine(
            TestStubAdvisoryBackend(override_raw_output="Sorry, I cannot output JSON {broken"),
            self.retriever,
        )
        resp_bad_json = bad_json_engine.advise(req)
        self.assertEqual(resp_bad_json.status, AdvisoryStatus.SEVERE_ESCALATION)
        self.assertTrue(resp_bad_json.escalate_to_kvk)
        self.assertIn("Tricyclazole", resp_bad_json.chemical_guidance_if_approved[0])
        self.assertIn("fallback", resp_bad_json.disclaimer.lower())

        # 3. Hallucinated unapproved pesticide (e.g. Monocrotophos / Carbendazim / invented ml/L) rejected
        hallucinated_payloads = [
            {
                "explanation": "Rice Blast detected.",
                "immediate_actions": ["Scout field."],
                "organic_cultural_steps": ["Sanitize bunds."],
                "chemical_guidance_if_approved": ["Spray Monocrotophos 36% SL immediately."],
                "preventive_measures": ["Use resistant seed."],
                "disclaimer": "Consult KVK.",
            },
            {
                "explanation": "Rice Blast detected.",
                "immediate_actions": ["Scout field."],
                "organic_cultural_steps": ["Spray Carbendazim 50% WP as organic step."],
                "chemical_guidance_if_approved": ["Tricyclazole 75% WP (verify with KVK)."],
                "preventive_measures": ["Use resistant seed."],
                "disclaimer": "Consult KVK.",
            },
            {
                "explanation": "Rice Blast detected.",
                "immediate_actions": ["Scout field."],
                "organic_cultural_steps": ["Sanitize bunds."],
                "chemical_guidance_if_approved": ["Mix Tricyclazole 75% WP at 5 ml/L and spray."],
                "preventive_measures": ["Use resistant seed."],
                "disclaimer": "Consult KVK.",
            },
        ]
        for bad_payload in hallucinated_payloads:
            hall_engine = AdvisoryEngine(
                TestStubAdvisoryBackend(override_raw_output=json.dumps(bad_payload)),
                self.retriever,
            )
            resp_hall = hall_engine.advise(req)
            self.assertEqual(resp_hall.status, AdvisoryStatus.SEVERE_ESCALATION)
            self.assertTrue(resp_hall.escalate_to_kvk)
            combined_out = " ".join(
                [
                    resp_hall.explanation,
                    *resp_hall.organic_cultural_steps,
                    *resp_hall.chemical_guidance_if_approved,
                ]
            ).lower()
            self.assertNotIn("monocrotophos", combined_out)
            self.assertNotIn("carbendazim", combined_out)
            self.assertNotIn("5 ml/l", combined_out)
            self.assertIn("fallback", resp_hall.disclaimer.lower())


class TestExecuTorchRuntimeWrappers(unittest.TestCase):
    def test_missing_runtime_errors_raised_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_bin = Path(tmpdir) / "missing_llama_main"
            fake_pte = Path(tmpdir) / "missing.pte"
            fake_tok = Path(tmpdir) / "missing_tokenizer.model"

            with self.assertRaises(MissingLlamaRuntimeError):
                ExecuTorchLlamaCliBackend(fake_bin, fake_pte, fake_tok)

            with self.assertRaises(MissingLlamaRuntimeError):
                ExecuTorchPythonRuntimeBackend(fake_pte, fake_tok)

            # Create files so Python runtime checks `executorch` import availability
            fake_pte.write_bytes(b"pte")
            fake_tok.write_bytes(b"tok")
            if importlib.util.find_spec("executorch") is None:
                with self.assertRaises(MissingLlamaRuntimeError):
                    ExecuTorchPythonRuntimeBackend(fake_pte, fake_tok)

    def test_cli_backend_subprocess_streaming_and_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mock_cli = tmp / "mock_llama_main.py"
            mock_cli.write_text(
                f"#!{sys.executable}\n"
                "import json, sys, time\n"
                "time.sleep(0.01)\n"
                "payload = {\n"
                "  'explanation': 'Rice Blast spotted via CLI runner.',\n"
                "  'immediate_actions': ['Avoid excess urea top-dressing.'],\n"
                "  'organic_cultural_steps': ['Maintain bund sanitation.'],\n"
                "  'chemical_guidance_if_approved': ['Tricyclazole 75% WP: verify label dose with KVK.'],\n"
                "  'preventive_measures': ['Monitor neck blast closely.'],\n"
                "  'disclaimer': 'Offline CLI test disclaimer.'\n"
                "}\n"
                "print(json.dumps(payload), flush=True)\n",
                encoding="utf-8",
            )
            os.chmod(mock_cli, os.stat(mock_cli).st_mode | stat.S_IXUSR)

            pte_file = tmp / "model.pte"
            tok_file = tmp / "tokenizer.model"
            pte_file.write_bytes(b"pte")
            tok_file.write_bytes(b"tok")

            cli_backend = ExecuTorchLlamaCliBackend(
                binary_path=mock_cli,
                pte_path=pte_file,
                tokenizer_path=tok_file,
                timeout_sec=10.0,
            )
            engine = AdvisoryEngine(cli_backend, OfflineKBRetriever())
            resp = engine.advise(
                AdvisoryRequest(
                    vision_finding=_make_finding(),
                    state_region="Punjab",
                    language=SupportedLanguage.EN,
                    kb_entry=None,
                )
            )
            self.assertEqual(resp.status, AdvisoryStatus.SUCCESS)
            self.assertGreater(resp.ttft_ms, 0.0)
            self.assertGreater(resp.tokens_per_sec, 0.0)
            self.assertIn("Tricyclazole", resp.chemical_guidance_if_approved[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
