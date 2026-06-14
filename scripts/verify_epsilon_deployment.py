from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def fetch_json(url: str) -> tuple[dict, str | None]:
    try:
        with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=10) as response:
            return json.loads(response.read().decode("utf-8")), None
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {}, str(exc)


def fetch_laravel_status_via_artisan() -> tuple[dict, str | None]:
    laravel_root = WORKSPACE_ROOT / "ReaDirect"
    command = [
        "php",
        "artisan",
        "tinker",
        "--execute=echo json_encode(app(\\App\\Services\\AI\\ReadirectAIService::class)->dashboardStatus());",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=laravel_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        output = result.stdout.strip()
        start = output.find("{")
        end = output.rfind("}")
        if start < 0 or end < start:
            return {}, f"Artisan did not return JSON: {output or result.stderr.strip()}"
        return json.loads(output[start : end + 1]), None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {}, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Epsilon deployment wiring.")
    parser.add_argument("--fastapi-url", default="http://127.0.0.1:8001")
    parser.add_argument("--laravel-url", default="http://127.0.0.1:8000")
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()

    model_path = Path(os.getenv("ASR_MODEL_PATH", PROJECT_ROOT / "models/asr/epsilon")).resolve()
    lm_path = Path(
        os.getenv(
            "ASR_LM_PATH",
            PROJECT_ROOT / "external_datasets/language_models/3-gram.pruned.1e-7.arpa",
        )
    ).resolve()
    required = ("config.json", "model.safetensors", "processor_config.json", "vocab.json")
    checks = {
        "model_path": str(model_path),
        "model_exists": model_path.is_dir(),
        "required_model_files": {
            name: (model_path / name).exists() for name in required
        },
        "lm_path": str(lm_path),
        "lm_exists": lm_path.is_file(),
        "startup_script_mentions_epsilon": "ASR_MODEL_NAME = \"epsilon\"" in (
            PROJECT_ROOT / "scripts/start_ai_service.ps1"
        ).read_text(encoding="utf-8"),
        "laravel_status_banner_mentions_decoder": "Decoder Backend" in (
            WORKSPACE_ROOT / "ReaDirect/resources/js/Components/AIServiceStatusBanner.vue"
        ).read_text(encoding="utf-8"),
        "laravel_ai_status_route_exists": "ai-status" in (
            WORKSPACE_ROOT / "ReaDirect/routes/web.php"
        ).read_text(encoding="utf-8"),
    }
    errors = []
    if not checks["model_exists"] or not all(checks["required_model_files"].values()):
        errors.append("Epsilon model folder is missing required runtime files.")
    if os.getenv("ASR_DECODE_MODE", "beam_lm") == "beam_lm" and not checks["lm_exists"]:
        errors.append("beam_lm is configured but the KenLM file is missing.")

    live = {}
    if not args.skip_live:
        health, health_error = fetch_json(f"{args.fastapi_url.rstrip('/')}/health")
        live["fastapi_health"] = health
        live["fastapi_error"] = health_error
        if health_error:
            errors.append(f"FastAPI health unavailable: {health_error}")
        else:
            expected = {
                "asr_model_name": "epsilon",
                "asr_model_loaded": True,
                "processor_loaded": True,
                "decode_mode": "beam_lm",
                "beam_search_enabled": True,
                "language_model_loaded": True,
                "decoder_backend": "pyctcdecode_with_lm",
            }
            for key, value in expected.items():
                if health.get(key) != value:
                    errors.append(
                        f"FastAPI health {key} expected {value!r}, got {health.get(key)!r}"
                    )

        admin, admin_error = fetch_json(f"{args.laravel_url.rstrip('/')}/admin/ai-status")
        if admin_error:
            admin, artisan_error = fetch_laravel_status_via_artisan()
            if artisan_error:
                admin_error = f"{admin_error}; Artisan fallback failed: {artisan_error}"
            else:
                admin_error = None
                live["admin_status_source"] = "artisan_service_fallback"
        live["admin_ai_status"] = admin
        live["admin_error"] = admin_error
        if admin_error:
            errors.append(f"Laravel admin AI status unavailable: {admin_error}")
        elif admin.get("asr_model_name") != "epsilon":
            errors.append("Laravel admin AI status does not report Epsilon.")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "healthy" if not errors else "error",
        "checks": checks,
        "live": live,
        "errors": errors,
    }
    output = PROJECT_ROOT / "reports/asr/deployment/epsilon_wiring_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved Epsilon wiring report to {output}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
