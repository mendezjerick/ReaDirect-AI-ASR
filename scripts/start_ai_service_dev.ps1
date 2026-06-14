$ErrorActionPreference = "Stop"
$Repo = (Get-Location).Path
$env:ASR_MODEL_NAME = "epsilon"
$env:ASR_MODEL_PATH = Join-Path $Repo "models\asr\epsilon"
$env:WAV2VEC2_ASR_MODEL_PATH = $env:ASR_MODEL_PATH
$env:ASR_MODEL_SIZE = $env:ASR_MODEL_PATH
$env:ASR_DECODE_MODE = "beam_lm"
$env:ASR_BEAM_WIDTH = "100"
$env:ASR_ALPHA = "0.5"
$env:ASR_BETA = "1.0"
$env:ASR_LM_PATH = Join-Path $Repo "external_datasets\language_models\3-gram.pruned.1e-7.arpa"
$env:ASR_HOTWORDS = ""
$env:ASR_HOTWORD_WEIGHT = "5.0"
$env:ASR_ALLOW_NO_LM_FALLBACK = "false"
$env:ASR_DEVICE = if ($env:ASR_DEVICE) { $env:ASR_DEVICE } else { "cuda" }
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Using local .venv"
    . .venv\Scripts\Activate.ps1
} else {
    Write-Warning ".venv not found; using current Python environment"
}
python -c "import sys; print('Python:', sys.executable)"
python scripts/validate_ai_service_startup.py
uvicorn api.main:app --reload --host 127.0.0.1 --port 8001
