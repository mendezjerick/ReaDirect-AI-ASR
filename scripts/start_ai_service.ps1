$ErrorActionPreference = "Stop"

$Python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
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

if (-not (Test-Path -LiteralPath $Python)) {
    throw "AI service virtual environment was not found at $Python. Create the .venv before starting ASR."
}

Write-Host "Using local .venv"
Write-Host "Active ASR model: $env:ASR_MODEL_NAME ($env:ASR_MODEL_PATH)"
Write-Host "Decoder: $env:ASR_DECODE_MODE; LM: $env:ASR_LM_PATH"
& $Python -c "import sys; print('Python:', sys.executable)"
& $Python scripts/validate_ai_service_startup.py
& $Python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
