$ErrorActionPreference = "Stop"
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Using local .venv"
    . .venv\Scripts\Activate.ps1
} else {
    Write-Warning ".venv not found; using current Python environment"
}
python -c "import sys; print('Python:', sys.executable)"
python scripts/validate_ai_service_startup.py
uvicorn api.main:app --reload --host 127.0.0.1 --port 8001
