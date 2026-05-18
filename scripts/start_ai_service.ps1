$ErrorActionPreference = "Stop"

$Python = Join-Path (Get-Location) ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "AI service virtual environment was not found at $Python. Create the .venv before starting ASR."
}

Write-Host "Using local .venv"
& $Python -c "import sys; print('Python:', sys.executable)"
& $Python scripts/validate_ai_service_startup.py
& $Python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
