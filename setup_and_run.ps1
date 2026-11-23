<#
 One-step setup and launch for Windows PowerShell.

 Run from the project folder:
     powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1

 Options:
     -SkipChecks   skip the offline self-check
     -NoRun        set everything up but do not start the app
     -Live         also run the live embedding + Groq check
#>
param(
    [switch]$SkipChecks,
    [switch]$NoRun,
    [switch]$Live
)

Set-Location $PSScriptRoot

function Invoke-SystemPython {
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 @args } else { & python @args }
}

Write-Host "== 1/5 Checking Python version" -ForegroundColor Cyan
$version = Invoke-SystemPython -c "import sys; print(sys.version_info[0]*100 + sys.version_info[1])"
if ($LASTEXITCODE -ne 0 -or -not $version) {
    Write-Host "Python was not found. Install Python 3.11+ from https://www.python.org/downloads/ (tick 'Add to PATH')." -ForegroundColor Red
    exit 1
}
if ([int]$version -lt 311) {
    Write-Host "Python 3.11 or newer is required (found $version). Please upgrade." -ForegroundColor Red
    exit 1
}

Write-Host "== 2/5 Creating virtual environment (.venv)" -ForegroundColor Cyan
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Invoke-SystemPython -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Host "Could not create the virtual environment." -ForegroundColor Red; exit 1 }
}
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "== 3/5 Installing dependencies (first time takes several minutes)" -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Dependency installation failed. See the error above." -ForegroundColor Red; exit 1 }

Write-Host "== 4/5 Checking .env" -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example." -ForegroundColor Yellow
}
$envText = Get-Content ".env" -Raw
$hasKey = $envText -match '(?m)^\s*GROQ_API_KEY\s*=\s*\S+'
$hasModel = $envText -match '(?m)^\s*GROQ_MODEL\s*=\s*\S+'
if (-not ($hasKey -and $hasModel)) {
    Write-Host "GROQ_API_KEY and/or GROQ_MODEL are empty in .env." -ForegroundColor Yellow
    Write-Host "  Key:   https://console.groq.com/keys"
    Write-Host "  Model: pick a current one from https://console.groq.com/docs/models"
    Start-Process notepad.exe ".env"
    Read-Host "Fill them in, SAVE the file, then press Enter to continue"
}

if (-not $SkipChecks) {
    Write-Host "== 5/5 Running self-check" -ForegroundColor Cyan
    if ($Live) { & $venvPython check_project.py --live } else { & $venvPython check_project.py }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Some checks failed. Fix them (see README 'Common errors') before relying on the app." -ForegroundColor Yellow
    }
}

if (-not $NoRun) {
    Write-Host "Starting the app. Press Ctrl+C in this window to stop it." -ForegroundColor Green
    & $venvPython -m streamlit run app.py
}
