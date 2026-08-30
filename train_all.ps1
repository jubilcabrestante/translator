# train_all.ps1
# -------------------------------------------------------------------------
# ONE COMMAND that trains the whole thing by itself:
#   prepare data  ->  fine-tune  ->  evaluate
#
# Usage (from the project folder):
#   .\train_all.ps1
#   .\train_all.ps1 -Model "Qwen/Qwen2.5-0.5B-Instruct"   # smaller, if out of memory
# -------------------------------------------------------------------------
param(
    [string]$Model = "Qwen/Qwen2.5-1.5B-Instruct"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot                      # run from the project root
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "ERROR: .venv not found. Set up the environment first (see README)." -ForegroundColor Red
    exit 1
}

Write-Host "==> [1/3] Preparing data (both directions + entity passthrough)..." -ForegroundColor Cyan
& $py scripts/prepare_data.py
if ($LASTEXITCODE -ne 0) { Write-Host "prepare_data failed." -ForegroundColor Red; exit 1 }

Write-Host "==> [2/3] Fine-tuning $Model (the long step)..." -ForegroundColor Cyan
& $py scripts/train.py --model $Model
if ($LASTEXITCODE -ne 0) { Write-Host "train failed (out of memory? try -Model Qwen/Qwen2.5-0.5B-Instruct)." -ForegroundColor Red; exit 1 }

Write-Host "==> [3/3] Evaluating on the held-out set..." -ForegroundColor Cyan
& $py scripts/evaluate.py --base-model $Model
if ($LASTEXITCODE -ne 0) { Write-Host "evaluate failed." -ForegroundColor Red; exit 1 }

Write-Host "`nAll done!  Translate with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe scripts/translate.py" -ForegroundColor Green
