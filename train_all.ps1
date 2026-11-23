# train_all.ps1
# -------------------------------------------------------------------------
# ONE COMMAND that trains the whole thing by itself:
#   check data  ->  prepare data  ->  fine-tune  ->  evaluate
#
# Usage (from the project folder):
#   .\train_all.ps1
#   .\train_all.ps1 -Model "Qwen/Qwen2.5-0.5B-Instruct"   # smaller, if out of memory
#
# Training needs an NVIDIA GPU. On a machine without one, use
# notebooks/train_colab.ipynb on Google Colab instead.
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

Write-Host "==> [0/3] Checking the data (report only; run lint_data.py --fix to apply safe fixes)..." -ForegroundColor Cyan
& $py scripts/lint_data.py --show 3
if ($LASTEXITCODE -ne 0) { Write-Host "lint_data failed." -ForegroundColor Red; exit 1 }

Write-Host "==> [1/3] Preparing data (all directions, names, stable train/val/test split)..." -ForegroundColor Cyan
& $py scripts/prepare_data.py
if ($LASTEXITCODE -ne 0) { Write-Host "prepare_data failed." -ForegroundColor Red; exit 1 }

Write-Host "==> [2/3] Fine-tuning $Model (the long step)..." -ForegroundColor Cyan
& $py scripts/train.py --model $Model
if ($LASTEXITCODE -ne 0) { Write-Host "train failed (no NVIDIA GPU? use the Colab notebook. Out of memory? try -Model Qwen/Qwen2.5-0.5B-Instruct)." -ForegroundColor Red; exit 1 }

Write-Host "==> [3/3] Evaluating (test set if filled in, else validation)..." -ForegroundColor Cyan
& $py scripts/evaluate.py --base-model $Model
if ($LASTEXITCODE -ne 0) { Write-Host "evaluate failed." -ForegroundColor Red; exit 1 }

Write-Host "`nAll done!  A review sheet for a Cuyonon speaker is under outputs\eval\." -ForegroundColor Green
Write-Host "Translate with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe scripts/translate.py" -ForegroundColor Green
