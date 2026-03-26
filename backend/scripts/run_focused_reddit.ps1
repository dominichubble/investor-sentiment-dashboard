# Run focused Reddit → FinBERT → Neon. Run from anywhere.
# Usage:
#   .\backend\scripts\run_focused_reddit.ps1
#   .\backend\scripts\run_focused_reddit.ps1 -Quick
#   .\backend\scripts\run_focused_reddit.ps1 -ExtraArgs @("--tickers","NVDA","PLTR")

param(
    [switch]$Quick,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

$pyArgs = @("backend\scripts\focused_reddit_pipeline.py")
if ($Quick) { $pyArgs += "--quick" }
if ($ExtraArgs.Count -gt 0) { $pyArgs += $ExtraArgs }

& python @pyArgs
exit $LASTEXITCODE
