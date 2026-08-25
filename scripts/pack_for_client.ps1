# Pack a client-ready zip (excludes .venv and fastf1_cache).
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File .\scripts\pack_for_client.ps1

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Parent = Split-Path $Root -Parent
$Stamp = Get-Date -Format "yyyyMMdd"
$ZipPath = Join-Path $Parent "F1_Analytics_Client_$Stamp.zip"
$Stage = Join-Path $env:TEMP "F1_Analytics_Client_stage_$Stamp"

if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Path $Stage | Out-Null

$ExcludeDirNames = @(
    ".venv",
    "fastf1_cache",
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".cursor",
    "agent-transcripts"
)

function Should-Skip([string]$fullPath) {
    $rel = $fullPath.Substring($Root.Path.Length).TrimStart("\", "/")
    foreach ($part in $rel.Split([char[]]@("\", "/"))) {
        if ($ExcludeDirNames -contains $part) { return $true }
        if ($part -eq "__pycache__") { return $true }
    }
    if ($rel -like "*.pyc") { return $true }
    if ($rel -like "*.zip") { return $true }
    return $false
}

Write-Host "Staging from $Root ..."
Get-ChildItem -Path $Root -Recurse -Force -File | ForEach-Object {
    if (Should-Skip $_.FullName) { return }
    $rel = $_.FullName.Substring($Root.Path.Length).TrimStart("\", "/")
    $dest = Join-Path $Stage $rel
    $destDir = Split-Path $dest -Parent
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    Copy-Item $_.FullName $dest -Force
}

# Drop internal agent briefs from the client package (keep Project.md + README)
Get-ChildItem -Path $Stage -Filter "F1_CURSOR_*.md" -File -ErrorAction SilentlyContinue |
    Remove-Item -Force

if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Write-Host "Compressing -> $ZipPath (this may take a few minutes) ..."
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $ZipPath -CompressionLevel Optimal

Remove-Item $Stage -Recurse -Force
$sizeMB = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host "Done: $ZipPath ($sizeMB MB)"
Write-Host "Send this zip to the client. They should unzip, create .venv, pip install -r requirements.txt, then run: python scripts/verify_install.py"
