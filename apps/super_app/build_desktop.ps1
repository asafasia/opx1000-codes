$ErrorActionPreference = "Stop"

$sourceRoot = $PSScriptRoot
$configuredProjectRoot = $env:OPX1000_PROJECT_ROOT
if ([string]::IsNullOrWhiteSpace($configuredProjectRoot)) {
    $repoCandidate = Join-Path $sourceRoot "..\.."
    $siblingCandidate = Join-Path $sourceRoot "..\opx1000-codes"
    if (Test-Path -LiteralPath (Join-Path $repoCandidate "profiles")) {
        $configuredProjectRoot = (Resolve-Path $repoCandidate).Path
    }
    elseif (Test-Path -LiteralPath (Join-Path $siblingCandidate "profiles")) {
        $configuredProjectRoot = (Resolve-Path $siblingCandidate).Path
    }
    else {
        throw "Set OPX1000_PROJECT_ROOT to the opx1000-codes repository."
    }
}
$projectRoot = (Resolve-Path $configuredProjectRoot).Path
$labPython = "C:\Users\owner\miniconda3\envs\opx1000_env\python.exe"
$iconPath = Join-Path $projectRoot "apps\visualiser\static\Q.png"

if (-not (Test-Path -LiteralPath $labPython)) {
    throw "Lab Python was not found at $labPython"
}

Push-Location $sourceRoot
try {
    & $labPython -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --windowed `
        --name "Quantum Coherence Lab" `
        --icon $iconPath `
        --add-data "$sourceRoot\static;static" `
        --distpath "$sourceRoot\dist" `
        --workpath "$sourceRoot\build" `
        --specpath "$sourceRoot\build" `
        "$sourceRoot\desktop.py"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
    Write-Host "Built: $sourceRoot\dist\Quantum Coherence Lab.exe"
}
finally {
    Pop-Location
}
