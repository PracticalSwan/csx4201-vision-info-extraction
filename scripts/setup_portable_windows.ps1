[CmdletBinding()]
param(
    [ValidateSet('gpu', 'cpu')]
    [string]$Device = 'cpu',
    [string]$Python310 = '',
    [switch]$ReuseExisting,
    [string]$ExistingOCRPython = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe',
    [string]$ExistingLayoutPython = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-layout\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $ProjectRoot '.runtime'
$AppRoot = Join-Path $RuntimeRoot 'app'
$OCRRoot = Join-Path $RuntimeRoot 'ocr'
$LayoutRoot = Join-Path $RuntimeRoot 'layout'
$AppPython = Join-Path $AppRoot 'Scripts\python.exe'
$OCRPython = Join-Path $OCRRoot 'Scripts\python.exe'
$LayoutPython = Join-Path $LayoutRoot 'Scripts\python.exe'

if ([string]::IsNullOrWhiteSpace($Python310)) {
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $Launcher) {
        $Python310 = (& $Launcher.Source -3.10 -c "import sys; print(sys.executable)" 2>$null | Select-Object -Last 1)
    }
}
if ([string]::IsNullOrWhiteSpace($Python310) -or -not (Test-Path -LiteralPath $Python310)) {
    throw 'Python 3.10 is required. Install it from python.org or pass -Python310 with its executable path.'
}
$Version = (& $Python310 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($Version -ne '3.10') {
    throw "Python 3.10 is required; resolved $Version at $Python310."
}

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
if (-not (Test-Path -LiteralPath $AppPython)) {
    & $Python310 -m venv $AppRoot
}
& $AppPython -m pip install --upgrade pip
& $AppPython -m pip install -r (Join-Path $ProjectRoot 'requirements-app.txt')

if ($ReuseExisting) {
    if (-not (Test-Path -LiteralPath $ExistingOCRPython)) {
        throw "Existing OCR Python not found: $ExistingOCRPython"
    }
    if (-not (Test-Path -LiteralPath $ExistingLayoutPython)) {
        throw "Existing layout Python not found: $ExistingLayoutPython"
    }
    $OCRPython = (Resolve-Path -LiteralPath $ExistingOCRPython).Path
    $LayoutPython = (Resolve-Path -LiteralPath $ExistingLayoutPython).Path
} else {
    if (-not (Test-Path -LiteralPath $OCRPython)) {
        & $Python310 -m venv $OCRRoot
    }
    if (-not (Test-Path -LiteralPath $LayoutPython)) {
        & $Python310 -m venv $LayoutRoot
    }
    & $OCRPython -m pip install --upgrade pip
    & $LayoutPython -m pip install --upgrade pip
    if ($Device -eq 'gpu') {
        & $OCRPython -m pip install 'paddlepaddle-gpu==3.3.0' -i 'https://www.paddlepaddle.org.cn/packages/stable/cu130/'
        $TorchIndex = 'https://download.pytorch.org/whl/cu128'
    } else {
        & $OCRPython -m pip install 'paddlepaddle==3.3.0' -i 'https://www.paddlepaddle.org.cn/packages/stable/cpu/'
        $TorchIndex = 'https://download.pytorch.org/whl/cpu'
    }
    & $OCRPython -m pip install --upgrade 'torch==2.8.0' --index-url 'https://download.pytorch.org/whl/cpu'
    & $LayoutPython -m pip install --upgrade 'torch==2.8.0' --index-url $TorchIndex
    & $OCRPython -m pip install -r (Join-Path $ProjectRoot 'requirements-ocr.txt')
    & $LayoutPython -m pip install -r (Join-Path $ProjectRoot 'requirements-layout.txt')
}

$BundledAssetRoot = Join-Path $ProjectRoot 'assets'
$LegacyAssetRoot = 'D:\CSX4201\vision-info-extraction-assets'
$AssetRoot = if (Test-Path -LiteralPath (Join-Path $BundledAssetRoot 'checkpoints\layoutxlm_multitask\final\model.safetensors')) {
    $BundledAssetRoot
} else {
    $LegacyAssetRoot
}
$Checkpoint = Join-Path $AssetRoot 'checkpoints\layoutxlm_multitask\final'
$DeviceValue = if ($Device -eq 'gpu') { 'gpu:0' } else { 'cpu' }
$RuntimeConfig = [ordered]@{
    schema_version = '1.0'
    config = (Join-Path $ProjectRoot 'config.yaml')
    ocr_python = $OCRPython
    layout_python = $LayoutPython
    model_setup = (Join-Path $ProjectRoot 'reports\ocr\model_setup.json')
    layout_checkpoint = $Checkpoint
    asset_root = $AssetRoot
    output_root = (Join-Path $ProjectRoot 'outputs')
    device = $DeviceValue
    machine_local = $true
}
$RuntimeConfig | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ProjectRoot 'runtime.local.json') -Encoding UTF8

& $AppPython (Join-Path $ProjectRoot 'doctor.py') --probe
if ($LASTEXITCODE -ne 0) {
    throw 'Runtime setup completed, but doctor checks failed. Review the output above.'
}
Write-Host ''
Write-Host 'OCR Model is ready.'
Write-Host "GUI:  $ProjectRoot\launch_windows.bat"
Write-Host "CLI:  $ProjectRoot\run_cli.bat <document>"
Write-Host 'No OpenAI API key was requested or configured.'
