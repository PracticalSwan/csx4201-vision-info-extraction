[CmdletBinding()]
param(
    [ValidateSet('gpu', 'cpu')]
    [string]$Device = 'gpu',
    [string]$AssetRoot = 'D:\CSX4201\vision-info-extraction-assets',
    [string]$Python310 = '',
    [switch]$SkipModels
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OCREnvironmentRoot = Join-Path $AssetRoot 'environments\ie-ocr'
$LayoutEnvironmentRoot = Join-Path $AssetRoot 'environments\ie-layout'
$OCRPython = Join-Path $OCREnvironmentRoot 'Scripts\python.exe'
$LayoutPython = Join-Path $LayoutEnvironmentRoot 'Scripts\python.exe'
$Directories = @(
    $AssetRoot,
    (Join-Path $AssetRoot 'cache\pip'),
    (Join-Path $AssetRoot 'cache\paddlex'),
    (Join-Path $AssetRoot 'cache\huggingface'),
    (Join-Path $AssetRoot 'cache\torch'),
    (Join-Path $AssetRoot 'cache\temp'),
    (Join-Path $AssetRoot 'models\paddleocr'),
    (Join-Path $AssetRoot 'models\layoutxlm'),
    (Join-Path $AssetRoot 'checkpoints'),
    (Join-Path $AssetRoot 'generated'),
    (Join-Path $AssetRoot 'ocr_cache'),
    (Join-Path $AssetRoot 'private_outputs'),
    (Join-Path $AssetRoot 'data\normalized_ie_annotations'),
    (Join-Path $AssetRoot 'data\model_datasets')
)
foreach ($Directory in $Directories) {
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
}

$CFreeGiB = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
$DFreeGiB = [math]::Round((Get-PSDrive D).Free / 1GB, 2)
if ($CFreeGiB -lt 15) {
    throw "C: free space is $CFreeGiB GiB; at least 15 GiB is required."
}
if ($DFreeGiB -lt 15) {
    throw "D: free space is $DFreeGiB GiB; insufficient for the model environment."
}
if ([string]::IsNullOrWhiteSpace($Python310)) {
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $Launcher) {
        $Python310 = (& $Launcher.Source -3.10 -c "import sys; print(sys.executable)" 2>$null | Select-Object -Last 1)
    }
}
if ([string]::IsNullOrWhiteSpace($Python310) -or -not (Test-Path -LiteralPath $Python310)) {
    throw 'Python 3.10 was not found via the Windows py launcher. Pass -Python310 with an explicit executable path.'
}
$ResolvedPythonVersion = & $Python310 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($ResolvedPythonVersion.Trim() -ne '3.10') {
    throw "The environment requires Python 3.10; resolved $ResolvedPythonVersion at $Python310."
}
if (-not (Test-Path -LiteralPath $OCRPython)) {
    & $Python310 -m venv $OCREnvironmentRoot
}
if (-not (Test-Path -LiteralPath $LayoutPython)) {
    & $Python310 -m venv $LayoutEnvironmentRoot
}

$env:PIP_CACHE_DIR = Join-Path $AssetRoot 'cache\pip'
$env:PADDLE_PDX_CACHE_HOME = Join-Path $AssetRoot 'cache\paddlex'
$env:HF_HOME = Join-Path $AssetRoot 'cache\huggingface'
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME 'transformers'
$env:TORCH_HOME = Join-Path $AssetRoot 'cache\torch'
$env:TEMP = Join-Path $AssetRoot 'cache\temp'
$env:TMP = $env:TEMP
$CudaBin = Join-Path $OCREnvironmentRoot 'Lib\site-packages\nvidia\cu13\bin\x86_64'
$CudnnBin = Join-Path $OCREnvironmentRoot 'Lib\site-packages\nvidia\cudnn\bin'
$env:PATH = "$CudaBin;$CudnnBin;$env:PATH"

& $OCRPython -m pip install --upgrade pip
& $LayoutPython -m pip install --upgrade pip
if ($Device -eq 'gpu') {
    & $OCRPython -m pip install 'paddlepaddle-gpu==3.3.0' -i 'https://www.paddlepaddle.org.cn/packages/stable/cu130/'
    $TorchIndex = 'https://download.pytorch.org/whl/cu128'
} else {
    & $OCRPython -m pip install 'paddlepaddle==3.3.0' -i 'https://www.paddlepaddle.org.cn/packages/stable/cpu/'
    $TorchIndex = 'https://download.pytorch.org/whl/cpu'
}
& $LayoutPython -m pip install --upgrade 'torch==2.8.0' --index-url $TorchIndex
$ExpectedTorchMode = if ($Device -eq 'gpu') { 'cuda' } else { 'cpu' }
$ActualTorchMode = & $LayoutPython -c "import torch; print('cuda' if torch.version.cuda else 'cpu')"
if ($ActualTorchMode.Trim() -ne $ExpectedTorchMode) {
    Write-Host "Replacing mismatched PyTorch $ActualTorchMode wheel with $ExpectedTorchMode wheel."
    & $LayoutPython -m pip install --upgrade --force-reinstall 'torch==2.8.0' --index-url $TorchIndex
}
& $OCRPython -m pip uninstall -y accelerate seqeval sentencepiece | Out-Null
& $OCRPython -m pip install --upgrade 'torch==2.8.0' --index-url 'https://download.pytorch.org/whl/cpu'
$OCRTorchMode = & $OCRPython -c "import torch; print('cuda' if torch.version.cuda else 'cpu')"
if ($OCRTorchMode.Trim() -ne 'cpu') {
    Write-Host "Replacing mismatched OCR PyTorch wheel with the required CPU-only wheel."
    & $OCRPython -m pip install --upgrade --force-reinstall 'torch==2.8.0' --index-url 'https://download.pytorch.org/whl/cpu'
}
& $OCRPython -m pip install -r (Join-Path $ProjectRoot 'requirements-ocr.txt')
& $LayoutPython -m pip install -r (Join-Path $ProjectRoot 'requirements-layout.txt')
& $OCRPython (Join-Path $PSScriptRoot 'print_environment_report.py') --layout-python $LayoutPython
& $OCRPython -c "import paddle; paddle.utils.run_check()"
if ($Device -eq 'gpu') {
    & $LayoutPython -c "import torch; assert torch.cuda.is_available(); print(torch.__version__)"
} else {
    & $LayoutPython -c "import torch; assert not torch.cuda.is_available(); print(torch.__version__)"
}

if (-not $SkipModels) {
    $PaddleDevice = if ($Device -eq 'gpu') { 'gpu:0' } else { 'cpu' }
    & $OCRPython (Join-Path $PSScriptRoot 'download_ocr_models.py') --model all --device $PaddleDevice
    & $OCRPython (Join-Path $PSScriptRoot 'verify_ocr_models.py') --device $PaddleDevice
}

Write-Host "OCR environment ready at $OCREnvironmentRoot"
Write-Host "Layout environment ready at $LayoutEnvironmentRoot"
Write-Host "C: free: $CFreeGiB GiB; D: free before installation: $DFreeGiB GiB"
