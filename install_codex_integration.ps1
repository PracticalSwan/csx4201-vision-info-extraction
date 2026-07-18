[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$AppPython = Join-Path $ProjectRoot '.runtime\app\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $AppPython)) {
    throw 'Run setup_windows.bat before installing the Codex integration.'
}
$Codex = Get-Command codex -ErrorAction SilentlyContinue
if ($null -eq $Codex) {
    throw 'Codex CLI is not installed or not available on PATH.'
}

# Remove only this integration when it already exists, then register the exact
# local STDIO command. This uses the signed-in Codex session, not an API key.
& $Codex.Source mcp remove ocr_model 2>$null | Out-Null
& $Codex.Source mcp add ocr_model --env "OCR_MODEL_HOME=$ProjectRoot" -- $AppPython (Join-Path $ProjectRoot 'mcp_server.py')
if ($LASTEXITCODE -ne 0) {
    throw 'Codex MCP registration failed.'
}
Write-Host 'Installed local MCP server: ocr_model'
Write-Host 'Open this folder in Codex, select GPT-5.6, and invoke $review-ocr-document.'
Write-Host 'No OpenAI API key was requested or configured.'
