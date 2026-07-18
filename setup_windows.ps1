[CmdletBinding()]
param(
    [ValidateSet('gpu', 'cpu')]
    [string]$Device = 'cpu',
    [string]$Python310 = '',
    [switch]$ReuseExisting
)

$Arguments = @{
    Device = $Device
    Python310 = $Python310
    ReuseExisting = $ReuseExisting
}
& (Join-Path $PSScriptRoot 'scripts\setup_portable_windows.ps1') @Arguments
