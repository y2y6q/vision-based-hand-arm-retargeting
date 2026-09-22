[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [string]$PythonPath = (Join-Path $PSScriptRoot '..\.venv-robosuite\Scripts\python.exe'),
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PythonPath = (Resolve-Path $PythonPath).Path
$Manager = Join-Path $ProjectRoot 'scripts\manage_3dxware_profile.py'
$Manifest = (Resolve-Path $Manifest).Path

$Arguments = @($Manager, 'restore', '--manifest', $Manifest)
if ($Force) {
    $Arguments += '--force'
}

& $PythonPath @Arguments
exit $LASTEXITCODE
