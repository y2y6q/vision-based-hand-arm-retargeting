[CmdletBinding()]
param(
    [string]$PythonPath = (Join-Path $PSScriptRoot '..\.venv-robosuite\Scripts\python.exe'),
    [string]$BackupRoot = (Join-Path $PSScriptRoot '..\outputs\3dxware_profiles\backups'),
    [switch]$AllowOverwrite
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PythonPath = (Resolve-Path $PythonPath).Path
$Manager = Join-Path $ProjectRoot 'scripts\manage_3dxware_profile.py'

$Arguments = @(
    $Manager,
    'install',
    '--target-exe', 'python.exe',
    '--target-exe', 'pycharm64.exe',
    '--backup-root', $BackupRoot
)
if ($AllowOverwrite) {
    $Arguments += '--allow-overwrite'
}

& $PythonPath @Arguments
exit $LASTEXITCODE
