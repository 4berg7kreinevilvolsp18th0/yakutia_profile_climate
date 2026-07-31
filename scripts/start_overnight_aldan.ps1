$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Process = Start-Process `
  -FilePath "cmd.exe" `
  -ArgumentList @("/d", "/c", "scripts\overnight_aldan.bat") `
  -WorkingDirectory $Root `
  -WindowStyle Hidden `
  -PassThru

Write-Output $Process.Id
