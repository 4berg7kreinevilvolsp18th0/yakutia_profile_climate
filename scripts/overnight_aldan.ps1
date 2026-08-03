# Ночная полная пересборка Алдана: decode -> plots -> daily -> dashboard
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = $Root
$LogDir = Join-Path $Root "gdex_outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$Log = Join-Path $LogDir "overnight_aldan_${Stamp}_${PID}.log"

function Write-Log([string]$msg) {
  $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Add-Content -Path $Log -Value $line -Encoding UTF8
  Write-Host $line
}

Write-Log "START overnight aldan root=$Root log=$Log"

Write-Log "STEP1 decode --fresh"
& py -3 .\run_fast_extract.py `
  --station aldan `
  --start-date 1999-10-01 `
  --end-date 2026-07-30 `
  --cycles 00,12 `
  --workers 8 `
  --checkpoint-every 100 `
  --fresh `
  --output gdex_outputs/результаты-алдан `
  *>> $Log
$rc1 = $LASTEXITCODE
Write-Log "STEP1 exit=$rc1"
if ($null -eq $rc1 -or $rc1 -ne 0) {
  Write-Log "ABORT after STEP1"
  exit $(if ($null -eq $rc1) { 1 } else { $rc1 })
}

Write-Log "STEP2 monthly plots"
& py -3 -m gdex_bufr monthly-profile-plots `
  --station aldan `
  --start-date 1999-10-01 `
  --end-date 2026-07-30 `
  --input gdex_outputs/результаты-алдан/profiles_long.csv `
  --metrics gdex_outputs/результаты-алдан/profile_metrics.csv `
  --output gdex_outputs/monthly_temperature_profiles `
  --set актуальное `
  *>> $Log
$rc2 = $LASTEXITCODE
Write-Log "STEP2 exit=$rc2"
if ($null -eq $rc2 -or $rc2 -ne 0) {
  Write-Log "ABORT after STEP2"
  exit $(if ($null -eq $rc2) { 1 } else { $rc2 })
}

Write-Log "STEP3 daily profiles json"
& py -3 scripts/build_daily_profiles.py `
  --long gdex_outputs/результаты-алдан/profiles_long.csv `
  --metrics gdex_outputs/результаты-алдан/profile_metrics.csv `
  --output gdex_outputs/результаты-алдан/daily_profiles.json `
  *>> $Log
$rc3 = $LASTEXITCODE
Write-Log "STEP3 exit=$rc3"
if ($null -eq $rc3 -or $rc3 -ne 0) {
  Write-Log "ABORT after STEP3"
  exit $(if ($null -eq $rc3) { 1 } else { $rc3 })
}

Write-Log "STEP4 offline dashboard"
& py -3 scripts/export_offline_dashboard.py *>> $Log
$rc4 = $LASTEXITCODE
Write-Log "STEP4 exit=$rc4"
if ($null -eq $rc4 -or $rc4 -ne 0) {
  Write-Log "ABORT after STEP4"
  exit $(if ($null -eq $rc4) { 1 } else { $rc4 })
}

Write-Log "DONE rc_decode=$rc1 rc_plots=$rc2 rc_daily=$rc3 rc_dash=$rc4"
exit 0
