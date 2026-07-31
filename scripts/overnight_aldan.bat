@echo off
setlocal
cd /d "%~dp0.."
set PYTHONPATH=%CD%
set LOGDIR=%CD%\gdex_outputs\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%I
set LOG=%LOGDIR%\overnight_aldan_%STAMP%.log
echo %DATE% %TIME% START > "%LOG%"
echo ROOT=%CD%>> "%LOG%"
echo STEP1 decode>> "%LOG%"
py -3 run_fast_extract.py --station aldan --start-date 1999-10-01 --end-date 2026-07-30 --cycles 00,12 --workers 14 --checkpoint-every 150 --fresh --output gdex_outputs/profile_climate/aldan >> "%LOG%" 2>&1
if errorlevel 1 goto :decode_failed
echo STEP1 exit=0>> "%LOG%"
echo STEP2 plots>> "%LOG%"
py -3 -m gdex_bufr monthly-profile-plots --station aldan --start-date 1999-10-01 --end-date 2026-07-30 --input gdex_outputs/profile_climate/aldan/profiles_long.csv --metrics gdex_outputs/profile_climate/aldan/profile_metrics.csv --output gdex_outputs/monthly_temperature_profiles >> "%LOG%" 2>&1
if errorlevel 1 goto :plots_failed
echo STEP2 exit=0>> "%LOG%"
echo STEP3 daily>> "%LOG%"
py -3 scripts/build_daily_profiles.py --long gdex_outputs/profile_climate/aldan/profiles_long.csv --metrics gdex_outputs/profile_climate/aldan/profile_metrics.csv --output gdex_outputs/profile_climate/aldan/daily_profiles.json >> "%LOG%" 2>&1
if errorlevel 1 goto :daily_failed
echo STEP3 exit=0>> "%LOG%"
echo STEP4 dashboard>> "%LOG%"
py -3 scripts/export_offline_dashboard.py >> "%LOG%" 2>&1
if errorlevel 1 goto :dashboard_failed
echo STEP4 exit=0>> "%LOG%"
echo DONE>> "%LOG%"
exit /b 0

:decode_failed
echo STEP1 FAILED; pipeline aborted>> "%LOG%"
exit /b 1

:plots_failed
echo STEP2 FAILED; pipeline aborted>> "%LOG%"
exit /b 1

:daily_failed
echo STEP3 FAILED; pipeline aborted>> "%LOG%"
exit /b 1

:dashboard_failed
echo STEP4 FAILED; pipeline aborted>> "%LOG%"
exit /b 1
