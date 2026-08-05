@echo off
setlocal
cd /d "%~dp0.."
set PYTHONPATH=%CD%
echo STEP compare CSV
py -3 scripts/compare_aldan_csv.py --old-metrics gdex_outputs/результаты-алдан/profile_metrics.csv --new-metrics gdex_outputs/результаты-алдан-полный/profile_metrics.csv --index gdex_outputs/результаты-алдан-полный/aldan_bufr_index.csv --output-dir gdex_outputs/результаты-алдан-полный
if errorlevel 1 exit /b 1
echo STEP sync extra BUFR from new metrics into bufr_алдан
py -3 scripts/copy_aldan_bufr.py --index gdex_outputs/результаты-алдан-полный/aldan_bufr_index.csv --dest-root gdex_data/bufr_алдан --manifest gdex_outputs/результаты-алдан-полный/bufr_алдан_manifest.csv
echo STEP assemble daily/dashboard/plots into сравнение
py -3 scripts/assemble_aldan_full.py --results-dir gdex_outputs/результаты-алдан-полный --plot-set полный_все_циклы
if errorlevel 1 exit /b 1
echo DONE
exit /b 0
