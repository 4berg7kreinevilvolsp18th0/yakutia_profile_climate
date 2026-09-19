# Пересборка с QC γ (актуально)

Полный пересчёт с `gamma_min_dz_m=20` лежит здесь: **`sample_output_qc/`**.

Старый `sample_output/` мог не обновиться из‑за блокировки `tables/interval_gammas.csv` (файл открыт в IDE) и нехватки места при двойной копии.

## Как заменить sample_output

1. Закройте `sample_output/tables/interval_gammas.csv` в редакторе.
2. В PowerShell из папки `Графики для статьи`:

```powershell
Remove-Item -Recurse -Force sample_output
Rename-Item sample_output_qc sample_output
```

или:

```powershell
B:\Kutunika` programmist\yakutia_profile_climate\.venv\Scripts\python.exe scripts\_copy_sample_output_qc.py
```

## Эффект QC

- интервалов с `|γ|≥20` в климатологии: **584 → 17**
- отфильтровано как `thin_interval` (dz &lt; 20 м): **4443**
- overflow-бин `≥20` в `gamma_counts_all.csv`: **14** (+3 в `≤-20`)

Графики γ (`type03_*`, `gamma_monthly*`, `gamma_by_year*`, 3D) пересобраны из QC-данных.
