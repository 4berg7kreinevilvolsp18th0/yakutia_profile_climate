# GRAPHICS DATA AUDIT

Строк метрик: 12

## 01_thickness / inversion_depth_vs_base_joint / base_height_agl_m
- formula: AGL от нижнего уровня профиля (z0)
- unit: m
- N=19996, median=908.65, q25=0.0, q75=2229.65
- N00=8938, N12=9053

## 01_thickness / inversion_depth_vs_base_joint / depth_m
- formula: top_height_agl_m - base_height_agl_m
- unit: m
- N=19996, median=236.5, q25=133.20000000000027, q75=395.5
- N00=8938, N12=9053

## 02_gamma_sfc_P / type03_gamma_annual_cycle_850_700_500 / gamma_sfc_850
- formula: 100*(T_850-T_sfc)/(H_850-H_sfc), no extrapolation
- unit: °C/100 m
- N=17991, median=-0.4716263502205943, q25=-0.7000259268861853, q75=-0.06188610878272788
- N00=8938, N12=9053

## 02_gamma_sfc_P / type03_gamma_annual_cycle_850_700_500 / gamma_sfc_700
- formula: 100*(T_700-T_sfc)/(H_700-H_sfc), no extrapolation
- unit: °C/100 m
- N=17991, median=-0.4980163754537018, q25=-0.6525616171873663, q75=-0.23209395163162047
- N00=8938, N12=9053

## 02_gamma_sfc_P / type03_gamma_annual_cycle_850_700_500 / gamma_sfc_500
- formula: 100*(T_500-T_sfc)/(H_500-H_sfc), no extrapolation
- unit: °C/100 m
- N=17991, median=-0.5352287567706444, q25=-0.6225110039823935, q75=-0.4033781432384086
- N00=8938, N12=9053

## 03_gamma_local / gamma_local_month_height_median / gamma_local_median
- formula: median over intervals: 100*ΔT/Δz
- unit: °C/100 m
- N=132, median=-0.5089769740071304, q25=-0.6419615829974901, q75=-0.21374285208936028
- N00=8938, N12=9053

## 03_gamma_local / gamma_local_month_height_median / gamma_local_intervals
- formula: 100*(T_{i+1}-T_i)/(z_{i+1}-z_i)
- unit: °C/100 m
- N=163729, median=-0.492610837438391, q25=-0.7474889044615743, q75=0.0
- N00=8938, N12=9053

## 04_multilayer / n_layers_histogram / n_inversion_layers
- formula: count v3 layers per eligible profile
- unit: count
- N=17991, median=1.0, q25=0.0, q75=2.0
- N00=8938, N12=9053

## 06_extra / violin_depth_by_month / depth_m
- formula: layer depth_m
- unit: m
- N=19996, median=236.5, q25=133.20000000000027, q75=395.5
- N00=8938, N12=9053

## diagnostic / qc_depth_histogram / depth_m
- formula: same as 01/06 depth_m
- unit: m
- N=19996, median=236.5, q25=133.20000000000027, q75=395.5
- N00=8938, N12=9053

## 02_gamma_sfc_P / type03_gamma_annual_cycle_850_700_500 / gamma_sfc_850
- formula: shared compute_sfc_level_gamma
- unit: °C/100 m
- N=17991, median=-0.4716263502205943, q25=-0.7000259268861853, q75=-0.06188610878272788
- N00=8938, N12=9053

## diagnostic / qc_eligible_counts / gamma_sfc_850
- formula: shared compute_sfc_level_gamma
- unit: °C/100 m
- N=17991, median=-0.4716263502205943, q25=-0.7000259268861853, q75=-0.06188610878272788
- N00=8938, N12=9053
