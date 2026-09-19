"""QC вертикального градиента температуры γ = 100·ΔT/Δz."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_gamma_qc(
    df: pd.DataFrame,
    *,
    z_bottom: str,
    z_top: str,
    t_bottom: str,
    t_top: str,
    min_dz_m: float = 20.0,
    review_abs_gamma: float = 20.0,
) -> pd.DataFrame:
    """Высоты: м; температуры: °C или K в одной шкале.

    gamma > 0 означает повышение температуры с высотой.
    Тонкие интервалы (dz < min_dz_m) не попадают в gamma_qc_*.
    """
    if not np.isfinite(min_dz_m) or min_dz_m <= 0:
        raise ValueError("min_dz_m должен быть конечным и > 0")
    if not np.isfinite(review_abs_gamma) or review_abs_gamma <= 0:
        raise ValueError("review_abs_gamma должен быть конечным и > 0")

    out = df.copy()
    cols = [z_bottom, z_top, t_bottom, t_top]
    values = out[cols].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)

    zb, zt, tb, tt = values.T
    dz = zt - zb
    dt = tt - tb

    finite = (
        np.isfinite(values).all(axis=1)
        & np.isfinite(dz)
        & np.isfinite(dt)
    )
    positive_dz = finite & (dz > 0)

    raw = np.full(len(out), np.nan)
    np.divide(100.0 * dt, dz, out=raw, where=positive_dz)

    reason = np.select(
        [
            ~finite,
            finite & (dz <= 0),
            positive_dz & (dz < min_dz_m),
        ],
        ["missing_or_nonfinite", "nonpositive_dz", "thin_interval"],
        default="passes_dz_check",
    )
    usable = reason == "passes_dz_check"

    out["dz_m"] = dz
    out["dt_c"] = dt
    out["gamma_raw_c_per_100m"] = raw
    out["gamma_qc_c_per_100m"] = np.where(usable, raw, np.nan)
    out["gamma_qc_reason"] = reason
    out["gamma_extreme_review"] = (
        np.isfinite(raw) & (np.abs(raw) >= review_abs_gamma)
    )
    out["gamma_min_dz_m"] = float(min_dz_m)
    return out
