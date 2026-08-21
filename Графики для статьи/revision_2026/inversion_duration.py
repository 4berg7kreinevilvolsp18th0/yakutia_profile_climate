"""Расчёт длительности инверсионных событий с interval censoring."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

EventType = Literal["ANY", "G", "E", "HE"]
STEP_HOURS = 12.0


def _positive_flags(profiles: pd.DataFrame, flags: pd.DataFrame, *, event_type: EventType) -> pd.DataFrame:
    keep = ["profile_id", "has_G", "has_E", "has_HE", "has_any_v3"]
    keep = [c for c in keep if c in flags.columns]
    p = profiles.merge(flags[keep], on="profile_id", how="left")
    if "has_any_v3" not in p.columns and all(c in p.columns for c in ["has_G", "has_E", "has_HE"]):
        p["has_any_v3"] = p[["has_G", "has_E", "has_HE"]].any(axis=1)
    if event_type == "ANY":
        p["positive"] = p["has_any_v3"].fillna(False).astype(bool)
    else:
        p["positive"] = p[f"has_{event_type}"].fillna(False).astype(bool)
    return p


def build_inversion_duration_events(
    profiles: pd.DataFrame,
    flags: pd.DataFrame,
    *,
    event_type: EventType = "ANY",
    allow_one_missing_sounding: bool = False,
) -> pd.DataFrame:
    """События по хронологии станции (00/12 UTC)."""
    use = profiles[profiles["cycle"].astype(str).str.zfill(2).str[-2:].isin(["00", "12"])].copy()
    use = _positive_flags(use, flags, event_type=event_type)
    use = use.sort_values("datetime_utc").reset_index(drop=True)
    if use.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    event_id = 0
    i = 0
    n = len(use)
    while i < n:
        if not use.loc[i, "positive"]:
            i += 1
            continue
        start_i = i
        while i + 1 < n and use.loc[i + 1, "positive"]:
            dt_h = (use.loc[i + 1, "datetime_utc"] - use.loc[i, "datetime_utc"]).total_seconds() / 3600.0
            if dt_h <= STEP_HOURS + 0.1:
                i += 1
                continue
            if allow_one_missing_sounding and dt_h <= 2 * STEP_HOURS + 0.1:
                i += 1
                continue
            break
        end_i = i

        t_first = use.loc[start_i, "datetime_utc"]
        t_last = use.loc[end_i, "datetime_utc"]
        n_pos = end_i - start_i + 1
        duration_lower = (t_last - t_first).total_seconds() / 3600.0

        prev_neg = (
            use.loc[start_i - 1, "datetime_utc"]
            if start_i > 0 and not use.loc[start_i - 1, "positive"]
            else pd.NaT
        )
        next_neg = (
            use.loc[end_i + 1, "datetime_utc"]
            if end_i + 1 < n and not use.loc[end_i + 1, "positive"]
            else pd.NaT
        )

        left_censored = start_i == 0 or pd.isna(prev_neg)
        right_censored = end_i == n - 1 or pd.isna(next_neg)

        duration_upper = np.nan
        duration_midpoint = np.nan
        if pd.notna(prev_neg) and pd.notna(next_neg):
            duration_upper = (next_neg - prev_neg).total_seconds() / 3600.0
            duration_midpoint = 0.5 * (duration_lower + duration_upper)
        elif pd.notna(prev_neg) or pd.notna(next_neg):
            duration_upper = duration_lower + 2 * STEP_HOURS
            duration_midpoint = duration_lower + STEP_HOURS

        gap_inside = False
        for j in range(start_i, end_i):
            dt_h = (use.loc[j + 1, "datetime_utc"] - use.loc[j, "datetime_utc"]).total_seconds() / 3600.0
            if dt_h > STEP_HOURS + 0.1 and not (
                allow_one_missing_sounding and dt_h <= 2 * STEP_HOURS + 0.1
            ):
                gap_inside = True
                break

        event_id += 1
        rows.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "start_first_detected_utc": t_first,
                "end_last_detected_utc": t_last,
                "previous_negative_utc": prev_neg,
                "next_negative_utc": next_neg,
                "n_positive_soundings": int(n_pos),
                "duration_lower_h": float(duration_lower),
                "duration_upper_h": float(duration_upper) if np.isfinite(duration_upper) else np.nan,
                "duration_midpoint_h": float(duration_midpoint) if np.isfinite(duration_midpoint) else np.nan,
                "left_censored": bool(left_censored),
                "right_censored": bool(right_censored),
                "missing_observation_inside": bool(gap_inside),
                "cycle_sequence": ",".join(use.loc[start_i : end_i + 1, "cycle"].astype(str)),
                "profile_ids": ";".join(use.loc[start_i : end_i + 1, "profile_id"].astype(str)),
            }
        )
        i = end_i + 1

    return pd.DataFrame.from_records(rows)


def build_all_duration_events(profiles: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    parts = [
        build_inversion_duration_events(profiles, flags, event_type=t)
        for t in ("ANY", "G", "E", "HE")
    ]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)
