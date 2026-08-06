from __future__ import annotations

import pandas as pd

from gdex_bufr.profile_climate.article_figures.config import InversionConfig
from gdex_bufr.profile_climate.article_figures.metrics import detect_surface_inversion_v2


def test_confirmed_inversion():
    levels = pd.DataFrame(
        {
            "pressure_hpa": [1000, 950, 900, 850, 820, 800],
            "temperature_c": [-30, -28, -25, -24, -26, -28],
            "height_m": [0, 400, 800, 1300, 1600, 1800],
        }
    )
    result = detect_surface_inversion_v2(levels, InversionConfig())
    assert result.detected
    assert result.quality == "confirmed"
    assert result.delta_t_c == 6


def test_rejected_inversion():
    levels = pd.DataFrame(
        {
            "pressure_hpa": [1000, 950, 900, 850, 820, 800],
            "temperature_c": [-30, -28, -25, -24, -26, -25],
            "height_m": [0, 400, 800, 1300, 1600, 1800],
        }
    )
    result = detect_surface_inversion_v2(levels, InversionConfig())
    assert not result.detected
    assert result.candidate
    assert result.quality == "rejected_no_lapse"
