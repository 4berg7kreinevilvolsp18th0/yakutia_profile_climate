"""Тесты аудита и инвариантов графиков статьи."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from revision_2026.metrics import gamma_local_interval, gamma_sfc_to_level, valid_layers


def test_gamma_sign_normal_lapse_negative():
    assert gamma_local_interval(0.0, -1.0, 0.0, 100.0) < 0


def test_gamma_sign_inversion_positive():
    assert gamma_local_interval(0.0, 1.0, 0.0, 100.0) > 0


def test_depth_equals_top_minus_base():
    layers = pd.DataFrame(
        {
            "base_height_agl_m": [0.0, 100.0],
            "top_height_agl_m": [200.0, 400.0],
            "depth_m": [200.0, 300.0],
            "month": [1, 2],
        }
    )
    out = valid_layers(layers)
    assert len(out) == 2
    assert bool(np.allclose(out["depth_m"], out["top_height_agl_m"] - out["base_height_agl_m"]))


def test_cycle_00_12_separation():
    df = pd.DataFrame({"cycle": ["00", "12", "06", "00"]})
    cy = df["cycle"].astype(str).str.zfill(2).str[-2:]
    assert set(cy[cy.isin(["00", "12"])]) == {"00", "12"}
    assert (cy == "06").sum() == 1


def test_gamma_sfc_no_extrapolation():
    assert np.isnan(gamma_sfc_to_level(0.0, -5.0, 0.0, -100.0))


def test_marginal_histogram_shared_bins():
    bins = np.linspace(0, 100, 11)
    a = np.array([10, 20, 30])
    b = np.array([15, 25])
    h1, _ = np.histogram(a, bins=bins)
    h2, _ = np.histogram(b, bins=bins)
    assert h1.shape == h2.shape
