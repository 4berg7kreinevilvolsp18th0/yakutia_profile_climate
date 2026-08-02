from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import export_offline_dashboard


def _write_payload(path: Path, *, schema: str = "observations_v1") -> None:
    payload = {
        "schema": schema,
        "station_name": "Aldan",
        "months": {
            "2026-01": {
                "days": [
                    {
                        "date": "2026-01-01",
                        "n_profiles": 1,
                        "inversion_detected": False,
                        "t_surface_c": -30.0,
                        "observations": [],
                        "day_mean": {
                            "heights_m": [100.0, 1000.0],
                            "temperature_c": [-30.0, -35.0],
                        },
                    }
                ]
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_offline_dashboard_supports_observations_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "daily_profiles.json"
    out_path = tmp_path / "dashboard.html"
    _write_payload(data_path)
    monkeypatch.setattr(export_offline_dashboard, "DATA_PATH", data_path)
    monkeypatch.setattr(export_offline_dashboard, "OUT_PATH", out_path)

    assert export_offline_dashboard.main() == 0

    html = out_path.read_text(encoding="utf-8")
    assert '<script src="https://cdn.plot.ly' not in html
    assert "function daySeries(day)" in html
    assert "day.day_mean || day" in html
    assert "Plotly.newPlot" in html


def test_offline_dashboard_rejects_old_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "daily_profiles.json"
    _write_payload(data_path, schema="daily_v0")
    monkeypatch.setattr(export_offline_dashboard, "DATA_PATH", data_path)

    with pytest.raises(ValueError, match="observations_v1"):
        export_offline_dashboard.main()
