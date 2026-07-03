"""Тесты месячных графиков profile_climate."""
from pathlib import Path

from gdex_bufr.profile_climate.plots import render_monthly_temperature_profiles


def _sample_rows():
    long_rows = []
    metrics_rows = []
    for day in range(1, 6):
        profile_id = f"31004_199901{day:02d}12_12"
        levels = [
            (1000, -30 + day * 0.1),
            (900, -25 + day * 0.1),
            (800, -28 + day * 0.1),
            (700, -32 + day * 0.1),
            (600, -36 + day * 0.1),
            (500, -40 + day * 0.1),
        ]
        for index, (pressure, temp) in enumerate(levels):
            long_rows.append({
                "station_id": "31004",
                "station_name": "Aldan",
                "datetime_utc": f"1999-01-{day:02d}T12:00:00Z",
                "year": 1999,
                "month": 1,
                "cycle": "12",
                "profile_id": profile_id,
                "level_index": index,
                "pressure_hpa": pressure,
                "temperature_c": temp,
                "height_m": None,
                "source_file": "test.bufr",
                "qc_flag": "",
            })
        metrics_rows.append({
            "profile_id": profile_id,
            "station_id": "31004",
            "station_name": "Aldan",
            "datetime_utc": f"1999-01-{day:02d}T12:00:00Z",
            "year": 1999,
            "month": 1,
            "cycle": "12",
            "profile_status": "good",
            "inversion_detected": day % 2 == 0,
        })
    return long_rows, metrics_rows


def test_render_monthly_png(tmp_path: Path):
    long_rows, metrics_rows = _sample_rows()
    output_path = tmp_path / "aldan_1999_01_temperature_profiles_to_500hpa.png"
    result = render_monthly_temperature_profiles(
        station_slug="aldan",
        station_name="Aldan",
        year=1999,
        month=1,
        long_rows=long_rows,
        metrics_rows=metrics_rows,
        output_path=output_path,
        min_profiles_per_month=3,
    )
    assert result is not None
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_empty_month_does_not_crash(tmp_path: Path):
    output_path = tmp_path / "empty.png"
    result = render_monthly_temperature_profiles(
        station_slug="aldan",
        station_name="Aldan",
        year=1999,
        month=2,
        long_rows=[],
        metrics_rows=[],
        output_path=output_path,
    )
    assert result is None
    assert not output_path.exists()
