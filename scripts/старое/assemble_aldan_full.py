"""Сборка daily/dashboard/PNG для результаты-алдан-полный (актуальное не трогаем)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> int:
    print("RUN", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def _build_daily(results: Path) -> int:
    return _run([
        sys.executable, "scripts/build_daily_profiles.py",
        "--long", str(results / "profiles_long.csv"),
        "--metrics", str(results / "profile_metrics.csv"),
        "--output", str(results / "daily_profiles.json"),
    ])


def _build_dashboard(results: Path, daily_json: Path) -> int:
    # export_offline_dashboard использует фиксированные пути — подменяем через inline-runner
    runner = f"""
from pathlib import Path
import scripts.export_offline_dashboard as m
m.DATA_PATH = Path(r"{daily_json.resolve()}")
m.OUT_PATH = Path(r"{(results / 'aldan_dashboard.html').resolve()}")
raise SystemExit(m.main())
"""
    return subprocess.call([sys.executable, "-c", runner], cwd=str(ROOT))


def _build_plots(args: argparse.Namespace, long_csv: Path, metrics_csv: Path) -> int:
    return _run([
        sys.executable, "-m", "gdex_bufr", "monthly-profile-plots",
        "--station", "aldan",
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--input", str(long_csv),
        "--metrics", str(metrics_csv),
        "--output", "gdex_outputs/monthly_temperature_profiles",
        "--set", args.plot_set,
    ])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="gdex_outputs/результаты-алдан-полный")
    p.add_argument("--start-date", default="1999-10-01")
    p.add_argument("--end-date", default="2026-07-19")
    p.add_argument("--plot-set", default="полный_все_циклы")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--skip-dashboard", action="store_true")
    args = p.parse_args(argv)

    results = Path(args.results_dir)
    long_csv = results / "profiles_long.csv"
    metrics_csv = results / "profile_metrics.csv"
    if not long_csv.exists() or not metrics_csv.exists():
        print(f"Нет CSV в {results}", file=sys.stderr)
        return 1

    # 1) daily JSON
    rc = _build_daily(results)
    if rc != 0:
        return rc

    # 2) offline dashboard
    if not args.skip_dashboard:
        rc = _build_dashboard(results, results / "daily_profiles.json")
        if rc != 0:
            return rc

    # 3) месячные PNG
    if not args.skip_plots:
        rc = _build_plots(args, long_csv, metrics_csv)
        if rc != 0:
            return rc

    print("OK assemble", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
