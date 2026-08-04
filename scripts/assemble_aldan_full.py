"""Сборка daily/dashboard/PNG для результаты-алдан-полный (актуальное не трогаем)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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

    daily_json = results / "daily_profiles.json"
    cmd_daily = [
        sys.executable, "scripts/build_daily_profiles.py",
        "--long", str(long_csv),
        "--metrics", str(metrics_csv),
        "--output", str(daily_json),
    ]
    print("RUN", " ".join(cmd_daily))
    rc = subprocess.call(cmd_daily, cwd=str(ROOT))
    if rc != 0:
        return rc

    if not args.skip_dashboard:
        # export_offline_dashboard uses fixed paths — override via env-less copy invoke
        # Write dashboard next to daily json by temporarily patching through CLI args if supported
        dash_script = ROOT / "scripts" / "export_offline_dashboard.py"
        text = dash_script.read_text(encoding="utf-8")
        # Prefer running with monkeypatched paths via small inline runner
        runner = f"""
from pathlib import Path
import scripts.export_offline_dashboard as m
m.DATA_PATH = Path(r"{daily_json.resolve()}")
m.OUT_PATH = Path(r"{(results / 'aldan_dashboard.html').resolve()}")
raise SystemExit(m.main())
"""
        rc = subprocess.call([sys.executable, "-c", runner], cwd=str(ROOT))
        if rc != 0:
            return rc

    if not args.skip_plots:
        cmd_plots = [
            sys.executable, "-m", "gdex_bufr", "monthly-profile-plots",
            "--station", "aldan",
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--input", str(long_csv),
            "--metrics", str(metrics_csv),
            "--output", "gdex_outputs/monthly_temperature_profiles",
            "--set", args.plot_set,
        ]
        print("RUN", " ".join(cmd_plots))
        rc = subprocess.call(cmd_plots, cwd=str(ROOT))
        if rc != 0:
            return rc

    print("OK assemble", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
