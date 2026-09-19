"""Copy rebuilt article figures from sample_output_qc into sample_output."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sample_output_qc"
DST = ROOT / "sample_output"
LOG = ROOT / "sample_output_qc" / "_copy_log.txt"


def main() -> None:
    lines: list[str] = []
    if not SRC.exists():
        lines.append(f"MISSING SRC {SRC}")
        LOG.write_text("\n".join(lines), encoding="utf-8")
        raise SystemExit(1)

    locked = 0
    copied = 0
    for name in ("tables", "figures"):
        s = SRC / name
        d = DST / name
        d.mkdir(parents=True, exist_ok=True)
        for p in s.rglob("*"):
            if not p.is_file():
                continue
            out = d / p.relative_to(s)
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(p, out)
                copied += 1
            except PermissionError:
                locked += 1
                alt = out.with_name(out.name + ".new")
                shutil.copy2(p, alt)
                lines.append(f"LOCKED -> {alt}")

    for extra in ("summary.json",):
        sp = SRC / extra
        if sp.exists():
            try:
                shutil.copy2(sp, DST / extra)
                copied += 1
            except PermissionError:
                shutil.copy2(sp, DST / f"{extra}.new")
                locked += 1
                lines.append(f"LOCKED -> {DST / f'{extra}.new'}")

    # quick QC check on destination (or .new)
    ig = DST / "tables" / "interval_gammas.csv"
    if not ig.exists():
        ig = DST / "tables" / "interval_gammas.csv.new"
    if ig.exists():
        import pandas as pd

        df = pd.read_csv(ig)
        usable = df["gamma_c_per_100m"].dropna()
        lines.append(f"interval_gammas rows={len(df)} usable={usable.notna().sum() if hasattr(usable,'notna') else len(usable)}")
        lines.append(f"thin={(df['gamma_qc_reason']=='thin_interval').sum() if 'gamma_qc_reason' in df.columns else 'n/a'}")
        lines.append(f"|g|>=20 usable={(usable.abs()>=20).sum()}")
        lines.append(f"|raw|>=20={(df['gamma_raw_c_per_100m'].abs()>=20).sum() if 'gamma_raw_c_per_100m' in df.columns else 'n/a'}")

    png_n = sum(1 for _ in (DST / "figures").rglob("*.png"))
    csv_n = sum(1 for _ in (DST / "tables").glob("*.csv"))
    lines.insert(0, f"copied={copied} locked={locked} dst_png={png_n} dst_csv={csv_n}")
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
