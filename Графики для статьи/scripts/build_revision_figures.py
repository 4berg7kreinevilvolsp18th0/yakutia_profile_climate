from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.profile_climate.article_figures.config import AnalysisConfig, FigureStyle, load_yaml_config
from revision_2026.pipeline import build_revision
from revision_2026.style import revision_style


def main() -> int:
    parser = argparse.ArgumentParser(description="Ревизия рисунков статьи (отдельные папки, без перезаписи sample_output)")
    parser.add_argument("--input", required=True, help="profiles_long.csv")
    parser.add_argument("--output", default="revision_2026/output")
    parser.add_argument("--config", help="YAML с параметрами анализа")
    args = parser.parse_args()

    if args.config:
        analysis, style = load_yaml_config(args.config)
    else:
        analysis, style = AnalysisConfig(), FigureStyle()
    style = revision_style(style)
    summary = build_revision(args.input, args.output, analysis, style)
    print(
        f"Готово: {summary['eligible_profiles']} пригодных профилей, "
        f"{summary['valid_layers']} слоёв; {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
