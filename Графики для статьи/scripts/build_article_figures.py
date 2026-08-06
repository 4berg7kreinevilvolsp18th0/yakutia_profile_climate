from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.profile_climate.article_figures.config import AnalysisConfig, FigureStyle, load_yaml_config
from gdex_bufr.profile_climate.article_figures.pipeline import build_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать все рисунки и таблицы для статьи по Алдану")
    parser.add_argument("--input", required=True, help="profiles_long.csv")
    parser.add_argument("--output", default="gdex_outputs/article_figures", help="Каталог результата")
    parser.add_argument("--config", help="YAML с параметрами")
    parser.add_argument("--show-title", action="store_true", help="Показывать заголовки внутри рисунков")
    parser.add_argument("--language", choices=["ru", "en"], default=None)
    args = parser.parse_args()

    if args.config:
        analysis, style = load_yaml_config(args.config)
    else:
        analysis, style = AnalysisConfig(), FigureStyle()
    style_dict = style.__dict__.copy()
    if args.show_title:
        style_dict["show_title"] = True
    if args.language:
        style_dict["language"] = args.language
    style = FigureStyle(**style_dict)

    summary = build_all(args.input, args.output, analysis, style)
    print(f"Готово: {summary['profiles']} профилей; результат: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
