"""Конфигурация метеодиаграмм (YAML + dataclass)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return default


@dataclass
class ThemeColors:
    temperature: str = "#FF4757"
    dewpoint: str = "#00CEC9"
    wind_speed: str = "#00B894"
    wind_direction: str = "#A29BFE"
    hodograph: str = "#6C5CE7"
    rh: str = "#00D2FF"
    map_stations: str = "#FF7675"
    barbs: str = "#2D3436"
    thermo_box: str = "#FFF3BF"
    height: str = "#E056FD"
    theta_e: str = "#FFA502"
    ttd: str = "#1E90FF"
    wind_shear: str = "#26DE81"
    axis_reference: str = "#636E72"


@dataclass
class PlotTheme:
    figure_facecolor: str = "#ffffff"
    axes_facecolor: str = "#f4f7fb"
    grid_color: str = "#b8c5d6"
    text_color: str = "#1a2332"
    colors: ThemeColors = field(default_factory=ThemeColors)


@dataclass
class SkewTStyle:
    figsize: tuple[float, float] = (10.0, 10.0)
    rotation: float = 45.0
    pressure_min_hpa: float = 100.0
    pressure_max_hpa: float = 1000.0
    temp_min_c: float = -50.0
    temp_max_c: float = 30.0
    show_barbs: bool = True
    barb_skip: int = 2
    show_dry_adiabats: bool = True
    show_moist_adiabats: bool = True
    show_mixing_lines: bool = True
    color_temperature: str = "#FF4757"
    color_dewpoint: str = "#00CEC9"
    color_barbs: str = "#2D3436"
    linewidth: float = 2.5
    marker: str = "o"
    markersize: float = 6.0
    title_template: str = "Skew-T {station_id} {datetime}"


@dataclass
class ProfileStyle:
    figsize: tuple[float, float] = (7.0, 9.0)
    color_temperature: str = "#c7392b"
    color_dewpoint: str = "#2980b9"
    marker: str = "o"
    markersize: float = 5.0
    linewidth: float = 1.5
    show_grid: bool = True
    grid_alpha: float = 0.3
    pressure_tick_step_hpa: float = 50.0
    title_template: str = "Vertical profile {station_id} {datetime}"


@dataclass
class WindStyle:
    figsize: tuple[float, float] = (10.0, 8.0)
    color_speed: str = "#16a085"
    color_direction: str = "#8e44ad"
    marker: str = "o"
    markersize: float = 5.0
    linewidth: float = 1.5
    show_grid: bool = True
    grid_alpha: float = 0.3
    title_template: str = "Wind profile {station_id} {datetime}"


@dataclass
class HodographStyle:
    figsize: tuple[float, float] = (7.0, 7.0)
    color: str = "#3C7461"
    marker: str = "o"
    markersize: float = 3.0
    linewidth: float = 1.5
    show_grid: bool = True
    grid_alpha: float = 0.3
    title_template: str = "Hodograph {station_id} {datetime}"


@dataclass
class RhStyle:
    figsize: tuple[float, float] = (6.0, 9.0)
    color: str = "#1abc9c"
    marker: str = "o"
    markersize: float = 5.0
    linewidth: float = 1.5
    rh_min: float = 0.0
    rh_max: float = 100.0
    show_grid: bool = True
    grid_alpha: float = 0.3
    title_template: str = "Relative humidity {station_id} {datetime}"


@dataclass
class HeightStyle:
    figsize: tuple[float, float] = (6.0, 9.0)
    color: str = "#6c3483"
    marker: str = "o"
    markersize: float = 5.0
    linewidth: float = 1.5
    show_grid: bool = True
    grid_alpha: float = 0.3
    title_template: str = "Geopotential height {station_id} {datetime}"


@dataclass
class ThetaEStyle:
    figsize: tuple[float, float] = (6.0, 9.0)
    color: str = "#ca6f1e"
    marker: str = "o"
    markersize: float = 5.0
    linewidth: float = 1.5
    show_grid: bool = True
    grid_alpha: float = 0.3
    title_template: str = "Equivalent potential temperature {station_id} {datetime}"


@dataclass
class TtdStyle:
    figsize: tuple[float, float] = (6.0, 9.0)
    color: str = "#2874a6"
    marker: str = "o"
    markersize: float = 5.0
    linewidth: float = 1.5
    show_grid: bool = True
    grid_alpha: float = 0.3
    title_template: str = "T-Td spread {station_id} {datetime}"


@dataclass
class WindShearStyle:
    figsize: tuple[float, float] = (8.0, 9.0)
    color: str = "#117a65"
    marker: str = "o"
    markersize: float = 5.0
    linewidth: float = 1.5
    layer_hpa: float = 50.0
    show_grid: bool = True
    grid_alpha: float = 0.3
    title_template: str = "Wind shear {station_id} {datetime}"


@dataclass
class MapStyle:
    figsize: tuple[float, float] = (12.0, 6.0)
    marker_size: float = 12.0
    marker_alpha: float = 0.7
    color: str = "#d35400"
    show_grid: bool = True
    grid_alpha: float = 0.3
    title_template: str = "Stations map ({count} profiles)"


@dataclass
class ThermoStyle:
    figsize: tuple[float, float] = (8.0, 6.0)
    show_cape_cin: bool = True
    show_lcl: bool = True
    show_lfc_el: bool = True
    font_size: float = 11.0
    box_facecolor: str = "#fef9e7"
    text_color: str = "#2c3e50"
    title_template: str = "Thermo indices {station_id} {datetime}"


@dataclass
class CompositeStyle:
    figsize: tuple[float, float] = (16.0, 12.0)
    include_skewt: bool = True
    include_profile: bool = True
    include_wind: bool = True
    include_hodograph: bool = True
    include_rh: bool = True


@dataclass
class PlotStyle:
    """Полный набор настроек визуализации."""

    plot_types: list[str] = field(
        default_factory=lambda: [
            "skewt", "profile", "wind", "hodograph", "rh", "thermo",
            "height", "theta_e", "ttd", "wind_shear", "composite", "map",
        ]
    )
    dpi: int = 200
    export_format: str = "png"
    max_profiles_per_file: int | None = 5
    min_levels: int = 5
    output_subdir_by_date: bool = True
    output_subdir_by_cycle: bool = False
    skip_existing: bool = True
    workers: int = 4
    export_xlsx: bool = True
    title_datetime_format: str = "%Y-%m-%d %HZ"
    pressure_tick_step_hpa: float = 50.0

    theme: PlotTheme = field(default_factory=PlotTheme)
    skewt: SkewTStyle = field(default_factory=SkewTStyle)
    profile: ProfileStyle = field(default_factory=ProfileStyle)
    wind: WindStyle = field(default_factory=WindStyle)
    hodograph: HodographStyle = field(default_factory=HodographStyle)
    rh: RhStyle = field(default_factory=RhStyle)
    height: HeightStyle = field(default_factory=HeightStyle)
    theta_e: ThetaEStyle = field(default_factory=ThetaEStyle)
    ttd: TtdStyle = field(default_factory=TtdStyle)
    wind_shear: WindShearStyle = field(default_factory=WindShearStyle)
    map: MapStyle = field(default_factory=MapStyle)
    thermo: ThermoStyle = field(default_factory=ThermoStyle)
    composite: CompositeStyle = field(default_factory=CompositeStyle)

    def format_title(self, template: str, profile) -> str:
        dt = profile.report_datetime_utc or ""
        if dt and self.title_datetime_format:
            try:
                from datetime import datetime

                parsed = datetime.fromisoformat(dt.replace("Z", "+00:00") if "T" in dt else dt)
                dt = parsed.strftime(self.title_datetime_format)
            except ValueError:
                pass
        return template.format(
            station_id=profile.station_id or "unknown",
            datetime=dt,
            count="{count}",
        )

    def output_suffix(self) -> str:
        fmt = self.export_format.lower().lstrip(".")
        return f".{fmt}" if fmt else ".png"


def apply_matplotlib_theme(theme: PlotTheme) -> None:
    """Применяю общие цвета фона и сетки ко всем графикам."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": theme.figure_facecolor,
        "axes.facecolor": theme.axes_facecolor,
        "axes.edgecolor": theme.text_color,
        "axes.labelcolor": theme.text_color,
        "text.color": theme.text_color,
        "xtick.color": theme.text_color,
        "ytick.color": theme.text_color,
        "grid.color": theme.grid_color,
    })


def _color(raw_colors: dict[str, Any], key: str, theme: PlotTheme, fallback: str) -> str:
    return str(raw_colors.get(key, getattr(theme.colors, key, fallback)))


def _load_theme(raw: dict[str, Any] | None) -> PlotTheme:
    raw = raw or {}
    colors_raw = raw.get("colors") or {}
    colors = ThemeColors(
        temperature=str(colors_raw.get("temperature", ThemeColors.temperature)),
        dewpoint=str(colors_raw.get("dewpoint", ThemeColors.dewpoint)),
        wind_speed=str(colors_raw.get("wind_speed", ThemeColors.wind_speed)),
        wind_direction=str(colors_raw.get("wind_direction", ThemeColors.wind_direction)),
        hodograph=str(colors_raw.get("hodograph", ThemeColors.hodograph)),
        rh=str(colors_raw.get("rh", ThemeColors.rh)),
        map_stations=str(colors_raw.get("map_stations", ThemeColors.map_stations)),
        barbs=str(colors_raw.get("barbs", ThemeColors.barbs)),
        thermo_box=str(colors_raw.get("thermo_box", ThemeColors.thermo_box)),
        height=str(colors_raw.get("height", ThemeColors.height)),
        theta_e=str(colors_raw.get("theta_e", ThemeColors.theta_e)),
        ttd=str(colors_raw.get("ttd", ThemeColors.ttd)),
        wind_shear=str(colors_raw.get("wind_shear", ThemeColors.wind_shear)),
        axis_reference=str(colors_raw.get("axis_reference", ThemeColors.axis_reference)),
    )
    return PlotTheme(
        figure_facecolor=str(raw.get("figure_facecolor", PlotTheme.figure_facecolor)),
        axes_facecolor=str(raw.get("axes_facecolor", PlotTheme.axes_facecolor)),
        grid_color=str(raw.get("grid_color", PlotTheme.grid_color)),
        text_color=str(raw.get("text_color", PlotTheme.text_color)),
        colors=colors,
    )


def _load_skewt(raw: dict[str, Any], theme: PlotTheme) -> SkewTStyle:
    colors = raw.get("colors") or {}
    return SkewTStyle(
        figsize=_pair(raw.get("figsize"), (10.0, 10.0)),
        rotation=float(raw.get("rotation", 45.0)),
        pressure_min_hpa=float(raw.get("pressure_min_hpa", raw.get("p_min", 100.0))),
        pressure_max_hpa=float(raw.get("pressure_max_hpa", raw.get("p_max", 1000.0))),
        temp_min_c=float(raw.get("temp_min_c", raw.get("t_min", -50.0))),
        temp_max_c=float(raw.get("temp_max_c", raw.get("t_max", 30.0))),
        show_barbs=bool(raw.get("show_barbs", True)),
        barb_skip=int(raw.get("barb_skip", 2)),
        show_dry_adiabats=bool(raw.get("show_dry_adiabats", True)),
        show_moist_adiabats=bool(raw.get("show_moist_adiabats", True)),
        show_mixing_lines=bool(raw.get("show_mixing_lines", True)),
        color_temperature=_color(colors, "temperature", theme, theme.colors.temperature),
        color_dewpoint=_color(colors, "dewpoint", theme, theme.colors.dewpoint),
        color_barbs=_color(colors, "barbs", theme, theme.colors.barbs),
        linewidth=float(raw.get("linewidth", 2.0)),
        marker=str(raw.get("marker", "o")),
        markersize=float(raw.get("markersize", 6.0)),
        title_template=str(raw.get("title_template", "Skew-T {station_id} {datetime}")),
    )


def _load_profile(raw: dict[str, Any], theme: PlotTheme) -> ProfileStyle:
    colors = raw.get("colors") or {}
    return ProfileStyle(
        figsize=_pair(raw.get("figsize"), (7.0, 9.0)),
        color_temperature=_color(colors, "temperature", theme, theme.colors.temperature),
        color_dewpoint=_color(colors, "dewpoint", theme, theme.colors.dewpoint),
        marker=str(raw.get("marker", "o")),
        markersize=float(raw.get("markersize", 5.0)),
        linewidth=float(raw.get("linewidth", 1.5)),
        show_grid=bool(raw.get("show_grid", True)),
        grid_alpha=float(raw.get("grid_alpha", 0.3)),
        pressure_tick_step_hpa=float(raw.get("pressure_tick_step_hpa", 50.0)),
        title_template=str(raw.get("title_template", "Vertical profile {station_id} {datetime}")),
    )


def _load_wind(raw: dict[str, Any], theme: PlotTheme) -> WindStyle:
    colors = raw.get("colors") or {}
    return WindStyle(
        figsize=_pair(raw.get("figsize"), (10.0, 8.0)),
        color_speed=_color(colors, "speed", theme, theme.colors.wind_speed),
        color_direction=_color(colors, "direction", theme, theme.colors.wind_direction),
        marker=str(raw.get("marker", "o")),
        markersize=float(raw.get("markersize", 5.0)),
        linewidth=float(raw.get("linewidth", 1.5)),
        show_grid=bool(raw.get("show_grid", True)),
        grid_alpha=float(raw.get("grid_alpha", 0.3)),
        title_template=str(raw.get("title_template", "Wind profile {station_id} {datetime}")),
    )


def _load_hodograph(raw: dict[str, Any], theme: PlotTheme) -> HodographStyle:
    return HodographStyle(
        figsize=_pair(raw.get("figsize"), (7.0, 7.0)),
        color=str(raw.get("color", theme.colors.hodograph)),
        marker=str(raw.get("marker", "o")),
        markersize=float(raw.get("markersize", 3.0)),
        linewidth=float(raw.get("linewidth", 1.5)),
        show_grid=bool(raw.get("show_grid", True)),
        grid_alpha=float(raw.get("grid_alpha", 0.3)),
        title_template=str(raw.get("title_template", "Hodograph {station_id} {datetime}")),
    )


def _load_rh(raw: dict[str, Any], theme: PlotTheme) -> RhStyle:
    return RhStyle(
        figsize=_pair(raw.get("figsize"), (6.0, 9.0)),
        color=str(raw.get("color", theme.colors.rh)),
        marker=str(raw.get("marker", "o")),
        markersize=float(raw.get("markersize", 5.0)),
        linewidth=float(raw.get("linewidth", 1.5)),
        rh_min=float(raw.get("rh_min", 0.0)),
        rh_max=float(raw.get("rh_max", 100.0)),
        show_grid=bool(raw.get("show_grid", True)),
        grid_alpha=float(raw.get("grid_alpha", 0.3)),
        title_template=str(raw.get("title_template", "Relative humidity {station_id} {datetime}")),
    )


def _load_height(raw: dict[str, Any], theme: PlotTheme) -> HeightStyle:
    return HeightStyle(
        figsize=_pair(raw.get("figsize"), (6.0, 9.0)),
        color=str(raw.get("color", theme.colors.height)),
        marker=str(raw.get("marker", "o")),
        markersize=float(raw.get("markersize", 5.0)),
        linewidth=float(raw.get("linewidth", 1.5)),
        show_grid=bool(raw.get("show_grid", True)),
        grid_alpha=float(raw.get("grid_alpha", 0.3)),
        title_template=str(raw.get("title_template", "Geopotential height {station_id} {datetime}")),
    )


def _load_theta_e(raw: dict[str, Any], theme: PlotTheme) -> ThetaEStyle:
    return ThetaEStyle(
        figsize=_pair(raw.get("figsize"), (6.0, 9.0)),
        color=str(raw.get("color", theme.colors.theta_e)),
        marker=str(raw.get("marker", "o")),
        markersize=float(raw.get("markersize", 5.0)),
        linewidth=float(raw.get("linewidth", 1.5)),
        show_grid=bool(raw.get("show_grid", True)),
        grid_alpha=float(raw.get("grid_alpha", 0.3)),
        title_template=str(raw.get("title_template", "Equivalent potential temperature {station_id} {datetime}")),
    )


def _load_ttd(raw: dict[str, Any], theme: PlotTheme) -> TtdStyle:
    return TtdStyle(
        figsize=_pair(raw.get("figsize"), (6.0, 9.0)),
        color=str(raw.get("color", theme.colors.ttd)),
        marker=str(raw.get("marker", "o")),
        markersize=float(raw.get("markersize", 5.0)),
        linewidth=float(raw.get("linewidth", 1.5)),
        show_grid=bool(raw.get("show_grid", True)),
        grid_alpha=float(raw.get("grid_alpha", 0.3)),
        title_template=str(raw.get("title_template", "T-Td spread {station_id} {datetime}")),
    )


def _load_wind_shear(raw: dict[str, Any], theme: PlotTheme) -> WindShearStyle:
    return WindShearStyle(
        figsize=_pair(raw.get("figsize"), (8.0, 9.0)),
        color=str(raw.get("color", theme.colors.wind_shear)),
        marker=str(raw.get("marker", "o")),
        markersize=float(raw.get("markersize", 5.0)),
        linewidth=float(raw.get("linewidth", 1.5)),
        layer_hpa=float(raw.get("layer_hpa", 50.0)),
        show_grid=bool(raw.get("show_grid", True)),
        grid_alpha=float(raw.get("grid_alpha", 0.3)),
        title_template=str(raw.get("title_template", "Wind shear {station_id} {datetime}")),
    )


def _load_map(raw: dict[str, Any], theme: PlotTheme) -> MapStyle:
    return MapStyle(
        figsize=_pair(raw.get("figsize"), (12.0, 6.0)),
        marker_size=float(raw.get("marker_size", 12.0)),
        marker_alpha=float(raw.get("marker_alpha", 0.7)),
        color=str(raw.get("color", theme.colors.map_stations)),
        show_grid=bool(raw.get("show_grid", True)),
        grid_alpha=float(raw.get("grid_alpha", 0.3)),
        title_template=str(raw.get("title_template", "Stations map ({count} profiles)")),
    )


def _load_thermo(raw: dict[str, Any], theme: PlotTheme) -> ThermoStyle:
    return ThermoStyle(
        figsize=_pair(raw.get("figsize"), (8.0, 6.0)),
        show_cape_cin=bool(raw.get("show_cape_cin", True)),
        show_lcl=bool(raw.get("show_lcl", True)),
        show_lfc_el=bool(raw.get("show_lfc_el", True)),
        font_size=float(raw.get("font_size", 11.0)),
        box_facecolor=str(raw.get("box_facecolor", theme.colors.thermo_box)),
        text_color=str(raw.get("text_color", theme.text_color)),
        title_template=str(raw.get("title_template", "Thermo indices {station_id} {datetime}")),
    )


def _load_composite(raw: dict[str, Any]) -> CompositeStyle:
    return CompositeStyle(
        figsize=_pair(raw.get("figsize"), (16.0, 12.0)),
        include_skewt=bool(raw.get("include_skewt", True)),
        include_profile=bool(raw.get("include_profile", True)),
        include_wind=bool(raw.get("include_wind", True)),
        include_hodograph=bool(raw.get("include_hodograph", True)),
        include_rh=bool(raw.get("include_rh", True)),
    )


def load_plot_style(raw: dict[str, Any] | None) -> PlotStyle:
    raw = raw or {}
    theme = _load_theme(raw.get("theme"))
    plot_types = raw.get("plot_types") or raw.get("enabled")
    if not plot_types:
        plot_types = [
            "skewt", "profile", "wind", "hodograph", "rh", "thermo",
            "height", "theta_e", "ttd", "wind_shear", "composite", "map",
        ]
    style = PlotStyle(
        plot_types=[str(p).strip() for p in plot_types if str(p).strip()],
        dpi=int(raw.get("dpi", 200)),
        export_format=str(raw.get("export_format", raw.get("format", "png"))),
        max_profiles_per_file=raw.get("max_profiles_per_file"),
        min_levels=int(raw.get("min_levels", 5)),
        output_subdir_by_date=bool(raw.get("output_subdir_by_date", True)),
        output_subdir_by_cycle=bool(raw.get("output_subdir_by_cycle", False)),
        skip_existing=bool(raw.get("skip_existing", True)),
        workers=int(raw.get("workers", 4)),
        export_xlsx=bool(raw.get("export_xlsx", raw.get("export_jsonl", True))),
        title_datetime_format=str(raw.get("title_datetime_format", "%Y-%m-%d %HZ")),
        pressure_tick_step_hpa=float(
            raw.get(
                "pressure_tick_step_hpa",
                (raw.get("profile") or {}).get("pressure_tick_step_hpa", 50.0),
            )
        ),
        theme=theme,
        skewt=_load_skewt(raw.get("skewt") or {}, theme),
        profile=_load_profile(raw.get("profile") or {}, theme),
        wind=_load_wind(raw.get("wind") or {}, theme),
        hodograph=_load_hodograph(raw.get("hodograph") or {}, theme),
        rh=_load_rh(raw.get("rh") or {}, theme),
        height=_load_height(raw.get("height") or {}, theme),
        theta_e=_load_theta_e(raw.get("theta_e") or {}, theme),
        ttd=_load_ttd(raw.get("ttd") or {}, theme),
        wind_shear=_load_wind_shear(raw.get("wind_shear") or {}, theme),
        map=_load_map(raw.get("map") or {}, theme),
        thermo=_load_thermo(raw.get("thermo") or {}, theme),
        composite=_load_composite(raw.get("composite") or {}),
    )
    apply_matplotlib_theme(style.theme)
    return style
