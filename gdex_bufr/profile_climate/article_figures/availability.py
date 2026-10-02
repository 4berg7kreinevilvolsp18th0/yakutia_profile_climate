"""Не публикуем рисунок, если на нём нет ни одной конечной точки данных."""
from __future__ import annotations

import numpy as np


def has_plotted_data(figure) -> bool:
    """Смотрим на данные artists, а не на цвет фона и наличие подписей осей.

    Нулевая частота при существующей выборке — допустимый результат. Полностью
    пустые оси, NaN-матрицы и заглушки «Нет данных» не являются рисунком данных.
    """
    for axis in figure.axes:
        if axis.get_label() == "<colorbar>":
            continue
        for line in axis.lines:
            # axhline/axvline — вспомогательные ориентиры, не наблюдения.
            if line.get_transform() is not axis.transData or line.get_gid() == "reference":
                continue
            xy = np.ma.asarray(line.get_xydata(), dtype=float).filled(np.nan)
            if xy.size and np.isfinite(xy).all(axis=1).any():
                return True
        for artist in axis.collections:
            array = artist.get_array()
            if array is not None:
                values = np.ma.asarray(array, dtype=float).filled(np.nan)
                if np.isfinite(values).any():
                    return True
            offsets = getattr(artist, "_offsets3d", None)
            if offsets is not None:
                if len(offsets[0]) and np.isfinite(np.asarray(offsets, dtype=float)).all(axis=0).any():
                    return True
            elif type(artist).__name__ == "PathCollection":
                xy = np.ma.asarray(artist.get_offsets(), dtype=float).filled(np.nan)
                if xy.size and np.isfinite(xy).all(axis=1).any():
                    return True
            elif array is None and artist.get_paths():
                # Заливки и контуры, уже построенные из конечных координат.
                if any(np.isfinite(p.vertices).all(axis=1).any() for p in artist.get_paths()):
                    return True
        for patch in axis.patches:
            # У Rectangle исходный путь всегда единичный квадрат, даже если
            # высота столбца NaN. Проверяем координаты после преобразования.
            vertices = patch.get_patch_transform().transform(patch.get_path().vertices)
            if np.isfinite(vertices).all():
                return True
        for artist in axis.images:
            values = np.ma.asarray(artist.get_array(), dtype=float).filled(np.nan)
            if np.isfinite(values).any():
                return True
    return False
