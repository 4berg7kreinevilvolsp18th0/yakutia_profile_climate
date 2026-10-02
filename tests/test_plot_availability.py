import matplotlib.pyplot as plt
import numpy as np

from gdex_bufr.profile_climate.article_figures.availability import has_plotted_data
from gdex_bufr.profile_climate.article_figures.pipeline import save_figure
from gdex_bufr.profile_climate.article_figures.config import FigureStyle


def test_empty_figure_not_exported(tmp_path):
    figure, axis = plt.subplots()
    axis.text(.5, .5, "Нет данных")
    axis.axhline(0)
    assert not has_plotted_data(figure)
    assert save_figure(figure, tmp_path / "empty", FigureStyle()) == []
    assert not (tmp_path / "empty.png").exists()


def test_all_missing_matrix_is_empty():
    figure, axis = plt.subplots()
    image = axis.imshow(np.full((3, 3), np.nan))
    figure.colorbar(image)
    assert not has_plotted_data(figure)
    plt.close(figure)


def test_zero_frequency_is_real_data():
    figure, axis = plt.subplots()
    axis.plot([1, 2], [0, 0])
    assert has_plotted_data(figure)
    plt.close(figure)


def test_missing_bars_do_not_count_as_observations():
    figure, axis = plt.subplots()
    axis.bar([1, 2], [np.nan, np.nan])
    assert not has_plotted_data(figure)
    plt.close(figure)


def test_stale_figure_archived_when_new_data_empty(tmp_path):
    previous = tmp_path / "empty.png"
    previous.write_bytes(b"old figure")
    figure, axis = plt.subplots()
    assert save_figure(figure, tmp_path / "empty", FigureStyle()) == []
    assert not previous.exists()
    archived = list((tmp_path / "_excluded").rglob("empty.png"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == b"old figure"
