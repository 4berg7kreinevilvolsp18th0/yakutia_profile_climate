"""Сводка hits без BUFR."""
from gdex_bufr.bufr_adapter import _station_filter_set
from gdex_bufr.profile_climate.export import split_rows_by_station
from gdex_bufr.station_index import summarize_hits


def test_split_rows_by_station():
    rows = [
        {"station_id": "31004", "x": 1},
        {"station_id": "24959", "x": 2},
        {"station_id": "31004", "x": 3},
    ]
    grouped = split_rows_by_station(rows, id_to_slug={"31004": "aldan", "24959": "yakutsk"})
    assert len(grouped["aldan"]) == 2
    assert len(grouped["yakutsk"]) == 1


def test_summarize_hits_per_station():
    rows = [
        {"station_id": "31004", "source_file": "a.bufr", "delayed": ""},
        {"station_id": "31004", "source_file": "b.bufr", "delayed": ""},
        {"station_id": "24959", "source_file": "a.bufr", "delayed": ""},
        {"station_id": "", "source_file": "c.bufr", "delayed": "ERROR:x"},
    ]
    stats = summarize_hits(
        rows,
        station_ids=["31004", "24959"],
        slug_by_id={"31004": "aldan", "24959": "yakutsk"},
    )
    assert stats["hits_total"] == 3
    assert stats["errors"] == 1
    assert stats["per_station"]["31004"]["hits"] == 2
    assert stats["per_station"]["31004"]["files"] == 2
    assert stats["per_station"]["24959"]["hits"] == 1


def test_station_filter_set_normalizes_ids():
    assert _station_filter_set("31004,24959") == {"31004", "24959"}
    assert _station_filter_set({"31004", "24959"}) == {"31004", "24959"}
    assert _station_filter_set(None) is None
