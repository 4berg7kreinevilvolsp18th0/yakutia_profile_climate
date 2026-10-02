"""Проверки адаптеров не требуют установки нативных библиотек в обычной CI."""
import json
from types import SimpleNamespace

import pytest

from gdex_bufr.decoder_backends import decode
from gdex_bufr.decoder_backends.common import framed_messages, profile_from_pairs
from gdex_bufr.decoder_backends.embedded_tables import normalize_ncep_sequences, load_wmo_tables, write_eccodes_definitions
from gdex_bufr.decoder_backends.ncep_backend import parse_mnemonics
from gdex_bufr.profile_climate.fast_pipeline import build_parser
from gdex_bufr.profile_climate.inversion import _strictly_above, detect_surface_inversion
from scripts.compare_decoders import compare_values


def test_parser_keeps_old_default_and_accepts_native_backends():
    assert build_parser().parse_args([]).decoder == "pybufrkit"
    assert build_parser().parse_args(["--decoder", "ncepbufr"]).decoder == "ncepbufr"


def test_invalid_backend_fails_explicitly(tmp_path):
    with pytest.raises(ValueError, match="Неизвестный"):
        decode(tmp_path / "input", backend="invented")


@pytest.mark.parametrize("backend", ["eccodes", "ncepbufr"])
@pytest.mark.parametrize("options", [{"collect_elements": True}, {"strict": False}, {"max_profiles": 1}])
def test_native_options_are_not_silently_ignored(tmp_path, backend, options):
    with pytest.raises(ValueError):
        decode(tmp_path / "input", backend=backend, **options)


def test_native_table_version_uses_only_requested_path(tmp_path, monkeypatch):
    import sys
    definitions = tmp_path / "defs"
    source = definitions / "bufr/tables/0/wmo/12"
    source.mkdir(parents=True)
    (source / "element.table").write_text("#comment\n001001|blockNumber|long|WMO BLOCK|Numeric|0|0|7|0|0|0\n")
    (source / "sequence.def").write_text('"301001" = [ 001001, 001002 ]\n')
    monkeypatch.setitem(sys.modules, "eccodes", SimpleNamespace(codes_definition_path=lambda: str(definitions)))
    registry = SimpleNamespace(tables_root=tmp_path / "missing")
    b, d = load_wmo_tables(registry, 0, 12)
    assert b["001001"] == ["WMO BLOCK", "Numeric", 0, 0, 7]
    assert d["301001"][1] == [1001, 1002]
    with pytest.raises(ValueError, match="0/99"):
        load_wmo_tables(registry, 0, 99)


def test_reserved_empty_sequences_not_emitted(tmp_path, monkeypatch):
    from gdex_bufr.decoder_backends import embedded_tables as tables
    monkeypatch.setattr(tables, "read_embedded_tables", lambda *args: ({}, {}, {(0, 36)}))
    monkeypatch.setattr(tables, "load_wmo_tables", lambda *args: ({}, {"303051": ["reserved", []], "301001": ["pair", [1001, 1002]]}))
    write_eccodes_definitions(b"", None, tmp_path)
    assert (tmp_path / "bufr/tables/0/wmo/36/sequence.def").read_text() == '"301001" = [ 001001, 001002 ]\n'


def test_bufr_framing_and_truncation():
    message = b"BUFR" + (12).to_bytes(3, "big") + b"\x03" + b"7777"
    assert list(framed_messages(b"padding" + message + b"\0\0" + message)) == [message, message]
    with pytest.raises(ValueError, match="Повреждён"):
        list(framed_messages(message[:-1]))
    with pytest.raises(ValueError, match="Нет читаемых"):
        list(framed_messages(b"not a message"))


def test_ncep_replication_prefix_inlined_without_flattening_other_sequences():
    tables = {"360001": ["repeat", [101000, 31002]],
              "361123": ["level", [8001, 7004]],
              "363001": ["report", [1001, 360001, 361123]]}
    normalized = normalize_ncep_sequences(tables)
    assert normalized["363001"][1] == [1001, 101000, 31002, 361123]
    assert tables["363001"][1] == [1001, 360001, 361123]


def test_ncep_mnemonics_are_read_from_embedded_table():
    text = "| WDIR1 | 011001 | OTHER |\n| WDIR | 011001 | WIND |\n| TMDB | 012225 | TEMP |"
    assert parse_mnemonics(text) == {11001: "WDIR", 12225: "TMDB"}


def test_numeric_comparison_does_not_hide_scientific_differences():
    field = "/metrics/inversion_from_top_tops"
    assert not compare_values(json.dumps([{"delta": .2}]), json.dumps([{"delta": .20000000000004}]), field)
    assert compare_values([{"quality": "confirmed"}], [{"quality": "rejected"}])
    assert compare_values([1., None], [1., 0.])
    assert compare_values([1.], [1., 2.])


@pytest.mark.parametrize("noise", [-5e-14, 0, 5e-14])
def test_temperature_threshold_not_dependent_on_last_float_bit(noise):
    assert not _strictly_above(.2 + noise, .2)
    assert _strictly_above(.201 + noise, .2)
    levels = [{"pressure_hpa": p, "temperature_c": t} for p, t in
              [(1000, 0.), (980, 1.), (960, .8 + noise), (940, .6 + noise)]]
    assert detect_surface_inversion(levels).inversion_detected
