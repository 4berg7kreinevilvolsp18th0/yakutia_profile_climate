"""Общая адаптация декодированных значений, без повторного чтения битов BUFR.

Все движки используют одни формулы единиц/высоты и одну сборку профилей.
Это позволяет отдельно сравнивать распаковку и научную обработку.
"""
from collections import defaultdict
from types import SimpleNamespace


def framed_messages(raw: bytes):
    """NCEP может вставлять служебные байты между BUFR. Внутри длина обязательна."""
    offset = 0
    count = 0
    while True:
        start = raw.find(b"BUFR", offset)
        if start < 0:
            break
        if start + 8 > len(raw):
            raise ValueError("Оборван заголовок BUFR")
        length = int.from_bytes(raw[start + 4:start + 7], "big")
        end = start + length
        if length < 12 or end > len(raw) or raw[end - 4:end] != b"7777":
            raise ValueError(f"Повреждён BUFR по смещению {start}")
        count += 1
        yield raw[start:end]
        offset = end
    if not count:
        raise ValueError("Нет читаемых BUFR-сообщений")


def profile_from_pairs(path, pairs, subset_index, header, registry, backend, collect_elements=False):
    """Сохраняем порядок дескрипторов: независимо собранные ряды дают сдвиг уровней."""
    from gdex_bufr.bufr_adapter import _decode_subset

    descriptors = [SimpleNamespace(id=int(code)) for code, _ in pairs]
    values = [value for _, value in pairs]
    rows = defaultdict(list)
    for code, value in pairs:
        rows[f"{int(code):06d}"].append(value)
    query_cache = {code: {subset_index: row} for code, row in rows.items()}
    data = SimpleNamespace(
        wire=lambda: None,
        decoded_descriptors_all_subsets=[[]] * subset_index + [descriptors],
        decoded_values_all_subsets=[[]] * subset_index + [values],
    )
    message = SimpleNamespace(template_data=SimpleNamespace(value=data))
    profile = _decode_subset(path, message, subset_index, registry=registry,
                             decode_mode="adpupa", header_meta=header,
                             query_cache=query_cache, collect_elements=collect_elements)
    profile.metadata["decoder"] = backend
    return profile
