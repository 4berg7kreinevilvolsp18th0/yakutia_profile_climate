"""Точка входа расшифровки и межпроцессная блокировка папки результата."""
from __future__ import annotations

import atexit
import os
from pathlib import Path

_RUN_LOCK_HANDLE = None


def _release_output_lock() -> None:
    global _RUN_LOCK_HANDLE

    current = _RUN_LOCK_HANDLE
    if current is None:
        return
    try:
        current.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(current.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - Windows is the production platform
            import fcntl

            fcntl.flock(current.fileno(), fcntl.LOCK_UN)
    finally:
        current.close()
        _RUN_LOCK_HANDLE = None


def _acquire_output_lock(output_dir: Path) -> None:
    """Запрещает двум дешифровщикам одновременно менять один output_dir."""
    global _RUN_LOCK_HANDLE

    if _RUN_LOCK_HANDLE is not None:
        raise RuntimeError("Этот процесс уже удерживает блокировку дешифровщика")

    lock_path = output_dir / ".fast_extract.lock"
    handle = lock_path.open("a+b")
    if lock_path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)

    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - Windows is the production platform
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            f"Дешифровщик уже запущен для {output_dir}. "
            "Дождитесь его завершения или выберите другую папку --output."
        ) from exc

    _RUN_LOCK_HANDLE = handle


atexit.register(_release_output_lock)


from gdex_bufr.profile_climate.fast_pipeline import build_parser, main

if __name__ == "__main__":
    raise SystemExit(main())
