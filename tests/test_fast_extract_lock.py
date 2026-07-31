from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import run_fast_extract


def test_output_lock_blocks_a_second_process(tmp_path: Path) -> None:
    run_fast_extract._acquire_output_lock(tmp_path)
    code = (
        "import sys; from pathlib import Path; import run_fast_extract; "
        "\ntry: run_fast_extract._acquire_output_lock(Path(sys.argv[1]))"
        "\nexcept RuntimeError: raise SystemExit(0)"
        "\nraise SystemExit(1)"
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
    finally:
        run_fast_extract._release_output_lock()


def test_output_lock_can_be_reacquired_after_release(tmp_path: Path) -> None:
    run_fast_extract._acquire_output_lock(tmp_path)
    run_fast_extract._release_output_lock()

    try:
        run_fast_extract._acquire_output_lock(tmp_path)
    finally:
        run_fast_extract._release_output_lock()
