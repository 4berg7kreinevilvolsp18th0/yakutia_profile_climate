from concurrent.futures import ProcessPoolExecutor
import os
import sys


def check():
    try:
        import gdex_bufr.profile_climate as pc
        ok = pc.__file__
    except Exception as e:
        ok = repr(e)
    return {
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "path_hits": [p for p in sys.path if "yakutia" in p.lower() or "Kutenika" in p],
        "import": ok,
    }


if __name__ == "__main__":
    print("parent", check())
    with ProcessPoolExecutor(2) as pool:
        print("child", pool.submit(check).result())
