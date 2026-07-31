from concurrent.futures import ProcessPoolExecutor


def check():
    from gdex_bufr.profile_climate.fast_worker import decode_one, worker_init
    return decode_one.__module__, worker_init.__name__


if __name__ == "__main__":
    print("parent", check())
    with ProcessPoolExecutor(2) as pool:
        print("child", pool.submit(check).result())
        from gdex_bufr.profile_climate.fast_worker import decode_one, worker_init
        print("submit worker", pool.submit(decode_one, "nosuch.bufr").result()[0:1] + pool.submit(decode_one, "nosuch.bufr").result()[-1:])
