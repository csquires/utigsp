import multiprocessing
import time


def test(sec):
    time.sleep(sec)
    print(sec)


with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
    list(pool.imap_unordered(test, [5, 1, 3]))

