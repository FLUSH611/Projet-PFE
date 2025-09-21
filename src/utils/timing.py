import time
from contextlib import contextmanager

@contextmanager
def timed(msg: str):
    start = time.time()
    yield
    dur = time.time() - start
    print(f"{msg} took {dur:.2f}s")