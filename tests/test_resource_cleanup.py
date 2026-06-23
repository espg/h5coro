"""Resource-management tests: executor shutdown, context manager, cache release.

Regression coverage for the leak reported in zagg#66 -- H5Coro must release its
cache lines and worker threads deterministically (via close()/context manager)
rather than relying on garbage-collection timing.
"""
import gc
import threading
import time

import pytest

from .test_helpers import h5coro, filedriver, download_hdf_to_local, DATASET_PATHS


@pytest.fixture(scope="module")
def local_file():
    return download_hdf_to_local()


def _read(h5obj):
    promise = h5obj.readDatasets(DATASET_PATHS, block=True)
    # touch every value so the read actually completes
    return {d: promise[d] for d in promise}


def test_close_clears_cache(local_file):
    h5obj = h5coro.H5Coro(local_file, filedriver.FileDriver)
    _read(h5obj)
    assert len(h5obj.cache) > 0, "expected cache lines after a read"
    h5obj.close()
    assert len(h5obj.cache) == 0, "close() must clear the cache lines"


def test_close_is_idempotent(local_file):
    h5obj = h5coro.H5Coro(local_file, filedriver.FileDriver)
    _read(h5obj)
    h5obj.close()
    h5obj.close()  # second call must not raise
    assert len(h5obj.cache) == 0


def test_context_manager_releases_cache(local_file):
    with h5coro.H5Coro(local_file, filedriver.FileDriver) as h5obj:
        _read(h5obj)
        assert len(h5obj.cache) > 0
    # cache released on __exit__
    assert len(h5obj.cache) == 0


def test_no_thread_accumulation_across_reads(local_file):
    """Opening/reading/closing many files must not accumulate worker threads."""
    baseline = threading.active_count()
    for _ in range(25):
        with h5coro.H5Coro(local_file, filedriver.FileDriver) as h5obj:
            _read(h5obj)
    gc.collect()
    # executor shutdown is async (wait=False); allow threads to wind down
    deadline = time.time() + 5.0
    while threading.active_count() > baseline + 1 and time.time() < deadline:
        time.sleep(0.05)
    assert threading.active_count() <= baseline + 1, (
        f"threads leaked: {threading.active_count()} vs baseline {baseline}"
    )
