"""
Hermetic tests for hyperslice reads on compact and contiguous layouts (issue #6):
h5py writes local files, h5coro reads them back through the FileDriver.
"""
import h5py
import numpy as np
import pytest

import h5coro
from h5coro import filedriver


DTYPES = ['<i2', '<i4', '<u1', '<f4', '<f8']

# (dataset shape, hyperslice) pairs: 1-D, 2-D outer-dimension only, 2-D with an
# inner-dimension range, 3-D, and slices touching the first and last rows
CASES = [
    ((1000,),       [[100, 110]]),
    ((1000,),       [[0, 7]]),
    ((1000,),       [[993, 1000]]),
    ((1000,),       [[0, 1000]]),
    ((50, 7),       [[10, 20]]),
    ((50, 7),       [[10, 20], [2, 5]]),
    ((50, 7),       [[49, 50], [6, 7]]),
    ((50, 7),       [[0, 50], [0, 7]]),
    ((6, 5, 4),     [[1, 4], [2, 5], [0, 4]]),
    ((6, 5, 4),     [[2, 3], [1, 2], [3, 4]]),
]


def fill(shape, dtype):
    size = int(np.prod(shape))
    if np.dtype(dtype).kind == 'f':
        return (np.arange(size, dtype='<f8') * 0.25 - 10.0).astype(dtype).reshape(shape)
    info = np.iinfo(dtype)
    return (np.arange(size) % (int(info.max) - int(info.min) + 1) + int(info.min)).astype(dtype).reshape(shape)


def create_compact(group, name, data):
    # h5py has no high-level switch for the compact layout; use the low level API
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_layout(h5py.h5d.COMPACT)
    space = h5py.h5s.create_simple(data.shape)
    dataset = h5py.h5d.create(group.id, name.encode('ascii'), h5py.h5t.py_create(data.dtype), space, dcpl=dcpl)
    dataset.write(h5py.h5s.ALL, h5py.h5s.ALL, data)


@pytest.fixture(scope='module')
def layout_file(tmp_path_factory):
    path = tmp_path_factory.mktemp('layouts') / 'layouts.h5'
    with h5py.File(path, 'w') as f:
        for dtype in DTYPES:
            for shape, _ in CASES:
                name = f'contiguous_{dtype[1:]}_{len(shape)}d'
                if name not in f:
                    f.create_dataset(name, data=fill(shape, dtype))
            # compact datasets must fit in the object header (64 KiB); keep them small
            create_compact(f, f'compact_{dtype[1:]}_1d', fill((100,), dtype))
            create_compact(f, f'compact_{dtype[1:]}_2d', fill((12, 5), dtype))
        f.create_dataset('shuffled_s16', data=np.array([f'row{i:02d}'.encode('ascii') for i in range(32)], dtype='S16'), chunks=(8,), shuffle=True, compression='gzip')
    return str(path)


@pytest.fixture
def h5obj(layout_file):
    obj = h5coro.H5Coro(layout_file, filedriver.FileDriver, errorChecking=True)
    yield obj
    obj.close()


def layout_of(path, name):
    with h5py.File(path, 'r') as f:
        return f[name].id.get_create_plist().get_layout()


class TestContiguousHyperslice:

    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('shape, hyperslice', CASES)
    def test_hyperslice_matches_h5py(self, h5obj, layout_file, dtype, shape, hyperslice):
        name = f'contiguous_{dtype[1:]}_{len(shape)}d'
        assert layout_of(layout_file, name) == h5py.h5d.CONTIGUOUS
        index = tuple(slice(a, b) for a, b in hyperslice)
        with h5py.File(layout_file, 'r') as f:
            expected = f[name][index]
        values = h5obj.readDatasets([{'dataset': name, 'hyperslice': hyperslice}], block=True)[name]
        assert values.shape == expected.shape
        np.testing.assert_array_equal(values, expected)

    @pytest.mark.parametrize('dtype', DTYPES)
    def test_full_read_unchanged(self, h5obj, layout_file, dtype):
        name = f'contiguous_{dtype[1:]}_2d'
        with h5py.File(layout_file, 'r') as f:
            expected = f[name][:]
        np.testing.assert_array_equal(h5obj.readDatasets([name], block=True)[name], expected)

    def test_partial_hyperslice_fills_missing_dimensions(self, h5obj, layout_file):
        # a hyperslice shorter than the dataset rank covers the trailing dimensions fully
        name = 'contiguous_f8_3d'
        with h5py.File(layout_file, 'r') as f:
            expected = f[name][1:4]
        values = h5obj.readDatasets([{'dataset': name, 'hyperslice': [[1, 4]]}], block=True)[name]
        np.testing.assert_array_equal(values, expected)


class TestCompactHyperslice:

    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('name_suffix, hyperslice', [('1d', [[17, 40]]), ('1d', [[0, 100]]), ('2d', [[3, 9], [1, 4]]), ('2d', [[11, 12]])])
    def test_hyperslice_matches_h5py(self, h5obj, layout_file, dtype, name_suffix, hyperslice):
        name = f'compact_{dtype[1:]}_{name_suffix}'
        assert layout_of(layout_file, name) == h5py.h5d.COMPACT
        index = tuple(slice(a, b) for a, b in hyperslice)
        with h5py.File(layout_file, 'r') as f:
            expected = f[name][index]
        values = h5obj.readDatasets([{'dataset': name, 'hyperslice': hyperslice}], block=True)[name]
        np.testing.assert_array_equal(values, expected)


def test_shuffle_word_wider_than_8_bytes(h5obj, layout_file):
    # the shuffle guard used to reject element sizes above 8 bytes for non-compound types
    with h5py.File(layout_file, 'r') as f:
        expected = f['shuffled_s16'][:]
    values = h5obj.readDatasets(['shuffled_s16'], block=True)['shuffled_s16']
    # arrays of fixed strings are still rendered by the existing string path (a
    # repr of the raw buffer); the point here is that the read succeeds and the
    # bytes were un-shuffled, so every row appears intact and in order
    assert values is not None
    positions = [values.find(row.decode('ascii')) for row in expected]
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions)
