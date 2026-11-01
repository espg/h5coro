"""
Hermetic tests for big-endian non-compound datasets (issue #7): h5py writes
local files, h5coro reads them back through the FileDriver.
"""
import h5py
import numpy as np
import pytest

import h5coro
from h5coro import filedriver
from h5coro.h5metadata import H5Metadata


BIG_ENDIAN = ['>i2', '>i4', '>f4', '>f8']
LITTLE_ENDIAN = ['<i2', '<i4', '<f4', '<f8']
N = 200


def values_for(dtype):
    if np.dtype(dtype).kind == 'f':
        return (np.linspace(-1e6, 1e6, N) / 7.0).astype(dtype)
    info = np.iinfo(dtype)
    return np.linspace(info.min, info.max, N).astype(dtype)


@pytest.fixture(scope='module')
def byteorder_file(tmp_path_factory):
    path = tmp_path_factory.mktemp('byteorder') / 'byteorder.h5'
    with h5py.File(path, 'w') as f:
        for dtype in BIG_ENDIAN + LITTLE_ENDIAN:
            tag = dtype.replace('>', 'be_').replace('<', 'le_')
            f.create_dataset(f'{tag}_contiguous', data=values_for(dtype))
            f.create_dataset(f'{tag}_chunked', data=values_for(dtype), chunks=(32,), shuffle=True, compression='gzip')
            f.create_dataset(f'{tag}_2d', data=values_for(dtype).reshape(20, 10), chunks=(8, 10), shuffle=True, compression='gzip')
        f['be_f8_contiguous'].attrs.create('be_attr', np.array([1.5, -2.25], dtype='>f8'))
        f['be_f8_contiguous'].attrs.create('be_scalar', np.array(42, dtype='>i4'))
    return str(path)


@pytest.fixture
def h5obj(byteorder_file):
    obj = h5coro.H5Coro(byteorder_file, filedriver.FileDriver, errorChecking=True)
    yield obj
    obj.close()


def tag_of(dtype):
    return dtype.replace('>', 'be_').replace('<', 'le_')


class TestBigEndian:

    @pytest.mark.parametrize('layout', ['contiguous', 'chunked', '2d'])
    @pytest.mark.parametrize('dtype', BIG_ENDIAN)
    def test_full_read_matches_h5py(self, h5obj, byteorder_file, dtype, layout):
        name = f'{tag_of(dtype)}_{layout}'
        with h5py.File(byteorder_file, 'r') as f:
            expected = f[name][:]
            assert f[name].dtype.byteorder == '>'
        values = h5obj.readDatasets([name], block=True)[name]
        assert values.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(values, expected)
        # the values are usable as numbers, not just byte-identical
        assert float(values.max()) == float(expected.max())

    @pytest.mark.parametrize('layout, hyperslice', [('contiguous', [[13, 77]]), ('chunked', [[13, 77]]), ('2d', [[3, 11], [2, 9]])])
    @pytest.mark.parametrize('dtype', BIG_ENDIAN)
    def test_hyperslice_matches_h5py(self, h5obj, byteorder_file, dtype, layout, hyperslice):
        name = f'{tag_of(dtype)}_{layout}'
        index = tuple(slice(a, b) for a, b in hyperslice)
        with h5py.File(byteorder_file, 'r') as f:
            expected = f[name][index]
        values = h5obj.readDatasets([{'dataset': name, 'hyperslice': hyperslice}], block=True)[name]
        np.testing.assert_array_equal(values, expected)

    def test_metadata_reports_byte_order(self, h5obj):
        _, _, meta = h5obj.inspectPath('be_i4_contiguous', w_attr=False)
        assert meta.byteOrder == 1
        assert meta.getNumpyType() == np.dtype('>i4')
        assert meta.numpyDtype == np.dtype('>i4')
        _, _, meta = h5obj.inspectPath('le_i4_contiguous', w_attr=False)
        assert meta.byteOrder == 0
        assert meta.getNumpyType() == np.dtype('<i4')

    def test_big_endian_attributes(self, h5obj):
        _, attributes, _ = h5obj.inspectPath('be_f8_contiguous', w_attr=True)
        np.testing.assert_array_equal(attributes['be_attr'], np.array([1.5, -2.25]))
        np.testing.assert_array_equal(attributes['be_scalar'], np.array([42]))


class TestLittleEndianUnchanged:

    @pytest.mark.parametrize('layout', ['contiguous', 'chunked', '2d'])
    @pytest.mark.parametrize('dtype', LITTLE_ENDIAN)
    def test_full_read_matches_h5py(self, h5obj, byteorder_file, dtype, layout):
        name = f'{tag_of(dtype)}_{layout}'
        with h5py.File(byteorder_file, 'r') as f:
            expected = f[name][:]
        values = h5obj.readDatasets([name], block=True)[name]
        # the little-endian dtype is the native one, exactly as before
        assert values.dtype == np.dtype(dtype)
        assert values.dtype.isnative
        np.testing.assert_array_equal(values, expected)
