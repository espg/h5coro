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
FILL_VALUES = {'>f8': -7.5, '>i4': -42}


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
        # big-endian datasets with a defined fill value and only the first third
        # written; with 10-element chunks the tail chunks stay unallocated
        for dtype, fill in FILL_VALUES.items():
            d = f.create_dataset(f'{tag_of(dtype)}_fill', shape=(N,), chunks=(10,), dtype=dtype, fillvalue=fill)
            d[:N // 3] = values_for(dtype)[:N // 3]
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

    @pytest.mark.parametrize('dtype', list(FILL_VALUES))
    def test_fill_value_for_unwritten_chunks(self, h5obj, byteorder_file, dtype):
        # Pins the fill pre-fill path for big-endian data. The fill bytes are
        # stored in the dataset's byte order; readFillValue decodes them as a
        # little-endian int and struct.pack('Q') re-encodes them little-endian,
        # so the original bytes round-trip unchanged and the '>' dtype reads
        # them correctly. If either side starts honouring the byte order alone,
        # the bytes get swapped once and the unwritten chunks decode as garbage.
        name = f'{tag_of(dtype)}_fill'
        with h5py.File(byteorder_file, 'r') as f:
            expected = f[name][:]
            assert f[name].id.get_num_chunks() < N // 10
        # h5py returns the fill value for the unwritten tail chunks
        assert (expected[N // 3 + 10:] == FILL_VALUES[dtype]).all()
        values = h5obj.readDatasets([name], block=True)[name]
        assert values.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(values, expected)
        assert (values[N // 3 + 10:] == FILL_VALUES[dtype]).all()

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
