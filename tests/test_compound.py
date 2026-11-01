"""
Hermetic tests for compound (struct) datatype support: h5py writes local
files, h5coro reads them back through the FileDriver, and the two are
compared.  No network access is required.
"""
import logging
import h5py
import numpy as np
import pytest

import h5coro
from h5coro import filedriver
from h5coro.h5metadata import H5Metadata


FLAT_DTYPE   = np.dtype([('a', '<i4'), ('b', '<f8'), ('c', '>f4'), ('s', 'S8')])
ARRAY_DTYPE  = np.dtype([('x', '<i2'), ('v', '<f4', (2,))])
NESTED_DTYPE = np.dtype([('p', FLAT_DTYPE), ('q', '<u1')])
ENUM_DTYPE   = np.dtype([('k', h5py.enum_dtype({'RED': 0, 'GREEN': 1}, basetype='<i1')), ('w', '<u2')])

NUM_RECORDS = 40

# dataset name -> (dtype, create_dataset keyword arguments)
DATASETS = {
    'flat':             (FLAT_DTYPE,   {}),
    'flat_chunked':     (FLAT_DTYPE,   {'chunks': (8,), 'shuffle': True, 'compression': 'gzip'}),
    'array_member':     (ARRAY_DTYPE,  {}),
    'array_chunked':    (ARRAY_DTYPE,  {'chunks': (8,), 'shuffle': True, 'compression': 'gzip'}),
    'nested':           (NESTED_DTYPE, {}),
    'nested_chunked':   (NESTED_DTYPE, {'chunks': (8,), 'shuffle': True, 'compression': 'gzip'}),
    'enum_member':      (ENUM_DTYPE,   {}),
}

# h5py's default libver writes datatype message version 1 (version 2 when a
# member is an array type); ('v108', 'v108') writes version 3 for every compound
LIBVERS = {
    'v1_v2': {},
    'v3':    {'libver': ('v108', 'v108')},
}


def make_records(dtype, n):
    rng = np.random.default_rng(1234)
    values = np.zeros(n, dtype=dtype)
    for name in dtype.names:
        sub = dtype[name]
        base = sub.subdtype[0] if sub.subdtype else sub
        if base.names:
            values[name] = make_records(base, n)
        elif base.kind == 'S':
            values[name] = [f'r{i:05d}'.encode('ascii') for i in range(n)]
        elif base.kind == 'f':
            values[name] = rng.standard_normal(values[name].shape).astype(base)
        else:
            info = np.iinfo(base)
            values[name] = rng.integers(info.min, info.max, size=values[name].shape, dtype=base, endpoint=True)
    return values


@pytest.fixture(scope='module', params=LIBVERS.keys())
def compound_file(request, tmp_path_factory):
    path = tmp_path_factory.mktemp('compound') / f'compound_{request.param}.h5'
    with h5py.File(path, 'w', **LIBVERS[request.param]) as f:
        for name, (dtype, kwargs) in DATASETS.items():
            f.create_dataset(name, data=make_records(dtype, NUM_RECORDS), **kwargs)
        f.create_dataset('grid', data=make_records(FLAT_DTYPE, 12).reshape(3, 4))
        f.create_dataset('plain', data=np.arange(NUM_RECORDS, dtype='<f4'))
    return str(path)


@pytest.fixture
def h5obj(compound_file):
    obj = h5coro.H5Coro(compound_file, filedriver.FileDriver, errorChecking=True, verbose=False)
    yield obj
    obj.close()


class TestCompoundMetadata:

    @pytest.mark.parametrize('name', [n for n in DATASETS if n != 'enum_member'] + ['grid'])
    def test_dtype_matches_h5py(self, h5obj, compound_file, name, caplog):
        with h5py.File(compound_file, 'r') as f:
            expected = f[name].dtype
            shape = f[name].shape

        with caplog.at_level(logging.WARNING):
            _, _, meta = h5obj.inspectPath(name)
            dtype = meta.numpyDtype
        # every member of these datasets is decodable, so nothing falls back to opaque
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

        assert meta.type == H5Metadata.COMPOUND_TYPE
        assert meta.typeSize == expected.itemsize
        assert tuple(meta.dimensions) == shape
        assert [m[0] for m in meta.members] == list(expected.names)
        assert [m[1] for m in meta.members] == [expected.fields[n][1] for n in expected.names]
        assert dtype == expected
        assert dtype.itemsize == expected.itemsize

    def test_member_byte_order(self, h5obj):
        _, _, meta = h5obj.inspectPath('flat', w_attr=False)
        byte_orders = {name: member.byteOrder for name, _, member in meta.members}
        assert byte_orders['a'] == 0
        assert byte_orders['b'] == 0
        assert byte_orders['c'] == 1
        assert meta.numpyDtype['c'] == np.dtype('>f4')
        assert meta.numpyDtype['a'] == np.dtype('<i4')

    def test_array_member(self, h5obj):
        _, _, meta = h5obj.inspectPath('array_member', w_attr=False)
        name, offset, member = meta.members[1]
        assert name == 'v'
        assert offset == 2
        assert member.type == H5Metadata.ARRAY_TYPE
        assert member.arrayDims == [2]
        assert member.typeSize == 8
        assert member.baseType.type == H5Metadata.FLOATING_POINT_TYPE
        assert member.baseType.typeSize == 4
        assert meta.numpyDtype['v'] == np.dtype(('<f4', (2,)))

    def test_nested_member(self, h5obj):
        _, _, meta = h5obj.inspectPath('nested', w_attr=False)
        name, offset, member = meta.members[0]
        assert name == 'p'
        assert offset == 0
        assert member.type == H5Metadata.COMPOUND_TYPE
        assert [m[0] for m in member.members] == list(FLAT_DTYPE.names)
        assert member.numpyDtype == FLAT_DTYPE
        assert meta.numpyDtype['p'] == FLAT_DTYPE

    def test_opaque_member_fallback(self, h5obj, compound_file, caplog):
        with h5py.File(compound_file, 'r') as f:
            expected = f['enum_member'].dtype
        with caplog.at_level(logging.WARNING):
            _, _, meta = h5obj.inspectPath('enum_member', w_attr=False)
            meta.numpyDtype
        assert 'treating as opaque' in caplog.text
        # the enum member is not decodable, it must still have the right size
        assert meta.numpyDtype['k'] == np.dtype('V1')
        assert meta.numpyDtype['w'] == np.dtype('<u2')
        assert meta.numpyDtype.itemsize == expected.itemsize
        assert [meta.numpyDtype.fields[n][1] for n in expected.names] == [expected.fields[n][1] for n in expected.names]

    def test_member_dtypes_and_repr(self, h5obj):
        _, _, meta = h5obj.inspectPath('flat', w_attr=False)
        assert list(meta.memberDtypes().keys()) == list(FLAT_DTYPE.names)
        assert meta.memberDtypes()['c'] == np.dtype('>f4')
        text = repr(meta)
        for name in FLAT_DTYPE.names:
            assert f'"{name}"' in text
        assert 'unsupported' not in text

    def test_list_reports_members(self, h5obj, caplog):
        with caplog.at_level(logging.WARNING):
            variables, _, _ = h5obj.list('/', w_attr=False)
            meta = variables['flat']['__metadata__']
            dtype = meta.numpyDtype
        # list() no longer warns about compound datasets; numpyDtype is built
        # lazily, so the enum_member opaque fallback is not triggered here either
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert meta.type == H5Metadata.COMPOUND_TYPE
        assert [m[0] for m in meta.members] == list(FLAT_DTYPE.names)
        assert dtype == FLAT_DTYPE
        # non-compound datasets in the same file are reported as before
        plain = variables['plain']['__metadata__']
        assert plain.type == H5Metadata.FLOATING_POINT_TYPE
        assert plain.typeSize == 4
        assert plain.getNumpyType() == np.dtype('<f4')

    def test_plain_dataset_unchanged(self, h5obj, compound_file):
        with h5py.File(compound_file, 'r') as f:
            expected = f['plain'][:]
        promise = h5obj.readDatasets(['plain'], block=True)
        np.testing.assert_array_equal(promise['plain'], expected)


def test_compound_attribute_v1(tmp_path):
    # attribute messages carry the datatype size, so errorChecking verifies that
    # the member parser consumed exactly the message (version 1 file only: version
    # 3 attribute messages are a pre-existing limitation)
    path = tmp_path / 'attr.h5'
    with h5py.File(path, 'w') as f:
        ds = f.create_dataset('x', data=np.arange(4, dtype='<i4'))
        ds.attrs.create('struct_attr', make_records(ARRAY_DTYPE, 3))
    obj = h5coro.H5Coro(str(path), filedriver.FileDriver, errorChecking=True)
    try:
        _, attributes, _ = obj.inspectPath('x', w_attr=True)
        assert 'struct_attr' in attributes
        meta = obj.metadataTable['x/struct_attr']
        assert meta.type == H5Metadata.COMPOUND_TYPE
        assert meta.numpyDtype == ARRAY_DTYPE
    finally:
        obj.close()


@pytest.mark.parametrize('itemsize', [255, 256, 65535, 65536, 70000])
def test_wide_struct_v3_offset_width(tmp_path, itemsize):
    # version 3 compound messages store member offsets in the fewest bytes that
    # can hold the struct size: sizes on both sides of each width boundary cover
    # the 1-, 2- and 3-byte widths
    dtype = np.dtype({'names': ['head', 'tail'], 'formats': ['<i4', '<f8'], 'offsets': [0, itemsize - 8], 'itemsize': itemsize})
    path = tmp_path / 'wide.h5'
    with h5py.File(path, 'w', libver=('v108', 'v108')) as f:
        f.create_dataset('wide', (2,), dtype=dtype)
    obj = h5coro.H5Coro(str(path), filedriver.FileDriver, errorChecking=True)
    try:
        _, _, meta = obj.inspectPath('wide', w_attr=False)
        assert meta.typeSize == itemsize
        assert [(m[0], m[1]) for m in meta.members] == [('head', 0), ('tail', itemsize - 8)]
        assert meta.numpyDtype == dtype
    finally:
        obj.close()
