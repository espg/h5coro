"""
Hermetic tests for compound (struct) datatype support: h5py writes local
files, h5coro reads them back through the FileDriver, and the two are
compared.  No network access is required.
"""
import logging
import re
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
        f.create_dataset('grid', data=make_records(FLAT_DTYPE, 12).reshape(3, 4), chunks=(2, 2))
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


###############################################################################
# decoding (issue #3)
###############################################################################

HYPERSLICE = [[7, 29]]
CHUNKED = [n for n in DATASETS if n.endswith('_chunked')]


class TestCompoundDecode:

    @pytest.mark.parametrize('name', [n for n in DATASETS if n != 'enum_member'])
    def test_full_read_matches_h5py(self, h5obj, compound_file, name):
        with h5py.File(compound_file, 'r') as f:
            expected = f[name][:]
        values = h5obj.readDatasets([name], block=True)[name]
        assert values.dtype == expected.dtype
        np.testing.assert_array_equal(values, expected)

    @pytest.mark.parametrize('name', CHUNKED)
    def test_hyperslice_read_matches_h5py(self, h5obj, compound_file, name):
        with h5py.File(compound_file, 'r') as f:
            expected = f[name][HYPERSLICE[0][0]:HYPERSLICE[0][1]]
        promise = h5obj.readDatasets([{'dataset': name, 'hyperslice': HYPERSLICE}], block=True)
        np.testing.assert_array_equal(promise[name], expected)

    @pytest.mark.xfail(strict=True, reason='hyperslices on contiguous layouts read from the dataset start for every datatype (pre-existing, not compound specific)')
    def test_hyperslice_read_contiguous(self, h5obj, compound_file):
        with h5py.File(compound_file, 'r') as f:
            expected = f['flat'][HYPERSLICE[0][0]:HYPERSLICE[0][1]]
        promise = h5obj.readDatasets([{'dataset': 'flat', 'hyperslice': HYPERSLICE}], block=True)
        np.testing.assert_array_equal(promise['flat'], expected)

    def test_two_dimensional_read(self, h5obj, compound_file):
        with h5py.File(compound_file, 'r') as f:
            expected = f['grid'][:]
        values = h5obj.readDatasets(['grid'], block=True)['grid']
        assert values.shape == (3, 4)
        np.testing.assert_array_equal(values, expected)
        sliced = h5obj.readDatasets([{'dataset': 'grid', 'hyperslice': [[1, 3], [0, 2]]}], block=True)['grid']
        assert sliced.shape == (2, 2)
        np.testing.assert_array_equal(sliced, expected[1:3, 0:2])

    def test_big_endian_member_round_trip(self, h5obj, compound_file):
        with h5py.File(compound_file, 'r') as f:
            expected = f['flat']['c']
        values = h5obj.readDatasets(['flat'], block=True)['flat']
        assert values['c'].dtype == np.dtype('>f4')
        np.testing.assert_array_equal(values['c'], expected)
        np.testing.assert_array_equal(values['c'].astype('<f4'), expected.astype('<f4'))

    def test_nested_and_array_members(self, h5obj, compound_file):
        with h5py.File(compound_file, 'r') as f:
            nested = f['nested'][:]
            array = f['array_member'][:]
        promise = h5obj.readDatasets(['nested', 'array_member'], block=True)
        np.testing.assert_array_equal(promise['nested']['p']['b'], nested['p']['b'])
        np.testing.assert_array_equal(promise['nested']['q'], nested['q'])
        assert promise['array_member']['v'].shape == (NUM_RECORDS, 2)
        np.testing.assert_array_equal(promise['array_member']['v'], array['v'])

    def test_opaque_member_reads_raw_bytes(self, h5obj, compound_file):
        with h5py.File(compound_file, 'r') as f:
            expected = f['enum_member'][:]
        values = h5obj.readDatasets(['enum_member'], block=True)['enum_member']
        np.testing.assert_array_equal(values['w'], expected['w'])
        assert values['k'].dtype == np.dtype('V1')
        np.testing.assert_array_equal(values['k'].view('i1'), expected['k'])

    def test_fields_selection(self, h5obj, compound_file):
        with h5py.File(compound_file, 'r') as f:
            expected = f['flat'][:]
        promise = h5obj.readDatasets([
            {'dataset': 'flat', 'fields': ['a', 'c']},
        ], block=True)
        values = promise['flat']
        assert values.dtype.names == ('a', 'c')
        np.testing.assert_array_equal(values['a'], expected['a'])
        np.testing.assert_array_equal(values['c'], expected['c'])

    def test_single_field_returns_plain_array(self, h5obj, compound_file):
        with h5py.File(compound_file, 'r') as f:
            expected = f['flat']['b']
        promise = h5obj.readDatasets([{'dataset': 'flat_chunked', 'fields': ['b'], 'hyperslice': HYPERSLICE}], block=True)
        values = promise['flat_chunked']
        assert values.dtype == np.dtype('<f8')
        assert values.dtype.names is None
        np.testing.assert_array_equal(values, expected[HYPERSLICE[0][0]:HYPERSLICE[0][1]])

    def test_nested_field_selection(self, h5obj, compound_file):
        with h5py.File(compound_file, 'r') as f:
            expected = f['nested'][:]
        values = h5obj.readDatasets([{'dataset': 'nested', 'fields': ['p']}], block=True)['nested']
        assert values.dtype == FLAT_DTYPE
        np.testing.assert_array_equal(values['s'], expected['p']['s'])

    def test_unknown_field_raises(self, h5obj):
        with pytest.raises(RuntimeError) as excinfo:
            h5obj.readDatasets([{'dataset': 'flat', 'fields': ['a', 'nope']}], block=True)
        message = str(excinfo.value)
        assert 'nope' in message
        for name in FLAT_DTYPE.names:
            assert name in message

    def test_fields_on_non_compound_raises(self, h5obj):
        with pytest.raises(RuntimeError, match='not a compound dataset'):
            h5obj.readDatasets([{'dataset': 'plain', 'fields': ['a']}], block=True)

    def test_fields_as_string(self, h5obj, compound_file):
        with h5py.File(compound_file, 'r') as f:
            expected = f['flat']['a']
        values = h5obj.readDatasets([{'dataset': 'flat', 'fields': 'a'}], block=True)['flat']
        np.testing.assert_array_equal(values, expected)

    def test_shuffled_wide_struct(self, h5obj, compound_file):
        # the shuffle word size is the 24-byte struct, wider than any atomic type
        with h5py.File(compound_file, 'r') as f:
            assert f['flat_chunked'].shuffle and f['flat_chunked'].compression == 'gzip'
            expected = f['flat_chunked'][:]
        np.testing.assert_array_equal(h5obj.readDatasets(['flat_chunked'], block=True)['flat_chunked'], expected)


def test_struct_fill_value(tmp_path):
    # an explicit struct fill value is longer than 8 bytes; it must neither stop
    # the object header parse nor corrupt the records that were written
    dtype = np.dtype([('a', '<i4'), ('b', '<f8')])
    fill = np.array((7, 1.5), dtype=dtype)
    path = tmp_path / 'fill.h5'
    with h5py.File(path, 'w') as f:
        ds = f.create_dataset('fv', (6,), dtype=dtype, fillvalue=fill, chunks=(2,))
        ds[:3] = make_records(dtype, 3)
    with h5py.File(path, 'r') as f:
        expected = f['fv'][:]
    obj = h5coro.H5Coro(str(path), filedriver.FileDriver, errorChecking=True)
    try:
        values = obj.readDatasets(['fv'], block=True)['fv']
        np.testing.assert_array_equal(values, expected)
    finally:
        obj.close()


class TestFieldSelectionOptions:

    def test_squeeze_fields_false_keeps_structure(self, h5obj, compound_file):
        with h5py.File(compound_file, 'r') as f:
            expected = f['flat']['b']
        values = h5obj.readDatasets([{'dataset': 'flat', 'fields': ['b'], 'squeeze_fields': False}], block=True)['flat']
        assert values.dtype.names == ('b',)
        np.testing.assert_array_equal(values['b'], expected)

    def test_multi_field_selection_is_a_view(self, h5obj):
        # numpy multi-field indexing keeps the record size and offsets (a view);
        # pin that so a future change to a packed copy is deliberate
        values = h5obj.readDatasets([{'dataset': 'flat', 'fields': ['a', 'c']}], block=True)['flat']
        assert values.dtype.itemsize == FLAT_DTYPE.itemsize
        assert values.dtype.fields['c'][1] == FLAT_DTYPE.fields['c'][1]

    @pytest.mark.parametrize('fields, pattern', [
        (['a', 'a'], re.escape("duplicate field(s) ['a']")),
        ([], 'empty field selection'),
        (5, re.escape('must be a str or a list of str')),
        (['a', 3], re.escape('must be a str or a list of str')),
    ])
    def test_malformed_fields_raise(self, h5obj, fields, pattern):
        with pytest.raises(RuntimeError, match=pattern):
            h5obj.readDatasets([{'dataset': 'flat', 'fields': fields}], block=True)

    def test_missing_dataset_with_fields_raises(self, h5obj):
        with pytest.raises(RuntimeError) as excinfo:
            h5obj.readDatasets([{'dataset': 'does_not_exist', 'fields': ['a']}], block=True)
        assert 'not found' in str(excinfo.value)

    def test_request_dict_is_not_modified(self, h5obj):
        request = {'dataset': 'flat_chunked', 'hyperslice': [[1, 4]], 'fields': ['a']}
        snapshot = {'dataset': 'flat_chunked', 'hyperslice': [[1, 4]], 'fields': ['a']}
        h5obj.readDatasets([request], block=True)
        assert request == snapshot

    def test_several_field_requests(self, h5obj, compound_file):
        with h5py.File(compound_file, 'r') as f:
            flat = f['flat'][:]
            nested = f['nested'][:]
        promise = h5obj.readDatasets([
            {'dataset': 'flat', 'fields': ['s']},
            {'dataset': 'nested', 'fields': ['q']},
            'array_member',
        ], block=True)
        np.testing.assert_array_equal(promise['flat'], flat['s'])
        np.testing.assert_array_equal(promise['nested'], nested['q'])
        assert promise['array_member'].dtype == ARRAY_DTYPE
