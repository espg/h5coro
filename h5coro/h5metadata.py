# Copyright (c) 2023, University of Washington
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the University of Washington nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy
from h5coro.logger import log

###############################################################################
# H5Metadata Class
###############################################################################

class H5Metadata:

    #######################
    # Constants
    #######################
    # filters
    DEFLATE_FILTER          = 1
    SHUFFLE_FILTER          = 2
    FLETCHER32_FILTER       = 3
    SZIP_FILTER             = 4
    NBIT_FILTER             = 5
    SCALEOFFSET_FILTER      = 6
    # datatypes
    FIXED_POINT_TYPE        = 0
    FLOATING_POINT_TYPE     = 1
    TIME_TYPE               = 2
    STRING_TYPE             = 3
    BIT_FIELD_TYPE          = 4
    OPAQUE_TYPE             = 5
    COMPOUND_TYPE           = 6
    REFERENCE_TYPE          = 7
    ENUMERATED_TYPE         = 8
    VARIABLE_LENGTH_TYPE    = 9
    ARRAY_TYPE              = 10
    UNKNOWN_TYPE            = 11
    # extended h5coro types
    VL_STRING_TYPE          = 12
    VL_SEQUENCE_TYPE        = 13
    # numpy type conversion
    TO_NUMPY_TYPE = {
        FIXED_POINT_TYPE: {
            True: {
                1:  numpy.int8,
                2:  numpy.int16,
                4:  numpy.int32,
                8:  numpy.int64
            },
            False: {
                1:  numpy.uint8,
                2:  numpy.uint16,
                4:  numpy.uint32,
                8:  numpy.uint64
            }
        },
        FLOATING_POINT_TYPE: {
            True: {
                4:  numpy.single,
                8:  numpy.double
            }
        }
    }

    #######################
    # Constructor
    #######################
    def __init__(self, address=0):
        self.ndims              = None
        self.dimensions         = []
        self.typeSize           = 0
        self.type               = None
        self.signedval          = True
        self.fillsize           = 0
        self.fillvalue          = None
        self.layout             = None
        self.size               = 0
        self.address            = address
        self.chunkElements      = 0
        self.chunkDimensions    = []
        self.elementSize        = 0
        self.isattribute        = False
        self.byteOrder          = 0     # 0: little endian, 1: big endian (fixed/floating point only)
        self.members            = []    # compound: list of (name, offset, H5Metadata) in file order
        self.arrayDims          = []    # array: dimension sizes of the array datatype
        self.baseType           = None  # array: H5Metadata of the element datatype
        self._numpyDtype        = None  # cache for the numpyDtype property
        self.filter             = {
            self.DEFLATE_FILTER:        False,
            self.SHUFFLE_FILTER:        False,
            self.FLETCHER32_FILTER:     False,
            self.SZIP_FILTER:           False,
            self.NBIT_FILTER:           False,
            self.SCALEOFFSET_FILTER:    False
        }

    #######################
    # representation
    #######################
    def __repr__(self):
        typestr = f'{self.type} (unsupported)'
        if self.type == self.FIXED_POINT_TYPE or self.type == self.FLOATING_POINT_TYPE:
            datatype = self.TO_NUMPY_TYPE[self.type][self.signedval][self.typeSize]
            typestr = f'{datatype}'
        elif self.type == self.STRING_TYPE or self.type == self.VL_STRING_TYPE:
            typestr = f'{str}'
        elif self.type == self.COMPOUND_TYPE:
            members = ', '.join([f'\"{name}\": {dtype}' for name, dtype in self.memberDtypes().items()])
            return f'{{\"type\": {self.numpyDtype}, \"members\": {{{members}}}, \"dims\": {self.dimensions}}}'
        return f'{{\"type\": {typestr}, \"dims\": {self.dimensions}}}'

    #######################
    # print
    #######################
    def __str__(self):
        return self.__repr__()

    #######################
    # numpyDtype
    #######################
    @property
    def numpyDtype(self):
        """
        The numpy dtype that maps onto this datatype's bytes in the file.

        Compound types become structured dtypes (names, formats, offsets and
        the struct's itemsize), nested compounds recurse, array members become
        sub-array fields, fixed length strings become S<n>, and any member that
        cannot be decoded becomes an opaque V<n> of the right size so that the
        itemsize of the enclosing struct is always correct.  The result is
        computed once (after the datatype message has been parsed) and cached.
        """
        if self._numpyDtype is None:
            self._numpyDtype = self._buildNumpyDtype()
        return self._numpyDtype

    def _buildNumpyDtype(self):
        if self.type == self.COMPOUND_TYPE:
            try:
                return numpy.dtype({
                    'names':    [name for name, _, _ in self.members],
                    'formats':  [member.numpyDtype for _, _, member in self.members],
                    'offsets':  [offset for _, offset, _ in self.members],
                    'itemsize': self.typeSize
                })
            except (ValueError, TypeError) as e:
                log.warning(f'compound datatype of size {self.typeSize} has members that do not fit a numpy dtype ({e}): treating as opaque')
                return numpy.dtype(f'V{self.typeSize}')
        elif self.type == self.FIXED_POINT_TYPE or self.type == self.FLOATING_POINT_TYPE:
            try:
                dtype = numpy.dtype(self.TO_NUMPY_TYPE[self.type][self.signedval][self.typeSize])
            except KeyError:
                log.warning(f'no numpy dtype for datatype class {self.type} of size {self.typeSize}: treating as opaque')
                return numpy.dtype(f'V{self.typeSize}')
            return dtype.newbyteorder('>' if self.byteOrder == 1 else '<')
        elif self.type == self.STRING_TYPE:
            return numpy.dtype(f'S{self.typeSize}')
        elif self.type == self.ARRAY_TYPE and self.baseType != None:
            return numpy.dtype((self.baseType.numpyDtype, tuple(self.arrayDims)))
        else:
            log.warning(f'datatype class {self.type} of size {self.typeSize} is not decodable: treating as opaque')
            return numpy.dtype(f'V{self.typeSize}')

    #######################
    # memberDtypes
    #######################
    def memberDtypes(self):
        """Ordered {member name: numpy dtype} for a compound type ({} otherwise)."""
        if self.type != self.COMPOUND_TYPE:
            return {}
        return {name: member.numpyDtype for name, _, member in self.members}

    #######################
    # getNumpyType
    #######################
    def getNumpyType(self):
        """Convert H5Metadata type to a NumPy-compatible dtype."""
        if self.type in [self.FIXED_POINT_TYPE, self.FLOATING_POINT_TYPE]:
            try:
                return numpy.dtype(self.TO_NUMPY_TYPE[self.type][self.signedval][self.typeSize])
            except KeyError:
                return numpy.float32  # Default fallback for unknown floating-point types

        elif self.type in [self.STRING_TYPE, self.VL_STRING_TYPE]:
            return numpy.dtype("U")  # Use NumPy's Unicode string type

        elif self.type == self.COMPOUND_TYPE:
            return self.numpyDtype  # structured dtype of the records

        else:
            return numpy.dtype("O")  # Default to object type for unsupported types

    #######################
    # getShape
    #######################
    def getShape(self):
        return tuple(self.dimensions) if self.ndims > 0 else (None,)

    #######################
    # getDims
    #######################
    def getDims(self):
        """Return dimension names based on metadata."""
        return [f"dim_{i}" for i in range(len(self.dims))] if self.dims else ["dim_0"]
