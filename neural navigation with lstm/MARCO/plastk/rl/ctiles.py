# This file was created automatically by SWIG.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.

import _ctiles

def _swig_setattr(self,class_type,name,value):
    if (name == "this"):
        if isinstance(value, class_type):
            self.__dict__[name] = value.this
            if hasattr(value,"thisown"): self.__dict__["thisown"] = value.thisown
            del value.thisown
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    self.__dict__[name] = value

def _swig_getattr(self,class_type,name):
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types


class intArray(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, intArray, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, intArray, name)
    def __repr__(self):
        return "<C intArray instance at %s>" % (self.this,)
    def __init__(self, *args):
        _swig_setattr(self, intArray, 'this', _ctiles.new_intArray(*args))
        _swig_setattr(self, intArray, 'thisown', 1)
    def __del__(self, destroy=_ctiles.delete_intArray):
        try:
            if self.thisown: destroy(self)
        except: pass
    def __getitem__(*args): return _ctiles.intArray___getitem__(*args)
    def __setitem__(*args): return _ctiles.intArray___setitem__(*args)
    def cast(*args): return _ctiles.intArray_cast(*args)
    __swig_getmethods__["frompointer"] = lambda x: _ctiles.intArray_frompointer
    if _newclass:frompointer = staticmethod(_ctiles.intArray_frompointer)

class intArrayPtr(intArray):
    def __init__(self, this):
        _swig_setattr(self, intArray, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, intArray, 'thisown', 0)
        _swig_setattr(self, intArray,self.__class__,intArray)
_ctiles.intArray_swigregister(intArrayPtr)

intArray_frompointer = _ctiles.intArray_frompointer

class floatArray(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, floatArray, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, floatArray, name)
    def __repr__(self):
        return "<C floatArray instance at %s>" % (self.this,)
    def __init__(self, *args):
        _swig_setattr(self, floatArray, 'this', _ctiles.new_floatArray(*args))
        _swig_setattr(self, floatArray, 'thisown', 1)
    def __del__(self, destroy=_ctiles.delete_floatArray):
        try:
            if self.thisown: destroy(self)
        except: pass
    def __getitem__(*args): return _ctiles.floatArray___getitem__(*args)
    def __setitem__(*args): return _ctiles.floatArray___setitem__(*args)
    def cast(*args): return _ctiles.floatArray_cast(*args)
    __swig_getmethods__["frompointer"] = lambda x: _ctiles.floatArray_frompointer
    if _newclass:frompointer = staticmethod(_ctiles.floatArray_frompointer)

class floatArrayPtr(floatArray):
    def __init__(self, this):
        _swig_setattr(self, floatArray, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, floatArray, 'thisown', 0)
        _swig_setattr(self, floatArray,self.__class__,floatArray)
_ctiles.floatArray_swigregister(floatArrayPtr)

floatArray_frompointer = _ctiles.floatArray_frompointer

MAX_NUM_VARS = _ctiles.MAX_NUM_VARS

mod = _ctiles.mod

GetTiles = _ctiles.GetTiles

GetTilesWrap = _ctiles.GetTilesWrap
def GetTiles(num_tilings,variables,mem_size,ints=[]):
    """
    usage:  tilings = GetTiles(num_tilings,variables,mem_size,hash1=-1,hash2=-1,hash3=-1)

      tilings = returned list of tilings
      num_tilings = numer of tilings to generate (should be a power of 2)
      variables = sequence of (float) variables to tile
      mem_size = memory size for hash (i.e. number of features)
    """
    if len(variables) > MAX_NUM_VARS:
        raise "GetTiles can only tile %d variables." % MAX_NUM_VARS
    tile_array = intArray(num_tilings)
    var_array = floatArray(len(variables))
    for i,x in enumerate(variables):
        var_array[i] = x
    int_array = intArray(len(ints))
    for i,x in enumerate(ints):
        int_array[i] = x
    _ctiles.GetTiles(tile_array,
                    num_tilings,
                    mem_size,
                    var_array,len(variables),
                    int_array,len(ints))
    return [tile_array[i] for i in range(num_tilings)]

def GetTilesWrap(num_tilings,variables,mem_size,wrap_widths,ints=[]):

    """
    usage:  tilings = GetTilesWrap(num_tilings,variables,mem_size,hash1=-1,hash2=-1,hash3=-1)

      tilings = returned list of tilings
      num_tilings = numer of tilings to generate (should be a power of 2)
      variables = sequence of (float) variables to tile
      mem_size = memory size for hash (i.e. number of features)
      wrap_widths = sequence of (integer) wrap_widths (i.e. number of tiles before wrapping)
    """
    assert(len(variables) == len(wrap_widths))

    if len(variables) > MAX_NUM_VARS:
        raise "GetTiles can only tile %d variables." % MAX_NUM_VARS
    # Make the return array
    tile_array = intArray(num_tilings)

    # Make and fill the value array
    var_array = floatArray(len(variables))
    for i,x in enumerate(variables):
        var_array[i] = x

    # Make and fill the widths array
    width_array = intArray(len(variables))
    for i,x in enumerate(wrap_widths):
        width_array[i] = x
    
    # Make and fill the int array (may be empty)
    int_array = intArray(len(ints))
    for i,x in enumerate(ints):
        int_array[i] = x
    _ctiles.GetTilesWrap(tile_array,
                        num_tilings,
                        mem_size,
                        var_array,len(variables),
                        width_array,
                        int_array,len(ints))
    return [tile_array[i] for i in range(num_tilings)]



