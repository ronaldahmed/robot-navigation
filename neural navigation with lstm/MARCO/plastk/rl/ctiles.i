/* -*- c++ -*- */
/* $Id: ctiles.i,v 1.1 2004/10/04 20:15:28 jp Exp $ */
%module ctiles
%{
    // This is C/C++ code to be included in the wrapper module
#include "tiles2.h"

extern  int mod(int,int);
%}

// Define some special array types to hold ints and floats
%include "carrays.i"
%array_class(int,intArray);
%array_class(float,floatArray);


// below are the things to be wrapped.
#define MAX_NUM_VARS 20

int mod(int n, int k);

void GetTiles(
	int tiles[],               // provided array contains returned tiles (tile indices)
	int num_tilings,           // number of tile indices to be returned in tiles       
    int memory_size,           // total number of possible tiles
	float floats[],            // array of floating point variables
    int num_floats,            // number of floating point variables
    int ints[],				  // array of integer variables
    int num_ints);             // number of integer variables

void GetTilesWrap(
	int tiles[],               // provided array contains returned tiles (tile indices)
	int num_tilings,           // number of tile indices to be returned in tiles       
    int memory_size,           // total number of possible tiles
	float floats[],            // array of floating point variables
    int num_floats,            // number of floating point variables
    int wrap_widths[],         // array of widths (length and units as in floats)
    int ints[],				  // array of integer variables
    int num_ints);             // number of integer variables


// Below are my redefinitions of the Python-side proxy functions
// that call the wrapped C functions.  I changed the interfaces to make them more
// Python-like.


%pythoncode %{
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

%}
       
