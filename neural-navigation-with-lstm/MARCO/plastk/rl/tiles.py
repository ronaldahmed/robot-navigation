#				Tiling routines
#
# $Id: tiles.py,v 1.5 2005/04/01 22:12:33 jp Exp $
#
# External documentation and recommendations on the use of this code is
# available at http://rlai.net/...

# This is an implementation of grid-style tile codings, _based originally on
# the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm). 
# Here we provide a function, "GetTiles", that maps floating and integer
# variables to a list of tiles. This function is memoryless and requires no
# setup...but only if hashing collisions are to be ignored.  This is a second 
# version that optionally checks for hashing collisions and handles them appropriately
# to simulate an infinite, accurate memory.  (Until we run out of memory
# of course.)  We use open addressing with double hashing indexing.
# Memory-size must be a power of 2.  

# The float variables will be gridded at unit intervals, so generalization
# will be by approximately 1 in each direction, and any scaling will have 
# to be done externally before calling tiles.

# It is recommended by the UNH folks that nunTilings be a power of 2, e.g., 16. 

#     Useful routines:
#		getTiles(numTilings, memSizeorCtable, floats, ints)
#      loadTiles(tiles, startElement, numTilings, memSizeorCtable, floats, ints)
#			both of these routines prepare numTiling tiles
#				getTiles returns them as a list
#				loadTiles loads them into array tiles starting from position startElement
#			if memSizeorCtable is a CollisionTable, 
#				hashing collisions are checked for and handled
#				otherwise it must be an integer power of 2 and collisions are ignored
#			floats is a list of real variables to be tiled
#			ints is an optional list of integer variables to be tiled
#		CollisionTable(size,safety) or makeCtable(size,safety)
#			size and safety are optional; size must be a power of 2; makeCtable checks for this

from plastk.rand import random, randrange
from math import floor, log

_maxNumFloats = 200					# maximum number of variables used in one grid
_maxLongint = 2147483647
_maxLongintBy4 = _maxLongint // 4          
_randomTable = [randrange(65536) for i in range(2048)]

# The following are temporary variables used by tiles.
_qstate = [0 for i in range(_maxNumFloats)]
_base = [0 for i in range(_maxNumFloats)]
_coordinates = [0 for i in range(1 + 2*_maxNumFloats)]

class CollisionTable:
	"Structure to handle collisions"
	def __init__(self, size_val=2048, safety_val='safe'):
		# if not power of 2 error
		if not powerOf2(size_val):
			print "error - size should be a power of 2"
		self.size = size_val				
		self.safety = safety_val			# one of 'safe', 'super safe' or 'unsafe'
		self.calls = 0
		self.clearhits = 0
		self.collisions = 0
		self.data = [-1 for i in range(self.size)]
		
	def info (self):
		"Prints info about collision table"
		print "usage", self.ctUsage(), "size", self.size, "calls", self.calls, "clearhits", self.clearhits, \
				"collisions", self.collisions, "safety", self.safety

	def reset (self):
		"Reset Ctable values"
		self.calls = 0
		self.clearhits = 0
		self.collisions = 0
		self.data = [-1 for i in range(self.size)]
	
	def stats (self):
		return self.calls, self.clearhits, self.collisions, self.usage

	def usage (self):
		use = 0
		for d in self.data:
			if d >= 0:
				use += 1
		return use

def startTiles (numTilings, floats, ints=[]):
	"Does initial assignments to _coordinates, _base and _qstate for both GetTiles and LoadTiles"
	numFloats = len(floats)
	i = numFloats + 1					# starting place for integers
	for v in ints:					# for each integer variable, store it
		_coordinates[i] = v				
		i += 1
	i = 0
	for float in floats:				# for real variables, quantize state to integers
		_base[i] = 0
		_qstate[i] = int(floor(float * numTilings))
		i += 1
		
def fixCoord (numTilings, numFloats, j):
	"Fiddles with _coordinates and _base - done once for each tiling"
	for i in range(numFloats):			# for each real variable
		_coordinates[i] = _qstate[i] - ((_qstate[i] - _base[i]) % numTilings)
		_base[i] += 1 + 2*i
	_coordinates[numFloats] = j
	
def hashTile (numTilings, memSizeorCtable, numCoordinates):
	"Chooses hashing method and applies"
	if isinstance(memSizeorCtable, CollisionTable):
		hnum = hash(_coordinates, numCoordinates, memSizeorCtable)
	else:
		hnum = hashUNH(_coordinates, numCoordinates, memSizeorCtable)
	return hnum
		
def getTiles (numTilings, memSizeorCtable, floats, ints=[]):
	"""Returns list of numTilings tiles corresponding to variables (floats and ints),
	    hashed down to memSize, using ctable to check for collisions"""
	numFloats = len(floats)
	numCoordinates = 1 + numFloats + len(ints)
	startTiles (numTilings, floats, ints)
	tlist = []
	for j in range(numTilings):				# for each tiling
		fixCoord(numTilings, numFloats, j)
		hnum = hashTile(numTilings, memSizeorCtable, numCoordinates)
		tlist.append(hnum)
	return tlist
	
def loadTiles (tiles, startElement, numTilings, memSizeorCtable, floats, ints=[]):
	"""Loads numTilings tiles into array tiles, starting at startElement, corresponding
	   to variables (floats and ints), hashed down to memSize, using ctable to check for collisions"""
	numFloats = len(floats)
	numCoordinates = 1 + numFloats + len(ints)
	startTiles (numTilings, floats, ints)
	for j in range(numTilings):
		fixCoord(numTilings, numFloats, j)
		hnum = hashTile(numTilings, memSizeorCtable, numCoordinates)
		tiles[startElement + j] = hnum

def hashUNH (ints, numInts, m, increment=449):
	"Hashing of array of integers into below m, using random table"
	res = 0
	for i in range(numInts):
		res += _randomTable[(ints[i] + i*increment) % 2048]
	return res % m

def hash (ints, numInts, ct):
	"Returns index in collision table corresponding to first part of ints (an array)"
	ct.calls += 1
	memSize = ct.size
	j = hashUNH(ints, numInts, memSize)
	if ct.safety == 'super safe':
		ccheck = [ints[i] for i in range(numInts)]
	else:											# safe or unsafe
		ccheck = hashUNH(ints, numInts, _maxLongint, 457)
	if ccheck == ct.data[j]:
		ct.clearhits += 1
	elif ct.data[j] < 0:
		ct.clearhits += 1
		ct.data[j] = ccheck
	elif ct.safety == 'unsafe':			# collison, but we aren't worried	
		ct.collisions += 1
	else:											# handle collision
		h2 = 1 + 2*hashUNH(ints, numInts, _maxLongintBy4)
		i = 1
		while ccheck != ct.data[j]:
			ct.collisions += 1
			j = (j + h2) % memSize
			if i > memSize:
				print "Out of memory"
				ct.data[j] = ccheck			# make it stop
			if ct.data[j] < 0:
				ct.data[j] = ccheck
			i += 1
	return j

def makeCtable (size=2048, safety='safe'):
	"Makes a collision table"
	# if not power of 2 error
	if not powerOf2(size):
		print "error - size should be a power of 2"
	else:
		return CollisionTable(size, safety)

def powerOf2 (n):
	lgn = log(n, 2)
	return (lgn - floor(lgn)) == 0
	

