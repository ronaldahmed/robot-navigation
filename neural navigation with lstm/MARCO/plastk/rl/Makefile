# -*- makefile -*-


CXX = g++
CXXFLAGS = -g -O2 -MD
SWIG = swig
SWIGFLAGS = -c++ -python 
PYTHON_INCL = -I/usr/local/include/python2.3 -I/usr/include/python2.3 -I/p/include/python2.3

all: _ctiles.so ctiles.py


_ctiles.so: ctiles_wrap.o tiles2.o tiles2.h
	${CXX} -shared ctiles_wrap.o tiles2.o -o _ctiles.so

ctiles_wrap.cxx: ctiles.i 
	${SWIG} ${SWIGFLAGS} ctiles.i


ctiles_wrap.o: ctiles_wrap.cxx tiles2.h
	${CXX} ${CXXFLAGS} ${PYTHON_INCL} -c $<

.o : .C
	${CXX} ${CXXFLAGS} -c $<

clean:
	rm -f *.so *_wrap.cxx ctiles.py *.o *.d