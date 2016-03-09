"""
Unit tests for PLASTK

$Id: __init__.py,v 1.5 2005/06/23 18:48:21 jp Exp $
"""

import unittest

import testtd
import testfnapprox
import testvectordb

suite = unittest.TestSuite()

def all(verbosity=1):
    return unittest.TextTestRunner(verbosity=verbosity).run(suite)

for key,val in locals().items():
    if type(val) == type(unittest) and val != unittest:
        try:
            print 'Checking module %s for test suite...' % key,
            suite.addTest(getattr(val,'suite'))
            print 'found.'
        except AttributeError,err:
            print err

