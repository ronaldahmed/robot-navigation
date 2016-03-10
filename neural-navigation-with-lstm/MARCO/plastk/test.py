"""
Script to run PLASTK unit tests.

$Id: test.py,v 1.5 2005/06/28 02:56:07 jp Exp $
"""
import sys,time,getopt,unittest,pdb
import plastk.tests
import plastk.base


USAGE = """
usage: python test.py [options]  [test-module1 [test-module2 ...]]

options:
   --debug  Run in debugging mode (stop on first failure)
   -v N     Unittest verbosity = N (default 1)
   -p LEVEL PLASTK print level = LEVEL (default WARNING)
"""

try:
    opts,args = getopt.getopt(sys.argv[1:],'v:p:',['debug'])
except getopt.GetoptError:
    sys.exit(USAGE)

opts = dict(opts)

print_level = opts.get('-p','WARNING')
verbosity = int(opts.get('-v', '1'))
                       
plastk.base.BaseObject.print_level = getattr(plastk.base,print_level)

if args:
    for test in args:
        suite = getattr(plastk.tests,test).suite
        if '--debug' in opts:
            suite.debug()
        else:
            unittest.TextTestRunner(verbosity=verbosity).run(suite)
else:
    if '--debug' in opts:
        plastk.tests.suite.debug()
    else:
        plastk.tests.all(verbosity)
    
