"""
Pickling enhancements.

Some utilities for easier pickling and file handling.

$Id: pkl.py,v 1.6 2006/03/20 21:03:40 jp Exp $
"""
import os,re,gzip
import cPickle as pk
#import pickle as pk

def load(str,msg=False):
    """
    Load the pickle from the file with name str. If msg is True,
    print a 'Loading' message.
    """
    if msg:
        print "Loading",str
    f = gzopen(str,'r')
    result =  pk.load(f)
    f.close()
    return result

def dump(x,name,mode='w'):
    """
    Dump the object x into the pickle file name. (a.k.a. pkl.save())
    """
    f = gzopen(name,mode)
    pk.dump(x,f)
    f.close()

save = dump

def files(dir='.'):
    """
    Return a list of the files in dir.
    """
    return files_matching('.*\.pkl(.gz)?$',dir=dir)

def list(dir='.'):
    """
    Print a list of the files in dir.
    """
    for f in files(dir):
        print f

def files_matching(pat,dir='.',regexp=False):
    """
    Return a list of the files in dir that match pattern pat.  If
    regexp is true, pat is assumed to be a regular expression,
    otherwise, it's assumed to be a shell-style file matching pattern
    """
    import fnmatch
    if not regexp:
        pat = fnmatch.translate(pat)
    return [f for f in os.listdir(dir) if re.match(pat,f)]

def load_all(pattern,msg=False,regexp=False):
    """
    Load (as pickles) all files matching pattern.
    """
    return [load(f,msg) for f in files_matching(pattern,regexp=regexp)]

def is_gzip(name):
    """
    True if the given name ends with '.gz'
    """
    return name.split('.')[-1] == 'gz'

def gzopen(name,mode):
    """
    Open the named file, with gzip if necessary, otherwise as a normal
    file.
    """
    if is_gzip(name):
        return gzip.open(name,mode)
    else:
        return open(name,mode)



class pgen(object):
    """
    A simple wrapper that makes a generator picklable.  Takes a generator
    function and arguments.

    WARNING: when the generator is unpickled, its state is restored by
    re-iterating up to the point at which the pickling occurred.  This
    may be computationally expensive, and may not generate correct
    output (or work at all) for some generator objects (e.g. those
    that depend on the concurrently changing state of some other
    object.
    """
    def __init__(self,fn,*args,**kw):
        self.fn = fn
        self.args = args
        self.kw = kw

        self.gen = fn(*args,**kw)
        self.pos = 0

    def next(self):
        self.pos += 1
        return self.gen.next()

    def __iter__(self):
        return self

    def __getstate__(self):
        return dict(fn   = self.fn,
                    args = self.args,
                    kw   = self.kw,
                    pos  = self.pos)

    def __setstate__(self,state):
        for k,v in state.iteritems():
            setattr(self,k,v)

        self.gen = self.fn(*self.args,**self.kw)

        for i in xrange(self.pos):
            self.gen.next()
            
