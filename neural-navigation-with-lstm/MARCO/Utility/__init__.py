class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)
    def __str__(self): return str(self.__dict__)

#http://www.dalkescientific.com/writings/diary/archive/2005/04/20/tracing_python_code.html
import sys, math, linecache

def traceit(frame, event, arg):
    if event == "line":
        lineno = frame.f_lineno
        filename = frame.f_globals["__file__"]
        if (filename.endswith(".pyc") or
            filename.endswith(".pyo")):
            filename = filename[:-1]
        name = frame.f_globals["__name__"]
        line = linecache.getline(filename, lineno)
        print "%s:%s: %s" % (name, lineno, line.rstrip())
    return traceit

def any(X,Y):
    for x in X:
        if x in Y: return True
    return False

def all(X,Y): 
    for x in X:
        if x not in Y: return False
    return True
    
def uniq(X):
    if not X: return X
    X.sort()
    Y = [X[0]]
    for x in X[1:]:
        if x != Y[-1]: Y.append(x)
    return Y

def choose2(A):
    length = len(A)
    for i in range(length):
        for j in range(i+1, length):
            yield A[i], A[j]

def entropy(L):
    return sum([-p*math.log(p,2) for p in L if p > 0])

def conditional_entropy_x(M):
    cond_entropy = 0.0
    for row in M:
        row_sum = sum(row)
        cond_entropy += row_sum*entropy([p/row_sum for p in row])
    return cond_entropy

def conditional_entropy(dist):
    """
    >>> X_Y = {1: [0.125, 0.0625, 0.03125, 0.03125],
    ...     2: [0.0625, 0.125, 0.03125, 0.03125],
    ...     3: [0.0625, 0.0625, 0.0625, 0.0625],
    ...     4: [0.25, 0, 0, 0]}
    >>> for y,row in X_Y.items():
    ...     print y,entropy(normalize_to_prob(row)),','
    ...
    1 1.75 , 2 1.75 , 3 2.0 , 4 0.0 ,
    >>> print 'H(X|Y) =', conditional_entropy(X_Y)
    H(X|Y) = 1.375
    >>> Y_X = {1: [1.0/8, 1.0/16, 1.0/16, 1.0/4],
    ...     2: [1.0/16, 1.0/8, 1.0/16, 0],
    ...     3: [1.0/32, 1.0/32, 1.0/16, 0],
    ...     4: [1.0/32, 1.0/32, 1.0/16, 0]}
    >>> for x,row in Y_X.items(): 
    ...     print x,entropy(normalize_to_prob(row)),',',
    ... 
    1 1.75 , 2 1.5 , 3 1.5 , 4 1.5 ,
    >>> print 'H(Y|X) =', conditional_entropy(Y_X)
    H(Y|X) = 1.625
    """
    cond_entropy = 0.0
    for key,row in dist.items():
        cond_entropy += sum(row)*entropy(normalize_to_prob(row))
    return cond_entropy

def normalize_to_prob(L):
    n=float(sum(L))
    if not n: return L
    return [l/n for l in L]

def percent(part,total):
    if total: return (100.0*part/total)
    else: return 0.0

def histogram( L, flAsList=False ):
    """Return histogram of values in list L."""
    H = {}
    for val in L:
        H[val] = H.get(val,0) + 1
    if flAsList:
    	return H.items()
    return H

def mode(L,min_occur=0):
    max_occur = min_occur
    mode_value = None
    for value,count in histogram(L).items():
        if count >= max_occur: mode_value = value
    return mode_value

def frequency_entropy(L):
    return entropy(normalize_to_prob(histogram(L).values()))

try:
    import pylab
    def stderr(X,Y=None):
        if len(X) <= 1: return 0.0
        stderr_x = pow(pylab.std(X),2)/len(X)
        if Y:
            if len(Y) <= 1: return 0.0
            stderr_y = pow(pylab.std(Y),2)/len(Y)
        else: stderr_y = 0
        return math.sqrt(stderr_x + stderr_y)

    def ttest(X,Y):
        """
        Takes two lists of values, returns t value
        
        >>> ttest([2, 3, 7, 6, 10], [11,2,3,1,2])
        0.77459666924148329
        """
        if len(X) <= 1 or len(Y) <= 1: return 0.0
        return ((pylab.mean(X) - pylab.mean(Y))
                / stderr(X,Y))

    def paired_ttest(X,Y=None):
        """
        Takes two lists of paired values, returns t value
        
        >>> A = pylab.array([63, 54, 79, 68, 87, 84, 92, 57, 66, 53, 76, 63])
        >>> B = pylab.array([55, 62, 108, 77, 83, 78, 79, 94, 69, 66, 72, 77])
        >>> paired_ttest(A,B)
        -1.486762837055869
        >>> paired_ttest(A,[1])
        0.0
        """
        if type(X) != 'array': X = pylab.array(X)
        if len(X) <= 1: return 0.0
        if Y:
            if type(Y) != 'array': Y = pylab.array(Y)
            if len(Y) <= 1: return 0.0
            Array = X-Y
        else: Array = X
        StdErr = stderr(Array)
        if StdErr == 0.0: return 0.0
        return (pylab.mean(Array) / StdErr)

    def ttest_sig(X, Y, paired=False, one_sided=False):
        if paired:
            dof = len(X) - 1
            t_val = paired_ttest(X, Y)
        else:
            dof = len(X) + len(Y) - 2
            t_val = ttest(X, Y)
        sig = -0.0000
        return dof, t_val, sig
        
    def runningAverage(X,n=10,avg=pylab.mean):
        return pylab.array([avg(X[i:i+n]) for i in range(len(X)-n)])
except: pass

try:
    import scipy.stats
    def ttest_sig(X, Y, paired=False, one_sided=False):
        """
        Takes two lists of values, returns dof, t value and significance
        
        >>> ttest_sig([2, 3, 7, 6, 10], [11,2,3,1,2])
        0.77459666924148329
        """
        if paired:
            dof = len(X) - 1
            t_val, sig = scipy.stats.ttest_rel(X, Y)
        else:
            dof = len(X) + len(Y) - 2
            t_val, sig = scipy.stats.ttest_ind(X, Y)
        if one_sided: sig /= 2
        return dof, float(t_val), sig
except: pass

def latex_ttest((dof, t_val, sig)):
    return r'\ttest{%d}{%3.1f}{%3.3f}' % ((dof, t_val, sig))

def latex_correlation((dof, r, sig)):
    return r'\correlation{%d}{%4.3f}{%3.3f}' % ((dof, r, sig))

def append_flat(l,val):
    if '__iter__' in dir(val):
        l.extend(val)
    else:
        l.append(val)

class Enum:
    class Type(str):
        def __init__(self, name):
            self.__name = name
        def __str__(self):
            return self.__name
    
    def __init__(self, *keys):
        self.__keys = []
        for key in keys:
            mytype = self.Type(key)
            self.__dict__[key] = mytype
            self.__keys.append(mytype)
    
    def __iter__(self):
        return  iter(self.__keys)

def reprVisible(self):
    return ''.join([self.__class__.__name__,
                    '( *(',
                    ', '.join([repr(getattr(self,name)) for name in self.__visible__]),
                    ') )'
                    ])

def strVisible(self):
    return ''.join([self.__class__.__name__,
                    '(',
                    ', '.join([''.join([name,'=',repr(getattr(self,name))])
                               for name in self.__visible__]),
                    ' )',
                    ])

import os
def lstail(directory,fileglob=None,number=None):
    files = os.listdir(directory)
    if fileglob: files = [file for file in files if fileglob.match(file)]
    files.sort()
    if number: return files[-number:]
    else: return files

def nextFileSeq(directory, filename):
    matches = [file for file in os.listdir(directory) if file.startswith(filename)]
    return len(matches)+1
