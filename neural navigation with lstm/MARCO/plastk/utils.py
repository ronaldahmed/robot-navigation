"""
Various utility functions.

$Id: utils.py,v 1.19 2006/04/07 23:24:31 jp Exp $
"""



import numpy
from numpy import ones,exp,sqrt,zeros,argsort
import sys,operator

import plastk.rand as rand

inf = (ones(1)/0.0)[0]


def enumerate(seq):
    """
    (Deprecated) duplicates enum generator.
    """
    i = 0
    for x in seq:
        yield i,x
        i+=1

enum = enumerate


def smooth(data,N=10,step=1):
    """
    Smooth sequence data using a moving window of width N.  Returns a
    list of tuples, <i, mu, stderr>, where i is the index of the last
    element of in the moving window, mu is the average of the window,
    and stderr is the standard error of the values in the window.

    If step > 1, the values are subsampled, and every step'th  value
    is returned.
    """
    results = []
    for i in xrange(N,len(data)):
        avg,var,stderr = stats(numpy.array(data[i-N:i]))
        results.append((i-1,avg,stderr))
    return results[0::step]

def median_filter(data,N=5):
    """
    Median filter sequence data with window of width N.
    """
    results = zeros(len(data-N))
    for i in xrange(N,len(data)):
        x = data[i-N:i]
        s = argsort(x)
        results[i] = x[s[(N/2)+1]]
    return results
        


def mmin(ar):
    """
    Take the min value over an array of arbitrary dimension. Works for
    slices and other arrays where ar.flat is undefined.
    """
    if len(ar.shape) == 1:
        return min(ar)
    else:
        return min([mmin(x) for x in ar])

def mmean(ar):
    """
    Take the average of an array of arbitrary dimension. Works for
    slices and other arrays where ar.flat is undefined.
    """
    from numpy import average
    if len(ar.shape) == 1:
        return average(ar)
    else:
        return average([mmean(x) for x in ar])

def msum(ar):
    """
    Take the sum of an array of arbitrary dimension. Works for
    slices and other arrays where ar.flat is undefined.
    """
    from numpy import sum
    if len(ar.shape) == 1:
        return sum(ar)
    else:
        return sum([msum(x) for x in ar])
    
def mvariance(data):
    """
    Take the variance of an array of arbitrary dimension. Works for
    slices and other arrays where ar.flat is undefined.
    """
    from numpy import multiply
    tmp = data - mmean(data)
    return mmean(multiply(tmp,tmp))
    
def mmax(ar):
    """
    Take the max of an array of arbitrary dimension. Works for
    slices and other arrays where ar.flat is undefined.
    """
    if len(ar.shape) == 1:
        return max(ar)
    else:
        return max([mmax(x) for x in ar])

def stats(data):
    """
    Assumes a matrix of data with variables on the columns
    and observations on the rows.  Returns the mean,
    variance and standard error of the data.
    """
    from numpy import average,sqrt
    mean = average(data)
    var  = average((data-mean)**2)
    stderr = sqrt(var)/sqrt(len(data))
    return mean,var,stderr

## def median(data,pivot_set=None):
##     """
##     Assumes a sequence of scalar data, returns the median.
##     """
##     if pivot_set is None:
##         pivot_set = data

##     if len(pivot_set) == 1:
##         pivot = pivot_set[0]
##         if len(data) % 2 == 0:
##             lt,ge = partition(data,pivot)
##             if len(lt) >= len(ge):
##                 x = max(lt)
##             else:
##                 x = min([x for x in ge if x != pivot])
##             return (pivot+x)/2.0
##         else:
##             return pivot
##     else:
            
##         pivot = pivot_set[rand.randint(len(pivot_set))]        
##         lt,ge = partition(data,pivot)
                
##         if len(lt) < len(ge):
##             return median(data,[x for x in pivot_set if x in ge])
##         else:
##             return median(data,[x for x in pivot_set if x in lt])


def median(data):
    N = len(data)
    inds = argsort(data)
    return data[inds[N/2+1]]

def partition(data,pivot):
    lt,ge = [],[]
    for x in data:
        if x < pivot:
            lt.append(x)
        else:
            ge.append(x)
    return lt,ge
    

def norm(ar,L=2):
    return sum(abs(ar)**L)**(1.0/L)

def L2norm(ar):
    from numpy import dot,sqrt
    return sqrt(sum(ar**2))

def matrixnorm(m):
    """
    L2 norm of each row of a matrix.
    """
    mm = m*m
    return numpy.sqrt(numpy.sum(m*m,axis=1))


def histogram(data,bins):
    xmin = min(data)
    xmax = max(data)
    xrange = xmax-xmin
    step = xrange/float(bins)

    result = [0 for x in range(bins)]

    for x in data:
        if x == xmax: result[-1] += 1
        else: result[ int((x-xmin)/step) ] += 1
    return [(i*step+xmin) for i in range(bins)],result

def plot_hist(bins,counts,fmt='%10.2f',bar_width=50):
    if max(counts) > bar_width:
        unit_size = max(counts)/bar_width
    else:
        unit_size = 1
    print "UNIT_SIZE =", unit_size

    for b,c in zip(bins,counts):
        prefix = fmt % b + ' : '
        print fmt % b,':', ''.join(['*' for x in range(c/unit_size)]), c


def normalize_sum(ar):
    d = float(sum(ar))
    if d == 0:
        return ar * 0.0
    return ar/d

def entropy(X):
    """
    Computes the entropy of a histogram contained in sequence X.
    """
    from numpy import log,sum
    def fn(x):
        if x == 0:
            return 0
        else:
            return x*(log(x)/log(2))

    P = X/float(sum(X))
    return -sum(map(fn,P))

def gini_coeff(X,pad_len=0,pad_value=0):
    """
    Computes the Gini coefficient of a set of values contained in the
    1-d array X.

    If pad_len > len(X), X is padded out to length pad_len with
    the value pad_value (default 0).
    """
    
    # from http://mathworld.wolfram.com/GiniCoefficient.html
    # note there is probably a more efficient (O(n log n)) computation using
    # argsort(X), but this one was easiest to implement.
    
    from numpy import argsort,zeros,concatenate as join 
    if pad_len > len(X):
        X = join((X,zeros(pad_len-len(X))+pad_value))
    G = 0.0
    n = len(X)
    for xi in X:
        for xj in X:
            G += abs(xi-xj)
    return G/(2*n*n*mmean(X)) * (n/(n-1))

def gaussian(dist,stddev):
    X = - dist**2/(2*stddev**2)

    # cut off exponent at -500 to prevent overflow errors
#    if type(X) == Numeric.ArrayType:
#        numpy.putmask(X,X<-500,-500)
#    elif X < -500:
#        X = -500
        
    return exp(X)

def simple_gaussian_kernel(X):

    # cut off exponent at -500 to prevent overflow errors
 #   if type(X) == Numeric.ArrayType:
 #       numpy.putmask(X,X<-500,-500)
 #   elif X < -500:
 #       X = -500
        
    return exp(X)

def decay(time,half_life):
    return 0.5**(time/float(half_life))



def choose(a,b,pred):
    if pred(a,b):
        return a
    else:
        return b

def best_N(seq,N=1,choice=max):
    from sets import Set

    if len(seq) <= N:
        return seq
    
    if not isinstance(seq,Set):
        seq = Set(seq)
    r = (reduce(choice,seq),)
    if N==1:
        return r
    else:
        return r + best_N(seq-Set(r),N-1,choice)

def best_N_destructive(seq,N=1,pred=operator.gt):
    if len(seq) > N:
        for i in xrange(N):
            for j in xrange(i+1,len(seq)):
                if pred(seq[j],seq[i]):
                    tmp = seq[j]
                    seq[j] = seq[i]
                    seq[i] = tmp
        del seq[N:]

    return seq



def weighted_sample(weights):
    return rand.sample_index(weights)


def analyse_transitions(T):
    from numpy import zeros
    states,actions,results = T.shape

    entropies = zeros((states,actions)) * 0.0
    counts = zeros((states,actions)) * 0.0
    max_prob = zeros((states,actions)) * 0.0
    max_act = zeros((states,actions)) * 0.0
    
    for s in range(states):
        for a in range(actions):
            entropies[s,a] = entropy(normalize_sum(T[s,a]))
            counts[s,a] = sum(T[s,a])
            max_prob[s,a] = max(normalize_sum(T[s,a]))
            max_act[s,a] = argmax(normalize_sum(T[s,a]))

    print '       : ',
    for c in 'FBLR':
        print '%10s' % c,
    print
    print '-------------------------------------------------------------'
    

    for r in range(states):        
        print '%6d' % r,': ',
        for c in range(actions):
            print '%6.2f (%2d)' % (max_prob[r,c],max_act[r,c]),
        print



def print_table(table,annote=None,
                width=10,fmt='%.2f',col_labels=None,row_labels=None,
                out=sys.stdout):


    rows,cols = table.shape

    if not col_labels:
        col_labels = map(str,range(cols))
    if not row_labels:
        row_labels = map(str,range(rows))
           
    sfmt = '%%%ds' % width

    out.write(sfmt % '')
    for s in col_labels:
        out.write(sfmt % s)
    out.write('\n')
    for i in range(width*(cols+1)):
        out.write('-')
    out.write('\n')
                   
    for r in range(rows):
        out.write(sfmt % ('%s :' % row_labels[r]))
        for c in range(cols):
            if annote:
                out.write(sfmt % (fmt % (table[r,c],annote[r,c])))
            else:
                out.write(sfmt % (fmt % table[r,c]))
        out.write('\n')
            
        
def get_schema_reliabilities(T,threshold=0.9):

    num_contexts,num_actions,num_results = T.shape
    
    for c in range(num_contexts):
        for a in range(num_actions):
            for r in range(num_results):
                if T[c,a,r] >= threshold:
                    print (c,a,r), ' => %.2f' % T[c,a,r]

def read_matlab(file):

    import sre
    f = open(file,'r')
    data = []
    str = f.readline()
    while str:
        l = sre.split(' ',str)
        del(l[-1])
        data.append(map(float,l))
        str = f.readline()

    return numpy.array(data)

def write_matlab(data,filename):
    """
    Assumes data is a 2D array (or list of lists).  Writes
    data to file, one row per line.
    """

    f = open(filename,'w')
    for row in data:
        for x in row:
            f.write(`x`)
            f.write(' ')
        f.write('\n')
    f.close()



def uniq(L):
    if not L: return []
    result = [ L[0] ]
    for x in L[1:]:
        if x != result[-1]:
            result.append(x)
    return result

def time_call(N,fn,*args,**kw):
    import time
    total = 0
    for i in range(N):
        start = time.clock()
        fn(*args,**kw)
        end = time.clock()
        total += (end-start)
    return total/N
    

####################
class Stack(list):
    __slots__ = []
    def top(self):
        return self[-1]
    def empty(self):
        return len(self) == 0
    def push(self,x):
        return self.append(x)
    
####################

class LogFile(object):

    def __init__(self,name,mode='w'):
        self._file = file(name,mode)
        
        
    def __getstate__(self):
        self._file.flush()
        state = self._file.name,self._file.tell()
        print "Logfile saving state:", state
        return state

    def __setstate__(self,state):
        print "LogFile restoring state:", state
        name,pos = state
        self._file = file(name,'a')
        self._file.seek(pos)
        self._file.truncate()

    def write(self,str):
        self._file.write(str)

    def writelines(self,seq):
        for l in seq:
            self.write(l)
            
