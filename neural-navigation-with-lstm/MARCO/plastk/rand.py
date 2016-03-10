"""
Random numbers for PLASTK.

$Id: rand.py,v 1.4 2005/08/20 16:46:14 jp Exp $
"""

#from RandomArray import *
import sys

def shuffle(L):
    """
    Return randomly permuted version of L.  (non-destructive)
    """
    return [L[i] for i in permutation(len(L))]


def strseed(s):    
    s1 = s[::2]
    s2 = s[1::2]

    seed(int(strhash(s1)),int(strhash(s2)))

def randrange(i,j=None,step=1):
    if j==None:
        r = range(0,i,step)
    else:
        r = range(i,j,step)
    return r[int(uniform(0,len(r)))]

def sample(seq,weights=[]):
    if not weights:
        return seq[randrange(len(seq))]
    else:
        assert len(weights) == len(seq)
        return seq[sample_index(weights)]

def sample_index(weights):
    total = sum(weights)
    if total == 0:
        return randrange(len(weights))
    index = random() * total
    accum = 0
    for i,x in enumerate(weights):
        accum += x
        if index < accum:
            return i

def strhash(s,base=31,mod=2147483647):
    return reduce(lambda x,y: (x*base+y)%mod,map(ord,s))
