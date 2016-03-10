"""
Searchable database classes for continuous vectors.

$Id: vectordb.py,v 1.10 2005/08/20 16:50:13 jp Exp $
"""

import plastk.base
from plastk.base import *
from plastk.params import *
from plastk.utils import best_N_destructive,norm,matrixnorm
from Numeric import sometrue,array,argsort,nonzero

class VectorDB(object):

    __slots__ = ['vector_len']

    def add(self,key,value):
        """
        Add key to the database with the given value
        """
        raise NYI

    def k_nearest(self,key,k):
        """
        Return a list of key,value pairs for the k keys in the db
        nearest to key.  If there are fewer than k pairs in the db,
        all available pairs will be returned.
        """
        raise NYI

    def nearest(self,key):
        """
        Return the (key,value) pair for the key in the db closest to key.
        """
        results,dists = self.k_nearest(key,1)
        return results[0],dists[0]

    def find_in_radius(self,key,radius):
        """
        Return a list of all (key,value) pairs in the db with keys
        within a given radius of key.  If none, [] is returned.
        """
        raise NYI
    
    def any_in_radius(self,key,radius):
        """
        Return true if any point in the db is within the given
        radius of key.  May be much faster than find_in_radius.
        """
        raise NYI
    def all(self):
        """
        Return  list of all (key,value) pairs in the db.
        """
        raise NYI

    def debug(self,*args):
        if plastk.base.min_print_level >= DEBUG:
            print ' '.join([str(x) for x in args])

    def __getstate__(self):
        slots = []
        for c in classlist(type(self)):
            try:
                slots += c.__slots__
            except AttributeError:
                pass
        return dict([(s,getattr(self,s)) for s in slots])
    
    def __setstate__(self,dict):
        for k,v in dict.items():
            setattr(self,k,v)


class FlatVectorDB(VectorDB):
    __slots__ = ['db']

    def __init__(self,vector_len=2):
        self.vector_len = vector_len
        self.db = []

    def add(self,key,value):
        if not self.db:
            self.db.append((key,value))
        else:
            (nkey,nval),ndist = self.nearest(key)
            if norm(nkey-key) > 0.0:
                self.db.append((key,value))

    def k_nearest(self,key,k):

        # TODO: These distance computations can be further optimized
        # if the keys are stored as a matrix instead of as separate vectors.
        # However that would require changes in the VectorTree class, too.
        if not self.db:
            return [],[]
        X = array([x for x,v in self.db])
        dists = matrixnorm(key-X)
        sorted_indices = argsort(dists)
        
        return ([self.db[i] for i in sorted_indices[:k]],
                [dists[i] for i in sorted_indices[:k]])


    def find_in_radius(self,key,radius):
        if not self.db:
            return [],[]
        X = array([x for x,v in self.db])
        dists = matrixnorm(key-X)

        close_enough = nonzero(dists <= radius)
        return ([self.db[i] for i in close_enough],
                [dists[i] for i in close_enough])

    def any_in_radius(self,key,radius):
        if not self.db:
            return False
        X = array([x for x,v in self.db])
        dists = matrixnorm(key-X)
        return sometrue(dists <= radius)

    def all(self):
        return [(k,v) for k,v, in self.db]

    def size(self):
        return len(self.db)

class VectorTree(VectorDB):
    """
    A K-D-Tree-like vector database structure that stores a
    FlatVectorDB in each leaf and splits leaves when they reach a
    given maximum size.
    """
    __slots__ = ['max_flat_size',
                 'd_stride',
                 '_depth',
                 '_flat',
                 '_split_val',
                 '_le',
                 '_gt' ]

    def __init__(self,depth=0,
                 vector_len=2,
                 max_flat_size=100,
                 d_stride=1):

        self.vector_len = vector_len
        self.max_flat_size = max_flat_size
        self.d_stride = d_stride
        self._depth = depth
        self._flat = FlatVectorDB()
        self._split_val = None
        self._le = None
        self._gt = None

    def __getstate__(self):
        state = super(VectorTree,self).__getstate__()
        del state['_flat']
        del state['_le']
        del state['_gt']
        del state['_split_val']
        state['all'] = self.all()
        return state
    def __setstate__(self,state):
        all = state['all']
        del state['all']
        super(VectorTree,self).__setstate__(state)
        self._flat = FlatVectorDB()
        self._flat.db = all
        self._split()
        
        
    def add(self,key,value):
        assert len(key) == self.vector_len
        if self._flat:
            self._flat.add(key,value)
            if self._flat.size() > self.max_flat_size:
                self._split()
        elif key[self._depth%len(key)] <= self._split_val:
            self._le.add(key,value)
        else:
            self._gt.add(key,value)

    def size(self):
        if self._flat:
            return self._flat.size()
        else:
            return self._le.size() + self._gt.size()

    def all(self):
        if self._flat:
            return self._flat.all()
        else:
            return self._le.all() + self._gt.all()
        
    def _split(self):
        assert self._flat

        self.debug("Splitting at depth",self._depth)
        db = self._flat.db
        self._flat = None
        index = self._split_index()

        splits = [x[0][index] for x in db]
        splits.sort()

        N = len(splits)
        if N%2 == 0:
            # if the list size is even
            self._split_val = (splits[N/2] + splits[N/2-1])/2
        else:
            # if the list size is odd
            self._split_val = splits[len(splits)/2]

        self._le = VectorTree(depth=self._depth+1,
                              vector_len=self.vector_len,
                              d_stride=self.d_stride,
                              max_flat_size=self.max_flat_size)
                              
        self._gt = VectorTree(depth=self._depth+1,
                              vector_len=self.vector_len,
                              d_stride=self.d_stride,
                              max_flat_size=self.max_flat_size)

        self._le._flat.db = [(k,v) for k,v in db if self._is_le(k)]
        self._gt._flat.db = [(k,v) for k,v in db if not self._is_le(k)]

        if self._le.size() > self.max_flat_size:
            self._le._split()
        if self._gt.size() > self.max_flat_size:
            self._gt._split()
        
            
    def _is_le(self,key):
        return key[self._split_index()] <= self._split_val
        
    def _split_index(self):
        return (self._depth*self.d_stride) % self.vector_len

    def k_nearest(self,key,k):
        if self._flat:
            return self._flat.k_nearest(key,k)
        elif self._is_le(key):
            close_branch = self._le
            far_branch = self._gt
        else:
            close_branch = self._gt
            far_branch = self._le

        # Get the k nearest from the close side of the split
        results,dists = close_branch.k_nearest(key,k)

        # if the distance to the farthest result is less than
        # the distance to the split then we don't need to check the other
        # branch
        if results and  not self._split_within_radius(key,max(dists)):
            return results,dists
        else:
            far_results,far_dists = far_branch.k_nearest(key,k)

            results += far_results
            dists += far_dists

            final = zip(dists,results)
            final.sort()

            return [(x,v) for d,(x,v) in final[:k]],[d for d,(x,v) in final[:k]]

    def find_in_radius(self,key,radius):
        if self._flat:
            return self._flat.find_in_radius(key,radius)
        elif self._split_within_radius(key,radius):
            results1,dists1 = self._le.find_in_radius(key,radius)
            results2,dists2 = self._gt.find_in_radius(key,radius)
            return results1+results2, dists1+dists2
        elif self._is_le(key):
            return self._le.find_in_radius(key,radius)
        else:
            return self._gt.find_in_radius(key,radius)
        

    def any_in_radius(self,key,radius):
        if self._flat:
            return self._flat.find_in_radius(key,radius)
        elif self._split_within_radius(key,radius):
            return (self._le.any_in_radius(key,radius) or
                    self._gt.any_in_radius(key,radius) )
        elif self._is_le(key):
            return self._le.any_in_radius(key,radius)
        else:
            return self._gt.any_in_radius(key,radius)
        
            
    def _split_within_radius(self,key,radius):            
        i = self._split_index()
        s = self._split_val
        return radius >= abs(s-key[i])

    def get_depth(self,cmp=max):
        if self._flat:
            return 1
        else:
            return 1 + cmp(self._le.get_depth(cmp),self._gt.get_depth(cmp))

    def rebalance(self):
        from plastk import rand
        data = rand.shuffle(self.all())
        self._flat = FlatVectorDB(vector_len = self.vector_len)
        self._flat.db = data
        self._le = self._gt = None
        self._split()



class KDTree(VectorDB):
    """
    A kd-tree vector database.
    """
    __slots__ = ['_key',
                 '_value',
                 '_le',
                 '_gt' ]

    def __init__(self,vector_len=2):

        self.vector_len = vector_len
        self._key = None
        self._value = None
        self._le = None
        self._gt = None

    def get_depth(self,cmp=max):
        if self._le and self._gt:
            return 1 + cmp(self._le.get_depth(cmp),self._gt.get_depth(cmp))
        elif self._le:
            return 1 + self._le.get_depth(cmp)
        elif self._gt:
            return 1 + self._gt.get_depth(cmp)
        else:
            if self._key:
                return 1
            else:
                return 0

    def add(self,key,value,depth=0):
        assert len(key) == self.vector_len

        d = depth % len(key)
        if self._key is None:
            self._key = key
            self._value = value
        elif key[d] <= self._key[d]:
            if not self._le:
                self._le = KDTree(vector_len = self.vector_len)
            self._le.add(key,value,depth+1)
        else:
            if not self._gt:
                self._gt = KDTree(vector_len = self.vector_len)
            self._gt.add(key,value,depth+1)

    def all(self,result=None):
        if result is None:
            result = []

        result.append( (self._key,self._value) )
        if self._le:
            self._le.all(result)
        if self._gt:
            self._gt.all(result)

        return result
    
    def k_nearest(self,key,k,depth=0):

        if self._key is None:
            results = []
        else:
            d = depth % self.vector_len

            if key[d] <= self._key[d]:
                close_branch = self._le
                far_branch = self._gt
            else:
                close_branch = self._gt
                far_branch = self._le

            # Get the k nearest from the close side of the split
            if close_branch is not None:
                results = close_branch.k_nearest(key,k,depth+1)
            else:
                results = []

            # if the distance to the farthest result is less than
            # the distance to the split then we don't need to check the other
            # branch
            if not results or (max([dist for v,dist in results]) >= abs(self._key[d] - key[d])):
                if far_branch is not None:
                    far_results = far_branch.k_nearest(key,k,depth+1)
                else:
                    far_results = []
                results.append( ((self._key,self._value), norm(self._key - key)) )            
                results.extend(far_results)
                #results = best_N_destructive(results,N=k,pred=lambda a,b:a[1] < b[1])
                results.sort(key = lambda p: p[1])
                results = results[:k]
        if depth == 0:
            return [v for v,d in results],[d for v,d in results]
        else:
            return results


class KDTreeIterative(KDTree):

    def k_nearest(self,key,k):

        stack = [(self,0,none)]
        vl = self.vector_len
        while stack:
            node, depth, branch = stack[-1]
            d = depth % vl

            if key[d] <= self._key[d]:
                close_branch
    

########################
def testflat(n=1000):
    from plastk.rand import uniform
    db = FlatVectorDB()
    for i in range(n):
        db.add(uniform(0,1,(2,)),None)
    return db
        
def testkd(n=1000):
    from plastk.rand import uniform
    db = KDTree()
    for i in range(n):
        db.add(uniform(0,1,(2,)),None)
    return db

def testvt(n=1000):
    from plastk.rand import uniform
    db = VectorTree()
    for i in range(n):
        db.add(uniform(0,1,(2,)),None)
    return db


def testall(n=1000,d=2):
    from plastk.rand import uniform
    
    vt = VectorTree()
    kd = KDTree()
    flat = FlatVectorDB()
    for i in range(n):
        v = uniform(0,1,(d,))
        flat.add(v,None)
        kd.add(v,None)
        vt.add(v,None)

    return flat,kd,vt

def k_nearest_timing_test(d=2,k=1,tests=[VectorTree]):
    from plastk.rand import uniform,seed
    from plastk.utils import time_call
    import time

    new_seed = time.time()
    for e in range(3,20):
        print '==================='
        for db_type in tests:
            print
            print "Testing",db_type
            db = db_type(vector_len=d)

            n = 2**e
            seed(int(new_seed),int(new_seed%1 * 1000000))
            print "Adding",n,"data points....",
            total_add_time = 0.0
            for i in range(n):
                x = uniform(0,1,(d,))
                start = time.clock()
                db.add(x,None)
                end = time.clock()
                total_add_time += end-start
            print "done. Average add time = %4.3f ms." %((total_add_time/n)*1000)

            print "Average search search time...",
            seed(0,0)
            print '%6.3f ms'% (1000*time_call(100,lambda: db.k_nearest(uniform(0,1,(d,)),k)))


    
def radius_timing_test(d=2,radius=0.1,tests=[VectorTree]):
    from plastk.rand import uniform,seed
    from plastk.utils import time_call
    import time
    
    for e in range(3,20):
        for db_type in tests:
            print
            print "Testing",db_type
            db = db_type(vector_len=d)

            n = 2**e
            print "Adding",n,"data points....",
            for i in range(n):
                x = uniform(0,1,(d,))
                db.add(x,None)
            print "done."

            print "Average search search time...",
            seed(0,0)
            start = time.clock()
            total_results = 0
            for i in range(100):
                results,dists = db.find_in_radius(uniform(0,1,(d,)),radius)
                total_results += len(results)
            end = time.clock()
            print (end-start)/100
            print "Average results size:", total_results/100.0



if __name__ == '__main__':
    from Numeric import *
    from pprint import *
    import pdb

    kd1 = KDTree(vector_len = 1)
    for x in arange(10):
        kd1.add(array([x]),None)

        
        
