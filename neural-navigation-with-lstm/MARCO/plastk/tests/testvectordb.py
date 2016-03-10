"""
Unit tests for vector databases.

$Id: testvectordb.py,v 1.3 2005/06/28 03:00:28 jp Exp $
"""
import unittest
import pickle
from plastk import rand
from plastk.fnapprox.vectordb import *
from plastk.utils import *
from Numeric import array,arange

class TestVectorDB(unittest.TestCase):

    dim = 2
    N = 1000
    probes = [array((x,y)) for x in arange(0,1.25,0.25) for y in arange(0,1.25,0.25)]

    
    def setUp(self):
        rand.seed(0,0)        
        self.data = [(rand.uniform(0,1,(self.dim,)),None) for i in range(self.N)]
        for x,y in self.data:
            self.db.add(x,y)
            
    def testKNearest(self):
        k = 10
        
        def probe_cmp(a,b,probe):
            ax,ay = a
            bx,by = b
            diff = norm(ax-probe)-norm(bx-probe)
            return int(diff/abs(diff))


        for p in self.probes:
            k_nearest,dists = self.db.k_nearest(p,k)
            self.data.sort(lambda a,b:probe_cmp(a,b,p))        
            assert k_nearest == self.data[:k]
                                  

    def testFindInRadius(self):
        r = 0.1
        for p in self.probes:
            found,dists = self.db.find_in_radius(p,r)
            dists = [norm(p-x) for x,y in self.data]
            truth = [(x,y) for d,(x,y) in zip(dists,self.data) if d <= r]
            for pair in truth:
                assert pair in found
            for pair in found:
                assert pair in truth
             
class TestFlatVectorDB(TestVectorDB):
    def setUp(self):
        self.db = FlatVectorDB()
        TestVectorDB.setUp(self)

class TestVectorTree(TestVectorDB):
    def setUp(self):
        self.db = VectorTree()
        TestVectorDB.setUp(self)

        
class TestPickleVectorTree(TestVectorTree):
    def setUp(self):
        TestVectorTree.setUp(self)
        s = pickle.dumps(self.db)
        self.orig_db = self.db
        self.db = pickle.loads(s)

    def testContents(self):
        for xo,vo in self.orig_db.all():
            (x,v),d = self.db.nearest(xo)
            self.assertEqual(x,xo)
        
cases = [TestFlatVectorDB,
         TestVectorTree,
         TestPickleVectorTree,
         ]
suite = unittest.TestSuite()
suite.addTests([unittest.makeSuite(case) for case in cases])
