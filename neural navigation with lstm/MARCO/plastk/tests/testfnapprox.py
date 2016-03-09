"""
Unit tests for Function approximators.

$Id: testfnapprox.py,v 1.6 2005/11/04 22:12:57 jp Exp $
"""

import unittest
from plastk.fnapprox.linear import LinearFnApprox
from plastk.fnapprox.lwl import *
from plastk import rand
from plastk.utils import norm
from Numeric import dot

class TestFnApprox(unittest.TestCase):

    num_inputs = 4
    num_outputs = 1

    training_size = 1000
    _test_size = 1000
    acceptable_error = 0.01
    acceptable_failure_rate = 0.05
    training_error = 0.01

    def setUp(self):
        self.training_data = [rand.uniform(-1,1,self.num_inputs)
                              for x in range(self.training_size)]
        self.test_data = [rand.uniform(-1,1,self.num_inputs)
                          for x in range(self._test_size)]
        self.W = rand.uniform(-1,1,(self.num_outputs,self.num_inputs))


    def ground_truth(self,X):
        return dot(self.W,X)

    def testUniform(self):

        # train
        for X in self.training_data:
            Y = self.ground_truth(X)
            e = rand.normal(0,self.training_error,self.num_outputs)
            self.fn.learn_step(X,Y + e)

        # test
        failures = 0
        total_err = 0
        for i,X in enumerate(self.test_data):
            Y_learned = self.fn(X)
            Y_true = self.ground_truth(X)
            err = norm(Y_learned-Y_true)
            total_err += err
            self.fn.verbose("err =",err)
            failures += (err > self.acceptable_error)
        self.fn.message( "avg err =", total_err/len(self.test_data))
        failure_rate = failures/float(len(self.test_data))
        self.fn.message("%.1f%% Failure" % (failure_rate * 100))
        assert failure_rate < self.acceptable_failure_rate
                
            

        

class TestLinearFnApprox(TestFnApprox):
    acceptable_error = 0.01
    def setUp(self):
        TestFnApprox.setUp(self)
        self.fn = LinearFnApprox(num_inputs = self.num_inputs,
                                 num_outputs = self.num_outputs,
                                 alpha = 0.1)

class TestLWA(TestFnApprox):
    num_inputs = 2
    training_size = 3000
    acceptable_error = 0.05
    def setUp(self):
        TestFnApprox.setUp(self)
        self.fn = LocallyWeightedAverage(num_inputs = self.num_inputs,
                                         num_outputs = self.num_outputs,
                                         bandwidth=0.1)
class TestKNA(TestFnApprox):
    training_size = 3000
    acceptable_error = 0.05
    num_inputs = 2
    def setUp(self):
        TestFnApprox.setUp(self)
        self.fn = KNearestLWA(num_inputs = self.num_inputs,
                                  num_outputs = self.num_outputs,
                                  bandwidth=0.1,
                                  k=10)

class TestLWLR(TestFnApprox):
    training_size = 3000
    acceptable_error = 0.05
    num_inputs = 2
    def setUp(self):
        TestFnApprox.setUp(self)
        self.fn = LWLR(num_inputs = self.num_inputs,
                       num_outputs = self.num_outputs,
                       bandwidth=0.1,
                       )
    
class TestKNearestLWLR(TestFnApprox):
    training_size = 3000
    acceptable_error = 0.01
    num_inputs = 2
    def setUp(self):
        TestFnApprox.setUp(self)
        self.fn = KNearestLWLR(num_inputs = self.num_inputs,
                               num_outputs = self.num_outputs,
                               bandwidth=0.2,
                               k = 10
                               )
    

cases = [TestLinearFnApprox,TestLWA,TestKNA,TestLWLR,TestKNearestLWLR]
suite = unittest.TestSuite()
suite.addTests([unittest.makeSuite(case) for case in cases])
 
