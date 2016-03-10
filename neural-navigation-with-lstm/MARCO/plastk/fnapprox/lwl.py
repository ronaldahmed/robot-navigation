"""
Locally Weighted Learning


$Id: lwl.py,v 1.11 2006/01/18 23:23:20 jp Exp $
"""

import Numeric
from Numeric import array,transpose
from Numeric import matrixmultiply as mult
from Numeric import concatenate as join
from LinearAlgebra import inverse,linear_least_squares

import plastk.fnapprox.vectordb
from plastk.fnapprox import FnApprox
from plastk.params import *
from plastk.utils import inf,stats,simple_gaussian_kernel
from plastk import rand

class LocalLearner(FnApprox):
    """
    Abstract base class for locally weighted learning.
    """
    input_weights = Parameter(default=[])

    db_type   = Parameter(default=plastk.fnapprox.vectordb.VectorTree)
    db_params = Parameter(default={})

    def __init__(self,**params):
        super(LocalLearner,self).__init__(**params)
        self.db = self.db_type(vector_len=self.num_inputs,**self.db_params)
        if self.input_weights:
            self.input_weight_vec = Numeric.array(self.input_weights)
        else:
            self.input_weight_vec = 1.0

    def learn_step(self,X,Y):
        X *= self.input_weight_vec
        self.db.add(X, Y)



class LocallyWeightedLearner(LocalLearner):

    bandwidth = Number(default=1.0)

    ###############
    
    def __call__(self,X,error=False):

        h = self.bandwidth
        X = X * self.input_weight_vec

        XYs,dists = self._query(X)

        if not XYs:
            if error:
                N = self.num_outputs
                XYs = [(X,Numeric.zeros(N))]
                dists = [inf]
            else:
                return None
        
        weights = self.kernel(Numeric.array(dists)/h)
        self.verbose("Combining",len(weights),"points.")
        result,err =  self._combine(X,
                                    [x for x,y in XYs],
                                    [y for x,y in XYs],
                                    weights)

        if error:
            return result,err
        else:
            return result

    def _query(self,X):
        h = self.bandwidth
        return self.db.find_in_radius(X, h*2)

    def _combine(self,q,Xs,Ys,weights):
        raise NYI

    def kernel(self,x):
        return simple_gaussian_kernel(-x**2)

    
class LocallyWeightedAverage(LocallyWeightedLearner):

    def _combine(self,q,Xs,Ys,weights):
        return weighted_average(Ys,weights)



def weighted_average(Ys,weights):
    sumw = sum(weights)

    if sumw == 0.0:
        N = len(Ys[0])
        avg = Numeric.zeros(N)
        err = Numeric.array([inf]*N)
    else:
        avg = Numeric.sum( [y*w for y,w in zip(Ys, weights)] ) / sumw
        var = sum( [w*(y-avg)**2 for y,w in zip(Ys,weights)] ) / sumw
        err = Numeric.sqrt(var)/Numeric.sqrt(sumw)

    return avg,err
    

LWA = LocallyWeightedAverage

class LocallyWeightedLinearRegression(LocallyWeightedLearner):
    """
    Locally Weighted Linear Regression algorithm taken from
    Atkeson, Moore, and Schall, "Locally Weighted Learning"
    """

    ridge_range = Number(default=0.0)

    def _combine(self,q,Xs,Ys,weights):
        q = array(q)
        X = array(Xs)

        rows,cols = X.shape
        
        if rows < cols:
            self.verbose("Falling back to weighted averaging.")
            return weighted_average(Ys,weights)
        
        Y = array(Ys)
        W = Numeric.identity(len(weights))*weights
        Z = mult(W,X)
        v = mult(W,Y)

        if self.ridge_range:
            ridge = Numeric.identity(cols) * rand.uniform(0,self.ridge_range,(cols,1))
            Z = join((Z,ridge))
            v = join((v,Numeric.zeros((cols,1))))
            

        B,residuals,rank,s = linear_least_squares(Z,v)

        if len(residuals) == 0:
            self.verbose("Falling back to weighted averaging.")
            return weighted_average(Ys,weights)
        
        estimate = mult(q,B)

        # we estimate the variance as the sum of the
        # residuals over the squared sum of the weights
        variance = residuals/sum(weights**2)

        stderr = Numeric.sqrt(variance)/Numeric.sqrt(sum(weights))

        return estimate,stderr
        
LWLR = LocallyWeightedLinearRegression        


class KNearestLearner(LocallyWeightedLearner):
    k = PositiveInt(default=3)

    def _query(self,X):
        return self.db.k_nearest(X,self.k)

class KNearestLWA(KNearestLearner,LWA): pass
class KNearestLWLR(KNearestLearner,LWLR): pass


class InputFilter(LocallyWeightedLearner):

    change_threshold = Magnitude(default=0.1)

    def learn_step(self,X,Y):

        h = self.bandwidth
        X = X * self.input_weight_vec

        XYs, dists = self._query(X)

        if not XYs:
            super(InputFilter,self).learn_step(X,Y)
        else:
            W1 = self.kernel(Numeric.array(dists)/h)
            Xs = [x for x,y in XYs]
            Ys = [y for x,y in XYs]

            result1,err1 = self._combine(X,Xs,Ys,W1)

            W2 = Numeric.concatenate((W1,[self.kernel(0)]))
            Xs.append(X)
            Ys.append(Y)

            result2,err2 = self._combine(X,Xs,Ys,W2)

            if (abs(err2-err1)/err1 > self.change_threshold
                or abs(result2-result1)/result1 >self.change_threshold):

                self.verbose("Adding point",(X,Y),".")
                super(InputFilter,self).learn_step(X,Y)
            else:
                self.verbose("Not adding point",(X,Y),"insufficient change.")


class FilteredLWA(InputFilter,LocallyWeightedAverage): pass
class FilteredKNLWA(InputFilter,KNearestLWA): pass
class FilteredLWLR(InputFilter,LocallyWeightedLinearRegression): pass
class FilteredKNLWLR(InputFilter,KNearestLWLR): pass            
            
