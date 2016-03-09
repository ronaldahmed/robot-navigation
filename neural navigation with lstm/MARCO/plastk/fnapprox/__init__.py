"""
Function approximators.

$Id: __init__.py,v 1.10 2005/11/14 17:43:54 jp Exp $
"""
__all__ = ['linear','vectordb','lwl']

from plastk import base, utils, params, rand

class FnApprox(base.BaseObject):

    num_inputs = params.PositiveInt(default=1)
    num_outputs = params.PositiveInt(default=1)
    batch_epochs = params.PositiveInt(default=1)

    range = params.Parameter(default=(0,1))

    def __init__(self,**params):
        super(FnApprox,self).__init__(**params)
        self.MSE=0

    def __call__(self,X,error=False):
        """
        Apply self to X.
        """
        raise base.NYI
    
    def learn_step(self,X,Y):
        """
        Learn step for self(X)=Y
        """
        raise base.NYI
        
    def learn_batch(self,data):
        """
        Learns on a batch of data, given as a sequence.  The batch is shuffled,
        then presented sequentially to self.learn_step()
        """
        import plastk.rand
        self.verbose("Training on",len(data),"examples.")
        for i in xrange(self.batch_epochs):
            data = rand.shuffle(data)        
            for X,Y in data:
                self.learn_step(X,Y)

    def scale_input(self,X):
        rmin,rmax = self.range
        return (X-rmin)/(rmax-rmin)

    def scale_output(self,X):
        rmin,rmax = self.range
        return X*(rmax-rmin) + rmin

