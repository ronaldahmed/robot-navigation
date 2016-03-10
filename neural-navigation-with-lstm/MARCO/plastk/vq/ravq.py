"""
PLASTK Wrappers for Pyrobot's RAVQ and ARAVQ implementations

$Id: ravq.py,v 1.1 2005/08/20 19:04:19 jp Exp $
"""

from plastk.vq import VQ
from plastk.params import *
from pyrobot.brain import ravq


class RAVQ_Base(VQ):
    buffer_size = PositiveInt(default = 100)
    epsilon = Number(default = 0.1)
    delta  = Number(default = 0.1)
    history_size = NonNegativeInt(default = 0)

    def __init__(self,**args):
        super(RAVQ_Base,self).__init__(**args)
        self.winner_index = None
        self.winner_vec = None

    def present_input(self,X):
        self.train(X)
    def train(self,X):
        self.winner_index,self.winner_vec = self._ravq.input(X)
    def winner(self):
        return self.winner_index
    def get_model_vector(self,index):
        return self._ravq.models[index].vector
    def num_model_vectors(self):
        return len(self._ravq.models)
    def get_activation(self):
        raise NYI

class RAVQ(RAVQ_Base):
    def __init__(self,**args):
        super(RAVQ,self).__init__(**args)
        self._ravq = ravq.RAVQ(self.buffer_size,self.epsilon,
                                self.delta,self.history_size,
                                )
        


class ARAVQ(RAVQ):
    
    alpha = Magnitude(default = 0.1)

    def __init__(self,**args):
        super(ARAVQ,self).__init__(**args)
        self._ravq = ravq.ARAVQ(self.buffer_size,self.epsilon,
                                self.delta,self.history_size,
                                self.alpha)

