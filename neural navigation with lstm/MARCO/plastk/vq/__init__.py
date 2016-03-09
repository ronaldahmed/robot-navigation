"""
Generic vector quantizer module.

$Id: __init__.py,v 1.6 2005/09/13 14:54:09 jp Exp $
"""

__all__ = ['som','gng','VQ','VQTrainingMemory']

from plastk.base import BaseObject
from plastk.params import *

NYI = "Method must be implemented in sublcass."

class VQ(BaseObject):
    """
    The abstract vector quantizer class.  A vector quantiser must
    iomplement these methods.
    """
    def present_input(self,X):
        raise NYI
    def train(self,X):
        raise NYI
    def winner(self):
        raise NYI
    def get_model_vector(self,index):
        raise NYI
    def num_model_vectors(self):
        raise NYI
    def get_activation(self):
        raise NYI


class VQTrainingMemory(VQ):

    memory_size = PositiveInt(default = 1000)

    def __init__(self,**args):

        super(VQTrainingMemory,self).__init__(**args)
        self._mem = []
        
    def train(self,X,**kw):
        from plastk import rand
        self._mem.append((X,kw))
        
        N = len(self._mem)
        i = rand.randrange(N)
        self.debug("Training on memory element",i)
        V,training_kw = self._mem[i]
        super(VQTrainingMemory,self).train(V,**training_kw)

        if N > self.memory_size:
            self._mem.pop(0)
            
        self.present_input(X)
            
        
