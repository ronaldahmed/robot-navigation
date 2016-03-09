"""
Transition Model

$Id: seqcount.py,v 1.2 2006/04/13 01:28:52 jp Exp $
"""

from plastk.base import BaseObject
from plastk.params import *
from plastk import utils
import Numeric

class SequenceCounter(BaseObject):
    seq_constructor = Parameter(default=list)
    
    def __init__(self,data=[],**params):
        super(SequenceCounter,self).__init__(**params)

        self._count = 0
        self.t_table = {}

        for seq,count in data:
            self.record_seq(seq,count=count)


    def record_seq(self,seq,count=1):

        self._count += count

        if seq:
            head = seq[0]
            tail = seq[1:]

            if head not in self.t_table:
                self.t_table[head] = SequenceCounter(seq_constructor=self.seq_constructor)
            self.t_table[head].record_seq(tail,count=count)


    def sub_counter(self,*seq):
        if not seq:
            return self
        else:
            return self.t_table[seq[0]].sub_counter(*seq[1:])

    def children(self):
        return self.t_table.keys()
    
    def get_seqs(self,prefix=[]):
        seqtype = self.seq_constructor
        if not prefix:
            if not self.t_table:
                result = [seqtype([])]
            else:
                result = []
                for head,table in self.t_table.iteritems():
                    for tail in table.get_seqs():
                        result.append(seqtype((head,)) + tail)
            return result
        else:
            return self.t_table[prefix[0]].get_seqs(prefix=prefix[1:])

    def count(self,prefix=[]):
        if not prefix:
            return self._count
        elif prefix[0] not in self.t_table:
            return 0
        else:
            return self.t_table[prefix[0]].count(prefix[1:])

    def subcounts(self,prefix=[]):
        if not prefix:
            return Numeric.array([m._count for m in self.t_table.itervalues()])
        elif prefix[0] not in self.t_table:
            return Numeric.array([0])
        else:
            return self.t_table[prefix[0]].subcounts(prefix[1:])

    

