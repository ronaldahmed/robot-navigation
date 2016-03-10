import re
import operator

def bitvec2indices(bitvec):
    indices = []
    for i,bit in enumerate(bitvec):
        if bit: indices.append(i)
    return indices

def indices2bitvec(indices,len):
    bitvec=[0]*len
    for index in indices: bitvec[index] = 1
    return bitvec

# Thanks to Tim Peters
def powerset(seq):
    if seq:
        head, tail = seq[:1], seq[1:]
        for smaller in powerset(tail):
            yield smaller
            yield head + smaller
    else:
        yield []

def matchPatterns(re1,re2):
    p1 = re1.pattern[3:-3].split(',')
    p2 = re2.pattern[3:-3].split(',')
    def matchPatts(a,b):
        if '|' in a or '|' in b:
            if '|' in a: a = a[1:-1].split('|')
            elif type(a) == str: a = [a]
            if '|' in b: b = b[1:-1].split('|')
            elif type(b) == str: b = [b]
            for ax in a:
                for bx in b:
                    if matchPatts(ax,bx): return True
            return False
        else:
            return (bool(re.match(a,b)) or bool(re.match(b,a)))
    matches = [matchPatts(a,b) for a,b in zip(p1,p2)]
    return reduce(operator.and_, matches)

def findCompatibleViewDescs(ViewDescs):
    CompatViewDescs = []
    for viewDesc1 in ViewDescs:
        CompatViewDescs.append([int(matchPatterns(viewDesc1,viewDesc2))
                                for viewDesc2 in ViewDescs])
    return CompatViewDescs

def compileViewDescs(ViewDescIndex):
    ViewDesc = {}
    for pose,viewDesc in ViewDescIndex.items():
        regexp = re.compile('^\('+viewDesc+'\)$')
        ViewDescIndex[pose] = regexp
        l = ViewDesc.setdefault(regexp,[])
        l.append(pose)
    return ViewDesc

class observation:
    def __init__(self,indices):
        self.indices = indices
    def __str__(self):
        return 'vdm_'+'_'.join([str(ind) for ind in self.indices])

class ViewDescriptionObservations:
    def __init__(self,State2ViewDesc):
        self.ObservationGenerators['*']=self.getView
        # Dict of {state : viewdesc}
        self.State2ViewDesc = State2ViewDesc
        # Dict of {viewdesc : [state1,state2...]}
        self.ViewDesc2States = compileViewDescs(self.State2ViewDesc)
        # List of [viewdesc,] for consistent numbering
        self.ViewDescEnum = self.ViewDesc2States.keys()
        # Binary Compatibility Matrix (List of Lists of binary values)
        self.CompatViewDescs = findCompatibleViewDescs(self.ViewDescEnum)
    
    def generateObservations(self):
        """Generates the set of osbservations.
        """
        for state in self.State2ViewDesc:
            for obs,prob in self.generateStateObservations(state):
                yield str(obs)
    
    def generateStateObservations(self,state):
        possViewDescs = bitvec2indices(self.CompatViewDescs[
            self.ViewDescEnum.index(self.State2ViewDesc[state])])
        prob = 1.0/(2**len(possViewDescs))
        for obs in powerset(possViewDescs):
            yield (observation(obs),prob)
    
    def getView(self,state):
        """Looks up a view visible from a state.
        """
        return [o for o in self.generateStateObservations(state)]

    def checkDescription(self,description)
        return matchPatterns(self.observe(),description)
