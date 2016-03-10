import re, operator
import Base, ViewDescription

class Instruction(Base.Action):
    def __init__(self,id,action,rewards,n):
        self.id = id
        self.action = action
        self.Instructed,self.NotInstructed = rewards
        self.NumInstructions = n
    def __hash__(self):
        return str(self)
    def __str__(self):
        return __name__+str(self.id)
    
    def act(self,state):
         if state == self.id:
             if state < self.NumInstructions-1:
                 return [(state, self.Instructed, 0.5),(state+1, self.Instructed, 0.5)]
             else:
                 return [(state, self.Instructed, 1.0),]
         else:
             return [(self.NumInstructions, self.NotInstructed, 1.0),]
    def rewards(self,States):
        for state in States:
            for state,reward,prob in self.act(state):
                yield (States[state],reward,prob)

class POMDP(ViewDescription.ViewDescriptionObservations,Base.POMDP):
    def __init__(self,ViewDescs,Actions,name):
        Base.POMDP.__init__(self,'Route_InstQueue_'+name)
        ViewDescription.ViewDescriptionObservations.__init__(self,ViewDescs)
        self.Actions = Actions
        self.NumInstructions=len(Actions)
    
    def generateStates(self):
        for i in xrange(self.NumInstructions):
            self.States[i] = 'Inst_%(i)i' % locals()
            yield self.States[i]
        self.States[self.NumInstructions] = 'Offroute' % locals()
        yield self.States[self.NumInstructions]
    
    def generateStart(self):
        yield '1.0'
        for inst in self.States.values()[1:]:
            yield '0.0'
