import random
import sys,os
import ipdb
marco_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(marco_dir)

from Utility import logger

def chooseFromDist(dist):
    if len(dist) == 1: return dist[0]
    r = random.random()
    cumProb = 0.0
    for option in dist:
        cumProb += option[-1]
        if cumProb > r:
            return option
    raise ArithmeticError('Reached end of sequence without accumulating probability: ', r, cumProb)

class Action:
    def __hash__(cls):
        return cls.__name__
    __hash__ = classmethod(__hash__)
    
    def __str__(cls):
        return cls.__name__
    __str__ = classmethod(__str__)
    
    def act(self,state):
        """Get resulting state of the action in a state.
        
        Takes a State.
        Returns a list of (State,Reward,Prob):
        State is the next state after taking the action
        Reward is an integer representation of the immediate reward for that transition
        and Prob is a float percentage likelihood of that reward.
        """
        raise NotImplementedError
    
    def rewards(self,States):
        """Generate the set of all reward producing conditions for this action.
        
        Returns a list of tuples of (State,Reward,Prob):
        State is either a wildcard or a State,
        Reward is an integer representation of the immediate reward
        and Prob is a float percentage likelihood of that reward.
        """
        for state,state_str in States.items():
            for state,reward,prob in self.act(state):
                yield (state_str,reward,prob)


class ConsistentCostAction(Action):
    def rewards(self,States):
        return [('*',self.cost,1.0),]

class POMDP:
    def __init__(self,name):
        self.name = name
        self.trueState = None
        self.observed = None
        self.Actions = {}
        self.States = {}
        self.ObservationGenerators = {}
        self.Discounts = ['0.95',]
        self.Values = ['reward']
        self.generators = {
            'discount' : self.generateDiscounts,
            'values' : self.generateValues,
            'states' : self.generateStates,
            'start' : self.generateStart,
            }
    
    def reset(self):
        self.Actions.clear()
        self.States.clear()
        self.ObservationGenerators.clear()
    
    def set(self,state):
        self.trueState = state
    
    def generateTransitionFn(self):
        """Generates transition lines for POMDP.
        
        Lines have the form:
        'T:', Action, ':', StartState, ':', EndState, Probability
        """
        for startState,startStateStr in self.States.items():
            for action in self.Actions.values():
                for endState,reward,prob in action.act(startState):
                    yield 'T: %s : %s : %s %1.3f\n' % (action,startStateStr,self.States[endState],prob)
    
    def generateObservationFn(self):
        """Generates observation lines for POMDP.
        
        Lines have the form:
        'O :', Action, ':', State, ':', Observation, Probability
        """
        for state,stateStr in self.States.items():
            for action,obsGenerator in self.ObservationGenerators.items():
                for observ,prob in obsGenerator(state):
                    yield 'O : %s : %s : %s %s\n' % (action,stateStr,observ,prob)
    
    def generateRewardFn(self): 
        """Generates reward lines for POMDP.
        
        Lines have the form:
        R: <action> : <start-state> : <end-state> : <observation> <reward>%f
        """
        for action in self.Actions.values():
            for state,reward,prob in action.rewards(self.States):
                yield 'R : %s : %s : * : * %i \n' % (action,state,reward)
    
    def generateDiscounts(self):
        for d in self.Discounts: yield d
    def generateValues(self):
        for v in self.Values: yield v
    
    def generateStates(self): raise NotImplementedError
    def generateStart(self): raise NotImplementedError
    
    def write(self,file=sys.stdout):
        file.write('actions: '+' '.join([str(act) for act in self.Actions.values()])+'\n')
        ObsSet = {}
        for obs in self.generateObservations(): ObsSet[obs]=1
        file.write('observations: '+' '.join([o.code() for o in ObsSet])+'\n')
        for name,generator in self.generators.items():
            file.write(name+': '+' '.join([str(val) for val in generator()])+'\n')
        for transition in self.generateTransitionFn(): file.write(transition)
        for observation in self.generateObservationFn(): file.write(observation)
        for reward in self.generateRewardFn(): file.write(reward)
    
    def writefile(self):
        file=open(self.name+'.pomdp','w')
        self.write(file)
        file.close()
    def __str__(self):
        self.write()
        return ''
    
    def act(self,action):
        """Change the state of the world by taking action.
        
        Caches and returns the true hidden state of the world
        by randomly picking a stochastic result of the action.
        """
        self.trueState,reward,prob = chooseFromDist(self.Actions[action].act(self.trueState))
        return self.trueState, reward
    
    def observe(self):
        """Observe the world.
        
        Caches and returns an observation of the world
        by randomly picking a stochastic observation, given the true state.
        """
        self.observed,prob = chooseFromDist(self.getView(self.trueState))
        return self.observed
    
    def perform(self,action):
        """Perform action and return the observation.
        
        Public interface, keeps true world state hidden (internal).
        """
        if hasattr(self,'trace'):
            self.trace('State', self.trueState)
            self.trace('Action', action)
        state,reward = self.act(action)
        self.observe()
        return reward,self.observed
