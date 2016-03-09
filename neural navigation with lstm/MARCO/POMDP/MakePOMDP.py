Actions = []
States = {}
Observations = {}
ObservationGenerators = {}

class Action:
    def __init__(self):
        Actions.append(self)
    def __hash__(cls):
        return cls.__name__
    __hash__ = classmethod(__hash__)
    def __str__(cls):
        return cls.__name__
    __str__ = classmethod(__str__)
    def act(self,state):
        """Get resulting state of the action in a state.

        Takes a State.
        Returns a list of (State,Prob):
        State is the next state after taking the action
        and Prob is a float percentage likelihood of that next state.
        """
        raise NotImplementedError
    def reward(self,state):
        """Get resulting reward of the action in a state.

        Takes a State.
        Returns a list of (Reward,Prob):
        Reward is an integer representation of the immediate reward
        and Prob is a float percentage likelihood of that reward.
        """
        raise NotImplementedError
    def rewards(self):
        """Generate the set of all reward producing conditions for this action.
        
        Returns a list of tuples of (State,Reward,Prob):
        State is either a wildcard or a State,
        Reward is an integer representation of the immediate reward
        and Prob is a float percentage likelihood of that reward.
        """
        for state in States:
            yield self.reward(state)

class ConsistentCostAction(Action):
    def reward(cls,State):
        return [(State,cls.ActionCosts[cls.__name__],1.0),]
    reward = classmethod(reward)
    def rewards(cls):
        return [('*',cls.ActionCosts[cls.__name__],1.0),]
    rewards = classmethod(rewards)

## Transition Function ##
def generateTransitionFn():
    """generates transition lines for POMDP.

    Lines have the form:
    print 'T:', Action, ':', StartState, ':', EndState, Probability
    """
    for startState,startStateStr in States.items():
        for action in Actions:
            for endState,prob in action.act(startState):
                yield 'T: %s : %s : %s %1.3f\n' % (action,startStateStr,States[endState],prob)

## Observation Function #
def generateObservationFn():
    """generates observation lines for POMDP.

    Lines have the form:
    print 'O :', Action, ':', State, ':', Observation, Probability
    """
    for state,stateStr in States.items():
        for action,obsGenerator in ObservationGenerators.items():
            for observ,prob in obsGenerator(state):
                yield 'O : %s : %s : %s %1.3f\n' % (action,stateStr,Observations[observ],prob)

## Reward Function ##
def generateRewardFn(): 
    """generates reward lines for POMDP.

    Lines have the form:
    R: <action> : <start-state> : <end-state> : <observation> <reward>%f
    """
    for action in Actions:
        for state,reward,prob in action.rewards():
            yield 'R : %s : %s : * : * : %i \n' % (action,state,reward)

def writePOMDPfile(FileName,POMDP_sets,d):
    locals().update(d)
    POMDPFILE=open(FileName,'w')
    for name in ['discount','values','actions','observations','states','start',]:
        generator = POMDP_sets[name]
        POMDPFILE.write('%(name)s: ' % locals())
        POMDPFILE.write(' '.join([str(val) for val in generator()]))
        POMDPFILE.write('\n')
    for transition in generateTransitionFn(): POMDPFILE.write(transition)
    for observation in generateObservationFn(): POMDPFILE.write(observation)
    for reward in generateRewardFn(): POMDPFILE.write(reward)
    POMDPFILE.close()

