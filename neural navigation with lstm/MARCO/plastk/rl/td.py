"""
Temporal difference agents.
$Id: td.py,v 1.13 2006/04/07 23:24:57 jp Exp $
"""

from plastk.base import *
from plastk.params import *
from plastk.rl import Agent,is_terminal
from plastk.utils import mmax,weighted_sample,inf
from plastk import rand

import Numeric
from Numeric import nonzero,putmask,array,zeros,dot,argmax
from math import exp,log

class TDAgent(Agent):
    """
    A generic temporal-difference (TD) agent with discrete actions.
    To create a new TD agent, override this class and implement the methods
    .Q(sensation,action=None) and .update_Q(sensation,action,delda,on_policy=True).

    Parameters:

    alpha  -- The learning rate, default = 0.1
    gamma  -- The discount factor, default = 1.0
    lambda_ -- The eligibility discount factor, default = 0.0.

    step_method -- The method for doing TD updates: 'sarsa' or 'q_learning'.
                     default = 'sarsa'

    action_selection -- The action selection method, default 'epsilon_greedy'.
                        To change action selection, set this to the name of the new method,
                        e.g. 'softmax'.

    initial_epsilon -- The starting epsilon for epsilon_greedy selection. (default=0.1)
    min_epsilon     -- The minimum (final) epsilon. (default = 0.0)
    epsilon_half_life -- The half-life for epsilon annealing. (default = 1)
    
    initial_temperature -- The starting temperature for softmax (Boltzman distribution)
                           selection. (default = 1.0)
    min_temperature     -- The min (final) temperature for softmax selection.
                           (default = 0.01)
    temperature_half_life -- The temperature half-life for softmax selection
                           (default = 1)

    actions -- The list of available actions - can be any Python object
               that is understood as an action by the environment
    """


    alpha =       Magnitude(default=0.1)
    gamma =       Magnitude(default=1.0)
    lambda_ =     Magnitude(default=0.0)

    step_method = Parameter(default="sarsa")

    action_selection = Parameter(default="epsilon_greedy")

    # epsilon-greedy selection parameters
    initial_epsilon =   Magnitude(default=0.1)
    min_epsilon =       Magnitude(default=0.0)
    epsilon_half_life = Number(default=1, bounds=(0,None))

    # softmax selection parameters
    initial_temperature =   Number(default=1.0, bounds=(0,None))
    min_temperature =       Number(default=0.01, bounds=(0,None))
    temperature_half_life = Number(default=1, bounds=(0,None))

    actions = Parameter(default=[])

    prune_eligibility = Magnitude(default=0.001)
    replacing_traces = Parameter(default=True)

    history_log = Parameter(default=None)
    allow_learning = Parameter(default=True)

    def __init__(self,**args):
        from plastk.utils import LogFile
        
        super(TDAgent,self).__init__(**args)
        self.nopickle.append('policy_fn')
        self.policy_fn = getattr(self,self.action_selection)
        
        self.total_steps = 0

        if isinstance(self.history_log,str):
            self._history_file = LogFile(self.history_log)
        elif isinstance(self.history_log,file) or isinstance(self.history_log,LogFile):
            self._history_file = self.history_log

    def unpickle(self):
        """
        Called automatically when the agent is unpickled.  Sets
        the action-selection function to its appropriate value.
        """
        super(TDAgent,self).unpickle()
        self.policy_fn = getattr(self,self.action_selection)


    def __call__(self,sensation,reward=None):
        """
        Do a step.  Calls the function selected in self.step_method
        and returns the action.
        """
        step_fn = getattr(self,self.step_method+'_step')

        action_index = step_fn(sensation,reward)
        if self.history_log:
            if reward is None:
                self._history_file.write('start\n')
            self._history_file.write(`sensation`+'\n')
            self._history_file.write(`reward`+'\n')
            if not is_terminal(sensation):
                self._history_file.write(`action_index`+'\n')
        return self.actions[action_index]

    def Q(self,sensation,action=None):
        """
        Return Q(s,a).  If action is None, return an array
        of Q-values for each action in self.actions
        with the given sensation.

        You must override this method to implement a TDAgent subclass.
        """
        raise NYI

    def update_Q(self,sensation,action,delta,on_policy=True):
        """
        Update Q(sensation,action) by delta.  on_policy indicates
        whether the step that produced the update was on- or
        off-policy.  Any eligibility trace updates should be done from
        within this method.

        You must override this method to implement a TDAgent subclass.
        """
        raise NYI

    def sarsa_step(self,sensation,reward=None):
        """
        Do a step using the SARSA update method.  Selects an action,
        computes the TD update and calls self.update_Q.  Returns the
        agent's next action.
        """
        if reward == None:
            return self._start_episode(sensation)

        rho = self.rho(reward)
        next_action = self.policy(sensation)

        if is_terminal(sensation):
            value = 0
        else:
            value = self.Q(sensation,next_action)

        last_value = self.Q(self.last_sensation,self.last_action)
        delta = rho + (self.gamma * value - last_value)

        self.verbose("controller step = %d, rho = %.2f"
                      % (self.total_steps,rho))
        self.verbose(("Q(t-1) = %.5f, Q(t) = %.5f, diff = %.5f,"+
                       "delta = %.5f, terminal? = %d")
                      % (last_value,value,value-last_value,
                         delta,is_terminal(sensation)))        

        if self.allow_learning:
            self.update_Q(self.last_sensation,self.last_action,delta)

        self.last_sensation = sensation
        self.last_action = next_action
        if isinstance(reward,list):
            self.total_steps += len(reward)
        else:
            self.total_steps += 1

        return next_action


    def q_learning_step(self,sensation,reward=None):
        """
        Do a step using Watkins' Q(\lambda) update method.  Selects an
        action, computes the TD update and calls
        self._q_learning_training.  Returns the agent's next action.
        """
        if reward == None:
            return self._start_episode(sensation)

        if self.allow_learning:
            self._q_learning_training(self.last_sensation,self.last_action,reward,sensation)
        
        self.last_sensation = sensation
        self.last_action = self.policy(sensation)
        if isinstance(reward,list):
            self.total_steps += len(reward)
        else:
            self.total_steps += 1
        return self.last_action

    def _q_learning_training(self,sensation,action,reward,next_sensation):
        """
        Do a single Q-lambda training step given (s,a,r,s').  Can be
        called from outside the q_learning_step method for off-policy
        training, experience replay, etc.
        """
        rho = self.rho(reward)

        last_Q = self.Q(sensation)
        last_value = last_Q[action]
        
        if is_terminal(next_sensation):
            value = 0
        else:
            value = max(self.Q(next_sensation))

        delta = rho + (self.gamma * value - last_value)
        
        self.verbose("r = %.5f, Q(t-1) = %.5f, Q(t) = %.5f, diff = %.5f, delta = %.5f, terminal? = %d"
                      % (rho,last_value,value,value-last_value,delta,is_terminal(next_sensation)))

        self.update_Q(sensation,action,delta,on_policy = (last_Q[action] == max(last_Q)))

        if delta:
            assert (self.Q(sensation,action) - last_value)/delta < 1.0
    
    def _start_episode(self,sensation):
        """
        Start a new episode.  Called from self.__call__ when the reward is None.
        """
        self.last_sensation = sensation
        self.last_action = self.policy(sensation)
        return self.last_action


    def policy(self,sensation):
        """
        Given a sensation, return an action.  Uses
        self.action_selection to get a distribution over the agent's
        actions.  Uses self.applicable_actions to prevent selecting
        inapplicable actions.

        Returns 0 if is_terminal(sensation).
        """
        if not is_terminal(sensation):
            actions = self.applicable_actions(sensation)
            return actions[weighted_sample(self.policy_fn(sensation,actions))]
        else:
            # In the terminal state, the action is irrelevant
            return 0
        
    def epsilon_greedy(self,sensation,applicable_actions):
        """
        Given self.epsilon() and self.Q(), return a distribution over
        applicable_actions as an array where each element contains the
        a probability mass for the corresponding action.  I.e.  The
        action with the highest Q gets p = self.epsilon() and the
        others get the remainder of the mass, uniformly distributed.
        """
        Q = array([self.Q(sensation,action) for action in applicable_actions])

        # simple epsilon-greedy policy
        # get a vector with a 1 where each max element is, zero elsewhere
        mask = (Q == mmax(Q))

        num_maxes = len(nonzero(mask))
        num_others = len(mask) - num_maxes

        if num_others == 0: return mask
        
        e0 = self.epsilon()/num_maxes
        e1 = self.epsilon()/num_others

        result = zeros(len(mask))+0.0
        putmask(result,mask,1-e0)
        putmask(result,mask==0,e1)
        return result

    def softmax(self,sensation,applicable_actions):
        """
        Given self.temperature() and self.Q(), return a Bolzman
        distribution over applicable_actions as an array where each
        element contains the a probability mass for the corresponding
        action.
        """
        temp = self.temperature()
        self.verbose("softmax, temperature = %.3f" % temp)
        Q = array([self.Q(sensation,action) for action in applicable_actions])
        return softmax(Q,temp)

    def normalized_softmax(self,sensation,applicable_actions):
        """
        Like softmax, except that the Q values are scaled into the
        range [0,1].  May make setting the initial temperature easier than with softmax.
        """
        temp = self.temperature()
        self.verbose("softmax, temperature = %.3f" % temp)
        Q = array([self.Q(sensation,action) for action in applicable_actions])
        return softmax(normalize_minmax(Q),temp)

    def temperature(self):
        """
        Using initial_temperature, min_temperature, and temperature_half_life,
        compute the temperature after self.total_steps, steps.
        """
        Ti = self.initial_temperature
        Tm = self.min_temperature
        decay = log(2)/self.temperature_half_life
        return Tm + (Ti - Tm) * exp( -decay * self.total_steps )

    def epsilon(self):
        """
        Using initial_epsilon, min_epsilon, and epsilon_half_life,
        compute epsilon after self.total_steps, steps.
        """
        Ei = self.initial_epsilon
        Em = self.min_epsilon
        decay = log(2)/self.epsilon_half_life
        return Em + (Ei - Em) * exp( -decay * self.total_steps )
    
    def rho(self,reward):
        """
        Compute the reward since the last step.
        
        IF the reward is a scalar, it is returned unchanged.

        If reward is a list, it is assumed to be a list of rewards
        accrued at a constant time step, and the discounted sum is
        returned.
        """
        if isinstance(reward,list):
            result = 0
            for r in reward:
                result = self.gamma*result + r
        else:
            result = reward
        return result

    def applicable(self,action,sensation):
        """
        If the given action has a method called 'applicable' return
        the value of action.applicable(sensation), otherwise return True.
        """
        if 'applicable' in dir(action):
            return action.applicable(sensation)
        else:
            return True

    def applicable_actions(self,sensation):
        """
        Return a list of the actions that are applicable to the given
        sensation.
        """
        return [a for a in range(len(self.actions))
                if self.applicable(self.actions[a],sensation)]




class LinearTDAgent(TDAgent):
    """
    A TD agent that takes a sensation as a 1D Numeric vector of
    features and computes Q as a linear function of that sensation,
    using simple gradient descent.  The function is stored in the
    weight matrix self.w, such that Q(s) can be computed as w*s.
    Assumes a discrete set of actions.  Uses replacing eligibility
    traces.

    Parameters:

    num_features = The number of input features (default = 1)
    initial_w = A scalar value with which to initialize the weight
                matrix.
    """
    num_features = PositiveInt(default=1)
    initial_w =   Number(default=0.0)

    def __init__(self,**params):
        super(LinearTDAgent,self).__init__(**params)
        self.reset_w()
        self.reset_e()

    def _start_episode(self,sensation):
        self.reset_e()
        return super(LinearTDAgent,self)._start_episode(sensation)
        
    def reset_w(self):
        """
        Reset the weight matrix to self.initial_w.
        """
        self.w = zeros((len(self.actions),self.num_features),'f') + self.initial_w
        
    def reset_e(self):
        """
        Reset the eligibility traces for self.w to all zeros.
        """
        self.e = zeros((len(self.actions),self.num_features),'f') + 0.0

    def Q(self,state,action=None):
        """
        Compute Q(s,a) from W*s.
        """
        if action is None:
            return dot(self.w, state)
        else:
            return dot(self.w[action],state)

    def update_Q(self,sensation,action,delta,on_policy=True):
        """
        Do a linear update of the weights.  
        """
        if self.lambda_ and on_policy:
            self.e *= self.lambda_
            if self.prune_eligibility > 0.0:
                self.e *= (self.e > self.prune_eligibility)
        else:
            self.e *= 0.0

        self.e[action] += sensation
        
        if self.replacing_traces:
            putmask(self.e,self.e > 1,1)
            
        self.w += self.e * (self.alpha/(sum(sensation))) * delta


class TabularTDAgent(TDAgent):
    """
    A TDAgent for environments with discrete states and actions.
    Sensations/states can be any hashable Python object, and the
    universe of sensations need not be specified in advance. The agent
    stores and updates a separate Q estimate for every (s,a) pair.

    Parameters:

    initial_q -- The initial Q estimate for each (s,a) pair. (default = 0.0)
    
    """

    initial_q = Number(default=0.0)
    
    def __init__(self,**params):
        super(TabularTDAgent,self).__init__(**params)
        self.reset_q()
        self.reset_e()

    def _start_episode(self,sensation):
        self.reset_e()
        return super(TabularTDAgent,self)._start_episode(sensation)
        
    def reset_q(self):
        self.q_table = {}
        
    def reset_e(self):
        self.e = {}

    def Q(self,s,a=None):
        if a is None:
            result = [self.Q(s,a) for a in range(len(self.actions))]
        else:
            result =  self.q_table.get((s,a),self.initial_q)
        self.debug('Q(',s,',',a,') = ',result)
        return result

    def update_Q(self,s,a,delta,on_policy=True):
        if not on_policy:
            self.reset_e()

        if (s,a) not in self.q_table:
            self.q_table[(s,a)] = self.initial_q
            

        if self.lambda_:
            to_be_deleted = []
            for x in self.e:
                self.e[x] *= self.lambda_
                if self.e[x] < self.prune_eligibility:
                    to_be_deleted.append(x)
            for x in to_be_deleted:
                del self.e[x]

        if self.replacing_traces:
            self.e[(s,a)] = 1
        else:
            self.e[(s,a)] += 1

        for x,e in self.e.iteritems():
            self.q_table[x] += self.alpha * e * delta

class TabularMemoryTDAgent(TabularTDAgent):

    """
    A Tabular TD agent that keeps a memory of the last N steps of
    sensations and actions, and does Q learning/sarsa using the
    contents of memory as its state.
    """
    memory_steps = NonNegativeInt(default=1)
    
    def __init__(self,**params):
        super(TabularMemoryTDAgent,self).__init__(**params)
        self._memory =   []

    def __call__(self,sensation,reward=None):
        if reward is None:
            self._memory = [sensation]
        else:
            self._memory.append(sensation)

        if is_terminal(sensation):
            return super(TabularMemoryTDAgent,self).__call__(sensation,reward)
        else:
            action = super(TabularMemoryTDAgent,self).__call__(tuple(self._memory),reward)
            assert self.actions[self.last_action] == action
            self._memory.append(self.last_action)

            if len(self._memory) > (2*self.memory_steps + 1):
                del self._memory[0:2]

            return action
            
class LinearTabularTDAgent(LinearTDAgent):
    """
    Subclass of LinearTDAgent for 'tabular' environments.  Assumes the
    state/sensation is a single integer.  Use the num_features
    parameter inherited from LinearTDAgent to specify the total number
    of states/sensations.
    """
    def __call__(self,sensation,reward=None):
        if not is_terminal(sensation):
            assert(type(sensation) == int)
            s = zeros(self.num_features)
            s[sensation] = 1.0
        else:
            s = sensation
        return super(LinearTabularTDAgent,self).__call__(s,reward)


class LinearListAgent(LinearTDAgent):
    """
    A subclss of LinearTDAgent where the sensation is assumed to be a
    list of discrete features.  For sparse feature representations,
    this is more compact than the feature-vector representation of
    LinearTDAgent. 
    """
    def __call__(self,sensation,reward=None):
        if is_terminal(sensation):
            new_sensation = sensation
        else:
            new_sensation = zeros(self.num_features,'f')
            for f in sensation:
                new_sensation[f] = 1
        return super(LinearListAgent,self).__call__(new_sensation,reward)


class UniformTiledAgent(LinearListAgent):
    """
    A LinearTDAgent subclass for continuous state spaces that
    automatically tiles the input space.  For high-dimensional inputs,
    the input can be separated into a several uniformly distributed
    'receptive fields' (rfs) that may overlap, and each rf is tiled
    separately.

    Parameters:
      num_rfs -- The number of receptive fields to use (default=1)
      rf_width -- The width of the receptive fields
                  (default=[D/num_rfs] where D = input dimensionality)
      num_tilings -- The number of tilings to use for each rf.
      tile_width  -- The width of each tile.
      num_features -- The total combined memory size for all rfs.

    Each separate rf is assumed to use the same tiling parameters.

    Examples:

    D = 9 , num_rfs = 3, rf_width = <default> will give the following

               |-rf0-|     |-rf2-| 
    Features: [ 0 1 2 3 4 5 6 7 8 ]
                     |-rf1-|     

    D = 10 , num_rfs = 3, rf_width = 4 will give the following

               |--rf0--|   |--rf2--| 
    Features: [ 0 1 2 3 4 5 6 7 8 9 ]
                     |--rf1--|     


    RF placements are determined with function place_rfs.
    
    """
    
    num_rfs =     PositiveInt(default=1)
    rf_width =    Parameter(None)
    num_tilings = PositiveInt(default=1)
    tile_width =  Number(default=1)

    def __init__(self,**args):
        super(UniformTiledAgent,self).__init__(**args)
        if not self.rf_width:
            self.rf_width = self.num_features/self.num_rfs
    
    def __call__(self,sensation,reward=None):
        if not is_terminal(sensation):
            sensation = tile_uniform_rfs(array(sensation)/self.tile_width,
                                         self.num_rfs,
                                         self.rf_width,
                                         self.num_tilings,
                                         self.num_features/self.num_rfs)
        return super(UniformTiledAgent,self).__call__(sensation,reward)


##################################
# utility functions

def softmax(ar,temp):
    """
    Given an array and a temperature, return the Boltzman distribution
    over that array.

    For an array X, and temp T returns a new array containing:

    exp(Xi/T)/sum_j(exp(Xj/T) for all Xi.

    If temp == 0 or any value in the array is inf, the function
    returns the limit value as T -> 0.
    """
    if temp == 0 or inf in ar:
        v = (ar == mmax(ar))
        return v/float(sum(v))
    else:
        numer = Numeric.exp(ar/float(temp))
        denom = Numeric.sum(numer)    
        return numer/denom

def normalize_minmax(ar):
    """
    Return the array ar scaled so that the min value is 0 and the max
    value is 1.
    """
    x = ar - min(ar)
    mmax = max(x)
    if mmax == 0:
        return x
    return x/mmax



def ranseq(x):
    """
    Generator that gives a random-length sequence of the integers
    ascending from 0.  The length is selected from the uniform
    distribution over the range [0,x).
    """
    for i in range(int(rand.uniform(0,x))):
        yield i


def tile_rfs(vec,specs):
    """
    Tile vec into several rfs, rfs are specified as a list of tuples:
    (slice_start,slice_end,num_tilings,memory_size).
    """
    from plastk.rl.tiles import getTiles
    result = []
    offset = 0
    for i,(start,end,num_tilings,memory_size) in enumerate(specs):
        F = getTiles(num_tilings,memory_size,vec[start:end])
        result += [x + offset for x in F]
        offset += memory_size
    return result

def tile_uniform_rfs(vec,num_rfs,rf_width,num_tilings,memory_size):
    """
    Tile vector vec into several approximately uniformly spaced
    receptive fields of equal width.

    num_rfs = the number of rfs
    rf_width = the width of each rf
    num_tilings = the number of tilings for each rf
    memory_size = the memory size for each rf

    Function uses place_rfs to determine rf positions.
    """
    specs = [(start,end,num_tilings,memory_size)
             for start,end in place_rfs(len(vec),num_rfs,rf_width)]
    return tile_rfs(vec,specs)


def place_rfs(length,count,width):
    """
    place-rfs - returns a list of receptive field index lists
    for use as rf-indices in an rf-array
        
    params
    length = the length of the input vector
    count  = the number of rfs
    width  = the width of each rf
    
    The rfs will be placed such that the first begins at 0 and the
    last ends at length - 1.  The rest will be (approximately) evenly
    spaced in between.  i.e. in 0..(length - width) step (length -
    width)/(count - 1)
    
    Note that they're assumed to overlap!
    """
    if count==1:
        return [(0,length)]

    end_pos = length-width
    step = int(round(end_pos / (count - 1.0)))
    pos = 0
    result = []
    for i in range(count-1):
        result.append((pos,pos+width))
        pos += step
    result.append((end_pos,end_pos+width))
    return result


