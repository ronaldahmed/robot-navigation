"""
Reinforcement Learning Interface and Classes

The top-level of the rl package contains an implementation of the
Sutton and Santamaria's Reinforcement Learning Interface (RLI), found
at rlai.net.

The modules within the package contain implementations of various
RL-related things, like temporal-difference learning (rl.td),
hierarchical agents (rl.options), tile-coding (rl.tiles), and some
example environments (rl.gridworld and rl.pendulum).

The class rl.RLI is a very simple RL Interface.  For an interface with
lots of bells and whistles, including a GUI and the ability to
log arbitrary run data into files, see rl.loggingrli.LoggingRLI.

$Id: __init__.py,v 1.17 2005/06/14 21:14:58 jp Exp $
"""

from plastk.base import BaseObject,NYI
from plastk.params import Number,Integer,Parameter,PositiveInt,NonNegativeInt,Magnitude
from numpy import zeros,argmax
from math import exp,log
import plastk.utils as utils
from plastk import rand

TERMINAL_STATE = "terminal"


class Agent(BaseObject):
    """
    The generic rl agent interface.  Subclass this and override
    __call__ to implement an Agent.
    """
    def __init__(self,**args):
        super(Agent,self).__init__(**args)
        self.sim = None
    def __call__(self,sensation,reward=None):
        """
        Do a single step and return an action. If reward is None, it
        is the first step of a new episode.
        """
        raise NYI


class Environment(BaseObject):
    """
    The generic rl environment interface, Subclass this and override
    __call__ to implement an Environment.
    """
    def __init__(self,**args):
        super(Environment,self).__init__(**args)
        self.sim = None
    def __call__(self,action=None):
        """
        Do a single environment step and return a (sensation,reward)
        pair. If action is None then it is the first step of a new episode.
        """
        raise NYI

class RLI(BaseObject):
    """
    The simple Reinforcement Learning Interface.  RLI instances manage
    the interaction between an Agent and an Environment.
    """
    def init(self,agent,env,init_agent=True,init_env=True):
        """
        Initialize the RLI with an agent and an environment.  If
        init_agent and init_env are true, then call .init() on the
        agent or env, respectively, if possible.  
        """
        self.agent = agent
        self.env = env
        env.sim = self
        agent.sim = self
        if init_env and 'init' in dir(env): env.init()
        if init_agent and 'init' in dir(agent): agent.init()
        self.last_sensation = TERMINAL_STATE

    def start_episode(self):
        """
        Start a new episode with the current agent and environment.
        """
        self.last_sensation = self.env()
        self.next_action = self.agent(self.last_sensation)


    def steps(self,num_steps):
        """
        Execute the given number of steps, starting new episodes as
        necessary.
        """
        if self.last_sensation == TERMINAL_STATE:
            self.start_episode()
        for step in range(num_steps):
            next_sensation,reward = self.env(self.next_action)
            self.collect_data(self.last_sensation, self.next_action, reward, next_sensation)
            self.next_action = self.agent(next_sensation,reward)
            self.last_sensation = next_sensation
            if self.last_sensation == TERMINAL_STATE:
                self.start_episode()
    
        
    def episodes(self, num_episodes, num_steps_per_episode):
        """
        Run the given number of episodes, of at most the given number
        of steps.
        """
        for ep in range(num_episodes):
            self.start_episode()
            for step in range(num_steps_per_episode):
                next_sensation,reward = self.env(self.next_action)
                self.collect_data(self.last_sensation, self.next_action, reward, next_sensation)
                self.next_action = self.agent(next_sensation,reward)
                self.last_sensation = next_sensation
                if self.last_sensation == TERMINAL_STATE:
                    break


    def collect_data(self,sensation,action,reward,next_sensation):
        """
        Collect data.  Override this method to do any data collection needed.
        """
        pass



def is_terminal(sensation):
    """
    Check whether a sensation is terminal.  This is more reliable than
    (sensation == 'terminal') if the sensation is a sequence, rather
    than a scalar.
    """
    return type(sensation) == type(TERMINAL_STATE) and sensation == TERMINAL_STATE



###############
from plastk.rl.loggingrli import LoggingRLI

