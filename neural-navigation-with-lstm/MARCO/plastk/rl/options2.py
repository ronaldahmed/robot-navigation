"""
An interface for Precup & Sutton's Options formalism for hierarchical RL

This is an alterative and improved version over options.py.  It does
not require a special HierarchicalAgent class to handle the option
stack. 

$Id: options2.py,v 1.3 2004/12/15 05:47:51 jp Exp $
"""

from rl import *
from utils import Stack

OPTION_TERMINATED = ":OPTION-TERMINATED"

class Option(Agent):

    def __init__(self,**params):
        super(Option,self).__init__(**params)
        self.__suboption_executing = None

    def applicable(self,sensation):
        return True
    
    def terminal(self,sensation):
        return False

    def reward(self,sensation,super_reward):
        return 0

    def start_episode(self,sensation):
        self.suboption_executing = None
        self.suboption_rewards = []
        action = self._start_episode(sensation)
        if isinstance(action,Option):
            self.__suboption_executing =

    def step(self,sensation,reward):       
        # store the reward 
        self.stored_rewards.append(reward)

        # if there's a suboption executing,
        if self.suboption_executing:
            # and call the suboption
            action = self.suboption_executing.step(sensation,reward)
            #if it didn't terminate
            if action != OPTION_TERMINATED:
                # then return it's result
                return action
        # otherwise do a step
        action = self._step(sensation,self.stored_rewards)
        # clear the stored rewards
        self.stored_rewards = []
        # if the action is a suboption,
        if isinstance(action,Option):
            # push it
            self.suboption_executing = action
            # and execute it
            action = self.suboption_executing.step(sensation,reward)
        return action
                
                                     

class PrimitiveOption(Option):
    value = Parameter(default=None)
    def start_episode(self,sensation):
        return self.value
    def _step(self,sensation,reward):
        return OPTION_TERMINATED

class GeneratorOption(Option):
    fn = Parameter(default=None)
    args = Parameter(default=[])

    def _start_episode(self,sensation):
        self.gen = self.fn(*self.args)
        try:
            return self.gen.next()
        except StopIteration:
            return OPTION_TERMINATED
    def _step(self,sensation,reward):        
        try:
            return self.gen.next()
        except StopIteration:
            return OPTION_TERMINATED
            
    
class OptionAdapter(Option):
    def __init__(self,agent,**args):
        super(OptionAdapter,self).__init__(**args)
        self.agent=agent
    def init(self):
        self.agent.init()
    def _start_episode(self,sensation):
        result = self.agent.start_episode(sensation)
        if sensation == TERMINAL_STATE:
            result = OPTION_TERMINATED
        return result
    def _step(self,sensation,reward):
        result = self.agent.step(sensation,reward)
        if sensation == TERMINAL_STATE:
            result = OPTION_TERMINATED
        return result
