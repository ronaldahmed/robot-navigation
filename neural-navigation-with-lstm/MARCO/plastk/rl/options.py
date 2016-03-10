"""
An interface for Precup & Sutton's Options formalism for hierarchical RL

$Id: options.py,v 1.14 2005/08/20 16:54:32 jp Exp $
"""

import plastk.rl as rl
from plastk.utils import Stack
from plastk.params import *

OPTION_TERMINATED = "option terminated"

class Option(rl.Agent):
    def applicable(self,sensation):
        return True
    def __call__(self,sensation,reward=None):
        raise NYI
    def reward(self,sensation,super_reward):
        return super_reward


class GeneratorOption(Option):
    fn =      Parameter(default=None)
    args =    Parameter(default=[])
    appl_fn = Parameter(default=None)

    def __init__(self,**params):
        super(GeneratorOption,self).__init__(**params)
        self.terminal = False
        self.gen = None
        self.nopickle += ['gen']
        
    def __call__(self,sensation,reward=None):
        if reward == None:
            self.setup_gen()
        if not rl.is_terminal(sensation):
            try:
                return self.gen.next()
            except StopIteration:
                return OPTION_TERMINATED

    def applicable(self,sensation):
        if self.appl_fn:
            return self.appl_fn(sensation)
        else:
            return True

    def setup_gen(self):
        self.gen = self.fn(*self.args)        
    def unpickle(self):
        self.warning("Unpickling generator option does not restore generator state.")
        self.setup_gen()
        super(GeneratorOption,self).unpickle()


class GeneratorMethodOption(GeneratorOption):
    obj =         Parameter(default=None)
    method =      Parameter(default=None)
    appl_method = Parameter(default=None)

    def __init__(self,**params):
        super(GeneratorMethodOption,self).__init__(**params)
        self.nopickle += ['fn','appl_fn']
    def __call__(self,sensation,reward=None):
        if reward == None:
            self.setup_fns()
        return super(GeneratorMethodOption,self).__call__(sensation,reward)
    def setup_fns(self):
        self.fn = getattr(self.obj,self.method)
        if self.appl_method:
            self.appl_fn = getattr(self.obj,self.appl_method)
    def unpickle(self):
        self.setup_fns()
        super(GeneratorMethodOption,self).unpickle()

class Macro(Option):
    sequence = Parameter(default = [])
    def __init__(self,**args):
        super(Macro,self).__init__(**args)
        self.pos = 0
    def __call__(self,sensation,reward=None):
        if reward == None:
            assert self.applicable(sensation)
            self.pos=0

        if self.pos < len(self.sequence):
            curr_step = self.sequence[self.pos]
            if (not isinstance(curr_step,Option)) or curr_step.applicable(sensation):
                self.pos += 1
                return curr_step
        return OPTION_TERMINATED

    def applicable(self,sensation):
        assert self.sequence
        if 'applicable' in dir(self.sequence[0]):
            return self.sequence[0].applicable(sensation)
        else:
            return True

class Primitive(Macro):
    action = Parameter(default=None)
    def __init__(self,**args):
        super(Primitive,self).__init__(**args)
        self.sequence = (self.action,)
    
    
class OptionAdapter(Option):
    agent = Parameter(default = None)
    def __init__(self,**args):
        super(OptionAdapter,self).__init__(**args)
    def __call__(self,sensation,reward=None):
        result = self.agent(sensation,reward)
        if rl.is_terminal(sensation):
            result = OPTION_TERMINATED
        return result


################################################################
class HierarchicalAgent(rl.Agent):
    """
    An agent that can use Options for hierarchical behavior.
    Internally it keeps a stack of running options, and calls
    the top option in the stack, pushing or popping
    from the stack as needed.
    """

    root_option = Parameter(default=None)
    
    def __init__(self,**args):
        super(HierarchicalAgent,self).__init__(**args)
        self.stack = Stack([])
    def __call__(self,sensation,reward=None):
        if reward == None:
            self.stack = Stack([])
            self.push_option(self.root_option)
    
        self.last_sensation = sensation
        self.last_reward = reward

        if rl.is_terminal(sensation):
            # unwind the stack giving everyone the current reward
            # TODO: when options get their own separate rewards, this may change
            while not self.stack.empty():
                option,reward_list = self.stack.pop()
                option(sensation,reward_list+[option.reward(sensation,reward)])
            return None
        else:
            for option,rewards in self.stack[:-1]:
                rewards.append(option.reward(sensation,reward))
            option,rewards = self.stack.top()
            return  self.haction(option(sensation,option.reward(sensation,reward)))

    def haction(self,action):
        self.debug("Doing haction: "+`action`)
        sensation = self.last_sensation
        reward = self.last_reward
        if  isinstance(action,Option):
            # The action is an option, so push it on the
            # stack and start it running.
            self.debug("pushing "+`action`)
            self.push_option(action)
            result = self.haction(action(sensation))
        elif action == OPTION_TERMINATED:
            # The top option on the stack terminated, so pop it
            # and do a step on the option below.
            self.debug("popping "+`self.stack.top()`)
            self.stack.pop()
            if len(self.stack) == 0:
                raise "Error, hierarchical stack is empty."
            option,rewards = self.stack.pop()
            # reset the list of rewards
            self.push_option(option)
            # assert reward != None
            result = self.haction(option(sensation,rewards))
        else:
            # It's not an option or OPTION_TERMINATED
            # so it's a primitive. Just return it.
            self.debug("Doing primitive: "+`action`)
            result =  action  

        return result

    def push_option(self,option):
        # The stack is a list of pairs, an option, and a list of rewards
        # that occurred since the option's last step.
        self.stack.append( (option, []) )
        
