"""
Thread-safe queues for communicating with agents/environments in other threads.

actionQ and observationQ are each a Queue.Queue
to allow threadsafe communication between the agent and environment.

$Id: queue.py,v 1.8 2006/03/24 23:52:33 adastra Exp $
"""

import plastk.rl as rl
import time

class QueueAgentProxy(rl.Agent):
    """
    Act like an rl agent, but forward everything over threading queues.
    """
    def __init__(self,actionQ,observationQ,**args):
        super(QueueAgentProxy,self).__init__(**args)
        self.actionQueue = actionQ
        self.observationQueue = observationQ
    def __call__(self,sensation,reward=None):
        self.observationQueue.put((sensation,reward))
        return self.actionQueue.get()

class QueueEnvironmentProxy(rl.Environment):
    """
    Act like an rl environment, but forward everything over threading queues.
    """
    def __init__(self,actionQ,observationQ,**args):
        super(QueueEnvironmentProxy,self).__init__(**args)
        self.actionQueue = actionQ
        self.observationQueue = observationQ
    def __call__(self,action=None):
        self.actionQueue.put((action,time.localtime()))
        return self.observationQueue.get()

class QueuePOMDPProxy(object):
    """
    Act like a POMDP from the robot's POV, but forward everything over threading queues.
    """
    def __init__(self,actionQ,observationQ,str2meaning,invert_reward=True):
        self.actionQueue = actionQ
        self.observationQueue = observationQ
        self.observe_wait = False
        self.str2meaning = str2meaning
        self.state = None
        self.pomdp_state = None
        self.invert_reward = invert_reward
    
    def setPOMDP(self,pomdp):
        self.name = pomdp.env + str(pomdp.PosSet)
        self.NumPoses = pomdp.NumPoses
        self.NumPlaces = pomdp.NumPlaces
        self.Positions = pomdp.Positions

    def setRoute(self,Start,Dest):
        print 'QueuePOMDPProxy.setRoute',(Start,Dest),time.localtime()
        while not self.actionQueue.empty(): self.actionQueue.get()
        while not self.observationQueue.empty():
            print 'QueuePOMDPProxy.setRoute: flushing',self.observationQueue.get()
        self.observed = self.reward = None
        self.actionQueue.put((('Route',(Start,Dest)),time.localtime()))
    
    def set(self,state):
        print 'QueuePOMDPProxy.set',(state),time.localtime()
        while not self.observationQueue.empty():
            print 'QueuePOMDPProxy.set: flushing',self.observationQueue.get()
        self.observed = self.reward = None
        self.state = state
        self.pomdp_state = None
        self.observe_wait = 'State'
        self.actionQueue.put((('State',state),time.localtime()))

    def is_state(self,observed):
        return (type(observed) == tuple and len(observed) == 2 and
                type(observed[0]) == int and type(observed[0]) == int)

    def observe(self):
        if self.actionQueue.empty() and not self.observe_wait:
            print 'perform:obs', 'Observe', time.localtime()
            self.observe_wait = 'Active Observe'
            self.actionQueue.put(('Observe', time.localtime()))
        observed,reward =  self.observationQueue.get()
        # Catch State change report
        if reward == None: # End of episode
            self.observed = self.reward = None
            print 'Queue.observe() => observed', observed, 'reward', reward,self.observe_wait
            if self.observe_wait not in ('Active Observe', ):
                self.observe_wait = False
        if rl.is_terminal(observed):
            print 'Queue.observe() observed is terminal:', observed,self.observe_wait
            self.observed,self.reward = [],reward
            observed = None
            if self.observe_wait not in ('Active Observe','State'):
                self.observe_wait = 'Terminal'
                if not self.actionQueue.full():
                    time.sleep(0.1)
                    print 'perform:obs', 'Observe', time.localtime()
                    self.actionQueue.put(('Observe', time.localtime()))
        if type(observed) == str:
            observed = self.str2meaning(observed)
        if self.is_state(observed):
            self.pomdp_state = observed
            print 'Queue.observe() observed is a state:', observed,self.observe_wait
            observed = None
            if self.observe_wait in ('State','Terminal'):
                self.observe_wait = False
        while not observed:
            observed = self.observe()
        self.observe_wait = False
        self.observed,self.reward = observed,reward
        if not self.reward: self.reward = 0
        return self.observed

    def perform(self,action):
        if action == None:
            while not self.observationQueue.empty(): self.observationQueue.get()
            while not self.actionQueue.empty(): self.actionQueue.get()
        if self.actionQueue.full():
            while not self.observationQueue.empty(): self.observationQueue.get()
        print 'perform',action,time.localtime()
        self.observe_wait = 'Action'
        self.actionQueue.put((action,time.localtime()))
        self.observe()
        if self.invert_reward: self.reward *= -1
        return self.reward,self.observed
