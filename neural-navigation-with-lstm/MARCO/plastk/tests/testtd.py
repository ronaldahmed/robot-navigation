"""
Unit tests for temporal-differences reinforcement learning.

$Id: testtd.py,v 1.3 2005/08/20 17:04:15 jp Exp $
"""
import unittest
from plastk import base,pkl
from plastk.rl import gridworld,LoggingRLI
from plastk.rl.td import *

import os,operator

class TestTDGridWorld(unittest.TestCase):
    '''
    An abstract test class for RL agent classes, assumes subclasses
    will define self.agent to be various kinds of agents.
    '''
    grid = ['S..',
            '...',
            '..G']


    correct_reward = -3
    correct_length = 4
    num_episodes = 300

    def setUp(self):
        base.BaseObject.print_level = base.WARNING

        TDAgent.action_selection = 'epsilon_greedy'
        TDAgent.initial_epsilon = 1.0
        TDAgent.min_epsilon = 0.0
        
        self.env = gridworld.GridWorld(grid=self.grid)
        self.setUp_agent()
        self.rli = LoggingRLI(name=self.__class__.__name__)
        self.rli.init(self.agent,self.env)
        self.rli.episodes(self.num_episodes,100)


    def testMain(self):
        self.myTestAgent()
        self.assertEqual(self.rli.episode_data.variables['length'][-1], self.correct_length)
        self.assertEqual(self.rli.episode_data.variables['reward'][-1], self.correct_reward)

        steps = self.rli.episode_data.variables['length']
        
        self.agent.sim = None
        name = tmpnam()
        pkl.save(self.agent,name)
        new_agent = pkl.load(name)
        new_agent.sim = self.rli
        self.rli.agent = new_agent
        self.rli.episodes(2,100)
        self.assertEqual(steps,self.rli.episode_data.variables['length'][-1])

    def myTestAgent(self): pass
    def tearDown(self):
        self.rli.episode_data.close()
        os.remove(self.rli.episode_filename)

        
#####################################################
# Tabular Agents:
#

class TestTabularAgent(TestTDGridWorld):
    def myTestAgent(self): pass
    
class TestTabularSarsa(TestTabularAgent):
    def setUp_agent(self):
        self.agent = TabularTDAgent(actions=self.env.actions,
                                    step_method = 'sarsa',
                                    lambda_ = 0.0,
                                    print_level=base.WARNING)

class TestTabularQ(TestTabularAgent):
    def setUp_agent(self):
        self.agent = TabularTDAgent(actions=self.env.actions,
                                    step_method = 'q_learning',
                                    lambda_ = 0.0,
                                    print_level=base.WARNING)

#####################################################
# Linear Agents:

class TestLinearAgent(TestTDGridWorld):
    def myTestAgent(self):
        num_actions,num_features = self.agent.w.shape
        self.assertEqual(num_features,self.env.num_states)
        self.assertEqual(num_actions,len(self.env.actions))
    
class TestLinearSarsa(TestLinearAgent):
    def setUp_agent(self):
        self.agent = LinearTabularTDAgent(num_features=self.env.num_states,
                                          actions = self.env.actions,
                                          step_method = 'sarsa',
                                          lambda_ = 0.9,
                                          print_level=base.WARNING)
class TestLinearQ(TestLinearAgent):
    def setUp_agent(self):
        self.agent = LinearTabularTDAgent(num_features=self.env.num_states,
                                          actions = self.env.actions,
                                          step_method = 'q_learning',
                                          lambda_ = 0.9,
                                          print_level=base.WARNING)

####################################################
# Hierarchical Agents

from plastk.rl.options import Option,HierarchicalAgent,OptionAdapter,OPTION_TERMINATED
class RepeatTilNoChange(Option):

    def __init__(self,action,**args):
        super(RepeatTilNoChange,self).__init__(**args)
        self.action = action
    def __call__(self,sensation,reward=None):
        if reward == None:
            self.last_sensation = not sensation
            
        if sensation == self.last_sensation:
            return OPTION_TERMINATED
        else:
            self.last_sensation = sensation
            return self.action


class TestHierarchicalAgent(TestTDGridWorld):

    correct_reward = -4
    correct_length = 5
    
    def setUp_agent(self):
#        actions = self.env.actions+[RepeatTilNoChange(a) for a in self.env.actions]
        actions = [RepeatTilNoChange(a) for a in self.env.actions]
        root = OptionAdapter(agent=LinearTabularTDAgent(num_features=self.env.num_states,
                                                        actions=actions,
                                                        step_method='sarsa',
                                                        lambda_=0.9,
                                                        print_level=base.WARNING))
        self.agent = HierarchicalAgent(root_option=root)
                                          
        
        
#########################################################
def tmpnam():
    return '/tmp/'+`os.getpid()`

cases = [
    TestLinearQ,
    TestTabularSarsa,
    TestTabularQ,
    TestLinearSarsa,
    TestHierarchicalAgent
    ]

suite = unittest.TestSuite()

suite.addTests([unittest.makeSuite(case) for case in cases])
 
