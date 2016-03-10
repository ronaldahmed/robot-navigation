"""
A simple gridworld demo.

This demo instantiates a gridworld and an agent that will learn to get
to the goal in the world using reinforcement learning.

$Id;$
"""

import pdb  # We may need the debugger

########################################################################
# Import what we need from PLASTK: the top-level modules,
# and stuff for gridworlds, temporal difference (td) agents, and
# an interface that supports GUIs and logging.
from plastk import *
from plastk.rl.facegridworld import *
from plastk.rl.td import *
from plastk.rl.loggingrli import *

base.min_print_level = base.DEBUG

########################################################################
# Set the TD agent parameters
#

TDAgent.alpha = 0.2
TDAgent.lambda_ = 0.9
TDAgent.gamma = 1.0
TDAgent.action_selection = 'epsilon_greedy'
TDAgent.update_method = 'sarsa'
TDAgent.initial_epsilon = 0.0
TDAgent.min_epsilon = 0.0
LinearTDAgent.initial_w = 0

#FaceGridWorld.correct_action_probability = 0.9

from POMDP.MarkovLoc_L import pomdp
pomdp.map_dir = '../Maps'
# Make a grid environment with the given grid.
env = MarkovLocPOMDPWorld(pomdp=pomdp)
env.setRoute(5,7)

# Make a linear-tabular agent, i.e. an agent that takes a single
# integer as the state and does linear updating
agent = TabularTDAgent(actions = env.actions, num_features = env.num_states)
##agent = MarcoAgent(actions = env.actions, num_features = env.num_states)

# set up the reinforcement-learning interface with agent and env
rli = LoggingRLI(name = "FaceGridworld Demo", rename_old_data = False)
rli.init(agent,env)

# Run the rli GUI with a FaceGridWorldDisplay widget and a widget that
# plots the length of each episode.
rli.gui(FaceGridWorldDisplay, EpisodeVarPlotter('length'))
