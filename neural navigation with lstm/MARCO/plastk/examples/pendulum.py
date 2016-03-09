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
from plastk.rl.pendulum import *
from plastk.rl.td import *
from plastk.rl.loggingrli import *


########################################################################
# Set the TD agent parameters
#


TDAgent.alpha = 0.1
TDAgent.lambda_ = 0.7
TDAgent.gamma = 1.0
TDAgent.action_selection = 'epsilon_greedy'
TDAgent.step_method = 'sarsa'
TDAgent.initial_epsilon = 0.0
TDAgent.min_epsilon = 0.0
TDAgent.epsilon_half_life = 5000
LinearTDAgent.initial_w = 0

Pendulum.initial_angle = pi
Pendulum.initial_velocity = 0.0
Pendulum.friction = 0.001
Pendulum.delta_t = 0.1
Pendulum.actions = [-7,7]   # Torque of +/- 7 (newton-meters?)




# Make a grid environment with the given grid.
env = PendulumEZ()


# Make a linear-tabular agent, i.e. an agent that takes a single
# integer as the state and does linear updating
agent = UniformTiledAgent(actions=env.actions,
                          num_tilings=8,
                          num_features=1024,
                          tile_width=2*pi/16)

# set up the reinforcement-learning interface with agent and env
rli = LoggingRLI(name = "Pendulum Demo",
                 rename_old_data = False)
rli.init(agent,env)




#######################################
# GUI Thread Debugging -jp
from threading import Thread
import threadframe
from traceback import print_stack
import time

def print_stacks():
    while True:
        print '=' * 72
        for id,frame in threadframe.dict().iteritems():
            print '----- [%s] ------' % id
            print_stack(frame)
        time.sleep(10)

dbthread = Thread(target=print_stacks,verbose=1)
dbthread.setDaemon(True)
dbthread.start()


# Run the rli GUI with a GridWorldDisplay widget and a widget that
# plots the length of each episode.
rli.gui(PendulumGUI,
        EpisodeVarPlotter('length'))
#rli.episodes(2,100000)
