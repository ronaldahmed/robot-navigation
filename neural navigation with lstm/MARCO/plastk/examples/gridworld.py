"""
A simple gridworld demo.

This demo instantiates a gridworld and an agent that will learn to get
to the goal in the world using reinforcement learning.

$Id;$
"""

import pdb  # We may need the debugger
import sys,getopt

########################################################################
# Import what we need from PLASTK: the top-level modules,
# and stuff for gridworlds, temporal difference (td) agents, and
# an interface that supports GUIs and logging.
from plastk import *
from plastk.rl.gridworld import *
from plastk.rl.td import *
from plastk.rl.loggingrli import *


########################################################################
# Set the TD agent parameters
#


TDAgent.alpha = 0.2
TDAgent.lambda_ = 0.9
TDAgent.gamma = 0.95
TDAgent.action_selection = 'epsilon_greedy'
TDAgent.step_method = 'sarsa'
TDAgent.initial_epsilon = 0.1
TDAgent.min_epsilon = 0.01
TDAgent.epsilon_half_life = 3000
LinearTDAgent.initial_w = 0

GridWorld.random_start_pos = True
GridWorld.correct_action_probability = 0.9
GridWorld.crumbs = True
GridWorld.step_reward = 0
GridWorld.goal_reward = 1

#
# The grid for the environment.  The agent starts in the upper left
# and must travel to the lower left.
#
grid1 = [
    '#############',
    '#S..........#',
    '#...........#',
    '#...........#',
    '#...........#',
    '#...........#',
    '######.######',
    '#...........#',
    '#...........#',
    '#...........#',
    '#...........#',
    '#G..........#',
    '#############',
    ]                

grid2 = [
    '#########################',
    '#S..........#...........#',
    '#...........#...........#',
    '#.......................#',
    '#...........#...........#',
    '#...........#...........#',
    '######.###########.######',
    '#...........#...........#',
    '#...........#...........#',
    '#.......................#',
    '#...........#...........#',
    '#...........#..........G#',
    '#########################',
    ]                

# Make a grid environment with the given grid.
colors = False
try:
    opts,args = getopt.getopt(sys.argv[1:],'c')
    opts = dict(opts)
    colors = '-c' in opts
except getopt.GetoptError,err:
    print err

if colors:
    env = GridWorld(grid=grid1, clear_crumbs_on_pose_set=False, recolor_crumbs_on_pose_set=True)
else:
    env = GridWorld(grid=grid1)

# Make a tabular agent. 
agent = TabularTDAgent(actions = env.actions)

# set up the reinforcement-learning interface with agent and env
rli = LoggingRLI(name = "Gridworld Demo",
                 rename_old_data = False)
rli.init(agent,env)

# Run the rli GUI with a GridWorldDisplay widget and a widget that
# plots the length of each episode.
rli.gui(GridWorldDisplay,
        EpisodeVarPlotter('length'))
