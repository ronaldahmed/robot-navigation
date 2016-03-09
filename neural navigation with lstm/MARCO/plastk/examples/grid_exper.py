"""
A demo experiment using gridworld, temporal-difference learners, and
the plastk.exper module to control the experimental conditions.

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
from plastk.rl.gridworld import *
from plastk.rl.td import *
from plastk.rl.loggingrli import *
from plastk.exper import *

########################################################################
# Set the TD agent parameters
#

TDAgent.alpha = 0.2
TDAgent.lambda_ = 0.9
TDAgent.gamma = 1.0
TDAgent.action_selection = 'epsilon_greedy'
TDAgent.step_method = 'sarsa'
TDAgent.initial_epsilon = 0.0
TDAgent.min_epsilon = 0.0
LinearTDAgent.initial_w = 0
TabularTDAgent.initial_q = 0

GridWorld.correct_action_probability = 0.9
GridWorld.count_wall_states = True

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


class TDExperiment(FactorExperiment):

    td_type = Factor(agent_type = [TabularTDAgent, LinearTabularTDAgent],
                     agent_name = ['tabular','linear'])
    grid = Factor(grid      = [grid1, grid2],
                  grid_name = ['grid1', 'grid2'])

    conditions = Cross(td_type, grid)

    def proc(self, condition_num,
             agent_type = None,
             agent_name = None,
             grid       = None,
             grid_name  = None ):


        # Make a grid environment with the given grid.
        env = GridWorld(grid=grid)


        # Make a tabular agent. 
        agent = agent_type(actions = env.actions,
                           num_features = env.num_states)

        # set up the reinforcement-learning interface with agent and env
        rli = LoggingRLI(name = "Gridworld Experiment",
                         filestem = self.make_filestem(condition_num),
                         rename_old_data = False)
        rli.init(agent,env)

        # Run the rli GUI with a GridWorldDisplay widget and a widget that
        # plots the length of each episode.
        if self.gui:
            rli.gui(GridWorldDisplay,
                    EpisodeVarPlotter('length'))
        else:
            rli.episodes(500,5000)

    def make_filestem(self,N):
        cc = self.conditions[N]
        agent = cc['agent_name']
        grid = cc['grid_name']
        return 'grid_exper-%s-%s-%d'%(agent,grid,N)

exp = TDExperiment()
if __name__ == '__main__':
    exp.main()
