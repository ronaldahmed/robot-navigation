"""
GNG vs SOM comparison example for PLASTK.

This script shows how to:
  - Train PLASTK vector quantizers (SOM and GNG)
  - Set default parameters
  - Create a simple agent and environment.
  - Run an interaction between the agent and the environment
    with a GUI.

$Id: gngsom.py,v 1.3 2006/02/17 19:40:09 jp Exp $
"""

# Import what we need from PLASTK

# All the top-level modules
from plastk import *

# Kohonen SOMs
from plastk.vq.som import SOM,SOM2DDisplay

# Growing Neural Gas
from plastk.vq.gng import EquilibriumGNG, GNG2DDisplay

# the python debugger
import pdb


###################################################################
# Set the PLASTK parameter defaults
# [ help('plastk.params') for more info on parameters ]

#
# SOM Defaults: 10x10 SOM, with 2-D inputs
#
SOM.xdim = 10
SOM.ydim = 10
SOM.dim = 2

#
# GNG defaults: 2-D inputs, maintain average discounted error below
# 0.002, grow at most every 200 steps, max connection age 100.
#

EquilibriumGNG.dim = 2
EquilibriumGNG.rmin = 0
EquilibriumGNG.rmax = 100
EquilibriumGNG.error_threshold = 0.002
EquilibriumGNG.lambda_ = 200
EquilibriumGNG.max_age = 50
EquilibriumGNG.e_b = 0.05
EquilibriumGNG.e_n = 0.001
EquilibriumGNG.print_level = base.VERBOSE

# Overwrite old data files, instead of renaming it.

LoggingRLI.rename_old_data = False


################################################################
# Create the agent and environment
# 

class SOMTestEnvironment(Environment):
    """
    A simple environment that generates 2D points sampled from a
    series of randomly generated normal distributions.  Reward is
    always 0.
    """    
    num_samples_per_distr = 5000

    def __init__(self,**args):

        # If we have an __init__ method on a plastk subclass
        # we must call the superclass constructor
        super(SOMTestEnvironment,self).__init__(**args)

        # create a Python generator object that generates
        # the points.
        self.gen = self.generate_points()
        
    def __call__(self,action=None):

        # The main step method for Environments.
        # return the a generated point, plus the reward, except
        # when the action is None (at the beginning of an episode).
        if action is None:
            return self.gen.next()
        else:
            return self.gen.next(),0
        
    def generate_points(self):

        # A python generator that produces an infinite series of 2D
        # points generated from a series of randomly generated normal
        # distributions. 
        while True:
            mean = rand.uniform(0,100,2)
            std  = rand.uniform(1,5,2)
            for i in xrange(self.num_samples_per_distr):
                yield rand.normal(mean,std,2)

class SOMTestAgent(Agent):
    """
    A simple agent that receives 2D points and trains a kohonen SOM
    and a Growing Neural Gas with them.  It produces no meanigful
    actions (i.e. it always emits 0).
    """
    def __init__(self,**args):

        # Call the superclass constructor
        super(SOMTestAgent,self).__init__(**args)

        # instantiate a SOM
        self.som = SOM()

        # intialize SOM training
        N = SOMTestEnvironment.num_samples_per_distr * 5        
        self.som.init_training(radius_0 = max(self.som.xdim,self.som.ydim),
                               training_length = N)

        # instantiate a Growing Neural Gas
        self.gng = EquilibriumGNG()

    def __call__(self,sensation,reward=None):

        # On receiving input train the SOM and GNG.
        self.som.train(sensation)
        self.gng.train(sensation)

        # Return 0 for the action.
        return 0


################################################
# Run the an interaction between the agent and environment.
#

# Instantiate an agent and an environment
agent = SOMTestAgent()
env = SOMTestEnvironment()

# Instantiate a Reinforcement Learning Interface.  An RLI controls the
# interaction between agent and environment, passing sensation and
# reward from the environment to the agent, and actions from the agent
# to the environment.  In this experiment, the actions and reward are
# meaningless, and only the sensations, 2D vectors, are important.
#
# The LoggingRLI class includes variable logging and GUI capabilities.

rli = LoggingRLI(name = 'GNGvSOM_experiment')

# Init the RLI
rli.init(agent,env)

# Run the RLI gui with two components, a SOM display and a GNG
# display.  The RLI gui takes a list of functions that take two
# parameters, a the rli's GUI frame (root) and the rli object (rli), and return
# instances of Tkinter.Frame that can be packed into the RLI's GUI frame.
# 
rli.gui(lambda root,rli:SOM2DDisplay(root,rli.agent.som),
        lambda root,rli:GNG2DDisplay(root,gng=rli.agent.gng))
