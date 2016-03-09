"""
The Python Learning Agent Software Toolkit

PLASTK is a class library of tools for building and studying learning
agents, including classes implementing reinforcement learning (RL)
algorithms and environments, hierarchical RL (Options), function
approximators, perceptual feature construction and learning (e.g. tile
coding and self-organzing feature maps.  An overview of the included
modules is below.  See individual module documentation (via pydoc or
help()) for details.

PLASTK depends on several other python modules and packages for
support: including Numeric Python, Scientific Python, Gnuplot.py,
Tkinter, Python MegaWidgets (Pmw), and BLT (for some graphing).

* SUBPACKAGES *

Reinforcement Learning:
rl           -- Generic RL agent, and environment classes, and a simple
                RL Interface (RLI) class, conforming to the RLI5 standard.
rl.td        -- Temporal-difference learning agents (Q-learning and Sarsa)
rl.options   -- Hierarchical reinforcement learning
rl.tiles     -- Pure python implementation of Sutton's tile coding
                lib.

rl.data      -- Data logging and graphing routines/classes for RL data
rl.gridworld -- Grid environments for reinforcement learning
rl.pendulum  -- A simple pendulum swing-up environment


Vector Quantization:
vq           -- Generic vector-quantizer class.  
vq.som       -- Self-organizing feature maps (Kohonen Maps)
vq.gng       -- Growing Neural Gas (incremental vector quantizer/self-organizing map)


Function Approximation:
fnapprox        -- Generic Function approximator class
fnapprox.linear -- Very simple linear gradient descent approximator
fnapprox.lwl    -- Locally-weighted learning.
fnapprox.vectordb -- Vector databases for lwl (KD-Trees, etc)

Utilities:
base      -- The BaseObject class from which all PLASTK classes inherit
params    -- Parameter descriptors used by PLASTK objects
plot      -- Plotting (requires Gnuplot.py)
exper     -- Constructing and running experiments over a variety of
             experimental conditions or parameter values
pkl       -- Routines for saving/restoring state (uses cPickle)
utils     -- Miscellaneous utility functions


$Id: __init__.py,v 1.14 2005/06/23 19:17:35 jp Exp $

"""


__all__ = ['base',
           'exper',
           'fnapprox',
           'pkl',
           'params', 
           'rand',
           'rl',
           'utils',
           'vq',
           'test',

           'Agent',
           'Environment',
           'RLI',
           'LoggingRLI',
           ]

from plastk.rl import RLI,Agent,Environment,LoggingRLI

def test(verbosity=1):
    import tests
    return tests.all(verbosity=verbosity)
    
