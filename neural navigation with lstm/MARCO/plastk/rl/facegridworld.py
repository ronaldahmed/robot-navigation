"""
Environment wrappers for turn/travel corridor navigation.

$Id: facegridworld.py,v 1.11 2006/03/24 23:53:55 adastra Exp $
"""

import random
from plastk.rl.gridworld import *
from plastk.display import pol2cart,pi

class FaceGridWorld(GridWorld):
    """
    FaceGridWorld is a dummy wrapper around GridWorld,
     which adds a (random) curr_face attribute to model the direction the agent is facing.
    FaceGridWorld is used for testing, but with different actions, could be a simple Turn,Travel Sim
    """
    turns = []
    directions = ['N','S','E','W']
    def __init__(self,start_pos=None,goal_pos=None,face=None,**args):
        super(FaceGridWorld,self).__init__(start_pos,goal_pos,**args)
        if face:
            self.start_face = face
        else:
            self.start_face = random.choice(self.directions)
        self.curr_face = self.start_face
        self.num_states *= 4
    
    def __call__(self,action=None):
        result = super(FaceGridWorld,self).__call__(action)
        self.curr_face = random.choice(self.directions)
        return result

class MarkovLocPOMDPWorld(rl.Environment):
    """
    MarkovLocPOMDPWorld wraps a POMDP_MarkovLoc_Antie object.
    POMDP_MarkovLoc_Antie models a corridors and intersections gridworld
    with discrete actions and complex, symbolic observations (views down the hallways).
    Views include observations of objects in the environment,
    such as furniture, flooring appearance, and wall paintings.

    The rli.env must have the following attributes:
    pomdp.Actions : A dictionary of string action names to action code objects
    pomdp.NumPlaces : Integer number of places
    pomdp.NumPoses : Integer number of poses (assumed 4).
    pomdp.env : string name of the environment
    pomdp.grid : metrical maps in the format specified for GridWorldDisplay
    pomdp.StartPlace : ID integer of place where the agent starts
    pomdp.DestPlace : ID integer of place where the agent should go
    pomdp.StartPose : Integer [0:3] or None indicating direction the agent starts facing
    
    pomdp.trueState : two integer tuple of agent's current place ID and pose direction
    
    pomdp.place2coords : function to convert PlaceNumber to grid metrical coordinates
    pomdp.coords2state : function to convert grid metrical coordinates to PlaceNumber
    pomdp.set : function to set pomdp.trueState
    pomdp.setRoute: function to set pomdp.StartPlace and pomdp.DestPlace and initialize for new episode
    pomdp.observe : function to return a current observation
    pomdp.perform : fuction which takes an action name string
                    and returns the symbolic observation and numeric reward

    the rli.env may have the following attributes:
    pomdp.map_image_file : the name of a graphical map image file
    pomdp.map_offsets : a tuple of offsets in pixels
                        of the (x_min,y_min,x_max,y_max) of the grid on the image.
    """
    directions = ['N','E','S','W']
    start_pos = Parameter(default=None)
    goal_pos  = Parameter(default=None)
    crumbs    = Parameter(default=False)
    clear_crumbs_on_pose_set = Parameter(default=True)
    recolor_crumbs_on_pose_set = Parameter(default=False)
    
    def __init__(self,**args):
        super(MarkovLocPOMDPWorld,self).__init__(**args)
        self.actions = self.pomdp.Actions.keys()
        self.num_states = self.pomdp.NumPlaces*self.pomdp.NumPoses
        self.name = self.pomdp.env + str(self.pomdp.PosSet)
        self.grid = self.pomdp.grid
        if 'map_image_file' in dir(self.pomdp) and self.pomdp.map_image_file:
            self.map_image_file = self.pomdp.map_dir+'/'+self.pomdp.map_image_file
        if 'map_offsets' in dir(self.pomdp) and self.pomdp.map_offsets:
            self.map_offsets = self.pomdp.map_offsets
        if self.crumbs:
            self.clear_crumbs = False
            self.recolor_crumbs = False
            self.connect_crumbs = True
        self.setRoute(self.pomdp.place2coords(self.pomdp.StartPlace),
                      self.pomdp.place2coords(self.pomdp.DestPlace))

    def reset_crumbs(self):
        if not self.crumbs: return
        if self.clear_crumbs_on_pose_set:
            self.clear_crumbs = True
        if self.recolor_crumbs_on_pose_set:
            self.recolor_crumbs = True
        self.connect_crumbs = False

    def start_episode(self):
        self.episode_steps = 0
        self.set((self.pomdp.StartPlace, self.pomdp.StartPose))
        self.reset_crumbs()

    def get_curr_pos(self):
        return self.pomdp.place2coords(self.pomdp.trueState[0])
    def set_curr_pos(self,pos):
        return self.pomdp.set((self.pomdp.coords2place(pos),self.pomdp.trueState[1]))
    curr_pos = property(get_curr_pos, set_curr_pos)

    def get_curr_face(self):
        return self.directions[self.pomdp.trueState[1]]
    def set_curr_face(self,direction):
        return self.pomdp.face(direction)
    curr_face = property(get_curr_face, set_curr_face)

    def setRoute(self,Start,Dest):
        self.pomdp.setRoute(Start,Dest)
        self.start_pos = self.pomdp.place2coords(self.pomdp.StartPlace)
        self.goal_pos = self.pomdp.place2coords(self.pomdp.DestPlace)
        print 'MarkovLocPOMDPWorld.setRoute',(Start,Dest),self.start_pos,self.goal_pos
        if not self.pomdp.StartPose:
            self.pomdp.StartPose = random.choice(range(self.pomdp.NumPoses))
        self.clear_crumbs_on_pose_set = True
        self.start_episode()
        self.clear_crumbs_on_pose_set = False

    def set(self,pose):
        print 'MarkovLocPOMDPWorld.set',(pose)
        self.pomdp.set(pose)
        self.reset_crumbs()

    def __call__(self,action=None):
        """In addition to the action name strings and None,
        also accepts an action tuple
        to change the route start_pos and goal_pos and move current pose to the start.
        """
        if type(action) == tuple:
            command,arg = action
            if command == 'Route':
                self.setRoute(*arg)
                return rl.TERMINAL_STATE,0
            elif command == 'State':
                self.set(arg)
                return str(self.pomdp.trueState),0
            else: raise ValueError, 'Unknown command %s with argument %s' % (command, arg)
        if action == None:
            self.start_episode()
            return str(self.pomdp.trueState)
        if action == 'Observe':
            observed,reward = str(self.pomdp.observe()),0
            return str(self.pomdp.observe()),0
        
        self.episode_steps += 1
        reward,observed = self.pomdp.perform(action)
        print 'MarkovLocPOMDPWorld(',action,') =>', str(self.pomdp.trueState), str(observed),reward
        if action == 'DeclareGoal': return rl.TERMINAL_STATE,reward
        return str(observed),reward

try:
    import Tkinter as Tk
    import Pmw
except ImportError,e:
    print "WARNING: Can't import Display Libraries:", e
else:
    class FaceGridWorldDisplay(GridWorldDisplay):
        """
        FaceGridWorldDisplay shows the agent's orientation on top of the GridWorldDisplay.
        The rli.env must have the following attributes:
        env.curr_face : one character string indicating direction the agent is facing.
         :: one of faces.keys(): ['N','E','S','W']
        """

        eye_width = GridWorldDisplay.cell_width/4
        eye_offsets = (pi/8,-pi/8)
        faces = { 'E' : 0.0,
                  'S' : pi/2,
                  'W' : pi,
                  'N' : -pi/2 }
        
        def position_eye(self,offset,position,face):
            x1,y1,x2,y2 = self.cell_coords(*position)
            midx,midy = x1+self.cell_width/2,y1+self.cell_width/2,
            dx,dy = pol2cart(self.cell_width*0.55, self.faces[face]+offset)
            return midx+dx, midy+dy
        
        def draw_env(self,event=None):
            GridWorldDisplay.draw_env(self,event)
            
            self.agent_face = self.rli.env.curr_face
            self.agent_eyes = {}
            for pose,color in zip(('Last','Curr'), ('purple','yellow')):
                for offset in self.eye_offsets:
                    x,y = self.position_eye(offset, self.rli.env.curr_pos, self.rli.env.curr_face)
                    eye = self.canvas.create_oval(x-self.eye_width,y-self.eye_width,
                                                  x+self.eye_width,y+self.eye_width,
                                                  fill=color,outline='purple',
                                                  tags=['agent_eye','all'])
                    self.agent_eyes[(pose,offset)] = eye

        def redraw(self):
            last_face = self.agent_face
            self.agent_face = self.rli.env.curr_face

            for pose,position,face in zip(
                ('Last', 'Curr'),
                (self.agent_pos, self.rli.env.curr_pos),
                (last_face, self.agent_face)):
                for offset in self.eye_offsets:
                    x,y = self.position_eye(offset,position,face)
                    self.canvas.coords(self.agent_eyes[(pose,offset)],
                                       x,y,
                                       x+self.eye_width,y+self.eye_width)
            GridWorldDisplay.redraw(self)
