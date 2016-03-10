"""
$Id: gridworld.py,v 1.19 2006/03/24 23:53:55 adastra Exp $
"""
from Numeric import array

import plastk.rl as rl
import plastk.rand as rand
import plastk.utils as utils
from plastk.params import *

FREE = '.'
WALL = '#'
START = 'S'
GOAL = 'G'

class GridWorld(rl.Environment):

    grid = Parameter(default=["############",
                              "#G.........#",
                              "#..........#",
                              "#..........#",
                              "#..........#",
                              "#..........#",
                              "#..........#",
                              "#..........#",
                              "#..........#",
                              "#..........#",
                              "#.........S#",
                              "############"])

    random_start_pos = Parameter(default=False)

    timeout = NonNegativeInt(default=0)
    
    actions = Parameter(default=['N','S','E','W'])
    action_map = Parameter(default={'N':(-1, 0),
                                    'S':( 1, 0),
                                    'E':( 0, 1),
                                    'W':( 0,-1) })
    
    correct_action_probability = Magnitude(default=1.0)
    step_reward = Number(default=-1)
    goal_reward = Number(default=0)

    start_pos = Parameter(default=None)
    goal_pos  = Parameter(default=None)
    crumbs    = Parameter(default=False)
    clear_crumbs_on_pose_set = Parameter(default=True)
    recolor_crumbs_on_pose_set = Parameter(default=False)

    count_wall_states = Boolean(default=False)
    
    def __init__(self,**args):
        super(GridWorld,self).__init__(**args)
        
        if self.crumbs:
            self.clear_crumbs = False
            self.recolor_crumbs = False
            self.connect_crumbs = True
        
        for r,row in enumerate(self.grid):
            if len(row) != len(self.grid[0]):
                raise "GridWorld error: grid rows must all be the same length."
            
            for c,cell in enumerate(row):
                if cell == START:
                    if self.start_pos:
                        raise "GridWorld error: grid has more than one start position."
                    self.start_pos = (r,c)
                elif cell == GOAL:
                    if self.goal_pos:
                        raise "GridWorld error: grid has more than one goal position."
                    self.goal_pos = (r,c)

        self.start_episode()
        if self.count_wall_states:
            self.num_states = sum([len(row) for row in self.grid])
        else:
            self.num_states = sum([row.count(FREE)+row.count(START)+row.count(GOAL)
                                   for row in self.grid])

    def __call__(self,action=None):
        if action == None:
            self.curr_pos = self.start_pos
            self.episode_steps = 0
            self.start_episode()
            return self.state()
        else:
            self.episode_steps += 1
            assert action in self.actions
            r,c = self.curr_pos
            p = self.correct_action_probability
            N = len(self.actions)
            distr = array([(1-p)/(N-1)] * N)
            distr[self.actions.index(action)] = p
            a = utils.weighted_sample(distr)

            dr,dc = self.action_map[self.actions[a]]

            if self.move_okay(r+dr,c+dc):
                r,c = self.curr_pos = (r+dr,c+dc)

        if (r,c) == self.goal_pos:
            self.verbose("!!! GOAL !!!")
            return rl.TERMINAL_STATE,self.goal_reward
        elif self.timeout and self.episode_steps > self.timeout:
            return rl.TERMINAL_STATE,self.step_reward
        else:
            return self.state(),self.step_reward

    def reset_crumbs(self):
        if not self.crumbs: return
        if self.clear_crumbs_on_pose_set:
            self.clear_crumbs = True
        if self.recolor_crumbs_on_pose_set:
            self.recolor_crumbs = True
        self.connect_crumbs = False

    def start_episode(self):
        if self.random_start_pos:
            while True:
                r = rand.randint(len(self.grid))
                c = rand.randint(len(self.grid[0]))
                g = self.grid[r][c]
                if g != WALL and g != GOAL:
                    self.curr_pos = self.start_pos = r,c
                    break
        else:
            self.curr_pos = self.start_pos

        self.episode_steps = 0
        self.reset_crumbs()

    def set_route(self,start_pos=None,goal_pos=None):
        if start_pos: self.start_pos = start_pos
        if goal_pos: self.goal_pos = goal_pos
        self.start_episode()

    def move_okay(self,r,c):
        rbound = len(self.grid)
        cbound = len(self.grid[0])
        return ( 0 <= r < rbound and
                 0 <= c < cbound and
                 self.grid[r][c] != WALL )
    
    def state(self):
        r,c = self.curr_pos
        return r*len(self.grid[0]) + c


try:
    import Tkinter as Tk
    import Pmw
    import itertools
except ImportError,e:
    print "WARNING: Can't import Display Libaries:", e
else:
    class GridWorldDisplay(Tk.Frame):
        """
        GridWorldDisplay show the agent's start, goal, and current position in a Cartesian grid world.
        The rli.env must have the following attributes:
        env.grid : a list of strings, one per row of the environment
         :: '.' is open, '#' is an obstacle, 'S' is Start and 'G' is Goal
         :: (defined as OPEN, WALL, START, and GOAL in plastk.rl.gridworld)
        env.name : string name of the environment
        env.start_pos : (r,c) row,column tuple indicating start position
        env.goal_pos : (r,c) row,column tuple indicating goal position
        env.curr_pos : (r,c) row,column tuple indicating current agent position

        The rli.env may have the following attributes:
        env.crumbs : Boolean indicating whether the Display should show the traveled track
        env.clear_crumbs : Boolean indicating to clear the traveled track
        env.map_image_file: path to an image file representing the environment
        env.map_offsets: tuple(x_min,y_min,x_max,y_max) of the bounding box in pixels
                         for mapping grid cells onto the map image.
        """
        cell_width = cell_height = _cell_width = _cell_height = 20 #pixels
        def __init__(self,root,rli,**tkargs):
            
            self.x_offset = self.y_offset = 0
            self.rli = rli
            if 'crumbs' in dir(rli.env) and self.rli.env.crumbs:
                self.crumb_width = self.cell_width/5
                self.crumb_offset_x = self.cell_width/2
                self.crumb_offset_y = self.cell_height/2
                colors = ['green','blue','gray','yellow','orange','purple','turquoise']
                self.crumb_lists_len = len(colors)
                self.crumb_colors = itertools.cycle(colors)
                self.crumb_color = self.crumb_colors.next()
            Tk.Frame.__init__(self,root,**tkargs)
            group = Pmw.Group(self,tag_text=rli.env.name)
            group.pack(side='top',fill='both',expand=1)

            grid = self.rli.env.grid
            
            self.canvas = Tk.Canvas(group.interior(),bg='white',
                                    width = len(grid[0]) * self.cell_width,
                                    height = len(grid) * self.cell_height)
            self.canvas.pack(side='top',expand=1,fill='both')

            self.draw_env()

        def resize_map(self):
            """
            Resize the map image to fit the current display.
            """
            w1 = self._cell_width * len(self.rli.env.grid[0])
            h1 = self._cell_height * len(self.rli.env.grid)
            if 'map_offsets' in dir(self.rli.env) and self.rli.env.map_offsets:
                x_min,y_min,x_max,y_max = self.rli.env.map_offsets
            else:
                x_min,y_min,x_max,y_max = 0,0,w1,h1
            w0 = x_max - x_min
            h0 = y_max - y_min
            if (w1,h1) != self.map.size:
                self.map = self.map.resize((w1,h1))
                self._cell_width = w0/(len(self.rli.env.grid[0])-1)
                self._cell_height = h0/(len(self.rli.env.grid)-1)
                self.cell_width = self._cell_width * w1/w0
                self.cell_height = self._cell_height * h1/h0
                self.x_offset = x_min * w1/w0
                self.y_offset = y_min * h1/h0

        def draw_env(self,event=None):
            # delete the old contents
            self.canvas.delete('all')

            if 'map_image_file' in dir(self.rli.env) and self.rli.env.map_image_file:
                from PIL.ImageTk import PhotoImage,Image
                self.map = Image.open(self.rli.env.map_image_file)
                self.photo = PhotoImage(self.map)
                self.resize_map()
                self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
            else: self.rli.env.map_image_file = None

            for r,row in enumerate(self.rli.env.grid):
                for c,cell in enumerate(row):
                    x1,y1,x2,y2 = self.cell_coords(r,c)
                    if cell == WALL:
                        if not self.rli.env.map_image_file:
                            self.canvas.create_rectangle(x1,y1,x2,y2,fill='black',outline='black',
                                                         tags=['wall','all'])
                    elif (r,c) == self.rli.env.start_pos:
                        self.start = self.canvas.create_oval(x1,y1,x2,y2,fill='green',outline='green',
                                                             tags=['start','all'])
                    elif (r,c) == self.rli.env.goal_pos:
                        self.goal = self.canvas.create_oval(x1,y1,x2,y2,fill='red',outline='red',
                                                            tags=['goal','all'])
                    elif cell == FREE:
                        continue
                    else: print "##WARNING##: Unknown cell type", cell

            self.agent_pos = self.rli.env.curr_pos
            x1,y1,x2,y2 = self.cell_coords(*self.agent_pos)

            self.agent_last = self.canvas.create_oval(x1,y1,x2,y2,
                                                      fill='#EEEEff',outline='#DDDDff',
                                                      tags=['agent','all'])
            self.agent_curr = self.canvas.create_oval(x1,y1,x2,y2,
                                                      fill='blue',outline='blue',
                                                      tags=['agent','all'])
            if 'crumbs' in dir(self.rli.env) and self.rli.env.crumbs:
                if hasattr(self,'crumb_lists'): self.clear_crumbs()
                else: self.crumb_lists = [[]]

        def redraw(self):
            if self.rli.env.map_image_file:
                self.resize_map()

            self.canvas.coords(self.agent_curr,*self.cell_coords(*self.rli.env.curr_pos))
            # update the last position marker
            self.canvas.coords(self.agent_last,*self.cell_coords(*self.agent_pos))
            # Drop a breadcrumb
            if 'crumbs' in dir(self.rli.env) and self.rli.env.crumbs:
                self.draw_crumb(self.agent_pos, self.rli.env.curr_pos)
            self.agent_pos = self.rli.env.curr_pos

            # Move start and goal
            self.canvas.coords(self.start,*self.cell_coords(*self.rli.env.start_pos))
            self.canvas.coords(self.goal,*self.cell_coords(*self.rli.env.goal_pos))

        def cell_coords(self,r,c):
            y1 = r * self.cell_height+self.y_offset
            y2 = y1 + self.cell_height
            x1 = c * self.cell_width+self.x_offset
            x2 = x1 + self.cell_width
            return x1,y1,x2,y2

        def draw_crumb(self,from_pt,to_pt):
            if self.rli.env.clear_crumbs: self.clear_crumbs()
            if self.rli.env.recolor_crumbs:
                self.crumb_color = self.crumb_colors.next()
                self.crumb_offset_x += self.cell_width/10
                self.crumb_offset_x %= self.cell_width
                self.crumb_offset_y += self.cell_height/10
                self.crumb_offset_y %= self.cell_height
                self.rli.env.recolor_crumbs = False
            if self.rli.env.connect_crumbs:
                x1,y1,x2,y2 = self.cell_coords(*from_pt)
                xC,yC = x1+self.crumb_offset_x, y1+self.crumb_offset_y
                self.crumb_lists[-1].append(self.canvas.create_oval(xC-self.crumb_width,yC-self.crumb_width,
                                                                    xC+self.crumb_width,yC+self.crumb_width,
                                                                    fill=self.crumb_color,outline='grey',
                                                                    tags=['crumbs','all']))
                x1,y1,x2,y2 = self.cell_coords(*to_pt)
                xL,yL = x1+self.crumb_offset_x, y1+self.crumb_offset_y
                self.crumb_lists[-1].append(self.canvas.create_line(xL,yL,xC,yC,
                                                                   fill=self.crumb_color,
                                                                   tags=['crumbs','all'],
                                                                   width=self.crumb_width))
            else:
                # Delete artifact singleton last list
                if self.crumb_lists and len(self.crumb_lists[-1])==1: self.clear_crumb_list(-1)
                if not self.crumb_lists or self.crumb_lists[-1]: self.crumb_lists.append([])
                if len(self.crumb_lists) >= self.crumb_lists_len: self.clear_crumb_list(0)
                self.rli.env.connect_crumbs = True 
        
        def clear_crumb_list(self,index):
            try:
                for j,crumb in enumerate(self.crumb_lists[index]):
                    self.canvas.delete(crumb)
                del self.crumb_lists[index]
            except IndexError,e:
                print 'GridWorldDisplay.clear_crumb_list(): Caught bad index',index, self.crumb_lists,e
                
        def clear_crumbs(self):
            for crumb_list_idx in range(len(self.crumb_lists)):
                self.clear_crumb_list(crumb_list_idx)
            self.crumb_lists = [[]]
            self.crumb_offset_x = self.cell_width/2
            self.crumb_offset_y = self.cell_height/2
            self.rli.env.clear_crumbs = False
