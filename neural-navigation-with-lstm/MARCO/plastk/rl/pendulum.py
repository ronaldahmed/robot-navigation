import plastk.rl
import plastk.rand as rand
from plastk.params import *
from math import pi,sin,cos
from Numeric import array,ArrayType
from math import atan2

class Pendulum(plastk.rl.Environment):

    initial_angle = Number(default=0)
    initial_velocity = Number(default=0)
    delta_t = Number(default=0.01)

    mass = Number(default=1.0,bounds=(0,None))   # mass in kg
    length = Number(default=1.0,bounds=(0,None))  # length in meters

    g = Number(default=9.8)  # gravitational acceleration in m/s^2

    friction = Number(default=0.01,bounds=(0,None))
    
    actions = Parameter([-1,0,1])   # actions are torques in newton-meters

    reward_exponent = Number(default=1)
    reward_velocity_threshold = Number(default=0.25)
    
    def __init__(self,**params):
        super(Pendulum,self).__init__(**params)
        self.angle = self.initial_angle
        self.velocity = self.initial_velocity
        self.last_action = 0

    def __call__(self,action=None):
        if action==None:
            self.angle = self.initial_angle
            self.velocity = self.initial_velocity
            return array((self.angle,self.velocity))
        else:
            if type(action) == ArrayType:
                action = action[0]
            self.last_action = action
            # setup constants
            m = self.mass
            g = self.g
            l = self.length
            dt = self.delta_t
            
            # Gravitational torque
            T_g = m * g * l * cos(pi/2-self.angle)
            # action torque
            T_a = action

            # rotational acceleration is Torque divided by moment-of-inertia
            a = (T_g + T_a)/(m*l)

            # update velocity from acceleration (including friction)
            self.velocity += (a * dt)
            self.velocity -= self.velocity * self.friction

            # update angle from velocity
            self.angle += (self.velocity * dt)/2

            return (array((normalize_angle(self.angle),self.velocity)),self.reward())


    def reward(self):

        if self.reward_velocity_threshold < abs(self.velocity) < 8 :
            return 0.0
        if abs(self.velocity) >= 8:
            return -( (abs(self.velocity)-8)**2)
        else:
            return cos(self.angle)**self.reward_exponent


class PendulumEpisodic(Pendulum):
    def __call__(self,action=None):        
        if action!=None and self.is_terminal():
            return plastk.rl.TERMINAL_STATE,0
        return super(PendulumEpisodic,self).__call__(action)
    
    def reward(self):
        return -1
    def is_terminal(self):
        return (abs(normalize_angle(self.angle)) < (10 * pi/180)
                and self.velocity < 0.1)
    
class PendulumEZ(PendulumEpisodic):
    def is_terminal(self):
        return abs(normalize_angle(self.angle)) < (10 * pi/180)


class PendulumUpright(PendulumEpisodic):
    initial_angle = Dynamic(lambda: rand.normal(0,5*pi/180))
    def reward(self):
        return 1
    def is_terminal(self):
        return abs(normalize_angle(self.angle)) > (90 * pi/180)



import Tkinter as Tk
import Pmw

class PendulumGUI(Tk.Frame):

    def __init__(self,parent,rli,**config):
        env = self.env = rli.env
        Tk.Frame.__init__(self,parent,**config)
        g = Pmw.Group(self,tag_text=env.name)
        g.pack(side='top',fill='both',expand=1)
        self.canvas = Tk.Canvas(g.interior())
        self.canvas.pack(side='top',fill='both',expand=1)

        self.line = None
        
        self.redraw()

    def redraw(self):
        from math import sin,cos
        
        xo,yo,xs,ys = self.origin_and_scale()    

        self.canvas.addtag_all('old')

        # get the position of the end of the pendulum
        r = self.env.length
        a = self.env.angle + pi/2


        # draw the velocity arc
        vel_arc_len = self.env.velocity/80 * 2 * pi 
        left,top = self.local2global(-r,-r)
        right,bottom = self.local2global(r,r)
        self.canvas.create_arc(left,top,right,bottom,
                               start=a*180/pi,
                               extent=-vel_arc_len*180/pi,
                               style='arc',
                               outline='blue',
                               width=2)
        # draw the torque arc
        torque_arc_len =  self.env.last_action/100.0 * 2 * pi
        left,top = self.local2global(-r/5,-r/5)
        right,bottom = self.local2global(r/5,r/5)
        self.canvas.create_arc(left,top,right,bottom,
                               start=a*180/pi,
                               extent=-torque_arc_len*180/pi,
                               style='arc',
                               outline='red',
                               width=5)
        # draw the pendulum bar
        x,y = r*cos(a), r*sin(a)
        xend,yend = self.local2global(x,y)
        self.canvas.create_line(xo,yo,xend,yend,width=4)

        # draw the pendulum ball
        ball_radius = self.env.length * 0.05
        ball_x1,ball_y1 = self.local2global(x-ball_radius, y-ball_radius)
        ball_x2,ball_y2 = self.local2global(x+ball_radius, y+ball_radius)
        self.canvas.create_oval(ball_x1,ball_y1,ball_x2,ball_y2,fill='red',width=2)
        
        self.canvas.delete('old')

    def local2global(self,x,y):
        xo,yo,sx,sy = self.origin_and_scale()
        return x*sx + xo, y*sy + yo

    def origin_and_scale(self):
        xo = self.canvas.winfo_reqwidth()/2
        yo = self.canvas.winfo_reqheight()/2

        xs = (min(xo,yo) * 0.9)/self.env.length
        ys = -xs
        
        return xo,yo,xs,ys


def normalize_angle(x):
    """
    Convert an angle in radians into the range (-pi,pi)
    """
    a = (x % (2*pi))
    if a > pi:
        a = a - 2*pi

    return a
        

def test(n=1000,gui=True):
    from plastk.rl.loggingrli import LoggingRLI
    p = Pendulum(initial_angle=pi,
                   initial_velocity=0.0,
                   friction=0.001,
                   delta_t=0.01,
                   actions=[-5,0,5])

    rli = LoggingRLI(running_plot=True)
    import os

    try: 
        os.remove(rli.episode_filename)
        os.remove(rli.step_filename)
    except OSError,err:
        print err
        
    agent = plastk.rl.UniformTiledSarsa(actions = p.actions,
                                        num_features = 1024,                                        
                                        num_rfs = 1,
                                        num_tilings = 1,
                                        tile_width = 2*pi/16,
                                        action_selection = 'epsilon_greedy',
                                        min_epsilon = 0.0001,
                                        initial_epsilon = 0.1,
                                        epsilon_half_life = 50000,
                                        alpha = 0.1,
                                        gamma = 0.99,
                                        lambda_ = 0.9,
                                        initial_w = 100.0/16
                                        #initial_w = 0
                                        )
                                 
    
    rli.add_step_variable('reward',lambda rli: rli.reward,'f')
    rli.add_step_variable('state',lambda rli: array((rli.env.angle,rli.env.velocity)),'d',2)
    rli.add_step_variable('action',lambda rli: rli.action or 0, 'd')
    rli.add_step_variable('Q', lambda rli: rli.agent.Q(rli.agent.last_sensation,rli.agent.last_action), 'd')
    
    rli.init(agent,p)
    #rli.init(lambda *x:1.0, p)

    if gui:
        rli.gui(None,PendulumGUI)
    else:
        rli.steps(1000000)


def torquetest(pendulum,torque):

    rli = plastk.rl.LoggingRLI(name = "Torque Test, torque = "+`torque`)

    import os
    try: 
        os.remove(rli.episode_filename)
        os.remove(rli.step_filename)
    except OSError,err:
        print err

    rli.init(lambda *x:torque,pendulum)
    rli.gui(None,PendulumGUI)

    
if __name__ == '__main__':
    import pdb,getopt,sys
    try:
        opts,args = getopt.getopt(sys.argv[1:],'g')
        opts = dict(opts)
        test(gui=('-g' not in opts))
    except getopt.GetoptError,err:
        print err

