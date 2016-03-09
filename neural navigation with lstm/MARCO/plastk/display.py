# $Id: display.py,v 1.1 2004/08/14 22:18:28 jp Exp $

from __future__ import generators
from Tkinter import *
from math import pi,sin,cos
from base import BaseObject

RADS_PER_DEG = pi/180
DEGS_PER_RAD = 180/pi

def pol2cart(r,theta):
    x = r * cos(theta)
    y = r * sin(theta)

    return x,y

def enum(seq):
    i = 0
    for x in seq:
        yield (i,x)
        i += 1

class LRFDisplay(Canvas,BaseObject):

    robot_radius = 0.2

    lrf_shape = (10.0,10.0)
    lrf_resolution = 1.0
    lrf_start_angle = -90
    point_width = 1.0
    
    def __init__(self,parent,**config):

        Canvas.__init__(self,parent,**config)

        self.channels = {None:([],{})}

        self.bind('Configure',self.reconfigure())
        self.canvas_shape = self.winfo_reqwidth(),self.winfo_reqheight()

        lw,lh = self.lrf_shape
        self.lrf_origin = (lw/2.0,self.robot_radius)
        self.draw_origin()

    def draw_origin(self):
        l,t = self.lrf2canvas(-self.robot_radius,self.robot_radius)
        r,b = self.lrf2canvas(self.robot_radius,-self.robot_radius)
        xo,yo = self.lrf2canvas(0,0)
        circle = self.create_oval(l,t,r,b,fill='red')
        line = self.create_line(xo,yo,xo,t)
        self.lift(line,circle)


    def reconfigure(self):
        self.canvas_shape = (self.winfo_width(),self.winfo_height())

        
    def draw_scan(self,scan,channel=None):

        if channel not in self.channels:
            self.init_channel(channel,fill='black',outline='black')

        points,config = self.channels[channel]
    
        if len(points) != len(scan):
            self.init_points(len(scan),channel=channel)
            points = self.channels[channel][0]
            
        theta = self.lrf_start_angle
        for i,r in enum(scan):
            x,y = pol2cart(r,(theta+90)*RADS_PER_DEG)
            x,y = self.lrf2canvas(x,y)
            left,top,right,bottom = self.coords(points[i])
            x_old = left+(right-left)
            y_old = top+(bottom-top)
            x_move = x - x_old
            y_move = y - y_old
            self.move(points[i],x_move,y_move)
            theta += self.lrf_resolution

    def draw_weights(self,w):
        self.draw_scan(w)

    def init_points(self,n,channel=None):
        points,config = self.channels[channel]
        points += [self.create_rectangle(0,0,self.point_width,self.point_width,**config)
                   for i in range(len(points),n)]
        
    def init_channel(self,channel=None,**config):
        self.channels[channel] = ([],config)
    
        
    def lrf2canvas(self,x_in,y_in):
        lrf_width,lrf_height = self.lrf_shape
        xo,yo = self.lrf_origin
        width,height = self.canvas_shape

        x_out = (x_in+xo) * (width/lrf_width)
        y_out = (y_in-(lrf_height-yo)) * -(height/lrf_height)

        return x_out,y_out

class SOMDisplay(Frame):

    lrf_params = {'bg'    : 'white',
                  'width' : 200,
                  'height': 100}
                  
    def __init__(self,parent,rows=6,cols=6,lrf_params={},**config):

        Frame.__init__(self,parent,**config)
        self.canvases = [[LRFDisplay(self,**self.lrf_params) for i in range(cols)] for j in range(rows)]
        for row in range(rows):
            for col in range(cols):
                self.canvases[row][col].grid(row=row,column=col)

    def postscript(self,filestem):
        i = 0
        for row in range(rows):
            for col in range(cols):
                self.canvases[row][col].postscript(filestem+'unit%0.3d'%i)
                i+=1

class SOMWindow(Toplevel):

    def __init__(self,root,som,**kw):

        Toplevel.__init__(self,root,**kw)
        self.title('SOM View')

        self.som = som
        
        Button(self,text="Refresh",command=self.redraw_som).pack()
        self.som_display = SOMDisplay(self,
                                      rows=self.som.ydim,
                                      cols=self.som.xdim)
        self.som_display.pack()

       
    def redraw_som(self):
        
        for y in range(self.som.ydim):
            for x in range(self.som.xdim):
                w = self.som.get_model_vector((x,y))
                self.som_display.canvases[x][y].draw_weights(w)
                


if __name__ == '__main__':

    root = Tk()
    LRFDisplay.lrf_shape = (20.0,10.0)
    lrf = LRFDisplay(root,bg='white',width=200,height=100)
    lrf.pack()

    lrf.draw_weights([5 for i in range(180)])


    som_window = Toplevel(root)
    som_disp = SOMDisplay(som_window,rows=4,cols=4)
    som_disp.pack()

#    root.mainloop()
