"""
SOM and NoveltySOM classes

$Id;$
"""
from Numeric import *

from plastk.vq import VQ
from plastk.base import BaseObject
from plastk.utils import norm,inf,gaussian,decay
from plastk.params import Number,Magnitude,PositiveInt,NonNegativeInt
import plastk.rand

class SOM(VQ):

    dim = NonNegativeInt(default=2)
    xdim = NonNegativeInt(default=1)
    ydim = NonNegativeInt(default=1)

    rmin = Number(default=0.0)
    rmax = Number(default=1.0)
    
    alpha_0 = Magnitude(default=0.5)
    radius_0 = Number(default=1.0,bounds=(0.0,None))
    
    response_exponent = Number(default=2)
    
    def __init__(self,**params):

        BaseObject.__init__(self,**params)

        self.weights = plastk.rand.uniform(self.rmin,self.rmax,
                                           (self.ydim,self.xdim,self.dim))

        self.activation = zeros( (self.ydim,self.xdim), 'f')
        self.count = 0

    ###########################################
    def init_training(self,alpha_0=None,
                      radius_0=None,
                      training_length=None):

        self.count = 0

        if alpha_0:
            self.alpha_0 = alpha_0
        if radius_0:
            self.radius_0 = radius_0

        self.half_life = training_length/8

    def alpha(self):
        return self.alpha_0 * decay(self.count,self.half_life)
    def radius(self):
        return self.radius_0 * decay(self.count,self.half_life)


    ##########################################
    def present_input(self,X):

        for y in range(self.ydim):
            for x in range(self.xdim):                
                self.activation[y,x] = norm(X - self.weights[y,x])**self.response_exponent

        self.activation = 1/self.activation
        
        if inf in self.activation:
            win = self.winner()
            self.activation.flat[win] = 0
            self.activation -= self.activation
            self.activation.flat[win] = 1.0
        else:
            self.activation /= sum(self.activation.flat)


    def train(self,X):

        self.present_input(X)

        wy,wx = self.winner_coords()

        self.debug("Winner coords = "+`(wy,wx)`)

        int_radius = floor(self.radius())

        self.debug("Training radius = %.2f" % self.radius())

        ymin = max(0,wy-int_radius)
        ymax = min(wy+int_radius,self.ydim-1)
        xmin = max(0,wx-int_radius)
        xmax = min(wx+int_radius,self.xdim-1)

        self.debug('y range = '+`(ymin,ymax)`)
        self.debug('x range = '+`(xmin,xmax)`)

        for y in range(ymin,ymax+1):
            for x in range(xmin,xmax+1):
                lattice_dist = sqrt((wx-x)**2 + (wy-y)**2)
                self.debug("Trying cell %d,%d"%(x,y))
                if  lattice_dist <= self.radius():
                    self.debug("Training cell %d,%d"%(x,y))
                    rate = self.alpha() * gaussian(lattice_dist,self.radius())
                    self.weights[y,x] += rate * (X - self.weights[y,x])
                                   
        self.count += 1 

    def train_batch(self,data,epochs):

        self.init_training(training_length=len(data)*epochs)

        for i in xrange(epochs):
            self.message("Starting epoch",i)
            for x in plastk.rand.shuffle(data):
                self.train(x)
            
    def winner(self):
        return argmax(self.activation.flat)

    def winner_coords(self):
        pos = argmax(self.activation.flat)
        return (pos/self.ydim, pos%self.xdim)

    def get_model_vector(self,index):
        if type(index) == int:
            y = index/self.ydim
            x = index%self.xdim
        else:
            # assume it's a tuple
            x,y = index
        return self.weights[y,x]

    def num_model_vectors(self):
        return len(self.activation.flat)
    def get_activation(self):
        return self.activation.flat


class NoveltySOM(SOM):

    alpha_gain =  Number(default=2.0,bounds=(0.0,None))
    radius_gain = Number(default=2.0,bounds=(0.0,None))


    def __init__(self,**params):

        SOM.__init__(self,**params)
        self.error_ratio = 1.0

    def present_input(self,X):
        
        SOM.present_input(self,X)        
        dist = norm( self.get_model_vector(self.winner()) - X )
        self.error_ratio = dist / norm(X)
    
    def alpha(self):
        return SOM.alpha(self) * tanh(self.error_ratio * self.alpha_gain)

    def radius(self):
        return SOM.radius(self) * tanh(self.error_ratio * self.radius_gain)



try:
    import Tkinter as Tk
    import Pmw
except ImportError:
    pass
else:
    class SOM2DDisplay(Tk.Frame):

        def __init__(self,root,som,**tkargs):

            self.som = som

            Tk.Frame.__init__(self,root,**tkargs)
            group = Pmw.Group(self,tag_text='2D SOM Display')
            group.pack(side='top',fill='both',expand=1)

            # create a Tk variable to link the update frequency
            self.update_period = Tk.StringVar()
            self.update_period.set('500')

            Pmw.EntryField(group.interior(),
                           label_text='Steps per update',
                           labelpos='w',
                           validate='numeric',
                           entry_textvariable=self.update_period).pack(side='top',fill='x')


            self.plot=Pmw.Blt.Graph(group.interior())
            self.plot.pack(side='top',expand=1,fill='both')
            self.plot.grid_on()

            self.messagebar = Pmw.MessageBar(group.interior(),
                                             entry_relief='groove',
                                             labelpos='w',
                                             label_text = 'Step:')
            self.messagebar.pack(side='bottom',fill='x')


            self.plot.line_create('units',label='',linewidth=0)
            for i in range(self.som.ydim):
                self.plot.line_create('row %d'%i,label='',symbol='')
            for i in range(self.som.xdim):
                self.plot.line_create('col %d'%i,label='',symbol='')            

            self.redraw()

        def redraw(self):

            period = int(self.update_period.get())

            if self.som.count % period == 0:
                self.messagebar.message('state',str(self.som.count))
            
                xs = self.som.weights[:,:,0]
                ys = self.som.weights[:,:,1]

                self.plot.element_configure('units',
                                            xdata=tuple(ravel(xs)),
                                            ydata=tuple(ravel(ys)))

                for i in range(self.som.ydim):
                    self.plot.element_configure('row %d'%i,
                                                xdata=tuple(ravel(xs[i])),
                                                ydata=tuple(ravel(ys[i])))

                for i in range(self.som.xdim):
                    self.plot.element_configure('col %d'%i,
                                                xdata=tuple(ravel(xs[:,i])),
                                                ydata=tuple(ravel(ys[:,i])))


if __name__ == '__main__':

    som = SOM(xdim=5,ydim=5)
    som.init_training(alpha_0=0.5, radius_0=5.0, training_length=1000)

    
    nsom = NoveltySOM(xdim=5,ydim=5)
    nsom.init_training(alpha_0=0.5, radius_0=5.0, training_length=1000)
