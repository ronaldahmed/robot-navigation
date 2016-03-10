"""
Growing Neural Gas class

$Id: gng.py,v 1.17 2006/03/20 21:13:24 jp Exp $
"""

from plastk.base import BaseObject
from plastk.vq import VQ
from plastk.utils import matrixnorm,norm,inf,gaussian
from Numeric import argsort,argmin,dot,sqrt,take,zeros,average,array,concatenate as join
from plastk.params import Parameter,PositiveInt,NonNegativeInt,Magnitude,Number

class GNG(VQ):

    dim  = PositiveInt(default=2)
    rmin = Number(default=0.0)
    rmax = Number(default=1.0)

    e_b = Magnitude(default=0.05)
    e_n = Magnitude(default=0.0006)
    lambda_ = PositiveInt(default=200)
    beta = Magnitude(default=0.0005)
    alpha = Magnitude(default=0.5)
    max_age = PositiveInt(default=100)
    
    response_exponent = Number(default=2)

    activation_function = Parameter(default='reciprocal')

    grow_callback = Parameter(default=None)
    shrink_callback = Parameter(default=None)

    initial_num_units = PositiveInt(default=2)
    initial_connections_per_unit = NonNegativeInt(default=0)

    normalize_error = Parameter(default=True)
    
    def __init__(self,**params):
        from plastk.rand import uniform
        from Numeric import zeros
        super(GNG,self).__init__(**params)

        N = self.initial_num_units
        
        self.weights = uniform(self.rmin,self.rmax,(N,self.dim))
        self.dists = zeros((N,1)) * 0.0
        self.error = zeros((N,1)) * 0.0

        self.connections = [{} for i in range(N)]
        
        self.last_input = zeros(self.dim)
        
        self.count = 0

        if self.initial_connections_per_unit > 0:
            for w in self.weights:
                self.present_input(w)
                ww = self.winners(self.initial_connections_per_unit+1)
                i = ww[0]
                for j in ww[1:]:
                    self.add_connection(i,j)

        self.nopickle += ['_activation_fn']
        self.unpickle()

    def unpickle(self):
        self._activation_fn = getattr(self,self.activation_function+'_activation')

        if hasattr(self,'units'):
            # if the gng has a units attrib, it's the old version,
            # so convert it to the new version.
            self.weights = array([u.weights for u in self.units])
            self.error = array([u.error for u in self.units])
            self.dists = array([u.distance for u in self.units])
            self.connections = []
            for u in self.units:
                conn_dict = {}
                for k,v in u.connections.iteritems():
                    conn_dict[self.units.index(k)] = v
                self.connections.append(conn_dict)
            del self.units

    def get_model_vector(self,i):
        return self.weights[i]
    def num_model_vectors(self):
        return len(self.weights)

    def add_connection(self,a,b):
        if b not in self.connections[a]:
            self.verbose("Adding connection between",a,"and",b)
        
        self.connections[b][a] = 0
        self.connections[a][b] = 0

        

    def del_connection(self,a,b):

        self.verbose("Deleting connection between",a,"and",b)
        
        del(self.connections[b][a])
        del(self.connections[a][b])

           
    def del_unit(self,x):

        self.verbose("Deleting unit",x)
        
        if self.shrink_callback:
            self.shrink_callback(x)

        # remove the connections for unit x
        del self.connections[x]

        # iterate through the connection dictionaries decrementing
        # all the connection numbers greater than x
        for i,conn_dict in enumerate(self.connections):
            new_dict = {}
            for k,v in conn_dict.items():
                assert x != k
                if k > x:
                    new_dict[k-1] = v
                else:
                    new_dict[k] = v
            self.verbose("old connections for unit",i,"=",conn_dict)
            self.verbose("new connections for unit",i,"=",new_dict)
            self.connections[i] = new_dict

        # set up slices for the items before and after
        # item x
        before = slice(0,x)
        after = slice(x+1,len(self.weights))
        
        # remove the weights for unit x
        self.weights = join((self.weights[before],self.weights[after]))

        # remove the error accumulator for unit x
        self.error = join((self.error[before],self.error[after]))
        
        # remove the distance value for unit x
        self.dists = join((self.dists[before],self.dists[after]))
    

    def present_input(self,X):
        self.dists = matrixnorm(self.weights-X)
        self.last_input = X
        self.new_input = True

    def get_activation(self):
        if self.new_input:
            self._activation_fn()
            self.new_input = False
        return self.__activation
    
    def reciprocal_activation(self):
        self.__activation = 1/self.dists
        
        if inf in self.__activation:
            win = self.winner()
            self.__activation.flat[win] = 0
            self.__activation -= self.__activation
            self.__activation.flat[win] = 1.0
        else:
            self.__activation /= sum(self.__activation.flat)
        return self.__activation

    def gaussian_activation(self):
        x = self.dists
        radii = zeros(self.dists.shape) * 0.0

        for u,conn_dict in enumerate(self.connections):
            neighbors = take(self.weights,conn_dict.keys())
            radii[u] = average(matrixnorm(neighbors-self.weights[u]))

        self.__activation = gaussian(x,radii/2)

    def uniform_gaussian_activation(self):
        x = self.dists

        total = 0.0
        count = 0

        for u,conn_dict in enumerate(self.connections):
            neighbors = take(self.weights,conn_dict.keys())
            total += sum(matrixnorm(neighbors-self.weights[u]))
            count += len(conn_dict)

        self.__activation = gaussian(x,(total/count)/2)

    def winner_take_all_activation(self):
        self.__activation = zeros(len(self.dists))
        self.__activation[argmin(self.dists)] = 1.0

    def dot_activation(self):
        self.__activation = dot(self.weights,self.last_input)

    def train(self,X,error=None):

        self.debug("Training on input:",X)
        self.present_input(X)
        self.count += 1
        
        # (roman numeral comments from fritzke's algorithm in
        # B. Fritzke, Unsupervised ontogenetic networks, in Handbook
        # of Neural Computation, IOP Publishing and Oxford University
        # Press, 1996)  [ replacing \zeta with X ]


        # (iii) Determine units s_1 and s_2 (s_1,s_2 \in A) such that
        #       |w_{s_1} - X| <= |w_c - X| (\forall c \in A)
        #   and
        #       |w_{s_2} - X| <= |w_c - X| (\forall c \in A\\s_1)

        s_1,s_2 = self.winners(2)

        # (iv) If it does not already exist, insert a connection between s1 and s2
	#   in any case, set the age of the connection to zero

        self.add_connection(s_1,s_2)

        # (v) Add the squared distance betwen the input signal and the
        # nearest unit in input space to a local error variable

        if error == None:
            error = self.dists[s_1]**2
            if self.normalize_error:
                error = sqrt(error)/norm(X)

        self.error[s_1] += error

        # (vi) Move s_i and its direcct topological neighbors towards
        # X by fractions e_b and e_n, respectively, of the total
        # distance.

        self.weights[s_1] += self.e_b * (X - self.weights[s_1])
        for n in self.connections[s_1]:
            self.weights[n] += self.e_n * (X - self.weights[n])

        # (vii) Increment the age of all edges emanating from s_1
        for n in self.connections[s_1]:
            self.connections[n][s_1] += 1
            self.connections[s_1][n] += 1
                                           

        # (viii) Remove edges with an age larger than max_age....  
        for a,connection_dict in enumerate(self.connections):
            for b,age in connection_dict.items():
                if age > self.max_age:
                    self.del_connection(a,b)

        # (viii) ... If this results in units having no emanating
        # edges, remove them as well.
        to_be_deleted = [a for a,d in enumerate(self.connections) if not d]
        #   sort the list in descending order, so deleting lower numbered
        #   units doesn't screw up the connections
        to_be_deleted.sort(reverse=True)
        if to_be_deleted:
            self.verbose("Deleting units",to_be_deleted)
        for a in to_be_deleted:
            self.del_unit(a)


                       
        # (ix) if the number of input signals so far is an integer
        # multiple of a parameter \lambda, insert a new unit as
        # follows.
        if self.time_to_grow():
            # o Determine the unit q with the maximum accumulated error.
            # o Interpolate a new unit r from q and its neighbor f with the largest
            #   error variable

            q,f = self.growth_pair()
            r = len(self.weights)
            
            new_weights = 0.5 * (self.weights[q] + self.weights[f])
            new_weights.shape = (1,self.dim)
            self.weights = join((self.weights,new_weights),axis=0)
            
            new_distance = norm(X-new_weights)
            self.dists = join((self.dists,new_distance),axis=0)

            self.connections.append({})

            # o Insert edges connecting the new unit r with unts q and f and
            #   remove the original edge between q and f.
            self.verbose("Adding unit",r,"between",q,"and",f,"--- count =",self.count)
            self.add_connection(q,r)
            self.add_connection(r,f)
            self.del_connection(q,f)

            # o Decrease the error variables of q and f
            self.error[q] += -self.alpha * self.error[q]
            self.error[f] += -self.alpha * self.error[f]

            # o Interpolate the error variable of r from q and f
            new_error = array(0.5 * (self.error[q] + self.error[f]))
            new_error.shape = (1,1)
            self.error = join((self.error,new_error))

            if self.grow_callback:
                self.grow_callback(q,f)

        # (x) Decrease the error variables of all units
        self.error += -self.beta * self.error
        return

    def winners(self,N=1):
        N = min(N,len(self.dists))
        indices = argsort(self.dists)
        return tuple(indices[:N])
        
    def winner(self):
        return argmin(self.dists)
    
    def time_to_grow(self):
        return (self.count%self.lambda_) == 0

    def growth_pair(self):
        def max_error(a,b):
            if self.error[a] > self.error[b]:
                return a
            else:
                return b
        q = reduce(max_error,range(len(self.error)))
        f = reduce(max_error,self.connections[q])
        return q,f

    def neighbors(self,i):
        return self.connections[i].keys()



class EquilibriumGNG(GNG):
    error_threshold = Number(default=1.0,bounds=(0,None))

    def time_to_grow(self):
        from Numeric import average,sqrt
        e = average(self.error*self.beta)[0]
        
        result = (e > self.error_threshold
                 and super(EquilibriumGNG,self).time_to_grow())
        if result:
            self.verbose("average error = %.4e"%e," -- Time to grow.")
        else:
            self.debug("average error = %.4e"%e," -- Not growing.")
            
        return result

class GNGUnit(BaseObject):
    __slots__ = ['connections',
                 'distance',
                 'error',
                 'weights']
    def __init__(self,weights=None,**params):
        super(GNGUnit,self).__init__(**params)
        self.connections = {}
        self.distance = None
        self.error = 0
        self.weights = weights
        
        
try:
    import Tkinter as Tk
    import Pmw
except ImportError:
    pass
else:
    class GNG2DDisplay(BaseObject,Tk.Frame):
        tkargs = Parameter(default={})
        update_period = NonNegativeInt(default=500)
        gng = Parameter(default=None)
        def __init__(self,root,**args):

            BaseObject.__init__(self,**args)
            Tk.Frame.__init__(self,root,**self.tkargs)
            group = Pmw.Group(self,tag_text='2D GNG Display')
            group.pack(side='top',fill='both',expand=1)

            # create a Tk variable to link the update frequency
            self.update_period_var = Tk.StringVar()
            self.update_period_var.set(str(self.update_period))            
            
            self.entry_field = Pmw.EntryField(group.interior(),
                                              label_text='Steps per update',
                                              labelpos='w',
                                              validate='numeric',
                                              entry_textvariable=self.update_period_var)
            self.entry_field.pack(side='top',fill='x')


            self.plot=Pmw.Blt.Graph(group.interior())
            self.plot.pack(side='top',expand=1,fill='both')
            self.plot.grid_on()
            
            self.messagebar = Pmw.MessageBar(group.interior(),
                                             entry_relief='groove',
                                             labelpos='w',
                                             label_text = 'Step:')
            self.messagebar.pack(side='bottom',fill='x')

            
            self.plot.line_create('units',label='',linewidth=0)

            self.num_units = 0
            self.redraw()

        def redraw(self):

            if self.entry_field.valid():
                period = int(self.update_period_var.get())
            else:
                period = self.update_period
                

            if self.gng.count % period == 0:
                self.messagebar.message('state',str(self.gng.count))

                # plot the points

                xs = self.gng.weights[:,0]
                ys = self.gng.weights[:,1]

                self.plot.element_configure('units',xdata=tuple(xs),ydata=tuple(ys))

                n = len(self.gng.dists)
                if self.num_units < n:
                    for i in range(self.num_units,n):
                        self.plot.line_create('edges %d'%i,label='',symbol='')
                elif self.num_units > n:
                    for i in range(n,self.num_units):
                        self.plot.element_delete('edges %d'%i)                
                self.num_units = n


                for i,(x1,y1) in enumerate(self.gng.weights):
                    xs,ys = [],[]
                    for j in self.gng.neighbors(i):
                        if j > i:
                            x2,y2 = self.gng.weights[j]
                            xs.extend((x1,x2))
                            ys.extend((y1,y2))

                    self.plot.element_configure('edges %d'%i,xdata=tuple(xs),ydata=tuple(ys))


if __name__ == '__main__':


    def train(N,gng=None,mu=0,stdev=5):
        from pdb import pm
        from pprint import pprint
        import plastk.rand
        import base
        if not gng:
            gng = EquilibriumGNG(lambda_ = 200,
                                 max_age = 100,
                                 error_threshold = 5.0,
                                 print_level=base.VERBOSE)
    

        for i in range(N):
            v = plastk.rand.normal(mu,stdev,2)
            gng.train(v)

        return gng
