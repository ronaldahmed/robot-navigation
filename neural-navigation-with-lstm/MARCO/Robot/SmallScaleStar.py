## Automatically adapted for numpy.oldnumeric May 17, 2011 by -c

import copy, math
import ViewCache
import POMDP.MarkovLoc_Antie
from Meanings import Back, Front, Left, Right, At, Side, Path, Flooring, Wall, Open, End, Unknown, opposite
from Utility import reprVisible, strVisible, uniq

# from HSSH/src/common/local_topology_data_types.hh

class gateway_type(object):
    """
    float x,y; //!< midpoint of gateway
    float radius; //!< estimated extent of gateway
    float leftx,lefty;
    float rightx,righty;
    
    //! heading on gateway's in direction.
    //! out_direction==heading+(2*PI)
    float heading;
    
    typedef enum {OneWalled, TwoWalled} gwtype; //Eventually add Doorway, Elevator, Ramp...?
    gwtype type;
    """
    
    OneWalled = 'OneWalled'
    TwoWalled = 'TwoWalled'
    
    __visible__ = ('x','y','radius','heading','gwtype')
    def __init__(self, x=0, y=0, radius=0, heading = 0.0, gwtype = None):
        self.x = x
        self.y = y
        self.radius = radius
        self.heading = heading
        
        self.leftx =  x + math.cos(heading) * radius/2
        self.lefty =  y + math.sin(heading) * radius/2
        self.rightx = x - math.cos(heading) * radius/2
        self.righty = y - math.sin(heading) * radius/2
        
        self.gwtype = gwtype or self.TwoWalled
    
    __repr__ = reprVisible
    __str__ = strVisible

class small_scale_star_tuple:
    """
    int gateway_index or Wall or Open
    int path_fragment_index
    bool direction_positive // path direction
    """
    
    __visible__ = ('gateway_index', 'path_fragment_index', 'direction_positive')
    def __init__(self, gateway_index=0, path_fragment_index=0, direction_positive=True):
        self.gateway_index = gateway_index
        self.path_fragment_index = path_fragment_index
        self.direction_positive = bool(direction_positive)
    
    __repr__ = reprVisible
    __str__ = strVisible
    
    def match(self,meaning):
        """
        >>> sss=SmallScaleStar([(0,-1),(-1,0),(0,1)],range(2),[(0,0,True),(1,1,True),(Wall,0,False),(2,1,False)])
        >>> for side in (Front,Right,Back,Left): print `side`,sss[side],'Path =',sss[side].match(Path),'Wall =',sss[side].match(Wall)
        ... 
        Front small_scale_star_tuple(gateway_index=Wall, path_fragment_index=0, direction_positive=False ) Path = False Wall = True
        Right small_scale_star_tuple(gateway_index=2, path_fragment_index=1, direction_positive=False ) Path = True Wall = False
        Back small_scale_star_tuple(gateway_index=0, path_fragment_index=0, direction_positive=True ) Path = True Wall = False
        Left small_scale_star_tuple(gateway_index=1, path_fragment_index=1, direction_positive=True ) Path = True Wall = False
        """
        if meaning == Path: return isinstance(self.gateway_index,int)
        if meaning == Wall: return self.gateway_index == Wall
        if meaning == Open: return self.gateway_index == Open
        raise ValueError, 'Unknown meaning %s' % (meaning)

class SmallScaleStar(object):
    """
    Model of the topology of a place: gateways and path fragments.
    Contains a circular list of (gateway, path_fragment, path_direction) tuples
    
    gateway_vec gateways
    vector<int> path_fragments
    int forward_path_fragment_direction

    vector<small_scale_star_tuple> table // First item in vector is entry gateway
    
    >>> SmallScaleStar([(0,-1),(-1,0),(0,1)],range(2),[(0,0,True),(1,1,True),(Wall,0,False),(2,1,False)])
    SmallScaleStar( *([gateway_type( *(0, -1, 0, 0.0, 'TwoWalled') ), gateway_type( *(-1, 0, 0, 0.0, 'TwoWalled') ), gateway_type( *(0, 1, 0, 0.0, 'TwoWalled') )], [0, 1], [small_scale_star_tuple( *(0, 0, True) ), small_scale_star_tuple( *(1, 1, True) ), small_scale_star_tuple( *(Wall, 0, False) ), small_scale_star_tuple( *(2, 1, False) )], 2) )
    """
    
    __visible__ = ('gateways', 'path_fragments', 'table', 'forward')
    def __init__(self, gateways=[], path_fragments=[], table=[], forward=None):
        
        if gateways and isinstance(gateways[0],gateway_type): self.gateways = gateways
        else: self.gateways = [gateway_type(*t) for t in gateways]
        
        self.path_fragments = path_fragments
        
        if table and isinstance(table[0],small_scale_star_tuple): self.table = table
        else: self.table = [small_scale_star_tuple(*t) for t in table]
        
        if forward != None: self.forward = forward
        else: self.forward = self.getOppositeIndex(0)
    
    __repr__ = reprVisible
    __str__ = strVisible
    
    directions = {Left: -1,
                  Right: 1,
                  Front: 0}
    
    def __getitem__(self,ssst_index):
        if ssst_index in self.directions:
            ssst_index = self.forward+self.directions[ssst_index]
        if ssst_index == Back:
            ssst_index = self.getOppositeIndex(self.forward)
        if ssst_index is None:
            print self, self.table, ssst_index
        return self.table[ssst_index]
    
    def find_gateway_pose(self,gw_index=0):
        """
        >>> sss=SmallScaleStar([(0,-1),(-1,0),(0,1)],range(2),[(0,0,True),(1,1,True),(Wall,0,False),(2,1,False)])
        >>> sss.find_gateway_pose(2)
        (0, 1)
        >>> sss.find_gateway_pose(3)
        Traceback (most recent call last):
          File "<stdin>", line 1, in ?
          File "/usr/tmp/python-vGDD5H.py", line 107, in find_gateway_pose
        IndexError: unknown gateway index 3
        """
        if (gw_index >= len(self.gateways)):
          raise IndexError, "unknown gateway index %r" % (gw_index)
        gw = self.gateways[gw_index] # gateway_type
        return (gw.x,gw.y) # posetype
    
    def getOpposite(self,gw_index):
        """
        >>> sss=SmallScaleStar([(0,-1),(-1,0),(0,1)],range(2),[(0,0,True),(1,1,True),(Wall,0,False),(2,1,False)])
        >>> for gw in range(3): print gw,sss.getOpposite(gw)
        ... 
        0 small_scale_star_tuple(gateway_index=Wall, path_fragment_index=0, direction_positive=False )
        1 small_scale_star_tuple(gateway_index=2, path_fragment_index=1, direction_positive=False )
        2 small_scale_star_tuple(gateway_index=1, path_fragment_index=1, direction_positive=True )
        >>> sss.getOpposite(3)
        Traceback (most recent call last):
            ...
        IndexError: unknown gateway index 3
        """
        reference = None
        # Find the small_scale_star_tuple associated with the gateway
        for ssst in self.table:
            if ssst.gateway_index == gw_index:
                reference = ssst
        if reference == None:
            raise IndexError, "unknown gateway index %r" % (gw_index)
        
        # Find the small_scale_star_tuple on the same path_fragment
        for ssst in self.table:
            if (ssst.path_fragment_index == reference.path_fragment_index
                and ssst.direction_positive != reference.direction_positive):
                return ssst
        return None
    
    def getOppositeIndex(self,ssst_index=0):
        if ssst_index >= len(self.table): return None
        this_ssst = self.table[ssst_index]
        for i,that_ssst in enumerate(self.table):
            if (this_ssst.path_fragment_index == that_ssst.path_fragment_index
                and this_ssst.direction_positive != that_ssst.direction_positive):
                return i
        return None
    
    def countPathFragments(self):
        return len(self.path_fragments)
    
    def countGateways(self):
        return len(uniq([ssst.gateway_index for ssst in self.table if ssst.match(Path)]))
    
    def rotate(self,direction):
        """
        >>> sss=SmallScaleStar([(0,-1),(-1,0),(0,1)],range(2),[(0,0,True),(1,1,True),(Wall,0,False),(2,1,False)])
        >>> sss.forward
        2
        >>> sss.rotate(Left); print sss.forward
        1
        >>> sss.rotate(Left); print sss.forward
        0
        >>> sss.rotate(Right); print sss.forward
        1
        >>> sss.rotate(Back); print sss.forward
        3
        >>> sss.rotate(Front); print sss.forward
        3
        >>> sss.rotate(At); print sss.forward
        Traceback (most recent call last):
            ...
        ValueError: Unknown direction At
        """ 
        if self.forward == None: return False
        if direction in self.directions: self.forward += self.directions[direction]
        elif direction == Back: self.forward = self.getOppositeIndex(self.forward)
        else: raise ValueError, 'Unknown direction %r' % (direction)
        if self.forward and len(self.table): self.forward %= len(self.table)
        return self
    
    def match(self,meaning,side=Front):
        if self.forward == None: return False
        if side in self.directions: index = self.forward + self.directions[side]
        elif side == Back: index = self.getOppositeIndex(self.forward)
        else: raise ValueError, 'Unknown side %r' % (side)
        return self.table[index].match(meaning)
    
    def convert(cls,obs,back,forward=1):
        """
        >>> for obs,back,name in ((observation([(Wall, Empty, Honeycomb, End, Wall, End)]),Wall, 'Dead End'),
        ...      (observation([(Wall, Hatrack, Wall, Fish, Wood, Fish)]),Wood, 'No Topological Place'),
        ...      (observation([(Wall, Empty, Cement, Butterfly, Brick, Butterfly)]),Wall, 'Corner'),
        ...      (observation([(BlueTile, Empty, Wall, Eiffel, Cement, Eiffel)]),Cement, 'T'),
        ...      (observation([(Grass, Empty, Grass, Eiffel, Brick, Eiffel)]),Brick, '4-way Intersection')):
        ...      sss = SmallScaleStar.convert(obs,back)
        ...      print name,'Gateways:',sss.countGateways(),'PathFragments:',sss.countPathFragments()
        ...      print sss
        ...
        Dead End Gateways: 1 PathFragments: 1
        SmallScaleStar(gateways=[gateway_type( *(1, 0, 0, 0.0, 'TwoWalled') )], path_fragments=[0], table=[small_scale_star_tuple( *(Wall, 0, True) ), small_scale_star_tuple( *(Wall, 1, True) ), small_scale_star_tuple( *(0, 0, False) ), small_scale_star_tuple( *(Wall, 1, False) )], forward=2 )
        No Topological Place Gateways: 2 PathFragments: 1
        SmallScaleStar(gateways=[gateway_type( *(0, 1, 0, 0.0, 'TwoWalled') ), gateway_type( *(0, -1, 0, 0.0, 'TwoWalled') )], path_fragments=[1], table=[small_scale_star_tuple( *(Wall, 0, True) ), small_scale_star_tuple( *(0, 1, True) ), small_scale_star_tuple( *(Wall, 0, False) ), small_scale_star_tuple( *(1, 1, False) )], forward=None )
        Corner Gateways: 2 PathFragments: 2
        SmallScaleStar(gateways=[gateway_type( *(0, 1, 0, 0.0, 'TwoWalled') ), gateway_type( *(1, 0, 0, 0.0, 'TwoWalled') )], path_fragments=[0, 1], table=[small_scale_star_tuple( *(Wall, 0, True) ), small_scale_star_tuple( *(0, 1, True) ), small_scale_star_tuple( *(1, 0, False) ), small_scale_star_tuple( *(Wall, 1, False) )], forward=2 )
        T Gateways: 3 PathFragments: 2
        SmallScaleStar(gateways=[gateway_type( *(-1, 0, 0, 0.0, 'TwoWalled') ), gateway_type( *(0, 1, 0, 0.0, 'TwoWalled') ), gateway_type( *(0, -1, 0, 0.0, 'TwoWalled') )], path_fragments=[0, 1], table=[small_scale_star_tuple( *(0, 0, True) ), small_scale_star_tuple( *(1, 1, True) ), small_scale_star_tuple( *(Wall, 0, False) ), small_scale_star_tuple( *(2, 1, False) )], forward=2 )
        4-way Intersection Gateways: 4 PathFragments: 2
        SmallScaleStar(gateways=[gateway_type( *(-1, 0, 0, 0.0, 'TwoWalled') ), gateway_type( *(0, 1, 0, 0.0, 'TwoWalled') ), gateway_type( *(1, 0, 0, 0.0, 'TwoWalled') ), gateway_type( *(0, -1, 0, 0.0, 'TwoWalled') )], path_fragments=[0, 1], table=[small_scale_star_tuple( *(0, 0, True) ), small_scale_star_tuple( *(1, 1, True) ), small_scale_star_tuple( *(2, 0, False) ), small_scale_star_tuple( *(3, 1, False) )], forward=2 )
        """
        if isinstance(obs,POMDP.MarkovLoc_Antie.observation):
            left,at,right,front_left,front,front_right = obs.view[0]
            gateways = []
            table = []
            for side,coords,path,dir in zip((left,front,right,back),
                                            ((-1,0),(0,1),(1,0),(0,-1)), #Canonical positions
                                            (0,1,0,1),
                                            (True,True,False,False)):
                if side.match(Flooring):
                    gateways.append(coords)
                    gateway_index = len(gateways)-1
                else:
                    gateway_index = side
                table.append((gateway_index,path,dir))
            return cls(gateways=gateways, table=table, forward=forward,
                       path_fragments=uniq([ssst[1] for ssst in table if ssst[0] != Wall]), )
        else: raise ValueError, 'Cannot convert %r to '+cls.__name__ % (obs)
    convert = classmethod(convert)

class ssst_observation(POMDP.MarkovLoc_Antie.observation):
    def __init__(self,view):
        if isinstance(view,SmallScaleStar): self.view = [view]
        elif isinstance(view,ssst_observation): self.view = view.view
        else: self.view = view
    
    def __getitem__(self,index):
        if isinstance(index,Side): return self.view[0][index]
        if isinstance(index,tuple):
            viewDist, viewLoc = index
            return self.view[0][viewDist]
        return ssst_observation(self.view[index])
    
    def repr(cls,view):
        if not view: return ''
        return '[\n\t'+'\n\t'.join([cls.repr(v) for v in view])+'\n\t]'
    repr = classmethod(repr)

class SmallScaleStarViewCache:
    """Local Topological Views
    >>> sssViewCache = SmallScaleStarViewCache(obs=observation([(Cement, Sofa, Wall, Butterfly, BlueTile, Butterfly), (Cement, Hatrack, Wall, Butterfly, BlueTile, Butterfly), (Cement, Empty, Wall, End, Wall, End)]))
    >>> sssViewCache.update(Right,observation([(BlueTile, Hatrack, BlueTile, Butterfly, Cement, Butterfly), (Wall, Empty, Cement, End, Wall, End)]))
    >>> print sssViewCache
    SmallScaleStarViewCache(local=SmallScaleStar( *([gateway_type( *(-1, 0, 0, 0.0, 'TwoWalled') ), gateway_type( *(0, 1, 0, 0.0, 'TwoWalled') ), gateway_type( *(1, 0, 0, 0.0, 'TwoWalled') ), gateway_type( *(0, -1, 0, 0.0, 'TwoWalled') )], [0, 1], [small_scale_star_tuple( *(0, 0, True) ), small_scale_star_tuple( *(1, 1, True) ), small_scale_star_tuple( *(2, 0, False) ), small_scale_star_tuple( *(3, 1, False) )], 1) ), cache={0: [], 1: [], 2: [], 3: []} )
    """
    __visible__ = ('local','cache')
    def __init__(self,local=None,cache=None,obs=None):
        # cache is the views out of each gateway
        self.cache = {}
        self.viewCache = ViewCache.ViewCache()
        self.reset(local,cache,obs)
    def reset(self,local=None,cache=None,obs=None):
        for key in self.cache.keys(): del self.cache[key]
        if local: self.local = local
        elif obs:
            self.viewCache.reset()
            self.update(At,obs)
        else: self.local = SmallScaleStar()
        if cache: self.cache = cache
        else:
            for index,gateway in enumerate(self.local.gateways): self.cache[index] = []# Path
    __repr__ = reprVisible
    __str__ = strVisible
    def __contains__(self,item): return self.cache.__contains__(item)
    def __delitem__(self,item): return self.cache.__delitem__(item)
    def __getitem__(self,index):
        if index == 0: return self.local
        if isinstance(index,Side):
            tmp=copy.deepcopy(self.local)
            tmp.rotate(index)
            if self.local[index].match(Path):
                view = self.cache[self.local[index].gateway_index]
            else: view = []
            return ssst_observation([tmp]+view)
        if isinstance(index, int): index -=1
        ssst = self.local[index]
        if ssst.match(Path):
            return self.cache[ssst.gateway_index]
        else: return [ssst.gateway_index]
    def __iter__(self): return self.cache.__iter__()
    def __len__(self): return self.cache.__len__()
    def __setitem__(self,item,value): return self.cache.__setitem__(item,value)
    def rotate(self,direction):
        self.local.rotate(direction)
        return self
    def update(self,direction,obs):
        if type(obs) == tuple or isinstance(obs,SmallScaleStar):
            obs = SmallScaleStar(obs)
            if direction in (Left,Right,At,Back):
                self.rotate(direction)
            elif direction in (Front):
                self = self.project()
            self.local = obs[0]
            if self.local[Front].gateway_index in self.local.gateways:
                self.cache[self.local[Front].gateway_index] = obs[1:]# or [Path]
        if isinstance(obs,POMDP.MarkovLoc_Antie.observation) or isinstance(obs,list):
            self.viewCache.update(direction,obs)
            self.reset(local=SmallScaleStar.convert(self.viewCache[Front][0],self.viewCache[Back][0,Front],1))
    def project(self,dist=1): return SmallScaleStarViewCache(self[dist])
    def lookToSide(self,desc,side,recFn):
        tmpVC = copy.deepcopy(self)
        try: tmpVC.rotate(side)
        except KeyError: raise ValueError('Unknown side %r', side)
        tmpDesc = copy.deepcopy(desc)
        tmpDesc.side = [desc.value.ViewPosition]
        return recFn(tmpDesc,tmpVC)
    def search(self,meaning,side,dist=''):
        try:
            view = [self.local] + self.cache[self.local.forward]
            if ':' in dist: viewParts = eval('view['+dist+']') #Grab sublist
            elif dist: viewParts = [eval('view['+dist+']')] #Listify item
            else: viewParts = view
        except IndexError,e: return False
        for sss in viewParts:
            if sss.match(meaning,side): return True
        return False

def _test(verbose=False):
    import doctest
    doctest.testmod(verbose=verbose)
