## Automatically adapted for numpy.oldnumeric May 17, 2011 by -c

import math, random, sys
import MarkovLoc
sys.path.append('..')
from Robot import Meanings
from Utility import logger
from Topology import *

class observation:
    def __init__(self,view):
        if type(view) is tuple: self.view = [view]
        elif hasattr(view,'view'): self.view = view.view
        else: self.view = view
    def __repr__(self): return '['+', '.join([repr(v) for v in self.view])+']'
    def __str__(self): return 'observation('+self.repr(self.view)+')'
    def __len__(self): return len(self.view)
    ViewLocations = {}
    for num,viewloc in enumerate((Meanings.Left,Meanings.At,Meanings.Right,
                                  Meanings.FrontLeft,Meanings.Front,Meanings.FrontRight)):
        ViewLocations[viewloc] = num
    
    def __getitem__(self,index):
        if isinstance(index, int) or type(index) is slice:
            return observation(self.view[index])
        if type(index) is tuple:
            viewDist, viewLoc = index
            return self.view[viewDist][self.ViewLocations[viewLoc]]
        if index in self.ViewLocations:
            return self.view[0][self.ViewLocations[index]]
        try:
            return observation(self.view[index])
        except TypeError: return observation(self.view[0])
    
    def repr(cls,view):
        if not view: return ''
        if type(view) is tuple:
            left,mid,right,fwdL,fwdM,fwdR= view
            return '('+',\n\t'.join((','.join([`a`.rjust(10) for a in (left,mid,right)]),
                                    ','.join([`a`.rjust(11) for a in (fwdL,fwdM,fwdR)])))+'), '
        else: return '[\n\t'+'\n\t'.join([cls.repr(v) for v in view])+'\n\t]'
    repr = classmethod(repr)
    
    def code(self,view=None):
        if view == None: view = self.view
        if not view: return ''
        if type(view) is tuple or type(view) is list:
            return ''.join([self.code(v) for v in view])
        if isinstance(view,Meanings.Meaning):
            return view.abbr
        else: return ''.join(view)
    
    def match(self,meaning,side=Meanings.Front):
        if side == Meanings.Sides:
            return self[Meanings.Left].match(meaning) or self[Meanings.Right].match(meaning)
        return self[side].match(meaning)
    
    def search(self,meaning,side,dist=''):
        """
        >>> View = observation([(Wall, Empty, BlueTile, Fish, Honeycomb, Fish),(Brick, Easel, Brick, End, Wall, End)])
        >>> View.search(Brick,Left)
        True
        >>> View.search(BlueTile,Left)
        False
        >>> View.search(BlueTile,Right)
        True
        >>> View.search(Honeycomb,Right)
        False
        >>> View.search(Brick,Left,'0')
        False
        >>> View.search(Brick,Left,'1')
        True
        >>> View.search(Brick,Left,'0:')
        True
        >>> View.search(Brick,Left,'1:')
        True
        >>> View.search(Brick,Left,':0')
        False
        >>> View.search(Brick,Left,'2')
        False
        """
        match = False
        if not isinstance(dist,str): dist=str(dist)
        try:
            if ':' in dist: viewParts = eval('self['+dist+']') #Grab sublist
            elif dist: viewParts = [eval('self['+dist+']')] #Listify item
            else: viewParts = self
        except IndexError,e: return False
        for viewPart in viewParts:
            match = viewPart.match(meaning,side)
            if match: break
        logger.debug('observation.search(%r, %r, %r, %r) => %r',self,meaning,side,dist, match)
        return match

class POMDP(MarkovLoc.POMDP):
    ### Setup methods
    def __init__(self,name):
        MarkovLoc.POMDP.__init__(self,name)
        self.env = name
        self.ObservationGenerators['*']=self.getViews
        self.plat_swap_axes = False
        self.getDirectionVal = 'Random'
        
    def initialize(self,Start,Dest):
        self.invertGateways()
        self.generatePaths()
        self.generateTextures()
        self.generatePictures()
        self.generateTopology()
        self.setRoute(Start,Dest)

    def setRoute(self,Start,Dest):
        if type(Start) == str:
            self.DestPlace = self.Positions[int(Dest)]
            self.StartPlace = self.Positions[int(Start)]
        elif isinstance(Start, int):
            self.DestPlace = self.Positions[Dest]
            self.StartPlace = self.Positions[Start]
        elif type(Start) == tuple:
            self.DestPlace = self.coords2place(Dest)
            self.StartPlace = self.coords2place(Start)
        else:
            print 'Unknown type for Start in pomdp.setRoute', Start
        
        self.StartPose = None
        self.Actions = {
            'TurnLeft' : MarkovLoc.TurnLeft(self.NumPoses,1),
            'TurnRight' : MarkovLoc.TurnRight(self.NumPoses,1),
            'TravelFwd' : MarkovLoc.TravelFwd(self.Gateways,(1,10)),
            'DeclareGoal' : MarkovLoc.DeclareGoal((self.DestPlace,None),(-500,500)),
            }
        #self.name = 'MarkovLoc_%s_%s_%s' % (self.env,Start,Dest)
    
    PeripheralViews = [Meanings.Hall,Meanings.Wall]
    
    def getDirection(self):
        if self.getDirectionVal == 'Random':
            return random.choice(range(self.NumPoses))
        else:
            return self.getDirectionVal
    
    def set(self,pose):
        if isinstance(pose, str):
            if pose.startswith('_'):
                self.face(self.platdir2orient(pose[1:]))
                return
            pose = int(pose)
        if isinstance(pose, int):
            pose = (self.Positions[pose], self.getDirection())
        self.trueState = pose

    def face(self,direction=None):
        if not direction: direction = self.getDirection()
        self.trueState = self.trueState[0],direction
    
    ### Conversion Methods

    def coords2place(self,coords):
        return [k for k,v in self.locations.items() if v == coords][0]

    def place2coords(self,place):
        return self.locations[place]

    def coords2plat(self,row,col,dir=None):
        if self.plat_swap_axes:
            row,col = col,row
        platRow = int(math.ceil(row / 2.0)) + self.plat_row_offset
        platCol = int(math.ceil(col / 2.0)) + self.plat_col_offset
        # Flip around middle column, since cols are up to down and rows are L -> R
        if self.plat_swap_axes:
            platCol -= int(2*(platCol-self.plat_mid_col))
        if dir is None:
            return platCol,platRow
        if type(dir) is str:
            platDir = self.plat_directions[dir]
        elif isinstance(dir, int):
            for platDir,gwDir in self.plat_orientations.items():
                if gwDir==dir: break
        else:
            platDir = None
        return platCol,platRow,platDir

    #added by David Chen
    def plat2place(self,platCol, platRow):
      return [k for k,v in self.locations.items() if self.coords2plat(v[0],v[1]) == (platCol,platRow)][0]

    def platdir2orient(self,platdir):
#        if isinstance(platdir, str: platdir = int)(platdir)
        platdir = int(platdir)
        return self.plat_orientations[platdir]
        
    def place2position(self,place):
#        if isinstance(place, str: place = int)(place)
        place = int(place)
        for k,v in self.Positions.items():
            if v == place:
                return k
        return None

    def position2place(self,position):
        return self.Positions[int(position)]

    def position2plat(self,position):
        return self.coords2plat(*self.place2coords(self.position2place(position)))

    def state2plat(self,(place,gateway)):
        row,col = self.place2coords(place)
        return self.coords2plat(row,col,gateway)
    
    ### Generation Methods

    def getPeripheralView(self,(place,pose)):
        if (place, pose%self.NumPoses) in self.Gateways:
            return Meanings.Hall
        else: return Meanings.Wall

    def getLocalView(self,(place,pose)):
        """Looks up a view immediately visible from a pose.

        Agent can see the presence (#) or absence (=) of walls to the sides.
        Agent can see the color of the hallway directly in front for one unit.
        Views are of the form (Left Hall|Wall)_(Front Wall| Color)_(Right Hall|Wall)
        """
        left = self.getPeripheralView((place, pose-1))
        if place in self.ObjectLocs:
            mid = Meanings.Object.Abbrevs[self.ObjectLocs[place]]
        else: mid = Meanings.Empty
        right = self.getPeripheralView((place, pose+1))
        if (place,pose) in self.TextureLocs:
            fwd = (Meanings.Picture.Abbrevs[self.PictureLocs[(place,pose)]],
                   Meanings.Texture.Abbrevs[self.TextureLocs[(place,pose)]],
                   Meanings.Picture.Abbrevs[self.PictureLocs[(place,pose)]])
        else:
            fwd = (Meanings.End,Meanings.Wall,Meanings.End)
        return [([(left,mid,right)+fwd],1.0),]

    def getView(self,(place,pose)):
        Views = self.getLocalView((place,pose))
        while 1:
            try:
                (place,pose) = self.Gateways[(place,pose)]
                NewViews=[]
                for nextView,nextViewProb in self.getLocalView((place,pose)):
                    NewViews.extend([(view+nextView,prob*nextViewProb) for (view,prob) in Views])
                Views = NewViews
            except KeyError:
                break
        return Views

    def getViews(self,state):
        return [(observation(view),prob) for (view,prob) in self.getView(state)]
    
    def generateObservations(self):
        """Generates the set of observations.
        """
        for pl_num in xrange(self.NumPlaces):
            place = pl_num + 1
            for pose in xrange(self.NumPoses):
                for view,prob in self.getViews((place,pose)):
                    yield view

    def generatePaths(self):
        for name,pathList in self.PathSpecs.items():
            revName = name[:-1]+'-'
            gwList = [(pl,self.oppositeGW(pl,gw)) for (pl,gw) in pathList]
            gwList.reverse()
            self.PathSpecs[revName] = gwList

    def generateTextures(self):
        for (pl,gw),view in self.TextureLocs.items():
            pl,gw = self.Gateways[(pl,gw)]
            gw = self.oppositeGW(pl,gw)
            self.TextureLocs[(pl,gw)] = view
        self.ForwardViews = [Meanings.Wall]+Meanings.Texture.Names.values()

    def generatePictures(self):
        for (pl,gw),view in self.PictureLocs.items():
            pl,gw = self.Gateways[(pl,gw)]
            gw = self.oppositeGW(pl,gw)
            self.PictureLocs[(pl,gw)] = view

    ### Topology

    def generateTopology(self):
#        Topology.__debug__ = 0
        TopoMap = Topology()
        PathIndex = {}
        TopoMap.place = {}
        for pathdir,pathList in self.PathSpecs.items(): # On, Order, and Terminates
            for GW in pathList: PathIndex[GW] = pathdir
            paName = pathdir[:-1]; paDir = pathdir[-1]
            if paDir == '-': continue
            path = Path(TopoMap,{'path':paName})
            lastPlace=None
            for (pl,gw) in pathList: # for pose along path
                plName = str(pl)
                place = TopoMap.place.setdefault(plName, Place(TopoMap,{'place':plName}))
                OnRelation(TopoMap,{'path':paName, 'thing':place.latex_str(), 'place':plName})
                if lastPlace is None:
                    TerminatesRelation(TopoMap,{'pathdir':paName+'-', 'place':plName})
                else:
                    OrderRelation(TopoMap,{'pathdir':pathdir,
                                           'thing':lastPlace.latex_str(), 'place':str(lastPlace),
                                           'thing2':place.latex_str(), 'place2':plName})
                lastPlace=place
            TerminatesRelation(TopoMap,{'pathdir':pathdir, 'place':plName})
        for Locations, Names in zip([self.TextureLocs, self.PictureLocs],
                                    [Meanings.Texture.Abbrevs, Meanings.Picture.Abbrevs]):
            for (pl,gw), appearance in Locations.items():
                AppearRelation(TopoMap,{'pathdir':PathIndex[(pl,gw)], 'place':plName, 'appearance':Names[appearance]})
        for pl,obj in self.ObjectLocs.items():
            AtRelation(TopoMap,{'place':str(pl), 'object':Meanings.Object.Abbrevs[obj]})
        for (pl,gw),pathdir in PathIndex.items():
            plName = str(pl)
            paName = pathdir[:-1]; paDir = pathdir[-1]
            for dir, nextGW in zip(('Right', 'Left'), (gw+1, gw-1)):
                pose = (pl,(nextGW)%self.NumPoses)
                if pose in PathIndex:
                    SideOfRelation(TopoMap,{'sideof':dir, 'place':plName, 'pathdir':pathdir, 'pathdir2':PathIndex[pose]})
        self.TopoMap = TopoMap

    def listTopology(self): 
        for k,v in self.TopoMap.__dict__.items():
            for key,topo in  v.items(): print topo.latex_repr()
