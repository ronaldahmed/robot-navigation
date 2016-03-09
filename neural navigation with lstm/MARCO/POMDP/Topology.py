## Automatically adapted for numpy.oldnumeric May 17, 2011 by -c

import re,string,os,sys

debug = False

class Topology(dict):
    '''Database modeling a topological map.'''
    
    Line = {}
    Content = {}
    RouteInstructions = []
    TopoAttrib = []
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
    
    def add(self,topo):
        #if __debug__: print 'Topology add',topo.name,topo
        if not self.__dict__.has_key(topo.name):
            self.__dict__[topo.name] = {}
            self.TopoAttrib.append(topo.name)
        self.__dict__[topo.name][topo.id] = topo
    
    def pprint(self):
        for k,v in self.__dict__.items():
            if type(v)==dict:
                print k,':\n\t', '\n\t'.join([str(k1)+'::'+str(v1) for k1,v1 in v.items()])
            else: print k,':',v
        for ri in self.RouteInstructions: print repr(ri)
    
    def reset(self):
        L=self.Line
        C=self.Content
        for k in self.TopoAttrib:
            del self.__dict__[k]
        self.TopoAttrib=[]
        self.RouteInstructions=[]
        self.Line=L
        self.Content=C
    
    def get_str_fn(self,name,value,fn='__str__'):
        if name[-1] in '0123456789': name = name[:-1]
        if self.__dict__.has_key(name):
            return eval('self.__dict__[name][value].'+fn+'()')
        else: return value

    def get_str(self,name,value):
        return self.get_str_fn(name,value,fn='__str__')

    def get_latex_str(self,name,value):
        return self.get_str_fn(name,value,fn='latex_str')

    def __getitem__(self,item): return self.__dict__[item]

TopoMap = Topology()

class Topological(object):
    '''Interface for topological entities.'''
    
    topology = None
    latex_markup=None
    latex_content=''
    id = ''
    
    def makeNamedPattern(name,regexp):
        return ''.join(['(?P<',name,'>',regexp,')'])
    makeNamedPattern = staticmethod(makeNamedPattern)
    
    evidenceRegexp=r'\s*\\ev\{(?P<evidence>[0-9,\*]*)\}\{'
    commaSpaceRegexp=',[\s~\\;\\!]*'
    
    def markupContent(Markup,Content): return ''.join([Markup,'\{',Content,'\}'])
    markupContent = staticmethod(markupContent)
    
    def makeLinePatt(cls,Name,LineID,Patt=None):
        if not Patt: Patt = Name
        return re.compile('(?P<'+LineID+'>'+cls.evidenceRegexp+Patt+'\(.*)')
    makeLinePatt = classmethod(makeLinePatt)
    
    def makeItemPatt(cls,Name,Content):
        return re.compile(cls.evidenceRegexp+Name+'\('+Content+'\)\}')
    makeItemPatt = classmethod(makeItemPatt)
    
    def initLineParser(cls,topology):
        cls.lineID = cls.name+'Line'
        cls.topology = topology
        #if __debug__: print cls.name,'initLineParser',cls.lineID
        cls.linePatt = cls.makeLinePatt(cls.name,cls.lineID,cls.name)
        #if __debug__: print 'makeLinePatt',cls.linePatt.pattern
        cls.itemPatt = cls.makeItemPatt(cls.name,cls.latex_content)
        #if __debug__: print 'makeItemPatt',cls.itemPatt.pattern
    initLineParser = classmethod(initLineParser)
    
    def parseLine(cls,results):
        #if __debug__: print 'parseLine',results[cls.lineID]
        for item in results[cls.lineID].split(';'):
            match = cls.itemPatt.match(item)
            #if __debug__: print item,match
            if match: cls(cls.topology,match.groupdict())
        del results[cls.lineID]
        del results['evidence']
    parseLine = classmethod(parseLine)
    
    evidence={}
    def __init__(self,topology,d):
        #if __debug__: print self.name,'__init__',d
        self.topology = topology
        self.evidence={}
        if d.has_key('evidence'):
            self.evidence[tuple([d[k] for k in self.latex_params])]=d['evidence']
        topology.add(self)
    
    def __hash__(self): return self.__str__().__hash__()

    def latex_str(self): return ''.join([self.latex_markup, '{', str(self.id), '}'])
    
    def latex_repr(self):
        '''Print LaTeX formatted string representation.'''
        return ''.join(['\\ev{', ','.join([e for e in self.evidence]), '}',
                        '{', self.name,'(', self.latex_str(), ')};',])
    
    def addRelation(self,relation,key,value=None):
        '''Adds a relation key/value pair to the topological entity.'''
        if debug: print 'Topological (', str(self ),')', relation,key,value
        if not self.__dict__.has_key(relation): self.__dict__[relation] = {}
        self.__dict__[relation][key] = value
    
    def unify(self,other):
        '''Combine two topological objects.
        
        @type other: Topological
        @param other: self and other must share an isa relationship.
        '''
        pass

class TopologicalEntity(Topological):
    local_topology = Topology()
    def __init__(self,topology,d):
        self.latex_params=[self.name]
        self.id = d[self.name]
        if debug: print 'TopologicalEntity (', str(self ),')'
        Topological.__init__(self,topology,d)
    def registerPattern(cls,topology):
        cls.initLineParser(topology) 
        topology.Line[cls.linePatt]=cls.parseLine
        topology.Content[cls.latex_markup]=cls.name
        #if __debug__: print cls.name,'registerPattern',cls.linePatt,cls.parseLine
    registerPattern = classmethod(registerPattern)

class GlobalDir:
    dirs = 'drul'
    numDirs = 4
    def __init__(self,dir=None):
        if type(dir) == type(None):
            self.dir = 0
        elif type(dir) == str:
            self.dir = dirs.find(dir)
        elif isinstance(dir, int):
            self.dir = dir
        elif type(dir) == __name__:
            self.dir = dir.dir
        else: raise TypeError, 'Unknown type '+str(type(dir))
        self.normalize()
    def normalize(self):
        self.dir %= self.numDirs
        return self
    def __str__(self):
        return dirs[self.dir]
    def __repr__(self):
        return dirs[self.dir]
    def __eq__(self,other):
        if type(other) == type(None): return False
        return self.dir == other.dir
    def __add__(self,other):
        if isinstance(other, int):
            return GlobalDir(self.dir + other)
        elif type(other) == __name__:
            return GlobalDir(self.dir + other.dir)
        elif type(other) == str:
            return dirs[self.dir] + other
        else: raise TypeError, 'Unknown type '+type(other)
    def __sub__(self,other):
        if isinstance(other, int):
            return GlobalDir(self.dir - other)
        elif type(other) == __name__:
            return GlobalDir(self.dir - other.dir)
        else: raise TypeError, 'Unknown type '+type(other)
    def __neg__(self):
        return GlobalDir(self.dir + self.numDirs/2)
    def __iadd__(self,other):
        if isinstance(other, int):
            self.dir += other
        elif type(other) == __name__:
            self.dir += other.dir
        else: raise TypeError, 'Unknown type '+type(other)
        return self.normalize()
    def __isub__(self,other):
        if isinstance(other, int):
            self.dir -= other
        elif type(other) == __name__:
            self.dir -= other.dir
        else: raise TypeError, 'Unknown type '+type(other)
        return self.normalize()
    def add(self,(x,y),d):
        if self.dir==0:   return (x,y-d) #d
        elif self.dir==1: return (x+d,y) #r
        elif self.dir==2: return (x,y+d) #u
        elif self.dir==3: return (x-d,y) #l

class Path(TopologicalEntity):
    '''Representation of an SSH topological path.'''
    
    latex_markup=r'\\sshpath'
    name='path'
    latex_content=Topological.markupContent(latex_markup,Topological.makeNamedPattern(name,r'[^\}]+'))
    def __init__(self,topology,d):
        self.dir = {}
        self.dir['+'] = PathDir(topology,d,'+')
        topology.add(self.dir['+'])
        self.dir['-'] = PathDir(topology,d,'-')
        topology.add(self.dir['-'])
        TopologicalEntity.__init__(self,topology,d)
        self.terminates = {}
    def fmt(cls,id):
        if type(id) == Path: return '"'+str(id)+'"'
        else: return '"Pa'+str(id)+'"'
    fmt = classmethod(fmt)
    def __str__(self): return 'Pa'+self.id
    def __repr__(self): return 'Path(\''+self.id+'\')'
    class partialOrder:
        def __init__(self,partialOrder): self.partialOrder = partialOrder
        def cmp(self,x,y):
            if (x,y) in self.partialOrder: return -1
            elif (y,x) in self.partialOrder: return 1
            else: return 0 # Arbitrary 
    
    def getTotalPlaceOrder(self):
        totalOrder = [self.topology.place[k].latex_str()[1:] for k in self.on]
        porder = self.partialOrder(self.dir['+'].order)
        totalOrder.sort(porder.cmp)
        prefixLen = len(Place.latex_markup)
        return [p[prefixLen:-1] for p in totalOrder]
    
    def writeMap(self,pathdir,drawn,placesOrder,arbitraryTurn,mapFile):
        # output ghost ends if not terminates
        if '-' not in self.terminates:
            termPlace = Place.fmt(placesOrder[0])
            mapFile.write(termPlace+' ['+str(-pathdir)+'] *{} ='+PathDir.fmt(self.id,'-')+'\n')
            mapFile.write(PathDir.fmt(self.id,'-')+':@{/--+} '+termPlace+'\n')
        if '+' not in self.terminates:
            termPlace = Place.fmt(placesOrder[-1])
            mapFile.write(termPlace+' ['+str(pathdir)+'] *{} ='+PathDir.fmt(self.id,'+')+'\n')
            mapFile.write(termPlace+':@{/--+} '+PathDir.fmt(self.id,'+')+'\n')
        if arbitraryTurn: #xyrefer 6.2
            turnFrom = drawn.get('Pa'+arbitraryTurn,1)
            arbTurnID = PathDir.fmt(arbitraryTurn,'at')
            mapFile.write(Place.fmt(placesOrder[0])+' ['+str(turnFrom)+str(turnFrom-1)+'(0.33)]'+
                          ' *{} ='+arbTurnID+'\n')
            mapFile.write(Place.fmt(placesOrder[0])+' ['+str(turnFrom)+str(turnFrom+1)+'(0.33)]'+
                          ' :@/'+str(turnFrom)+'/@{ *{?} } '+arbTurnID+'\n')
        #Sanity check ends
        mapFile.write(Place.fmt(placesOrder[0])+':@^{/:+} ')
        for place in placesOrder[1:-1]: mapFile.write('\''+Place.fmt(place)+' ')
        mapFile.write(Place.fmt(placesOrder[-1])+' ')
        mapFile.write('^{'+self.latex_str()[1:]+'}\n')
        for place1,place2 in zip(placesOrder[:-1],placesOrder[1:]):
            mapFile.write(Place.fmt(place1)+' :@^{} '+Place.fmt(place2)+' ')
            dist = ('?','Moves')
            if self.__dict__.has_key('pathDistance'):
                if (place1,place2) in self.pathDistance: dist = self.pathDistance[(place1,place2)]
                elif (place2,place1) in self.pathDistance: dist = self.pathDistance[(place2,place1)]
            dist = dist[0] #' '.join(dist)
            mapFile.write(' ^(.33){-'+dist+'-}\n')
        if 'onObject' in self.__dict__:
            mapFile.write(Place.fmt(placesOrder[0])+' :@^{} '+Place.fmt(placesOrder[-1])+' ')
            mapFile.write('^(0.75){\\textsf{\\txt{'+',\\\\ '.join(self.onObject.keys())+'}}}\n')
        annotations = []
        for d in ['+','-']:
            for annote in ['appear','pathType']:
                if annote in self.dir[d].__dict__:
                    for place,annotation in self.dir[d].__dict__[annote].items():
                        annotations.append(annotation)
        if annotations:
            mapFile.write(Place.fmt(place1)+' :@^{} '+Place.fmt(place2)+' _{\\texttt{\\txt{'+',\\\\'.join(annotations)+'}}}\n')
        drawn[str(self)] = pathdir
Path.registerPattern(TopoMap)

class Place(TopologicalEntity):
    '''Representation of an SSH topological place.'''
    
    latex_markup=r'\\sshplace'
    name='place'
    latex_content=Topological.markupContent(latex_markup,Topological.makeNamedPattern(name,r'[^\}]+'))
    def fmt(cls,id):
        if type(id) == Place: return '"'+str(id)+'"'
        else: return '"Pl'+str(id)+'"'
    fmt = classmethod(fmt)
    def __str__(self): return 'Pl'+self.id
    def __repr__(self): return 'Place(\''+self.id+'\')'

    def writeMap(self,drawn,currentLoc,dist=1,pathDir=None,context=''):
        if not dist: dist = 3 # Arbitrary placement leaves room for visible gap.
        if pathDir == None: pathDir = GlobalDir()
        drawn[str(self)] = pathDir.add(currentLoc,dist)
        if dist == 0: disp = ''
        if context: context = Place.fmt(context)+' '
        return ''.join([context,'[',str(pathDir)*dist,']',' '*(10-dist),
                       '*{',self.latex_str()[1:],'}*++[o]{}*+\\frm{o} =',self.fmt(self.id),'\n'])

    def layout(self,drawn,currentLoc,dist=2,pathdir=None,mapFile=sys.stdout,context=''):
        mapFile.write(self.writeMap(drawn,currentLoc,dist,pathdir,context))
        if self.__dict__.has_key('at'):
            dir = 'rd' #[,'ru','ld','lu']
            objects = ','.join(self.at.keys())
            mapFile.write(self.fmt(self.id)+' [] !{'+self.fmt(self.id)+'!/'+dir+' 24pt/} *++={\\txt{'+objects+'}}\n')

    def layoutPaths(self,drawn,currentLoc,pathdir=None,mapFile=sys.stdout,context=''):
        drawn[self.id+':Paths'] = True
        paths = self.on.keys()
        paths.sort()
        refpath = paths[0]
        for path in paths[1:] or paths:
            if len(paths)>1 and path == refpath: continue
            dir=drawn.get('Pa'+refpath,pathdir)
            pathObj = self.topology.path[path]
            side = None
            if refpath != path:
                if self.__dict__.has_key('sideof'): side = self.sideof.get((path+'+',refpath+'+'),None)
                if side == 'Left': dir -= 1
                elif side == 'Right': dir += 1
                else: dir -= 1# Arbitary left        ### OR fork map
            pathOrder = pathObj.getTotalPlaceOrder()
            print 'layoutPaths',self,path,'+',refpath,'+',side,str(dir),str(drawn.get('Pa'+refpath,pathdir)),pathOrder
            for place in pathOrder:
                if 'Pl'+place not in drawn:
                    if 'pathDistance' in pathObj.__dict__:
                        if (self.id,place) in pathObj.pathDistance:
                            distCount,distUnit = pathObj.pathDistance[(self.id,place)]
                        elif (place,self.id) in pathObj.pathDistance:
                            distCount,distUnit = pathObj.pathDistance[(place,self.id)]
                        else: distCount,distUnit = (2,'Move')
                        distCount = int(distCount)
                    else: distCount = None
                    self.topology.place[place].layout(drawn,currentLoc,distCount,dir,mapFile,self)
            if str(pathObj) not in drawn:
                if refpath != path and not side:
                    print 'Arbitrary.  Draw fuzzy turn for', pathObj, 'from', 'Pa'+refpath
                    arbitraryTurn=refpath
                else: arbitraryTurn=None
                pathObj.writeMap(dir,drawn,pathOrder,arbitraryTurn,mapFile)
            for place in pathOrder:
                if place+':Paths' not in drawn:
                    self.topology.place[place].layoutPaths(drawn,currentLoc,dir,mapFile,self)
            refpath = path
Place.registerPattern(TopoMap)

class Place2(Place):
    name='place2'
    latex_content=re.sub('>','2>',Place.latex_content)

class PathDir(TopologicalEntity):
    '''Representation of an SSH topological path with direction.'''
    
    latex_markup=r'\\sshpathdir'
    name='pathdir'
    latex_content=Topological.markupContent(
        latex_markup,
        Topological.makeNamedPattern(name,
                                     Topological.makeNamedPattern('path',r'[^\}]+')+'\}\{'+
                                     Topological.makeNamedPattern('dir','(\+|\-)')) )
    latex_params=['path']
    id_values=['path','dir']
    def __init__(self,topology,d,dir=None):
        if d.has_key('pathdir'):
            if '}{' in d['pathdir']:
                d['pathdir'] = d['pathdir'][:-3]+d['pathdir'][-1]
            d['path'] = d['pathdir'][:-1]
            d['dir']= d['pathdir'][-1]
        elif dir: self.dir=dir
        elif d.has_key('dir'): self.dir = d['dir']
        else: self.dir = '+'
        self.path=d['path']
        Topological.__init__(self,topology,d)
        self.id=self.path+self.dir
    def fmt(cls,id,dir): return '"Pa'+str(id)+dir+'"'
    fmt = classmethod(fmt)
    def __str__(self): return 'Pa'+self.id
    def __repr__(self): return 'PathDir(\''+self.path+','+self.dir+'\')'
    def latex_str(self): return ''.join([self.latex_markup, '{', str(self.path), '}{',str(self.dir),'}'])
    def extractPathDir(pathdirstr): return re.split('[\{\}]+',pathdirstr)
    extractPathDir=staticmethod(extractPathDir)
PathDir.registerPattern(TopoMap)

class PathDir2(PathDir):
    name='pathdir2'
    latex_content=re.sub('>','2>',PathDir.latex_content)

class PathFragment(TopologicalEntity):
    '''Representation of an SSH topological path fragement.'''
    
    latex_markup=r'\\sshpathfragment'
    name='pathfragment'
    latex_content=Topological.markupContent(latex_markup,PathDir.latex_content+','+Place.latex_content)
    def __str__(self): return ''.join('Pa',self.path,self.dir,'@','Pl'+place)
    def __repr__(self): return ''.join('Path(\'',self.path,self.dir,',',self.place,'\')')
PathFragment.registerPattern(TopoMap)

class Object(TopologicalEntity):
    '''Representation of an SSH topological path with direction.'''
    
    latex_markup=None
    name='object'
    latex_content=Topological.makeNamedPattern(name,'[^\)\{\}]+')
    def __str__(self): return 'Obj_'+self.id
    def __repr__(self): return 'Object(\''+self.id+'\')'
Object.registerPattern(TopoMap)

class Thing:
    '''Regular expression covering things that can be on paths.'''
    
    name = 'thing'
    Things = [Object,Place]
    latex_content=Topological.makeNamedPattern(name,'|'.join([t.latex_content for t in Things]))

class Thing2(Thing):
    name='thing2'
    latex_content=re.sub('>','2>',Thing.latex_content)

class Appear:
    name = 'appearance'
    latex_content=Topological.makeNamedPattern(name,r'[^\}]+')

class PathType:
    name = 'type'
    latex_content=Topological.makeNamedPattern(name,r'[^\}]+')

class LocalTopology:
    name = 'localTopology'
    latex_content=Topological.makeNamedPattern(name,r'[^\}]+')

class PathDistVal:
    name = 'pathDistVal'
    latex_content=Topological.makeNamedPattern(name,
                              Topological.makeNamedPattern('distCount',r'[^\},]+')+','+\
                              Topological.makeNamedPattern('distUnit',r'[^\},]+'))

class TopologicalRelation(Topological):
    '''Abstract class for topological relations.'''
    
    def __init__(self,topology,d):
        if d.has_key('pathdir'):
            if '}{' in d['pathdir']:
                d['pathdir'] = d['pathdir'][:-3]+d['pathdir'][-1]
            d['path'] = d['pathdir'][:-1]
            d['dir']= d['pathdir'][-1]
        if d.has_key('pathdir2'):
            if '}{' in d['pathdir2']:
                d['pathdir2'] = d['pathdir2'][:-3]+d['pathdir2'][-1]
            d['path2'] = d['pathdir2'][:-1]
            d['dir2']= d['pathdir2'][-1]
        Topological.__init__(self,topology,d)
        self.id = tuple([topology.get_str(k,d[k]) for k in self.latex_params])
        self.latex_markup=None
        for k,v in d.items(): self.__dict__[k] = v
        self.assertRelation()
    def __str__(self): return self.name+str(self.id)
    def __repr__(self): return self.name+'('+str(self.__dict__)+')'
    def registerPattern(cls,topology):
        cls.latex_content=cls.commaSpaceRegexp.join([p.latex_content for p in cls.params])
        cls.latex_params=[]
        for p in cls.params:
            if p.__dict__.has_key('id_values'): cls.latex_params+=p.id_values
            else: cls.latex_params.append(p.name)
        cls.initLineParser(topology) 
        topology.Line[cls.linePatt] = cls.parseLine
        topology.Content[cls.latex_markup]=cls.name
    registerPattern = classmethod(registerPattern)
    def assertRelation(self): pass
    def latex_str(self): return ',~'.join([self.topology.get_latex_str(p.name,self.__dict__[p.name]) for p in self.params])

class OnRelation(TopologicalRelation):
    '''Asserts a place is _on_ a path.'''
    
    name = 'on'
    params=[Path,Thing]
    def assertRelation(self):
        if self.__dict__.has_key('place') and self.place:
            self.topology.place[self.place].addRelation(self.name,self.path,self.evidence)
            self.topology.path[self.path].addRelation(self.name,self.place,self.evidence)
        elif self.__dict__.has_key('object') and self.object:
            #self.topology.object[self.object].addRelation(self.name,self.path,self.evidence)
            self.topology.path[self.path].addRelation(self.name+'Object',self.object,self.evidence)
OnRelation.registerPattern(TopoMap)

class AppearRelation(TopologicalRelation):
    '''Asserts the appearance of a path segment.'''
    
    name = 'appear'
    params=[PathDir,Place,Appear]
    def assertRelation(self):
        self.topology.place[self.place].addRelation(self.name,self.path+self.dir,self.appearance)
        self.topology.pathdir[self.path+self.dir].addRelation(self.name,self.place,self.appearance)
AppearRelation.registerPattern(TopoMap)

class OrderRelation(TopologicalRelation):
    name = 'order'
    params=[PathDir,Thing,Thing2]
    def assertRelation(self):
        self.topology.pathdir[self.path+self.dir].addRelation(self.name,(self.thing,self.thing2))
OrderRelation.registerPattern(TopoMap)

class SideOfRelation(TopologicalRelation):
    name = 'sideof'
    params=[Place,PathDir,PathDir2]
    def assertRelation(self): 
        self.topology.place[self.place].addRelation(self.name,(self.path+self.dir,self.path2+self.dir2),self.sideof)
        if self.sideof == 'Left': other = 'Right'
        elif self.sideof == 'Right': other = 'Left'
        else: other = 'UNK'
        self.topology.place[self.place].addRelation(self.name,(self.path2+self.dir2,self.path+self.dir),other)
        self.name='tothe'+self.sideof+'Of'
        self.topology.place[self.place].addRelation(self.name,(self.path+self.dir,self.path2+self.dir2))
    def registerPattern(cls,topology):
        cls.latex_content=cls.commaSpaceRegexp.join([p.latex_content for p in cls.params])
        cls.latex_params=[]
        for p in cls.params:
            if p.__dict__.has_key('id_values'): cls.latex_params+=p.id_values
            else: cls.latex_params.append(p.name)
        cls.initLineParser(topology) 
        cls.lineID = cls.name+'Line'
        cls.topology = topology
        cls.linePatt = cls.makeLinePatt(cls.name,cls.lineID,'tothe(?P<sideof>(Right|Left))Of')
        #if __debug__: print 'makeLinePatt',cls.linePatt.pattern
        cls.itemPatt = cls.makeItemPatt('tothe(?P<sideof>(Right|Left))Of',cls.latex_content)
        topology.Line[cls.linePatt] = cls.parseLine
        topology.Content[cls.latex_markup]=cls.name
    registerPattern = classmethod(registerPattern)
SideOfRelation.registerPattern(TopoMap)

class PathTypeRelation(TopologicalRelation):
    name='pathType'
    params=[PathDir,Place,PathType]
    def assertRelation(self):
        self.topology.pathdir[self.path+self.dir].addRelation(self.name,self.place,self.type)
PathTypeRelation.registerPattern(TopoMap)

class TerminatesRelation(TopologicalRelation):
    name='terminates'
    params=[PathDir,Place]
    def assertRelation(self):
        self.topology.pathdir[self.path+self.dir].addRelation(self.name,self.place)
        self.topology.place[self.place].addRelation(self.name,self.path+self.dir)
        self.topology.path[self.path].terminates[self.dir] = self.place
TerminatesRelation.registerPattern(TopoMap)

class AtRelation(TopologicalRelation):
    name='at'
    params=[Place,Object]
    def assertRelation(self):
        self.topology.place[self.place].addRelation(self.name,self.object)
AtRelation.registerPattern(TopoMap)

class LocalTopologyRelation(TopologicalRelation):
    name='localTopology'
    params=[Place,LocalTopology]
    def assertRelation(self):
        self.topology.place[self.place].addRelation(self.name,self.localTopology)
LocalTopologyRelation.registerPattern(TopoMap)

class PathDistanceRelation(TopologicalRelation):
    name='pathDistance'
    params=[Path,Place,Place2,PathDistVal]
    def assertRelation(self):
        self.topology.path[self.path].addRelation(self.name,(self.place,self.place2),(self.distCount,self.distUnit))
PathDistanceRelation.registerPattern(TopoMap)

class Instruction(Topological):
    class ISlot(Topological):
        latex_markup=r'\\islot'
        name='islot'
        latex_content=Topological.markupContent(latex_markup,Topological.makeNamedPattern(name,r'[-\w]+'))+\
                       Topological.markupContent('',Topological.makeNamedPattern('islotVal','[^, ]+'))
        pattern=re.compile(latex_content)
        islotValPatt=re.compile('('+\
                                Topological.makeNamedPattern('islotValName','[^\{]+')+'\{'+\
                                Topological.makeNamedPattern('islotVal1','[^\}]+')+'\}'+\
                                '(\{'+Topological.makeNamedPattern('islotVal2','[^\}]+')+'\})?'+\
                                ')|'+Topological.makeNamedPattern('islotPlain','[^\\\{\}]+'))
    
    name='instruction'
    latex_markup=r'\\instruct'
    params=[ISlot]
    param_name='islots'
    latex_params=[param_name]
    itemPatt = Topological.makeItemPatt('instruction',
                                        Topological.markupContent(latex_markup, Topological.makeNamedPattern(name,'[-\w]+'))+
                                        Topological.markupContent('',Topological.makeNamedPattern(param_name,'.*')))
    linePatt=re.compile(r'\\item'+Topological.makeLinePatt(name,name+'Line',name).pattern)
    
    def __init__(self,topology,d):
        Topological.__init__(self,topology,d)
        self.action = d['instruction']
        self.id = str(len(topology.RouteInstructions)+1)
        for islot in d['islots'].split(', '):
            m = self.ISlot.pattern.match(islot)
            #if __debug__: print 'Instruction __init__:', islot,m
            if m:
                #if __debug__: print 'Instruction __init__ match dict:', islot,m.groupdict()
                d = m.groupdict()
                islotmatch=self.ISlot.islotValPatt.match(d['islotVal'])
                #if __debug__: print 'Instruction islotmatch:',islotmatch
                #if __debug__ and islotmatch: print 'Instruction islotmatch dict:',islotmatch.groupdict()
                if islotmatch and islotmatch.groupdict()['islotValName']:
                    isv = islotmatch.groupdict()
                    slot = topology.Content['\\'+str(isv['islotValName'])],
                    #if __debug__: print 'Instruction islotVal slot:',slot[0],isv
                    #if __debug__: topology.pprint()
                    if isv['islotVal2']: islotVal=isv['islotVal1']+isv['islotVal2']
                    else: islotVal=isv['islotVal1']
                    self.addRelation(d['islot'], topology.__dict__[slot[0]][islotVal])
                else: self.addRelation(d['islot'], d['islotVal'])
        topology.RouteInstructions.append(self)
        #if __debug__: print repr(self)
    
    def __str__(self): return 'RI'+self.id
    def __repr__(self): return 'RI(\''+self.id+','+str(self.__dict__)+'\')'
    def registerPattern(cls,topology):
        topology.Line[cls.linePatt] = cls.parseLine
        cls.lineID = cls.name+'Line'
        cls.topology = topology
        topology.Content[cls.latex_markup]=cls.name
        #if __debug__: print 'Instruction: line pattern:', cls.linePatt.pattern
        #if __debug__: print 'Instruction: item pattern:', cls.itemPatt.pattern
    registerPattern = classmethod(registerPattern)
Instruction.registerPattern(TopoMap)

def dontCareLine(results): print '.',
#TopoMap.Line[re.compile('.*')]=dontCareLine
TopoMap.Line[re.compile('^.topolist.*$')] = dontCareLine
TopoMap.Line[re.compile('^.raggedright$')] = dontCareLine
TopoMap.Line[re.compile('^\s*.(begin|end)\{enumerate\}\s*$')] = dontCareLine
TopoMap.Line[re.compile('^(\}\{?|\s*)(%.*)?$')] = dontCareLine               #End of one block
TopoMap.Line[re.compile('^\s*..\s*$')] = dontCareLine                        #Newline
TopoMap.Line[re.compile('^\s*$')] = dontCareLine                             #Blank line
TopoMap.Line[re.compile('^.emph\{.*$')] = dontCareLine                       #Comment

def getRouteSequence(riList):
    routePlaces = []
    for ri in riList:
        if ri.action == 'turn':
            place = ri.at.keys()[0]
        elif ri.action == 'travel' or ri.action == 'find':
            place = ri.__dict__['from'].keys()[0]
        elif ri.action == 'declare-goal': pass
        routePlaces.append(Place.fmt(place.id)+' ')
        if ri.action == 'travel' or ri.action == 'find':
            place = ri.to.keys()[0]
            routePlaces.append(place.fmt(place.id)+' ')
    uniqPlaces = [routePlaces[0]]
    for i in range(len(routePlaces)-1):
        if routePlaces[i] != routePlaces[i+1]:
            uniqPlaces.append(routePlaces[i+1])
    return uniqPlaces

def printGraphicalMap(TopoMap,filename,pathdir=None):
    pathdir = GlobalDir(pathdir)
    print '='*8,file,'Graphical Map','='*8
    drawn = {}
    mapFile = open(filename,'w')
    mapFile.write('\\[\\xy \\xygraph{\n')
    
    # Draw connected places
    TopoMap.place.items()[0][1].layout(drawn,(0,0),dist=0,pathdir=pathdir,mapFile=mapFile)
    TopoMap.place.items()[0][1].layoutPaths(drawn,(0,0),pathdir=pathdir,mapFile=mapFile)
    
    # Look for and draw any unconnected places
    drawnPlaces = [k for k in drawn if k.startswith('Pl')]
    drawnPlaces.sort()
    for place in TopoMap.place.values():
        if str(place) not in drawnPlaces: #Arbitrary 3 dist, Arbitrary direction
            place.layout(drawn,(3,3),3,pathdir,mapFile,drawnPlaces[-1][2:])
            place.layoutPaths(drawn,(3,3),pathdir,mapFile,drawnPlaces[-1][2:])
            drawnPlaces = [k for k in drawn if k.startswith('Pl')]
            drawnPlaces.sort()
    
    # Mark route on map
    routePlaces = getRouteSequence(TopoMap.RouteInstructions)
    if routePlaces:
        mapFile.write(routePlaces[0]+' :@{(~)} ')
        for rp in routePlaces[1:-1]:
            mapFile.write('\''+rp)
        mapFile.write(routePlaces[-1])#+' _{route}\n')
    
    mapFile.write('} \\endxy \\]\n')
    mapFile.close()

if __name__ == '__main__':
    from Parser import parseFile
    results={}
    for Giver in ['EDA','EMWC','KLS','KXP','TJS','WLH']:
        suffix=Giver+'_Grid0_4_5_Dirs_1.txt.tex'
        file = 'Topo_'+suffix
        print '\n',file
        # initialize TopoMap fn
        TopoMap.reset()
        parseFile(TopoMap.Line, os.path.join('Directions','CorrFullTrees',file), results)
        print; TopoMap.pprint(); print
        printGraphicalMap(TopoMap,os.path.join('Directions','CorrFullTrees','TopoMap_'+suffix),'l')
