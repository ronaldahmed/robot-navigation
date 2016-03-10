import random, re, copy, math, string, weakref

from nltk_contrib.pywordnet import tools

from Options import Options
from Sense import SurfaceSemanticsStructure, SSS, saveFrame
from Sense import splitSurfaceSemantics,parseSurfaceSemantics,printSurfaceSemantics
from Utility import *
from Robot.Meanings import *

SynSetKB = {}
for word,meaning in KB.items():
    for sense in parseSurfaceSemantics(word):
        SynSetKB[sense.synset.offset] = meaning

def getPOS(sss,POS): return [f for f in sss.feature_names() if f.endswith('_'+POS)]
def getNouns(sss):  return getPOS(sss,'n')
def getVerbs(sss):  return getPOS(sss,'v')
def getPreps(sss):  return getPOS(sss,'p')

def getTurnVerbs(sss):
    return [f for f in sss.feature_names() if f in Turn.knownVerbs]
def getTravelVerbs(sss):
    return [f for f in sss.feature_names() if f in Travel.knownVerbs]

def listPrepositions(phrases): return tuple([prep+'_p' for prep in phrases])
def listNouns(nouns): return tuple([obj+'_n' for obj in nouns])

class CompoundActionSpecification:
    casKnownAttributes = (',', 'Agent', 'Det', 'cost', 'observations', 'sss', 'Meta', 'S', 'Punct', 'Then',
                          'plan', 'source','index')
    def __init__(self, sss=[], **kargs):
        self.__dict__.update(kargs)
        self.sss = sss
        self.cost = 0
        self.observations = []
    def __repr__(self):
        return '%s(%s)'%(self.__class__.__name__,
                         ', '.join(["%s=%r"%(k,v) for k,v in self.__dict__.items()
                                    if v and k not in self.casKnownAttributes])
                         )
    
    def __str__(self,indent=0,margin=70):
      rep = repr(self)
      if indent+len(rep) < margin or indent > 2*margin: return rep
      namelen = len(self.name())
      s = self.name()+'(\n'+' '*(indent+namelen)
      linelen = indent+namelen
      for k,v in self.__dict__.items():
          if isinstance(v,CompoundActionSpecification):
              s += (k + '\n' + ' '*(indent+namelen+1)
                    + '=' + v.__str__((indent+namelen),margin)
                    + ',\n' + ' '*(indent+namelen))
              linelen=indent
          elif v and k not in self.casKnownAttributes:
              if type(v) == list:
                  linelen += len(k)+2
                  V_Strs=[]
                  for item in v:
                      if isinstance(item,CompoundActionSpecification):
                          V_Strs.append(item.__str__(indent+namelen+len(k),margin))
                      else: V_Strs.append(repr(item))
                  if s[-len(' '*(indent+namelen))-1]!='\n':
                      lnbreak = '\n'+' '*(indent+namelen)
                      linelen = len(lnbreak)
                  else:
                      lnbreak = ''
                      linelen += len(V_Strs[-1])
                  s += '%s%s=[%s],' % (lnbreak,k,(',\n'+' '*(indent+namelen+len(k)+2)).join(V_Strs))
                  if lnbreak: s += '\n'+' '*(indent+namelen)
              else:
                  rep=repr(v)
                  linelen += len(k)+namelen+len(rep)
                  if linelen > margin:
                      s+= '\n'+' '*(indent)
                      linelen=len(' '*(indent))
                  s += k+'='+repr(v)+','
          if linelen > margin:
              s+='\n'+' '*(indent+namelen)
              linelen=indent
      return s+')'
    def __lt__(self,other):
        if isinstance(other,CompoundActionSpecification):
            return ([(k,v) for k,v in self.__dict__.items() if v and k not in self.casKnownAttributes]
                    < [(k,v) for k,v in other.__dict__.items() if v and k not in other.casKnownAttributes])
        else: return repr(self) < repr(other)
    def name(self): return self.__class__.__name__
    def checkUnknownAttrs(self):
        if not self.sss: return
        if isinstance(self.sss,SurfaceSemanticsStructure):
            attrList = self.sss.feature_names()
        else:
            attrList = [d for d in self.__dict__.items()]
            attrList = [k for k,v in attrList if v]
        unknownAttrs = [a for a in attrList if a not in self.knownAttrs + self.casKnownAttributes]
        for a in unknownAttrs:
            for legit in self.knownAttrs + self.casKnownAttributes:
                if valMatch(legit,a):
                    unknownAttrs.remove(a)
                    break
        if unknownAttrs:
            logger.warning('Unknown attributes for %s: %r\nin %r',self.name(),unknownAttrs, self.sss)
        for attr in self.__dict__:
            attval = getattr(self,attr)
            if isinstance(attval, list):
                setattr(self,attr,uniq(attval))
    def update(self,event,(incCost,obs),robot):
        """Record cost and observation stream."""
        self.cost += incCost
        self.observations.append(obs)
        logger.info('After %s %d %s',event,incCost,len(obs)>2 and `obs[:1]`+'...' or obs)
        if robot.NLUQueue:
            robot.NLUQueue.put(('CompoundAction', '%s : %s' % (self.name(),event)))
        return self.cost,self.observations[-1]
    def model(self):
        """Return topological entities and relationships from this cas."""
        pass
    def execute(self,robot):
        """Act out the utterence in the current context."""
        pass

SideDist ={
    'Against' : ([Back], '0:'),
    'Along' : ([Front],'0'),
    'Arrive_v' : ([At],'0'),
    'At' : ([At],'0'),
    'Away' : ([Back], '0:'),
    'Between' : ([Between], '1:'),
    'Loc_p' : ([At],'0'),
    'Onto' : ([Front],'0'),
    'Past' : ([At],'0'),
    'Pass_v' : ([At],'0'),
    'Toward' : ([Front], '1:'),
    'Until' : ([At],'0'),
    }
for phrase in SideDist.keys(): SideDist[phrase+'_p'] = SideDist[phrase]

def getSideDist(sss):
    side,dist = ([],'')
    if not isinstance(sss,SurfaceSemanticsStructure): return side,dist
    features = sss.feature_names()
    for feature in [f for f in features if f in SideDist]:
        side,dist = SideDist[feature]
    if len([f for f in features if f in SideDist]) > 1:
        logger.warning('More than one default side and dist for %r %r',sss,[f for f in features if f in SideDist])
    if 'Dir' in features: side = meanings(sss,('Dir'),Directions)
    elif 'Side' in features or 'Side_p' in features:
        for subsss in grab(sss,'Side'):
            if not 'feature_names' in dir(subsss):
                side = subsss
                continue
            subfeatures = subsss.feature_names()
            if 'Dir' in subfeatures:
                side = meanings(subsss,('Dir'),Directions)
            elif 'Side_n' in subfeatures:
                side = meanings(subsss,('Side_n'),Directions)
            elif 'Obj' in subfeatures:
                side = (meanings(subsss,('Obj','Obj_n'),Directions) or
                        meanings(subsss,('Obj_n'),Directions))
                if side == [Wall]:  side = [FrontLeft,FrontRight]
            elif 'Side_p' in subfeatures:
                side = lookupMeaning(subsss['Side_p'],Directions)
        dist = '0:'
    return side,dist

def setSideDist(Things,side=None,dist=None,overwrite=True):
    for thing in Things:
        if not isinstance(thing,Thing): continue
        if side: thing.side = side
        if dist:
            if (not overwrite and hasattr(thing,'dist')
                and dist == '0' and '1' in thing.dist): #Fix compatability check.
                continue
            thing.dist = dist
        for prep in Thing.knownPrepPhrases:
            if hasattr(thing,prep):
                setSideDist(getattr(thing,prep),dist=dist,overwrite=False)

class Thing(CompoundActionSpecification):
    """
    >>> Thing(SSS.parse("[Obj=[Detail=[Detail_p=[MEAN='OF'], Obj=[On=[On_p=[MEAN='ON'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN=\\"walls_N_('wall', [1])\\"]]], Obj_n=[MEAN=\\"butterflies_N_('butterfly', [1])\\"]]], Obj_n=[MEAN=\\"pictures_N_('picture', [2])\\"], On=[On_p=[MEAN='IN'], Path=[Det=[MEAN='THE'], Path_n=[MEAN='hall_N_1']]], Side=[Dir=[MEAN='right_ADV_4'], Side_p=[MEAN='TO']]]]"))
    Thing(On=[Thing(value=Path, type='Path')], Detail=[Thing(On=[Thing(dist='0', value=Wall, type='Obj')], value=Butterfly, type='Obj')], value=Pic, side=[Right], type='Thing')
    >>> Thing(SSS.parse("[Path=[Det=[MEAN='THE'], Detail=[Detail_p=[MEAN='WHICH'], Is_v=[MEAN='be_V_1,2,3,4,5'], Side=[Dir=[Cc=[MEAN='AND'], Det=[MEAN='YOUR'], Dir=[MEAN='left_ADV_1'], Dir_21=[MEAN='right_ADV_4']], Side_p=[MEAN='TO']]], Obj=[Appear=[MEAN='blue_ADJ_1'], Obj_n=[MEAN='carpet_N_1']], Path_n=[MEAN='alley_N_1']]]"))
    Thing(value=Path, type='Thing', side=[Left, Right])
    >>> Thing(SSS.parse("[Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='wall hanging_N_1']]]"))
    Thing(value=Pic, type='Thing')
    >>> Thing(SSS.parse("[Struct=[Det=[MEAN='THE'], Struct_n=[MEAN='dead end_N_1']]]"))
    Thing(value=DeadEnd, type='Thing')
    >>> Thing(SSS.parse("[Path=[Det=[MEAN='A'], Appear=[Appear=[MEAN='blue_ADJ_1'], Hyphen=[MEAN='-'], Appear_17=[MEAN='tiled_ADJ_1']], Path_n=[MEAN='hallway_N_1'], Side=[Side_p=[MEAN='TO'], Det=[MEAN='EITHER'], Side_n=[MEAN='side_N_1'], Rel=[Rel_p=[MEAN='OF'], Obj=[Ref=[MEAN='IT']]]]]]"))
    Thing(Appear=[BlueTile, Flooring], value=Path, side=[Sides], type='Thing')
    >>> Thing(SSS.parse("[Path=[Det=[MEAN='THE'], Structural=[MEAN='long_ADJ_2'], Obj=[Obj_n=[MEAN='butterfly_N_1']], Path_n=[MEAN='hallway_N_1'], ,=[MEAN=','], Detail=[Detail_p=[MEAN='WITH'], Obj=[Appear=[MEAN='blue_ADJ_1'], Obj_n=[MEAN=\"walls_N_('wall', [1])\"]]]]]"))
    Thing(Detail=[Thing(dist='0', Appear=[BlueTile], value=Wall, type='Obj'), [Thing(value=Butterfly, type='Obj')]], value=Path, type='Thing', Structural=['Long'])
    """
    knownPrepPhrases = ('Across', 'Against', 'Along', 'Away', 'Between', 'Detail', 'Loc', 'On', 'Onto', 
                        'Part', 'Past', 'Side', 'Toward', 'Until',)
    knownAdj = ('Appear', 'Count', 'Obj_adj', 'Order_adj', 'Reldist', 'Structural', 'Struct_type', 'Not', 'Det')
    knownClauses = tuple() #('S', 'Desc')
    knownAttrs = (('dist', 'negate', 'side', 'type', 'value') + knownAdj
                  + listPrepositions(knownPrepPhrases) + knownPrepPhrases
                  + ObjectTypes + listNouns(ObjectTypes))
    referenceCache = {}
    def __init__(self,sss=[], type='Thing', **kargs):
        if type == Back: return
        self.type = type
        self.negate = False
        CompoundActionSpecification.__init__(self,sss,**kargs)
        try:
            if not hasattr(self,'value'): self.value = []
            for typeAttr in [t for t in self.__dict__ if t.startswith(type+'_n')]:
                value = getattr(self,typeAttr)
                if isinstance(value,list) and value:
                    value = value[0]
                if isinstance(value,Meaning):
                    self.value = value
                    setattr(self,typeAttr,self.value)
                if (isinstance(value,SurfaceSemanticsStructure)
                    and typeAttr in value.feature_names()):
                    self.value = value[typeAttr]
        except AttributeError, e: logger.error("Can't find %s",e)
        if sss: self.interpret(sss)
        self.condSetSideDist(getSideDist(self.sss))
        if not self.value:
            if self.type == 'Boolean': self.value = str(self.value)
            elif isinstance(self.type,str): self.value = Defaults.get(self.type,Furniture)
            elif isinstance(self.type,Meaning): self.value = self.type
            else: self.value = Furniture
            logger.debug("Thing.init %r %r", self.type,self.value)
        if isinstance(self.value,Meaning) and not hasattr(self.value, 'ViewPosition'):
            self.value.ViewPosition = At
        if hasattr(self,'Side') and self.Side:
            self.side = self.Side
            self.type = 'Side'
            del self.Side
        if hasattr(self,'Obj_adj') and self.Obj_adj:
            if not hasattr(self,'Detail'): self.Detail = []
            self.Detail.append(Thing(value=self.Obj_adj[0], type='Obj'))
            del self.Obj_adj
        if hasattr(self,'Obj') and self.Obj and Options.RecognizeNounNounCompound:
            if not hasattr(self,'Detail'): self.Detail = []
            if isinstance(self.Obj,list): self.Detail.extend(self.Obj)
            else: self.Detail.append(self.Obj)
            del self.Obj
        Thing.referenceCache[Thing] = self
        Thing.referenceCache[self.value] = self
        self.checkUnknownAttrs()

    def interpret(self,sss):
        logger.debug("Thing.interpret 1 %r", sss)
        features = sss.feature_names()
        for thingPhrase in [p for p in features if valCanonical(p) in ObjectTypes]:
            self.value = (meanings(sss,(thingPhrase,valCanonical(thingPhrase)+'_n'),KB) or
                          meanings(sss,(thingPhrase,valCanonical(thingPhrase)+'_v'),KB))
            #self.value = meanings(sss,(thingPhrase,KB))
            logger.debug("Thing.interpret 2 %s", self.value)
            if isinstance(self.value,list) and self.value: self.value = self.value[0]
            if isinstance(self.value,str) or isinstance(self.value,SurfaceSemanticsStructure):
                if self.value in KB: self.value = KB[self.value]
                else: self.value = []
            
            # Anaphora resolution
            if (not self.value and 'Ref' in sss[thingPhrase].feature_names()
                or ('Det' in sss[thingPhrase].feature_names()
                    and sss[(thingPhrase,'Det','MEAN')] in ('THIS','THAT'))
                and Options.ReferenceResolution):
                resolution = Thing.referenceCache.get(Defaults[thingPhrase],[])
                for attr in [a for a in self.knownAttrs if hasattr(resolution,a)]:
                    setattr(self, attr, copy.deepcopy(getattr(resolution,attr)))
                logger.debug('Thing.interpret resolving reference to %s %r = %r', thingPhrase, self.value, resolution)
            
            for predPhrase in [p for p in sss[thingPhrase].feature_names()
                               if isCanonical(p) and p in self.knownPrepPhrases+self.knownClauses+self.knownAdj]:
                logger.debug('Thing.interpret 3 %s %r', predPhrase,grab(sss,(thingPhrase,predPhrase)))
                if (predPhrase == 'Loc'
                    and any(sss[(thingPhrase,predPhrase)].feature_names(), ('Position','Ref'))):
                    continue
                if predPhrase == 'Structural' and not (Options.RecognizeStructural and Options.ImplicitTurn):
                    continue
                if predPhrase == 'Order_adj' and 'Approx' in sss[thingPhrase].feature_names():
                    continue
                if predPhrase in ('Not','NOT') and Options.RecognizeNegative:
                    self.negate = True
                    continue
                if predPhrase == 'Det':
                    if (sss[(thingPhrase,predPhrase,'MEAN')] in ('ANOTHER', 'OTHER', 'THAT', 'THE OTHER')
                        and Options.RecognizeDistalDeterminers):
                        self.dist = '1:'
                    continue
                setattr(self,predPhrase,meanings(sss,(thingPhrase,predPhrase),KB))
                pred = getattr(self,predPhrase)
                logger.debug('Thing.interpret 3.5 %r',self)
                for detail in pred[:]:
                    if isinstance(detail,Side):
                        if hasattr(self,'side'): self.side.append(detail)
                        else: self.side = [detail]
                        pred.remove(detail)
                    elif isinstance(detail,Verify):
                        pred.extend(detail.desc[:])
                        pred.remove(detail)
                    elif isinstance(detail,Meaning) and predPhrase not in self.knownAdj:
                        pred.append(Thing(value=detail))
                        pred.remove(detail)
                    elif detail == 'NOT':
                        pred.remove(detail)
                        for p in pred: p.negate = True
                    elif isinstance(detail,str) and not detail in Structurals.values():
                        logger.debug('Thing.interpret 3.75 removing %r',detail)
                        pred.remove(detail)
                logger.debug('Thing.interpret 4 %r',self)
        for thing_n in [p for p in listNouns(ObjectTypes) if p in features]:
            try:
                self.value = meanings(sss,thing_n,KB)[0]
            except IndexError:
                logger.warning('Thing.interpret : Cannot interpret %s from %s',thing_n,sss)
        if any(features,('Not','NOT')): self.negate = True
        if self.value == Wall: self.dist = '0'
        logger.debug('Thing.interpret 5 %r',self)
    
    def condSetSideDist(self,(side,dist)):
        if not hasattr(self,'side') or not self.side: self.side = side
        if not hasattr(self,'dist') or not self.dist: self.dist = dist

class Verify(CompoundActionSpecification):
    """
    >>> Verify(SSS.parse("[S=[Obj=[Det=[MEAN='THE', TAG='DT', TEXT='the'], Obj_n=[MEAN='wall_N_1', TAG='NN', TEXT='wall']], Is_v=[MEAN='be_V_[1,2,3,4,5]', TAG='VBZ', TEXT='is'], Side=[Side_p=[MEAN='ON', TAG='IN', TEXT='on'], Dir=[MEAN='left_V_1', TAG='VBN', TEXT='left', Det=[MEAN='YOUR', TAG='PRP$', TEXT='your']]]]]"))
    Verify(desc=[Thing(dist='0', value=Wall, type='Obj', side=[Left])])
    >>> Verify(SSS.parse("[Ref=[MEAN='THIS', TAG='DT', TEXT='this'], Is_v=[MEAN='be_N_[1,2,3,4,5]', TAG='VBZ', TEXT='is'], Position=[P_name=[MEAN='2', TAG='CD', TEXT='2'], Pos_n=[MEAN='position_N_1', TAG='NN', TEXT='position']], Punct=[MEAN='.', TAG='.', TEXT='.']]"))
    Verify(goal=DeclareGoal(goal=['2']))
    >>> #Verify(goal=DeclareGoal(cond=Travel(until=Verify(desc=[Thing(dist='0', type='Obj', side=[Front], value=Wall)])), goal=['2']))
    >>> Verify(desc=[Thing(type='Obj', side=[Front], value=Wall)])
    Verify(desc=[Thing(value=Wall, type='Obj', side=[Front])])
    >>> Verify(SSS.parse("[S=[,=[MEAN=','], Loc=[Ref=[MEAN='HERE']], S=[Path=[Appear=[MEAN='grassy_ADJ_1'], Det=[MEAN='THE']], Path_8=[Appear=[MEAN='yellow_ADJ_1'], Det=[MEAN='THE'], Path_n=[MEAN='path_N_2,3,4']], Struct_v=[MEAN='meet_V_3']]], S_14=[Agent=[MEAN='YOU'], Is_v=[MEAN='be_V_1,2,3,4,5'], Loc=[Loc_p=[MEAN='AT'], Position=[P_name=[MEAN='4'], Pos_n=[MEAN='position_N_1']]]], Then=[MEAN='AND']]"))
    Verify(desc=[Thing(Struct_n=Intersection, dist='0', type='Struct', Detail=[Thing(dist='0', Appear=[Grass], value=Path, type='Path', side=[At]), Thing(dist='0', Appear=[Honeycomb], value=Path, type='Path', side=[At])], value=Intersection, side=[At])])
    >>> Verify(SSS.parse("[Ex=[MEAN='THERE'], Is_v=[MEAN=\\"are_V_('be', [1, 2, 3, 4, 5])\\"], Obj=[Count=[MEAN='two_ADJ_1'], Obj_n=[MEAN=\\"pictures_N_('picture', [2])\\"], Detail=[Detail_p=[MEAN='OF'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='eiffel_N_1']]], On=[On_p=[MEAN='ON'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN=\\"walls_N_('wall', [1])\\"]]]]]"))
    Verify(desc=[Thing(Count=[2], On=[Thing(dist='0', value=Wall, type='Obj')], Detail=[Thing(value=Eiffel, type='Obj')], value=Pic, type='Obj')])
    >>> Verify(SSS.parse("[,=[MEAN=','], Agent=[MEAN='YOU'], Cond=[Arrive=[Agent=[MEAN='YOU'], Arrive_v=[MEAN='reach_V_1'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='stool_N_1']]], Cond_p=[MEAN='WHEN']], Is_v=[MEAN=\\"are_V_('be', [1, 2, 3, 4, 5])\\"], Loc=[Loc_p=[MEAN='AT'], Position=[P_name=[MEAN='7'], Pos_n=[MEAN='POSTION']]]]"))
    Verify(goal=DeclareGoal(cond=Travel(until=Verify(desc=[Thing(dist='0', value=Barstool, type='Obj', side=[At])])), goal=['7']))
    >>> Verify(SSS.parse("[Arrive_v=[MEAN='reach_V_1'], Struct=[Det=[MEAN='AN'], Struct_n=[MEAN='intersection_N_2'], Part=[Part_p=[MEAN='WITH'], Path=[Appear=[Appear=[MEAN='black_ADJ_1'], Cc=[MEAN='AND'], Appear_10=[Appear=[MEAN='normal_ADJ_1'], Appear_10=[MEAN='concrete_ADJ_2']]]]]]]"))
    Verify(desc=[Thing(dist='0', Part=[Thing(dist='0', Appear=[Cement, Stone], value=Path, type='Path')], value=Intersection, type='Struct', side=[At])])
    >>> Verify(SSS.parse("[Obj=[Det=[MEAN='THE'], Obj_n=[MEAN=\\"butterflies_N_('butterfly', [1])\\"]], Is_v=[MEAN=\\"are_V_('be', [1, 2, 3, 4, 5])\\"], On=[On_p=[MEAN='ON'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='wall_N_1']]]]"))
    Verify(desc=[Thing(On=[Thing(dist='0', value=Wall, type='Obj')], dist='0', value=Butterfly, type='Obj')])
    >>> Verify(SSS.parse("[Loc=[Between=[Between_p=[MEAN='BETWEEN'], Obj=[Obj=[Count=[MEAN='two_ADJ_1'], Appear=[MEAN='black_ADJ_1'], Obj_n=[MEAN=\\"floors_N_('floor', [1])\\"]], Cc=[MEAN='AND'], Obj_17=[Count=[MEAN='one_ADJ_1,2'], Appear=[MEAN='green_ADJ_1'], Obj_n=[MEAN='floor_N_1']]]]]]"))
    Verify(desc=[Thing(dist='0', value=Position, between=[Thing(Count=[1], dist='1:', Appear=[Greenish], value=Path, type='Obj', side=[Between]), Thing(Count=[2], dist='1:', Appear=[Stone], value=Path, type='Obj', side=[Between])], type='Position', side=[At])])
    >>> Verify(SSS.parse("[Struct_s=[Agent=[MEAN='YOU'], Travel_v=[MEAN='go_V_1'], Dir=[MEAN='left_ADV_1'], Cc=[MEAN='OR'], Toward=[Toward_p=[MEAN='TOWARDS'], Struct=[Det=[MEAN='A'], Struct_n=[MEAN='dead end_N_1']]]]]"))
    Verify(desc=[Thing(value=Intersection, Part=[Thing(dist='1:', value=DeadEnd, type='Struct', side=[Front]), Thing(dist='0', Path_n=Path, value=Path, type='Path', side=[Left])], type='Struct')])
    >>> Verify(SSS.parse("[Ex=[MEAN='THERE'], Is_v=[MEAN=\"are_V_('be', [1, 2, 3, 4, 5])\"], Obj=[Cc=[MEAN='AND'], Obj=[Appear=[MEAN='blue_ADJ_1'], Obj_n=[MEAN='segment_N_1']], Obj_2=[Appear=[MEAN='pink_ADJ_1'], Obj_n=[MEAN=\"flowers_N_('flower', [1])\"]]]]"))
    Verify(desc=[Thing(Appear=[BlueTile], value=Segment, type='Obj'), Thing(Appear=[Rose], value=Rose, type='Obj')])
    """
    knownClauses = ('Arrive','Cond','Desc','Purpose','Struct_s','S')
    knownVerbs = ('Has_v','Is_v','See_v','Arrive_v','Struct_v')
    knownAttrs = (('Cc', 'Dir', 'Ex', 'MEAN', 'Position', 'Punct', 'Ref', 'Side_n', 'Orient_v', 'Turn',
                   'desc', 'View')
                  +ObjectTypes+Thing.knownPrepPhrases+Thing.knownAdj+knownVerbs+knownClauses+
                  listPrepositions(Thing.knownPrepPhrases+knownClauses))
    
    def __init__(self, sss=[], **kargs):
        CompoundActionSpecification.__init__(self,sss,**kargs)
        if not hasattr(self,'desc') or not self.desc: self.desc = []
        self.checkUnknownAttrs()
        if not sss or not sss.feature_names(): return
        
        if not sss.has_key('Loc'):
            self.dist = []
            self.between = []
        else:
            self.dist=grab(sss,('Loc','Dist'))
            if self.dist:
                self.dist = Distance(self.dist)
                sss['Loc']=[]
            self.between=grab(sss,('Loc','Between'))
            if self.between:
                desc = Verify(self.between).desc
                setSideDist(desc,[Between],'1:')
                self.desc.append(Thing(between=desc,value=Position, type='Position', side=[At], dist='0'))
                del self.between
                sss['Loc']=[]

        # Handle "where you (can) turn|travel to the right..."
        turnIntDesc = None
        if (all(('Agent','Turn_v'),sss.feature_names())
            or 'Turn_v' in sss.feature_names() and 'Dir' in sss.feature_names() and Options.RecognizeDirTurn):
            turnIntDesc = Face(sss)
        elif all(('Agent','Travel_v'), sss.feature_names()):
            turnIntDesc = Travel(sss)
            turnIntDesc = hasattr(turnIntDesc,'face') and turnIntDesc.face or []
            if (grab(sss, ('Travel_v', 'Not'))
                or (grab(sss, ('Travel_v', 'Aux_v')) and grab(sss, ('Travel_v', 'Aux_v')).endswith('not'))
                and turnIntDesc.face and Options.RecognizeNegativeCompound):
                turnIntDesc.negate = True
        elif 'Turn' in sss.feature_names() and Options.RecognizeDirTurn:
            self.desc.extend(Turn(grab(sss,'Turn')).location.until.desc)

        if turnIntDesc and Options.RecognizeFictiveTurnIntersections:
            pathDesc = hasattr(turnIntDesc,'faced') and turnIntDesc.faced and turnIntDesc.faced.desc or []
            if hasattr(turnIntDesc,'direction'):
                pathDesc.append(Thing(value=Path, Path_n=Path, type='Path', dist='0', side=turnIntDesc.direction))
            if hasattr(turnIntDesc, 'Order_adj'):
                Order_adj = turnIntDesc.Order_adj
            else: Order_adj = []
            self.desc.append(Thing(value=Intersection, type='Struct', Part=pathDesc, Order_adj=Order_adj))
            return

        logger.debug('Verify(%r,%r)', sss,kargs)
        self.descSSS = SurfaceSemanticsStructure()
        if not (isinstance(sss,list) or isinstance(sss,SurfaceSemanticsStructure)): sss = [sss]
        for key,subsss in sss.items():
            if ((not uninterpretable(key) or key in ('Arrive_v','Orient_v','Struct_v'))
                and isCanonical(key)):
                self.descSSS[key] = meanings(sss,key,KB)
                logger.debug('Verify[%s] = %r)', key, self.descSSS[key])
        logger.debug('Verify => %r', self.descSSS)
        self.interpretUtterance(self.descSSS)

        singular_pronoun = (grab(sss,('Ref')) and grab(sss,('Ref','MEAN'))
                            and grab(sss,('Ref','MEAN'))['MEAN'] in ('THIS','IT'))
        self.goal = (grab(sss,'Position') or grab(sss,('Loc','Position')) or singular_pronoun)
        if self.goal:
            if sss.has_key('Is_v') or sss.has_key('Has_v'): setSideDist(self.desc,dist='0')
            if (Options.DeclareGoalIdiom and 'Ref' in self.descSSS.feature_names() and singular_pronoun):
                # Treat reference as referring to current spot, not last described object
                for d in self.descSSS['Ref']:
                    if d in self.desc: self.desc.remove(d)
            self.goal = DeclareGoal(goal=self.goal,cond=self.desc,dist=self.dist)
            self.dist = self.desc = []
        del self.descSSS
    
    def interpretUtterance(self,descSSS):
        self.clauseInterpreted = False
        for key,clauseList in descSSS.items():
            logger.debug('Verify.interpretUtterance %r = %r', key, clauseList)
            if isinstance(clauseList,list): self.interpretClauseList(descSSS,key,clauseList)
        if self.clauseInterpreted:
            features = descSSS.feature_names()
            self.interpretClauseThings(descSSS,features)
            self.interpretClauseModifiers(descSSS)
        del self.clauseInterpreted
        logger.debug('Verify.interpretUtterance %r', descSSS)
        logger.debug('Verify.interpretUtterance => %r', self.desc)
    
    def interpretClauseList(self,descSSS,clause,clauseSSS):
        if clause in ('Against', 'Side'): return
        self.clauseInterpreted = True
        for item in clauseSSS:
            logger.debug('Verify.interpretClauseList pre item %r %r', item, self.desc)
            if isinstance(item,Verify):
                if isinstance(item.desc[0],Verify): item = item.desc[0]
                self.desc.extend(item.desc)
                if hasattr(item,'declareGoal'): self.declareGoal = item.declareGoal
            if isinstance(item,Travel):
                if hasattr(item,'until') and item.until:
                    self.desc.extend(item.until.desc)
            elif isinstance(item,Thing):
                if clause in SideDist:
                    #item.side,item.dist = SideDist[clause]
                    item.condSetSideDist(SideDist[clause])
                self.desc.append(item)
            elif isinstance(item,Meaning):
                pass
##            elif isinstance(item,SurfaceSemanticsStructure):
##                item_meaning = interpret(item, 'Ref')
##                if item_meaning: self.desc.append(item_meaning)
##                logger.debug('Verify.interpretClauseList found SSS %r, interpretted as %r, %r', item, item_meaning, self.desc)
            else:
                errorString = 'Verify.interpretClauseList() Unknown item %r, type %r' % (item, type(item))
                logger.debug(errorString)
                #raise ValueError(errorString)
            logger.debug('Verify.interpretClauseList post item %r %r', item, self.desc)
    
    def interpretClauseThings(self,descSSS,features):
        logger.debug('Verify.interpretClauseThings %r', self.desc)
        for thingName in [p for p in features if p in ObjectTypes]:
            for thing in [t for t in self.desc if t.type == thingName]:
                if not isinstance(thing,Thing):
                    logger.warning('Not updating thing %r with predicates',thing)
                    return
                thing.condSetSideDist(getSideDist(descSSS))
                for Pred in [p for p in features if p in Thing.knownPrepPhrases + Thing.knownAdj]:
                    logger.debug('Verify.interpretClauseThings predicate: %s %r %r', Pred,thing,descSSS[Pred])
                    if Pred == 'Side':
                        sideVals = descSSS[Pred]
                        if isinstance(sideVals,list):
                            if len(sideVals)>1 and Sides in sideVals: sideVals.remove(Sides)
                            thing.side = sideVals
                        else: thing.side = [sideVals]
                    elif Pred == 'Loc' and descSSS[Pred] and isinstance(descSSS[Pred][0],str):
                        pass
                    elif Pred == 'Between':
                        pass
                    elif Pred == 'Obj_adj' and descSSS[Pred]:
                        if not hasattr(thing,'Detail'): thing.Detail = []
                        thing.Detail.append(Thing(value=descSSS[Pred][0], type='Obj',dist=thing.dist,side=thing.side))
                    else:
                        for predVal in descSSS[Pred]:
                            if isinstance(predVal,Verify): predVal = predVal.desc[0]
                            if predVal == thing: continue
                            if hasattr(predVal,'side') and hasattr(predVal,'dist'):
                                thing.condSetSideDist((predVal.side,predVal.dist))
                            if hasattr(thing,Pred) and isinstance(getattr(thing,Pred),list):
                                getattr(thing,Pred).append(predVal)
                            else: setattr(thing,Pred,[predVal])
                            if predVal in self.desc: self.desc.remove(predVal)
                logger.debug('Verify.interpretClauseThings %r',thing)
        for thing in self.desc:
            if thing.value == Wall: thing.dist = '0'
    
    def interpretClauseModifiers(self,descSSS):
        modifier = ''
        features = descSSS.feature_names()
        if 'Arrive_v' in features and Options.RecognizeArriveFrame:
            setSideDist(self.desc,[At],'0')
            modifier = 'Arrive_v'
        if 'Orient_v' in features:
            for d in self.desc:
                if not d.side: d.side = [Front]
            modifier = 'Orient_v'
        if 'Struct_v' in features and Options.RecognizeStructFrame:
            setSideDist(self.desc,[At],'0')
            self.desc = [Thing(value=descSSS['Struct_v'],
                               Struct_n=descSSS['Struct_v'],
                               type='Struct',
                               side = [At],
                               dist = '0',
                               Detail=self.desc,
                               )]
            modifier = 'Struct_v'
        if 'Against' in features: #Handle with the chair to your back
            descList = []
            for desc in copy.deepcopy(self.desc):
                for againstObj in descSSS['Against']:
                    if isinstance(desc.value,Side): againstObj.side = [desc.value]
                    againstObj.dist = '0:'
                    if againstObj.value == Wall: againstObj.dist = '0'
                    descList.append(againstObj)
            self.desc = descList
            modifier = 'Against'
        if 'Side' in features:
             setSideDist(self.desc,descSSS['Side'])
             modifier = 'Side'
        if modifier:
            self.clauseInterpreted = True
            logger.debug('Verify.interpretModifiers %s = %s', modifier, self.desc)
    
    def execute(self,robot):
        if hasattr(self,'goal') and self.goal:
            return self.update('declareGoal',self.goal.execute(robot),robot)
        try:
            Success = robot.recognize(self.desc)
        except KeyError,e:
            raise KeyError('Unknown recognizer',str(e))
        return Success

class Turn(CompoundActionSpecification):
    """
    >>> Turn(SSS.parse("[Turn_v=[MEAN='turn_N_1,2', TAG='VB', TEXT='turn'], Purpose=[Purpose_p=[MEAN='SO', TAG='RB', TEXT='so'], S=[Obj=[Det=[MEAN='THE', TAG='DT', TEXT='the'], Obj_n=[MEAN='wall_N_1', TAG='NN', TEXT='wall']], Is_v=[MEAN='be_N_[1,2,3,4,5]', TAG='VBZ', TEXT='is'], Side=[Side_p=[MEAN='ON', TAG='IN', TEXT='on'], Det=[MEAN='YOUR', TAG='PRP$', TEXT='your'], Side_n=[MEAN='back_N_1', TAG='RB', TEXT='back']]]], Punct=[MEAN='.', TAG='.', TEXT='.']]"))
    Turn(face=Face(faced=Verify(desc=[Thing(dist='0', value=Wall, type='Obj', side=[Back])])))
    >>> Turn(SSS.parse("[Cond=[Cond_p=[MEAN='WITH', TAG='IN', TEXT='with'], Obj=[Det=[MEAN='YOU', TAG='PRP', TEXT='you'], Obj_n=[MEAN='back_ADV_1,2', TAG='RB', TEXT='back']], Against=[Against_p=[MEAN='TO', TAG='TO', TEXT='to'], Obj=[Det=[MEAN='THE', TAG='DT', TEXT='the'], Obj_n=[MEAN='wall_N_1', TAG='NN', TEXT='wall']]]], Turn_v=[MEAN='turn_N_1,2', TAG='NN', TEXT='turn'], Dir=[MEAN='left_V_1', TAG='VBN', TEXT='left'], Punct=[MEAN='.', TAG='.', TEXT='.']]"))
    Turn(direction=[Left], precond=Face(faced=Verify(desc=[Thing(dist='0', value=Wall, type='Obj', side=[Back])])))
    >>> Turn(SSS.parse("[Cond=[Cond_p=[MEAN='WITH', TAG='IN', TEXT='with'], Obj=[Det=[MEAN='YOU', TAG='PRP', TEXT='you'], Obj_n=[MEAN='back_ADV_1,2', TAG='RB', TEXT='back']], Against=[Against_p=[MEAN='TO', TAG='TO', TEXT='to'], Obj=[Det=[MEAN='THE', TAG='DT', TEXT='the'], Obj_n=[MEAN='easel_N_1', TAG='NN', TEXT='easel']]]], Turn_v=[MEAN='turn_N_1,2', TAG='NN', TEXT='turn'], Dir=[MEAN='left_V_1', TAG='VBN', TEXT='left'], Punct=[MEAN='.', TAG='.', TEXT='.']]"))
    Turn(direction=[Left], precond=Face(faced=Verify(desc=[Thing(dist='0:', value=Easel, type='Obj', side=[Back])])))
    >>> Turn(SSS.parse("[,=[MEAN=','], Cond=[Arrive=[Agent=[MEAN='YOU'], Arrive_p=[MEAN='TO'], Arrive_v=[MEAN='come_V_1,2,3,4'], Path=[Appear=[Appear=[MEAN='red_ADJ_1'], Appear_7=[MEAN='brick_N_1'], Path_n=[MEAN='path_N_2,3,4']], Det=[MEAN='A'], Path_n=[MEAN='path_N_2,3,4']]], Cond_p=[MEAN='WHEN']], Dir=[MEAN='left_ADV_1'], Turn_v=[MEAN='go_V_1']]"))
    Turn(direction=[Left], precond=Travel(until=Verify(desc=[Thing(dist='0', Appear=[Brick], value=Path, type='Path', side=[At])])))
    >>> Turn(SSS.parse("[Turn_v=[MEAN='turn_N_1,2'], Dir=[MEAN='left_ADV_1'], Purpose=[Purpose_p=[MEAN='TO'], Orient_v=[MEAN='face_V_5'], Path=[Det=[MEAN='THE'], Appear=[MEAN='yellow_ADJ_1'], Path_n=[MEAN='hallway_N_1']]]]"))
    Turn(face=Face(faced=Verify(desc=[Thing(Appear=[Honeycomb], value=Path, type='Path', side=[Front])]), direction=[Left]))
    >>> Turn(SSS.parse("[Turn_v=[MEAN='turn_N_1,2'], Dir=[MEAN='right_ADV_4'], View=[Agent=[MEAN='YOU'], See_v=[MEAN='see_V_1'], Obj=[Appear=[MEAN='gray_ADJ_1'], Obj_n=[MEAN='floor_N_1'], Detail=[Detail_p=[MEAN='WITH'], Obj=[Det=[MEAN='THE'], Appear=[MEAN='red_ADJ_1'], Obj_n=[MEAN='brick_N_1'], On=[On_p=[MEAN='ON'], Path=[Det=[MEAN='THE'], Order_adj=[MEAN='next_ADJ_3'], Path_n=[MEAN='alley_N_1']]]]]]]]"))
    Turn(face=Face(faced=Verify(desc=[Thing(Appear=[Gray], Detail=[Thing(On=[Thing(value=Path, Order_adj=[1], type='Path')], Appear=[Brick], value=Brick, type='Obj')], value=Path, type='Obj', side=[Front])]), direction=[Right]))
    >>> Turn(SSS.parse("[Turn_v=[MEAN='make_V_16'], Turn=[Det=[MEAN='THE'], Order_adj=[MEAN='first_ADJ_1'], Dir=[MEAN='left_ADV_1']]]"))
    Turn(direction=[Left], location=Travel(until=Verify(desc=[Thing(dist='0', Part=[Thing(dist='0', Path_n=Path, value=Path, type='Path', side=[Left])], value=Intersection, Order_adj=[1], type='Struct', side=[At])])))
    >>> Turn(SSS.parse("[Turn_v=[MEAN='take_V_1,11,40'], Turn=[Det=[MEAN='A'], Dir=[MEAN='right_ADV_4']], Onto=[Onto_p=[MEAN='ONTO'], Path=[Appear=[MEAN='black_ADJ_1'], Det=[MEAN='THE'], Path_n=[MEAN='path_N_2,3,4']]], Dist=[MEAN='ALL THE WAY DOWN'], Until=[Until_p=[MEAN='UNTIL'], Arrive=[Agent=[MEAN='YOU'], Arrive_v=[MEAN='reach_V_1'], Obj=[Appear=[MEAN='black_ADJ_1'], Det=[MEAN='THE'], Obj_n=[MEAN='easel_N_1']]]]]"))
    Turn(postcond=Travel(distance=[Distance()], until=Verify(desc=[Thing(dist='0', Appear=[Stone], value=Easel, type='Obj', side=[At])])), face=Face(faced=Verify(desc=[Thing(dist='0', Appear=[Stone], value=Path, type='Path', side=[Front])]), direction=[Right]), location=Travel(until=Verify(desc=[Thing(dist='0', Part=[Thing(dist='0', Path_n=Path, value=Path, type='Path', side=[Right])], value=Intersection, type='Struct', side=[At])])))
    """
    knownVerbs = ('Turn_v',)
    knownClauses = ('View',)
    knownFacePhrases = ('Away','Onto','Purpose','Toward','Until','View')
    knownPhrases = ('Across','Cond', 'Dir', 'Loc','Turn')
    knownModelAttribs = ('Dist','Side','Struct_s','direction','face','location','precond','postcond','view')
    viewPhrases = ('View','Agent')
    knownAttrs = knownModelAttribs+knownFacePhrases+knownPhrases+knownVerbs+knownClauses
    view = False
    def __init__(self, sss=[], **kargs):
        for attrib in Turn.knownModelAttribs: setattr(self,attrib,[])
        CompoundActionSpecification.__init__(self,sss,**kargs)
        self.checkUnknownAttrs()
        if not sss or not sss.feature_names():
            return
        
        if Options.TurnPreResetCache: Thing.referenceCache.clear()
        
        if grab(sss,('Struct_s','Agent')) and Options.RecognizeStructAgentFrame:
            sss = sss['Struct_s']
        
        self.direction = meanings(sss,'Dir',Directions) or meanings(sss,('Side','Dir'),Directions)

        self.precond = (grab(sss,'Cond') or grab(sss,('S','Cond')))
        if self.precond:
            if ((grab(self.precond,('Cond','Arrive')) or grab(self.precond,('Cond','Struct_s')))
                 and Options.RecognizeArriveFrame and Options.ImplicitTravel):
                self.precond = Travel(SSS(Until=self.precond['Cond']))
            else:
                self.precond = Face(faced=Verify(self.precond))
        
        if not grab(sss,('Loc','Position')):
            self.location = grab(sss,'Loc')
        if self.location:
            if grab(sss,('Loc','Ref')): self.location = []
            else: self.location = Travel(SSS(Until=self.location['Loc']))

        turnSSS = grab(sss,'Turn')
        if turnSSS and Options.RecognizeTakeTurnFrame and Options.ImplicitTravel:
            turnInt = Travel(SSS(Until=SSS(Struct=turnSSS['Turn'])))
            if 'Dir' in turnSSS['Turn'].feature_names():
                side = meanings(turnSSS,('Turn','Dir'),KB)
                self.direction = side
            else:
                side = [Sides]
            if turnInt.until.desc:
                turnInt.until.desc[0].Part = [Thing(value=Path, Path_n=Path, type='Path', dist='0', side=side)]
            if self.location:
                if not self.location.until:
                    self.location.until = turnInt.until
                else: self.location.until.desc.extend(turnInt.until.desc)
            else: self.location = turnInt
        
        if Options.TurnTermResetCache: Thing.referenceCache.clear()
        self.postcond = grab(sss,'Dist')
        if grab(sss,'Until') and not (grab(sss,('Until','View')) or grab(sss,('Until','S'))):
            if self.postcond: self.postcond['Until'] = grab(sss,'Until')['Until']
            else: self.postcond = grab(sss,'Until')
        if grab(sss,'Across'):
            if self.postcond: self.postcond['Across'] = grab(sss,'Across')['Across']
            else: self.postcond = grab(sss,'Across')
        if self.postcond and Options.TurnPostcond:
            self.postcond = Travel(self.postcond)
        
        if not hasattr(self,'faced'): #self is not already Face
            if not (hasattr(self,'face') and self.face): self.face = SSS()
            for pred in sss.feature_names():
                if (pred in Turn.knownFacePhrases
                    and getattr(Options, 'Face'+pred)
                    and not uninterpretable(pred)
                    and not (pred == 'Until' and (self.postcond or grab(sss,('Until','View')) or grab(sss,('Until','S'))))
                    ):
                    self.face[pred] = grab(sss,pred)
                if pred in self.viewPhrases: self.view = True
                if pred == 'Until' and (grab(sss,('Until','View')) or grab(sss,('Until','S'))) and Options.RecognizeUntilView:
                    self.face['View'] = grab(sss,('Until','View')) or grab(sss,('Until','S'))
                logger.debug('Turn.Face interpretting sss: %r %r',pred,self)
        if self.face:
            self.face = Face(faced=Verify(self.face),direction=self.direction)
            #Remove null Verify faces
            if self.face.faced and not self.face.faced.desc: self.face=[]
            elif self.face.direction: self.direction=[]
        if Options.TurnPostResetCache: Thing.referenceCache.clear()
    
    pathDetect = Verify(desc=[Thing(dist='0', value=Path, type='Path', side=[])])
    def getDirection(self,robot):
        if (not self.direction or self.direction in ([Back],[Front]) or not Options.TurnDirection):
            direction = random.choice((Left,Right))
            self.pathDetect.desc[0].side = [direction]
            if Options.TurnTowardPath and self.pathDetect.execute(robot):
                return direction
            else: return opposite(direction)
        elif len(self.direction)==1:
            if (hasattr(self, 'index') and self.index != len(self.plan())-1
                and isinstance(self.plan()[self.index+1], Travel)
                and Options.ImplicitTravel and Options.ReverseTurn):
                self.pathDetect.desc[0].side = self.direction
                if Options.TurnTowardPath and not self.pathDetect.execute(robot):
                    return opposite(self.direction[0])
            return self.direction[0]
        else:
            logger.warning("Multiple directions: %s", self.direction)
            return self.direction[0]
    
    def execute(self,robot):
        try: consecTurns = self.index > 1 and isinstance(self.plan()[self.index-1], Turn)
        except AttributeError: consecTurns = False
        if self.location and Options.TurnLocation and Options.ImplicitTravel:
            if hasattr(self,'index') and hasattr(self,'plan') and Options.PropagateContextInfo:
                self.location.index, self.location.plan = self.index, self.plan
            self.update('location',self.location.execute(robot),robot)
        if self.precond and Options.TurnPrecond:
            if ((isinstance(self.precond,Travel) and Options.ImplicitTravel)
                or (isinstance(self.precond,Turn) and Options.ImplicitTurn)):
                self.update('precond',self.precond.execute(robot),robot)
        
        if self.face: self.update('face',self.face.execute(robot),robot)
        elif self.direction != [Front]:
            if (consecTurns and not self.location and not self.precond
                and Options.TravelBetweenTurns and Options.ImplicitTravel):
                self.update('Traveling for consecutive turns', robot.travel(), robot)
            direction = self.getDirection(robot)
            self.update('turn',robot.turn(direction),robot)
            if self.direction == [Back]:
                self.update('turn again',robot.turn(direction),robot)
        if self.postcond and Options.TurnPostcond and Options.ImplicitTravel:
            self.update('postcond',self.postcond.execute(robot),robot)

class Face(Turn):
    """
    >>> Face(SSS.parse("[Against=[Against_p=[MEAN='TO'], Struct=[Det=[MEAN='THE'], Struct_n=[MEAN='intersection_N_2'], Structural=[MEAN='t_N_5']]], Obj=[Det=[MEAN='YOUR'], Obj_n=[MEAN='back_N_1']], Orient_v=[MEAN='place_N_1']]"))
    Face(faced=Verify(desc=[Thing(dist='0:', value=Intersection, type='Struct', side=[Back], Structural=[T_Int])]))
    >>> Face(SSS.parse("[Orient_v=[MEAN='face_N_5'], Path=[Appear=[MEAN='octagon_ADJ_1'], Det=[MEAN='THE'], Path_n=[MEAN='carpet_N_1']]]"))
    Face(faced=Verify(desc=[Thing(Appear=[Honeycomb], value=Path, type='Path', side=[Front])]))
    >>> Face(SSS.parse("[Orient_v=[MEAN='orient_V_3'],Agent=[MEAN='YOURSELF'], On=[On_p=[MEAN='ALONG'], Path=[Appear=[MEAN='stone_ADJ_1'], Det=[MEAN='THE'], Path_n=[MEAN='hallway_N_1']]], Cond=[Cond_p=[MEAN='WITH'], Struct=[Det=[MEAN='THE'], Struct_n=[MEAN='end_N_1'], Detail=[Has_v=[MEAN=\\"containing_V_('contain', [1])\\"], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='easel_N_1'], Side=[Side_p=[MEAN='TO'], Det=[MEAN='YOUR'], Side_n=[MEAN='back_N_1']]]]]]]"))
    Face(faced=Verify(desc=[Thing(Detail=[Thing(value=Easel, side=[Back], type='Obj')], value=End, type='Struct', side=[Front]), Thing(Appear=[Stone], value=Path, type='Path', side=[Front])]), precond=Face(faced=Verify(desc=[Thing(Detail=[Thing(value=Easel, side=[Back], type='Obj')], value=End, type='Struct', side=[Front])])))
    >>> Face(SSS.parse("[Loc=[Loc_p=[MEAN='AT'], Position=[P_name=[MEAN='ONE']]], Orient_v=[MEAN='face_V_5'], Path=[Det=[MEAN='THE'], Structural=[MEAN='long_ADJ_2'], Appear=[MEAN='brick_N_1'], Path_n=[MEAN='alley_N_1']]]"))
    Face(faced=Verify(desc=[Thing(dist='0', Appear=[Brick], value=Path, type='Path', side=[Front], Structural=['Long'])]))
    """
    knownClauses = ('Against','Orient')
    knownVerbs = ('Orient_v',)
    knownModelAttribs = ('faced',)
    knownAttrs = (CompoundActionSpecification.casKnownAttributes
                  +Turn.knownAttrs+Verify.knownAttrs
                  +knownModelAttribs+knownVerbs+knownClauses)
    def __init__(self, sss=[], faced=[], **kargs):
        for attrib in self.knownModelAttribs: setattr(self,attrib,[])
        if not self.faced: self.faced = Verify()
        Turn.__init__(self,sss,**kargs)
        if faced:
            if isinstance(faced,list):
                self.faced.desc.extend(Verify(desc=faced).desc)
            elif isinstance(faced,Verify):
                self.faced.desc.extend(faced.desc)
            else:
                raise TypeError('Face',faced)
                logger.warning('Face: Unknown type for %r', faced)
        if sss and Options.FaceExplicit:
            descSSS = SurfaceSemanticsStructure()
            for key,subsss in sss.items():
                if not uninterpretable(key):
                    descSSS[key] = grab(sss,key)
                    if key in self.viewPhrases: self.view = True
            self.faced.desc.extend(Verify(descSSS).desc)
            logger.debug('Face interpretting sss: %r %r',descSSS,self)
        for d in self.faced.desc:
            if d.side in ([At],[]) and d.value not in Objects.values(): d.side = [Front]
        self.checkUnknownAttrs()
    
    def travelToDistantView(self,immediateViewDescs,robot):
        logger.info("Can't find %r, trying to travel To DistantView %r", immediateViewDescs, self)
        sides = [d.side for d in immediateViewDescs if d.value]
        # Turn to distant view
        setSideDist(immediateViewDescs, dist='1:')
        faceDist = Face(faced=immediateViewDescs,inTravelToDistantView=True)
        self.update('travelToDistantView:Face',faceDist.execute(robot),robot)

        sideView = Verify()
        # Travel until the desired view is at the sides
        if Options.PerspectiveTaking: setSideDist(immediateViewDescs,[At],'0')
        sideView.desc = immediateViewDescs
        travel = Travel(until=sideView)
        self.update('travelToDistantView:Travel',travel.execute(robot),robot)

        for viewDesc,side in zip(immediateViewDescs,sides): viewDesc.side = side
        # Turn to face the desired view
        self.inTravelToDistantView=True
        self.update('travelToDistantView:reFace',self.execute(robot),robot)
        del self.inTravelToDistantView
    
    def execute(self,robot):
        if self.direction == [Front] and not self.faced.desc: return 0,[]        
        elif (self.direction and self.direction[0] not in (Front,Back)
              and Options.TurnExplicit and Options.TurnDirection):
            direction = self.direction[0]
            # Needed for Bad Queue Voodoo
            self.pathDetect.desc[0].side = [direction]
            self.pathDetect.execute(robot)
            self.update('Face: explicit direction',robot.turn(direction),robot)
        else:
            direction = self.getDirection(robot)
            if not self.faced.desc: self.update('Face: null desc',robot.turn(direction),robot)
        count = 0
        if not Options.FaceDescription: return count,[]
        while not self.faced.execute(robot):
            self.update('face',robot.turn(direction),robot)
            if count >= Options.FaceMaxTurns:
                immediateViewDescs = [d for d in self.faced.desc if d.dist == '0']
                if (immediateViewDescs and not hasattr(self,'inTravelToDistantView')
                    and Options.TravelToDistantView  and Options.PerspectiveTaking
                    and (Options.ImplicitTravel or Options.FindFaceTravel)):
                    self.travelToDistantView(immediateViewDescs,robot)
                elif (Options.FindFace and Options.UseFind
                      and (Options.ImplicitTravel or Options.FindFaceTravel)):
                    self.update('Face: find view',Find(until=self.faced, distant=True).execute(robot), robot)
                    break
                else:
                    #logger.info("Ignoring desc %s not in view.", self.faced)
                    #return count,[]
                    raise LookupError("Can't find", self.faced, "in", count, "turns.")
            count += 1
        if len(self.observations) > 1: obs = self.observations[-1]
        else: obs = []
        return self.cost,obs

class Distance(CompoundActionSpecification):
    """
    >>> Distance(SSS.parse("[Dist=[Count=[MEAN='three_ADJ_1', TAG='CD', TEXT='three'], Dist_unit=[MEAN=\\"segments_N_('segment', [1])\\", TAG='NNS', TEXT='segments']]]"))
    Distance(count=3)
    >>> Distance(SSS.parse("[Dist=[Count=[MEAN='one_ADJ_1,2'], Dist_unit=[Path_n=[MEAN='alley_N_1']], Past=[Past_p=[MEAN='PAST'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='hatrack_N_1']]]]]"))
    Distance(count=1, past=Travel(until=Verify(desc=[Thing(dist='0', value=Hatrack, type='Obj', side=[At])])), distUnit=Verify(desc=[Thing(dist='0', value=Path, type='Path', side=[Sides])]))
    >>> Distance(SSS.parse("[Dist=[Count=[MEAN='one_ADJ_1,2'], Dist_unit=[Struct_n=[MEAN='block_N_2']], Before=[Before_p=[MEAN='BEFORE'], Struct=[Det=[MEAN='THE'], Struct_n=[MEAN='end_N_1']]]]]"))
    Distance(count=1, distUnit=Verify(desc=[Thing(dist='0', value=Block, type='Struct', side=[At])]), before=Travel(until=Verify(desc=[Thing(dist='1', value=End, type='Struct', side=[At])])))
    >>> Distance(SSS.parse("[Dist=[Count=[MEAN='two_ADJ_1'], Dist_unit=[Path_n=[MEAN='alley_N_1']], Away=[Away_p=[MEAN='AWAY'], Obj=[Det=[MEAN='A'], Obj_n=[MEAN='lamp_N_2']]]]]"))
    Distance(count=2, away=Verify(desc=[Thing(dist='0', value=Lamp, type='Obj', side=[At])]), distUnit=Verify(desc=[Thing(dist='0', value=Path, type='Path', side=[Sides])]))
    >>> Distance(SSS.parse("[Dist=[Count=[MEAN='one_ADJ_1,2'], Dist_unit=[Struct_n=[MEAN='intersection_N_2']], Past=[Past_p=[MEAN='PAST'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='chair_N_1']]]]]"))
    Distance(count=1, past=Travel(until=Verify(desc=[Thing(dist='0', value=GenChair, type='Obj', side=[At])])), distUnit=Verify(desc=[Thing(dist='0', value=Intersection, type='Struct', side=[At])]))
    """
    knownPrepPhrases = ('Away', 'Before', 'Past',)
    knownClauses = tuple() #('S', 'Desc')
    knownAttrs = (('Approx','Dist','approx','count', 'dist_unit', ) + listPrepositions(knownPrepPhrases) + knownPrepPhrases)
    def __init__(self, sss=[], count=0, distUnit=None, **kargs):
        CompoundActionSpecification.__init__(self,sss,**kargs)
        self.count = count
        self.distUnit = distUnit
        self.before = None
        self.past = None
        self.approx = False
        if not sss: return
        
        self.count = Distance.getCount(meanings(sss,('Dist','Count'),Counts)
                                       or meanings(sss,('Dist','Order_adj'),Counts)
                                       or 0)
        self.distUnit = grab(sss,('Dist','Dist_unit'))
        if self.distUnit and getNouns(self.distUnit['Dist_unit']):
            desc = [Thing(self.distUnit['Dist_unit'],
                          type=getNouns(self.distUnit['Dist_unit'])[0][:-2])]
            self.distUnit = self.makeVerify(desc)
            logger.debug('Distance.__init__ distUnit found %r',self.distUnit)
        else: self.distUnit = None
        self.approx = bool(grab(sss,('Dist','Approx')))
        
        for phrase in self.knownPrepPhrases:
            setattr(self, phrase.lower(),
                    self.makeVerify(meanings(sss, ('Dist',phrase), KB)))
        if self.before and Options.PerspectiveTaking:
            setSideDist(self.before.desc, dist=str(self.count))
            self.before = Travel(until=self.before)
        if self.past: self.past = Travel(until=self.past)
        self.checkUnknownAttrs()
    
    @classmethod
    def getCount(cls, count_list):
        count = 0
        if not count_list: logger.debug('Distance.__init__ no count found')
        elif len(count_list)>1:
            logger.warning('Multiple Counts for Distance %s',count_list)
            if [n for n in count_list if n<0] and Options.RecognizeLast:
                count = 1
                for n in count_list: count *= n
            else:
                count = count_list[0]
        elif len(count_list)==1: count = count_list[0]
        else: logger.error("Impossible count %s",count_list)
        return count
    
    def makeVerify(self,description):
        if not description: return
        for desc in description[:]:
            if not isinstance(desc, Thing):
                if isinstance(desc,Verify):
                    description.extend(desc.desc)
                    description.remove(desc)
                else:
                    description.remove(desc)
                    logger.error("Distance.makeVerify desc %s is not a Thing",desc)
                continue
            #Don't verify follower in on a path, but that it's at another!
            if desc.value == Path:
                desc.side = [Sides]
            else:
                desc.side = [At]
            desc.dist = '0'
        return Verify(desc=description)
    
    def execute(self,travel,robot):
        distUnitCount = 0

        if self.before and Options.DistanceBefore: return self.before.execute(robot)
        if self.past and Options.DistancePast: self.past.execute(robot)
        if not Options.DistanceCount:
            travel.update('distance <no count>',robot.travel(),robot)
        if self.count < 0 and Options.RecognizeLast:
            last = True
            logger.info('Travel Distance: handling Order_adj last')
            if self.distUnit:
                for d in self.distUnit.desc[:]:
                    # Go until we're at one of this description
                    # and there are n-1 in front for nth to last (0 for last)
                    descCp = copy.deepcopy(d)
                    descCp.dist = '1:'
                    descCp.Count = [-1*self.count-1]
                self.distUnit.desc.append(descCp)
                self.count = 1
                #del self.distUnit.Count
        else: last = False
        while ((distUnitCount < self.count) and not self.approx):
            cost,obs = robot.travel()
            travel.update('distance',(cost,obs),robot)
            if travel.past: passed = travel.past.execute(robot)
            if not self.distUnit or self.distUnit.execute(robot):
                distUnitCount += 1
            logger.info('Travel Distance: %r Count is %r of %r', self.distUnit, distUnitCount, self.count)
            if cost > 1:
                raise Warning('Hit a wall')
        return True

class FaceVerify(Face): pass

class Travel(CompoundActionSpecification):
    """
    >>> Travel(SSS.parse("[Travel_v=[MEAN='take_V_1,11,40', TAG='VB', TEXT='take'], Along=[Path=[Det=[MEAN='THE', TAG='DT', TEXT='the'], Appear=[MEAN='green_ADJ_1', TAG='JJ', TEXT='green'], Path_n=[MEAN='path_N_2,3,4', TAG='NN', TEXT='path']]], Until=[Until_p=[MEAN='TO', TAG='TO', TEXT='to'], Struct=[Det=[MEAN='THE', TAG='DT', TEXT='the'], Appear=[Appear=[MEAN='red_ADJ_1', TAG='JJ', TEXT='red'], Appear_8=[MEAN='brick_N_1', TAG='NN', TEXT='brick']], Struct_n=[MEAN='intersection_N_2', TAG='NN', TEXT='intersection']]], Punct=[MEAN='.', TAG='.', TEXT='.']]"))
    Travel(face=Face(faced=Verify(desc=[Thing(dist='0', Appear=[Greenish], value=Path, type='Path', side=[Front])])), until=Verify(desc=[Thing(dist='0', Appear=[Brick], value=Intersection, type='Struct', side=[At])]))
    >>> Travel(SSS.parse("[Travel_v=[MEAN='head_N_1', TAG='NN', TEXT='head'], Along=[Along_p=[MEAN='DOWN', TAG='RB', TEXT='down']], Toward=[Toward_p=[MEAN='TOWARD', TAG='IN', TEXT='toward'], Obj=[Det=[MEAN='THE', TAG='DT', TEXT='the'], Obj_n=[MEAN='futon_N_1', TAG='NN', TEXT='futon'], Punct=[MEAN='/', TAG='NN', TEXT='/'], Obj_n_7=[MEAN='bench_N_1', TAG='NN', TEXT='bench']]], Punct=[MEAN='.', TAG='.', TEXT='.']]"))
    Travel(distance=[Distance(count=1, distUnit=Verify(desc=[Thing(dist='0', value=TopoPlace, type='Struct', side=[At])]))], face=Face(faced=Verify(desc=[Thing(dist='1:', value=Furniture, type='Obj', side=[Front])])))
    >>> Travel(SSS.parse("[Travel_v=[MEAN='go_N_1', TAG='VB', TEXT='go'], Dir=[MEAN='forward_ADV_1', TAG='RB', TEXT='forward'], Dist=[Count=[MEAN='three_ADJ_1', TAG='CD', TEXT='three'], Dist_unit=[MEAN=\\"segments_N_('segment', [1])\\", TAG='NNS', TEXT='segments']], Along=[Along_p=[MEAN='DOWN', TAG='RB', TEXT='down'], Path=[Det=[MEAN='THE', TAG='DT', TEXT='the'], Appear=[Appear=[MEAN='yellow_ADJ_1', TAG='JJ', TEXT='yellow'], Hyphen=[MEAN='-', TAG=':', TEXT='-'], Appear_9=[MEAN='tiled_ADJ_1', TAG='NN', TEXT='tiled']], Path_n=[MEAN='hall_N_1', TAG='NN', TEXT='hall']]], Punct=[MEAN=',', TAG=',', TEXT=','], Past=[Pass_v=[MEAN=\\"passing_N_('pass', [1])\\", TAG='NN', TEXT='passing'], Obj=[Det=[MEAN='THE', TAG='DT', TEXT='the'], Obj_n=[MEAN='bench_N_1', TAG='NN', TEXT='bench']]], Punct_15=[MEAN='.', TAG='.', TEXT='.']]"))
    Travel(distance=[Distance(count=3)], face=Face(faced=Verify(desc=[Thing(dist='0', Appear=[Flooring, Honeycomb], value=Path, type='Path', side=[Front])]), direction=[Front]), past=Verify(desc=[Thing(dist='0', value=Sofa, type='Obj', side=[At])]))
    >>> Travel(SSS.parse("[Along=[Path=[Appear=[MEAN='grey_ADJ_1'], Det=[MEAN='THE'], Path_n=[MEAN='hall_N_1']]], Dist=[MEAN='ALL'], Then=[MEAN='THEN']]"))
    Travel(distance=[Distance()], face=Face(faced=Verify(desc=[Thing(dist='0', Appear=[Gray], value=Path, type='Path', side=[Front])])), until=Verify(desc=[Thing(value=Wall, dist='0', Obj_n=Wall, type='Obj', side=[Front])]))
    >>> Travel(SSS.parse("[Inf=[MEAN='TO'], Pathdir=[Count=[MEAN='one_ADJ_1,2'], Det=[MEAN='ONLY'], Pathdir_n=[MEAN='way_N_6']], Travel_v=[MEAN='go_N_1']]"))
    Travel(distance=[Distance()], face=Face(faced=Verify(desc=[Thing(Count=[1], dist='0', value=PathDir, type='Pathdir', side=[Front])])), until=Verify(desc=[Thing(Count=[1], dist='0', value=PathDir, type='Pathdir', side=[At])]))
    >>> Travel(SSS.parse("[Cond=[Against=[Against_p=[MEAN='TO'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='wall_N_1']]], Cond_p=[MEAN='WITH'], Obj=[Det=[MEAN='YOUR'], Obj_n=[MEAN='back_N_1']]], Dir=[MEAN='forward_ADV_1'], Travel_v=[MEAN='move_N_4'], Until=[Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='wall_N_1']], Until_p=[MEAN='TO']]]"))
    Travel(face=Face(faced=Verify(desc=[Thing(dist='0', value=Wall, type='Obj', side=[Back])])), until=Verify(desc=[Thing(dist='0', value=Wall, type='Obj', side=[At])]))
    >>> Travel(SSS.parse("[Along=[Path=[Appear=[Appear=[MEAN='red_ADJ_1'], Appear_4=[MEAN='brick_ADJ_1']], Det=[MEAN='THE'], Path_n=[MEAN='path_N_2,3,4']]], Dir=[MEAN='straight_ADV_1,3'], Dist=[MEAN='ALL'], Then=[MEAN='AND'], Toward=[Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='lamp_N_2']], Toward_p=[MEAN='TOWARDS']], Travel_v=[MEAN='take_N_1,11,40'], Until=[Struct=[Det=[MEAN='THE'], Part=[Part_p=[MEAN='OF'], Path=[Det=[MEAN='THE'], Path_n=[MEAN='hall_N_1']]], Struct_n=[MEAN='end_N_1']], Until_p=[MEAN='UNTIL']]]"))
    Travel(distance=[Distance()], face=Face(faced=Verify(desc=[Thing(dist='0', Appear=[Brick], value=Path, type='Path', side=[Front]), Thing(dist='1:', value=Lamp, type='Obj', side=[Front])]), direction=[Front]), until=Verify(desc=[Thing(dist='0', Part=[Thing(dist='0', value=Path, type='Path')], value=End, type='Struct', side=[At])]))
    >>> Travel(SSS.parse("[Along=[Path=[Appear=[Appear=[MEAN='blue_ADJ_1'], Appear_5=[MEAN='tiled_ADJ_1'], Hyphen=[MEAN='-']], Det=[MEAN='THE'], Path_n=[MEAN='hallway_N_1']]], Loc=[Loc_p=[MEAN='AT'], Struct=[Det=[MEAN='THE'], Reldist=[MEAN='FAR'], Struct_n=[MEAN='end_N_1']]], Travel_v=[MEAN='follow_N_4'], Until=[Path=[Appear=[Appear=[MEAN='grass_ADJ_1'], Appear_10=[MEAN='floored_ADJ_1']], Det=[MEAN='THE'], Path_n=[MEAN='hallway_N_1']], Until_p=[MEAN='TO']]]"))
    Travel(location=Travel(until=Verify(desc=[Thing(Reldist=['FAR'], dist='0', value=End, type='Struct', side=[At])])), face=Face(faced=Verify(desc=[Thing(dist='0', Appear=[BlueTile, Flooring], value=Path, type='Path', side=[Front])])), until=Verify(desc=[Thing(dist='0', Appear=[Flooring, Grass], value=Path, type='Path', side=[At])]))
    >>> Travel(SSS.parse("[Travel_v=[MEAN='walk_N_1'], Until=[Until_p=[MEAN='UNTIL'], View=[Agent=[MEAN='YOU'], See_v=[MEAN='see_N_1'], Path=[Det=[MEAN='AN'], Appear=[Adv=[MEAN='almost_ADV_1'], Appear=[MEAN='black_ADJ_1']], Path_n=[MEAN='carpet_N_1']]]]]"))
    Travel(until=Verify(desc=[Thing(dist='0', Appear=[Stone], value=Path, type='Path', side=[At])]))
    >>> Travel(SSS.parse("[Along=[Path=[Appear=[MEAN='black_ADJ_1'], Det=[MEAN='THIS'], Path_n=[MEAN='path_N_2,3,4']]], Travel_v=[MEAN='take_N_1,11,40'], Until=[Arrive=[Agent=[MEAN='YOU'], Arrive_v=[MEAN='reach_N_1'], Struct=[Det=[MEAN='THE'], Part=[Part_p=[MEAN='WITH'], Path=[Appear=[Appear=[MEAN='white_ADJ_1'], Appear_13=[MEAN='cement_ADJ_1']], Det=[MEAN='THE']]], Struct_n=[MEAN='intersection_N_2']]], Until_p=[MEAN='UNTIL']]]"))
    Travel(face=Face(faced=Verify(desc=[Thing(dist='0', Appear=[Stone], value=Path, type='Path', side=[Front])])), until=Verify(desc=[Thing(dist='0', Part=[Thing(dist='0', Appear=[Cement], value=Path, type='Path')], value=Intersection, type='Struct', side=[At])]))
    >>> Travel(SSS.parse("[Travel_v=[MEAN='walk_N_1'], Dir=[MEAN='forward_ADV_1'], Dist=[Count=[MEAN='three_ADJ_1'], Dist_unit=[MEAN=\\"times_N_('time', [1])\\"]], Cc=[MEAN='OR'], Until=[Until_p=[MEAN='UNTIL'], S=[Agent=[MEAN='YOU'], Is_v=[MEAN=\\"are_N_('be', [1, 2, 3, 4, 5])\\"], Loc=[Loc_p=[MEAN='AT'], Struct=[Det=[MEAN='THE'], Struct_n=[MEAN='intersection_N_2'], Part=[Part_p=[MEAN='OF'], Path=[Appear=[Appear=[MEAN='green_ADJ_1'], Cc=[MEAN='AND'], Appear_15=[MEAN='yellow_ADJ_1']]]]]]]]]"))
    Travel(distance=[Distance(count=3)], until=Verify(desc=[Thing(dist='0', Part=[Thing(dist='0', Appear=[Greenish, Honeycomb], value=Path, type='Path')], value=Intersection, type='Struct', side=[At])]))
    >>> Travel(SSS.parse("[Travel_v=[MEAN='take_V_1,11,40'], Along=[Path=[Det=[MEAN='THE'], Appear=[MEAN='yellow_ADJ_1'], Path_n=[MEAN='path_N_2,3,4']]], Until=[Until_p=[MEAN='TO'], Struct=[Det=[MEAN='THE'], Part=[Path=[Appear=[MEAN='pink_ADJ_1'], Path_n=[MEAN='path_N_2,3,4']]], Struct_n=[MEAN='intersection_N_2']]]]"))
    Travel(face=Face(faced=Verify(desc=[Thing(dist='0', Appear=[Honeycomb], value=Path, type='Path', side=[Front])])), until=Verify(desc=[Thing(dist='0', Part=[Thing(dist='0', Appear=[Rose], value=Path, type='Path')], value=Intersection, type='Struct', side=[At])]))
    >>> Travel(SSS.parse("[Stop_v=[MEAN='stop_V_1'], Loc=[Loc_p=[MEAN='AT'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='hatrack_N_1']]]]"))
    Travel(until=Verify(desc=[Thing(dist='0', value=Hatrack, type='Obj', side=[At])]))
    >>> Travel(SSS.parse("[Travel_v=[MEAN='follow_V_4'], Along=[Path=[Det=[MEAN='THE'], Appear=[MEAN='yellow_ADJ_1'], Path_n=[MEAN='path_N_2,3,4']]], Until=[Until_p=[MEAN='UNTIL'], Struct_s=[Part=[Path=[Ref=[MEAN='IT']]], Struct_v=[MEAN=\\"crosses_V_('cross', [1, 2])\\"], Part_9=[Path=[Det=[MEAN='THE'], Appear=[MEAN='black_ADJ_1']]]]]]"))
    Travel(face=Face(faced=Verify(desc=[Thing(dist='0', Appear=[Honeycomb], value=Path, type='Path', side=[Front])])), until=Verify(desc=[Thing(Struct_n=Intersection, dist='0', type='Struct', Detail=[Thing(dist='0', Appear=[Stone], value=Path, type='Path', side=[At]), Thing(dist='0', Appear=[Honeycomb], value=Path, type='Path', side=[At])], value=Intersection, side=[At])]))
    >>> Travel(SSS.parse("[Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='hatrack_N_1']]]"))
    Travel(until=Verify(desc=[Thing(dist='0', value=Hatrack, type='Obj', side=[At])]))
    >>> Travel(SSS.parse("[Path=[Appear=[MEAN='pink_ADJ_1'], Path_n=[MEAN='path_N_2,3,4']]]"))
    Travel(distance=[Distance(count=1, distUnit=Verify(desc=[Thing(dist='0', value=TopoPlace, type='Struct', side=[At])]))], face=Face(faced=Verify(desc=[Thing(Appear=[Rose], value=Path, type='Path', side=[Front])])))
    >>> Travel(SSS.parse("[Travel_v=[MEAN='move_N_4'], Between=[Between_p=[MEAN='BETWEEN'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='butterfly_N_1'], Obj_n_5=[MEAN=\\"pictures_N_('picture', [2])\\"]]]]"))
    Travel(until=Verify(desc=[Thing(dist='0', value=Position, between=[Thing(dist='1:', value=Butterfly, type='Obj', side=[Between])], type='Struct', side=[At])]))
    >>> Travel(SSS.parse("[Stop_v=[MEAN='stop_V_1'], Loc=[Dist=[Count=[MEAN='one_ADJ_1,2'], Dist_unit=[Path_n=[MEAN='alley_N_1']], Past=[Past_p=[MEAN='PAST'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='lamp_N_2']]]]]]"))
    Travel(distance=[Distance(count=1, past=Travel(until=Verify(desc=[Thing(dist='0', value=Lamp, type='Obj', side=[At])])), distUnit=Verify(desc=[Thing(dist='0', value=Path, type='Path', side=[Sides])]))])
    >>> model([SSS.parse("[Cond=[Cond_p=[MEAN='WITH'], Obj=[Det=[MEAN='YOUR'], Obj_n=[MEAN='back_N_1']], Against=[Against_p=[MEAN='TO'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='wall_N_1']]]], S=[Travel_v=[MEAN='move_N_4'], Dir=[MEAN='forward_ADV_1'], Along=[Along_p=[MEAN='ALONG'], Path=[Det=[MEAN='THE'], Appear=[MEAN='octagon_N_1'], Path_n=[MEAN='carpet_N_1']]], Dist=[Count=[MEAN='one_ADJ_1,2'], Dist_unit=[Path_n=[MEAN='alley_N_1']]]]]")])
    ([Travel(distance=[Distance(count=1, distUnit=Verify(desc=[Thing(dist='0', value=Path, type='Path', side=[Sides])]))], face=Face(faced=Verify(desc=[Thing(dist='0', Appear=[Honeycomb], value=Path, type='Path', side=[Front]), Thing(dist='0', value=Wall, type='Obj', side=[Back])]), direction=[Front]))], None, '')
    >>> Travel(SSS.parse("[Travel_v=[MEAN='go_V_1'], Dir=[MEAN='straight_ADV_1,3'], Until=[Until_p=[MEAN='UNTIL'], Struct_s=[Agent=[MEAN='YOU'], Travel_v=[MEAN='go_V_1'], Dir=[MEAN='left_ADV_1'], Cc=[MEAN='OR'], Toward=[Toward_p=[MEAN='TOWARDS'], Struct=[Det=[MEAN='A'], Struct_n=[MEAN='dead end_N_1']]]]]]"))
    Travel(until=Verify(desc=[Thing(value=Intersection, Part=[Thing(dist='1:', value=DeadEnd, type='Struct', side=[Front]), Thing(dist='0', Path_n=Path, value=Path, type='Path', side=[Left])], dist='0', type='Struct', side=[At])]))
    """
    knownVerbs = ('Arrive_v', 'Pass_v', 'Stop_v', 'Travel_v')
    knownClauses = tuple() #'Arrive'
    knownFacePhrases = ('Along', 'Away', 'Dir', 'Pathdir', 'Onto', 'Side', 'Toward')
    knownPhrases = ('Across', 'Between', 'Cond', 'Dist', 'Loc', 'Past', 'Until')
    knownModelAttribs = ('distance','face','location','past','until','follow')
    knownAttrs = ('Cc', 'Inf', 'Punct')+knownModelAttribs+knownPhrases+knownFacePhrases+knownVerbs+knownClauses
    end = Verify(desc=[Thing(type='Obj', side=[Front], Obj_n=Wall, dist='0')])
    def __init__(self, sss=[], **kargs):
        for attrib in self.knownModelAttribs: setattr(self,attrib,[])
        CompoundActionSpecification.__init__(self,sss,**kargs)
        self.checkUnknownAttrs()
        
        if not sss or not sss.feature_names(): return
        if grab(sss,('Struct_s','Agent')) and Options.RecognizeStructAgentFrame:
            sss = sss['Struct_s']
        if not isinstance(sss,list): sss = [sss]
        
        for subsss in sss:
            features = subsss.feature_names()
            if 'Loc' in features and not subsss['Loc'].has_key('Position'):
                if 'Stop_v' in features:
                    if not Options.StopCond: continue
                    if subsss['Loc'].has_key('Dist'):
                        self.distance.append(Distance(grab(subsss,('Loc','Dist'))))
                    else:
                        self.until = Verify(grab(subsss,('Loc')))
                        setSideDist(self.until.desc, dist='0')
                    self.location = []
                else:
                    self.location = Travel(SSS(Until=grab(subsss,('Loc'))))
            
            if Options.FaceTravelArgs and any(features,self.knownFacePhrases):
                faced = SurfaceSemanticsStructure()
                for key in [k for k in features if k in self.knownFacePhrases]:
                    if key == 'Pathdir' and not (Options.RecognizePathdir and Options.ImplicitTurn):
                        continue
                    faced[key] = grab(subsss,key)[key]
                    logger.debug('Travel: Added attribute %r to Face %r', key,self.face)
                self.face = Face(faced)
                # If only facing Front
                if (not hasattr(self.face,'faced') or
                     (self.face.faced and not self.face.faced.desc
                      and (not self.face.direction or self.face.direction == [Front]))):
                    self.face = []
            
            if 'Cond' in features:
                cond = Verify(grab(subsss,('Cond')))
                if 'Stop_v' in features:
                    if not Options.StopCond: continue
                    self.until = cond
                    setSideDist(self.until.desc, dist='0')
                elif ((subsss['Cond'].has_key('Arrive') or subsss['Cond'].has_key('Struct_s'))
                       and Options.RecognizeArriveFrame and Options.ImplicitTravel):
                    self.location = Travel(SSS(Until=grab(subsss,('Cond'))))
                elif Options.FaceTravelArgs:
                    if not self.face: self.face=Face()
                    self.face.faced.desc.extend(cond.desc)
            
            if 'Until' in features:
                if subsss['Until'].has_key('Dist'):
                    self.distance.append(Distance(grab(subsss,('Until','Dist'))))
                elif grab(subsss,('Until','Loc','Dist')) and Options.RecognizeUntilLocDist:
                    self.distance.append(Distance(grab(subsss,('Until','Loc','Dist'))))
                elif grab(subsss,('Until','S','Loc','Dist')) and Options.RecognizeUntilLocDist:
                    self.distance.append(Distance(grab(subsss,('Until','S','Loc','Dist'))))
                elif grab(subsss,('Until','Arrive','Loc','Dist')) and Options.RecognizeUntilLocDist:
                    self.distance.append(Distance(grab(subsss,('Until','Arrive','Loc','Dist'))))
                elif (any(subsss['Until'].feature_names(),ObjectTypes)
                    or any(subsss['Until'].feature_names(),('Arrive','Cond','Loc','Ref','S','Struct_s','View'))):
                    self.until = Verify(grab(subsss,('Until')))
                    setSideDist(self.until.desc, dist='0')
            if subsss.has_key('Dist'):
                self.distance.append(Distance(grab(subsss,'Dist')))
            elif subsss.has_key('Stop_v') and not (subsss.has_key('Loc') or subsss.has_key('Cond')):
                self.distance.append(Distance(count=0))
            
            if 'Arrive' in features and Options.RecognizeStandaloneArrive:
                self.until = Verify(subsss['Arrive'])
            
            if 'Across' in features and Options.RecognizeAcross:
                counts = meanings(grab(subsss,('Across','Segment','Count')) or grab(subsss,('Across','Count')),'Count',KB) or [1]
                distUnit = Verify(subsss['Across'])
                for d in distUnit.desc:
                    if hasattr(d,'Count'): del d.Count
                self.distance.append(Distance(count=counts[0], distUnit=distUnit))
                if not self.face: self.face=Face()
                face = copy.deepcopy(distUnit.desc)
                for d in face: d.side,d.dist = ([Front],'0')
                self.face.faced.desc.extend(face)
            
            if 'Past' in features:
                if 'Pass_v' in features and not Options.RecognizePassFrame: continue
                self.past = Verify(grab(subsss,('Past')))
            
            if 'Between' in features or grab(subsss,('Until','Between')):
                betweenPlace = Thing(dist='0', value=Position, type='Struct', side=[At])
                betweenPlace.between = Verify(grab(subsss,('Between')) or subsss['Until']).desc
                if not self.until: self.until = Verify(desc=[betweenPlace])
                else: self.until.desc.append(betweenPlace)
            
#             if not self.until and grab(subsss,('Dist','MEAN')):
#                 self.until = self.end
            
            if 'Dist' in features and not 'Count' in subsss['Dist'].feature_names():
                 dist=grab(subsss,('Dist','MEAN'))
                 if dist and dist['MEAN']=='ALL':
                     self.distance = [Distance()]
            
            if 'Inf' in features:
                self.distance.append(Distance())
                if hasattr(self,'face') and self.face: setSideDist(self.face.faced.desc, dist='0')
            
            for objectType in [o for o in ObjectTypes if o in features]:
                objects = meanings(subsss, objectType ,KB)
                if Options.FaceTravelArgs and objectType == 'Path':
                    if not self.face: self.face=Face()
                    setSideDist(objects, [Front])
                    self.face.faced.desc.extend(objects)
                else:
                    if not self.until: self.until=Verify()
                    setSideDist(objects, [At])
                    self.until.desc.extend(objects)
            
            if self.until:
                for d in self.until.desc: d.condSetSideDist(([At],'0'))
            
            if ((grab(subsss, ('Along','Path','Structural'))
                 and grab(subsss, ('Along','Path','Structural','MEAN'))['MEAN'].startswith('winding'))
                or ('Dir' in features and grab(subsss,('Dir','MEAN'))
                    and subsss['Dir','MEAN'].split('_')[0] in ('out','around'))
                or ('Travel_v' in features and subsss['Travel_v','MEAN'].split('_')[0] in ('follow','wind')
                    and (not self.distance or not self.distance[0].count))):
                if self.until == self.end:
                    self.follow = Follow()
                else:
                    self.follow = Follow(until=self.until)
            
            logger.debug('Travel.__init__ interpreted %r to %r', subsss,self)
    
    def faceDistFeatures(self,robot):
        faceDesc = []
        faceAttribs = {}

        for distance in self.distance:

            if not distance.count or not Options.FaceDistance or distance.approx: continue
            faceAttribs['Distance'] = True
            if distance.distUnit:
                for distUnit in distance.distUnit.desc:
                    distUnitCp = copy.deepcopy(distUnit)
                    if Options.PerspectiveTaking: setSideDist([distUnitCp], dist='1:')
                    if distUnitCp.side == [At]: distUnitCp.side = [Front]
                    faceDesc.append(distUnitCp)
            else:
                faceDesc.append(Thing(dist=str(distance.count)+':', value=Position, type='Struct', side=[Front]))
        if self.until and self.until.desc and Options.FaceUntil:
            try: lastTurn = self.index == 0 or isinstance(self.plan()[self.index-1], Turn)
            except AttributeError: lastTurn = False
            if (lastTurn
                and (not hasattr(self.until.desc[0],'Order_adj') or not self.until.desc[0].Order_adj)
                and self.until.execute(robot) and Options.ImplicitTravel and Options.TravelToNext):
                self.until.desc[0].Order_adj = [1]
            needCount = False
            for desc in self.until.desc:
                faceAttribs['Until'] = True
                if hasattr(desc,'Reldist') and desc.Reldist:
                    desc.oldDist,desc.dist = desc.dist,desc.Reldist[0]
                if hasattr(desc,'Order_adj'): needCount = True
            if (needCount or not self.until.execute(robot)):
                for i,desc in enumerate(copy.deepcopy(self.until.desc)):
                    if hasattr(desc,'Order_adj') and desc.Order_adj:
                        count = Distance.getCount(desc.Order_adj)
                        orderDist = Distance(count=count, distUnit=Verify(desc=[copy.deepcopy(desc)]))
                        added = False
                        if self.distance and not self.distance[0].count: # Replace null distance (e.g. all the way down)
                            self.distance = [orderDist]
                            added = True
                        elif not self.distance:
                            self.distance.append(orderDist)
                            added = True
                        if added:
                            del self.until.desc[i]
                            logger.info('Travel: Added distance %r from %r', self.distance[-1],self.until.desc)
                    if (hasattr(desc,'Reldist') or hasattr(desc,'Order_adj')): dist = '1:'
                    else: dist = '0:'
                    if Options.PerspectiveTaking:
                        setSideDist([desc],dist=dist)
                        if desc.side == [At]: desc.side = [Front]
                    faceDesc.append(desc)
            for desc in self.until.desc:
                if hasattr(desc,'Reldist') and desc.Reldist:
                    desc.Reldist,desc.dist = desc.dist,desc.oldDist
            if not self.until.desc: self.until = []
        
        unPassedConds=[]
        if self.past:
            for desc in self.past.desc:
                faceAttribs['Past'] = True
                passCond = copy.deepcopy(desc)
                # Check to see if already past the condition.
                if Options.PerspectiveTaking: setSideDist([passCond],[Back],'0:')
                if not Verify(desc=[passCond]).execute(robot):
                    unPassedConds.append(Verify(desc=[desc]))
                    if Options.FacePast:
                        desc = copy.deepcopy(desc)
                        if Options.PerspectiveTaking: setSideDist([desc],[Front],'1:')
                        faceDesc.append(desc)
        # Check to make sure facing explicit and implicit distant features
        if faceDesc:
            if self.face and Options.TravelPrecond:
                faceDesc.extend(self.face.faced.desc)
                faceAttribs['Face'] = True
            log_text = '/'.join(faceAttribs.keys())
            logger.info('Facing the %s conditions in the distance %r', log_text, faceDesc)
            if Options.ImplicitTurn:
                self.update('Faced '+log_text, Face(faced=faceDesc).execute(robot), robot)
        return unPassedConds
    
    def execute(self,robot):
        try: consecTravels = self.index > 1 and isinstance(self.plan()[self.index-1], Travel)
        except AttributeError: consecTravels = False
        
        if (not self.until and not self.past and
            (not self.distance or not self.distance[0].count or self.distance[0].approx)):
            if Options.LookAheadForTravelTerm and Options.ImplicitTravel:
                try: self.plan().getUntilFromNextAction(self)
                except AttributeError: self.until = self.end
        
        if self.location and Options.TravelLocation and Options.ImplicitTravel:
            self.update('loc',self.location.execute(robot),robot)
        if self.face and Options.TravelPrecond and Options.ImplicitTurn:
            # Handles turn directions (e.g. "Go left onto...")
            self.update('face',self.face.execute(robot),robot)
        
        if hasattr(self,'follow') and self.follow and self.follow.until:
            self.follow.plan = self.plan
            self.follow.index=self.index
            self.follow.source=self.source
            return self.follow.execute(robot)
        
        if not hasattr(self,'until') or not self.until and not self.distance: return 0,[]
        
        unPassedConds = self.faceDistFeatures(robot)
        if (consecTravels and not self.location and not self.face and not self.until
            and not (self.distance and hasattr(self.distance[0], 'distUnit') and self.distance[0].distUnit)
            and Options.TurnBetweenTravels and Options.ImplicitTravel):
            logger.info('Turning for consecutive travels')
            Turn().execute(robot)
        dist_execute_state = 'OK'
        if self.distance:
            #for dist in self.distance: dist.execute(self,robot)
            self.distance[0].execute(self,robot)
            if self.until:
                dist_execute_state = 'Distance'
                for desc in self.until.desc:
                    if hasattr(desc,'Order_adj'): del desc.Order_adj
        if self.until:
            while not self.until.execute(robot) and Options.TravelUntil:
                if (self.distance and dist_execute_state == 'Distance' and Options.FaceUntil
                    and Options.FaceUntilPostDist):
                    postUntilFace = Verify(desc=copy.deepcopy(self.until.desc))
                    for desc in postUntilFace.desc:
                        if Options.PerspectiveTaking:
                            setSideDist([desc],dist='0:')
                            if desc.side == [At]: desc.side = [Front]
                    logger.info('Testing for over-shooting destination')
                    if not postUntilFace.execute(robot):
                        logger.info('Turning for over-shooting destination')
                        Turn(direction=[Back]).execute(robot)
                        dist_execute_state = 'Turned Around'
                    else: dist_execute_state = 'OK'
                cost,obs = robot.travel()
                self.update('until', (cost,obs) ,robot)
                if cost > 1: raise Warning('Hit a wall')
                for cond in unPassedConds[:]:
                    if cond.execute(robot): unPassedConds.remove(cond)
            if dist_execute_state == 'Turned Around':
                logger.info('Turning back around at dest after over-shooting destination')
                Turn(direction=[Back]).execute(robot)
                dist_execute_state = 'OK'
        elif not self.distance:
            if self.past and Options.TravelPast:
                while unPassedConds and not self.end.execute(robot):
                    for cond in unPassedConds[:]:
                        if cond.execute(robot): unPassedConds.remove(cond)
                    cost,obs = robot.travel()
                    self.update('past', (cost,obs) ,robot)
                    if cost > 1: raise Warning('Hit a wall')
            elif self.face and Options.TravelNoTermination and Options.ImplicitTravel:
                self.update('travel',robot.travel(),robot)
            elif Options.TravelEmpty and Options.ImplicitTravel:
                logger.info('empty travel %r',self)
                self.update('travel',robot.travel(),robot)
        if len(self.observations) > 1: obs = self.observations[-1]
        else: obs = []
        return self.cost,obs

class Find(Face):
    """
    >>> Find(SSS.parse("[Find=[Find_v=[MEAN='find_V_3'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='easel_N_1'], Loc=[Loc_p=[MEAN='IN'], Region=[Det=[MEAN='THIS'], Region_n=[MEAN='area_N_1']]]]]]"))
    Find(until=Verify(desc=[Thing(Loc=[Thing(dist='0', Appear=[Brick], value=Path, type='Path', side=[Front], Structural=['Long'])], dist='0', value=Easel, type='Obj', side=[At])]))
    """
    knownVerbs = ('Travel_v','Find_v','Follow_v')
    knownClauses = ('Find',)
    knownPhrases = ('Until',)
    knownModelAttribs = ('until',)
    knownAttrs = ('Cc', 'Inf', 'Punct')+knownModelAttribs+knownPhrases+knownVerbs+knownClauses
    def __init__(self, sss=[], **kargs):
        for attrib in self.knownModelAttribs: setattr(self,attrib,[])
        self.until = Verify()
        self.direction = []
        CompoundActionSpecification.__init__(self,sss,**kargs)
        if not sss or not sss.feature_names(): return
        if sss:
            if 'Find' in sss.feature_names(): sss = sss['Find']
            
            descSSS = SurfaceSemanticsStructure()
            for key,subsss in sss.items():
                if not uninterpretable(key):
                    descSSS[key] = grab(sss,key)
            self.until.desc.extend(Verify(descSSS).desc)
        setSideDist(self.until.desc, [At], '0')
        self.checkUnknownAttrs()
    
    stepPath = Travel(distance=[Distance(count=1)])
    facePath = Turn(direction=[Right])
    checkPath = Verify(desc=[Thing(dist='1:', value=Intersection, type='Struct', side=[Front])])
    
    def choose(self, robot, directions, count):
        if Front in directions and not self.checkPath.execute(robot):
            directions.remove(Front)
        if not directions: direction = face = []
        else: direction = random.choice(directions)
        if direction == Front:
            return
        if direction: self.facePath.direction[0] = direction
        turns = 0
        #Are we where we want to be or do we see it?
        while not (self.until.execute(robot) and
                   (not (hasattr(self,'distant') and self.distant) and self.distUntil.execute(robot))):
            self.facePath.execute(robot)
            turns += 1
            if turns != 2 and self.checkPath.execute(robot): break
    
    def execute(self,robot):
        if ((not hasattr(self,'until') or not self.until)
            and Options.LookAheadForTravelTerm and Options.ImplicitTravel):
            self.plan().getUntilFromNextAction(self)
            if not hasattr(self,'until') or not self.until: return 0,[]
        
        count = 0
        self.distUntil = copy.deepcopy(self.until)
        if Options.PerspectiveTaking: setSideDist(self.distUntil.desc, [Front], '1:')
        
        initQuitChance = 0.9999
        travelUntil = Travel(until=self.until)
        if not getattr(Options,'Use'+self.name()):
            return self.update(self.name()+': No Use'+self.name()+'.  Falling back to Travel',  
                               travelUntil.execute(robot),robot)
        while not self.until.execute(robot):
            if not (hasattr(self,'distant') and self.distant) and self.distUntil.execute(robot):
                self.distant = True
                self.update(self.name()+': Saw distant until', travelUntil.execute(robot),robot)
                break
            directions = []
            for direction in (Front,Left,Right):
                self.pathDetect.desc[0].side = [direction]
                if self.pathDetect.execute(robot): directions.append(direction)
            self.choose(robot, directions, count)
            if self.until.execute(robot) and Options.CheckAfterTurnFind:
                break
            self.stepPath.execute(robot)
            count += 1
            if not hasattr(self, 'currQuitChance'): self.currQuitChance = 1.0

            # Slowly exponentially increasing chance of giving up.
            self.currQuitChance *= initQuitChance
            if (random.random() > self.currQuitChance):
                raise LookupError("Can't find", self.until, "in", count, "movements.")
        if len(self.observations) > 1: obs = self.observations[-1]
        else: obs = []
        return count,obs

class Follow(Find,Travel):
      """
      >>> Follow(SSS.parse("[Travel_v=[MEAN='follow_V_4'], Along=[Path=[Det=[MEAN='THIS'], Path_n=[MEAN='hallway_N_1']]], Dir=[MEAN='out_ADV_1,2,3,4']]"))
      Follow(distance=[Distance(count=1, distUnit=Verify(desc=[Thing(dist='0', value=TopoPlace, type='Struct', side=[At])]))])
      >>> Follow(SSS.parse("[Travel_v=[MEAN='follow_V_4'], Along=[Path=[Det=[MEAN='THIS'], Path_n=[MEAN='hallway_N_1']]], Until=[Until_p=[MEAN='TO'], Struct=[Det=[MEAN='THE'], Part=[Path=[Appear=[MEAN='pink_ADJ_1'], Path_n=[MEAN='path_N_2,3,4']]], Struct_n=[MEAN='intersection_N_2']]]]"))
      Follow(face=Face(faced=Verify(desc=[Thing(dist='0', Part=[Thing(Appear=[Rose], value=Path, type='Path')], value=Intersection, type='Struct', side=[Front])])), until=Verify(desc=[Thing(dist='0', Part=[Thing(dist='0', Appear=[Rose], value=Path, type='Path')], value=Intersection, type='Struct', side=[At])]))
      >>> Follow(SSS.parse("[Travel_v=[MEAN='follow_V_4'], Along=[Path=[Det=[MEAN='THIS'], Path_n=[MEAN='hallway_N_1']]], Until=[Until_p=[MEAN='TO'], Struct=[Det=[MEAN='THE'], Part=[Path=[Appear=[MEAN='pink_ADJ_1'], Path_n=[MEAN='path_N_2,3,4']]], Struct_n=[MEAN='intersection_N_2']]]]"))
      Follow(face=Face(faced=Verify(desc=[Thing(dist='0', Part=[Thing(Appear=[Rose], value=Path, type='Path')], value=Intersection, type='Struct', side=[Front])])), until=Verify(desc=[Thing(dist='0', Part=[Thing(dist='0', Appear=[Rose], value=Path, type='Path')], value=Intersection, type='Struct', side=[At])]))
      """
      knownVerbs = ('Follow_v','Travel_v')
      knownClauses = tuple()
      knownPhrases = ('Cond', 'Dist', 'Loc', 'Past', 'Until')
      knownModelAttribs = ('distance','face','loc','past','until')
      knownAttrs = ('Cc', 'Inf', 'Punct')+knownModelAttribs+knownPhrases+Travel.knownFacePhrases+knownVerbs+knownClauses
      def __init__(self, sss=[], **kargs):
          for attrib in self.knownModelAttribs: setattr(self,attrib,[])
          Travel.__init__(self,sss,**kargs)
      
      def choose(self, robot, directions, count):
          # Environoment constrains choice
          if len(directions) == 1:
              direction = directions[0]
              if direction != Front:
                  self.facePath.direction[0] = direction
                  self.facePath.execute(robot)
          # End of path, with no visible paths or no default choice (facing wall at T).
          elif len(directions) == 0 or Front not in directions:
              raise Warning("End of winding path.")

Verify.allKnownClauses = uniq(list(Verify.knownClauses+Turn.knownPhrases+Travel.knownPhrases+Face.knownClauses))

class DeclareGoal(CompoundActionSpecification):
    """
    """
    knownAttrs = ('cond','goal')
    def __init__(self, sss=[], **kargs):
        self.dist = []
        for attrib in DeclareGoal.knownAttrs: setattr(self,attrib,[])
        CompoundActionSpecification.__init__(self,sss,**kargs)
        self.checkUnknownAttrs()
        if self.cond:
            self.cond = Travel(until=Verify(desc=self.cond))
            
            if self.dist: self.cond.distance=[self.dist]
        elif self.dist: self.cond = Travel(distance=[self.dist])
        if self.goal: self.goal = meanings(self.goal,('Position','P_name'),Counts)
        self.dist=[]
        
        if not sss or not sss.feature_names(): return

    def execute(self,robot):
        if self.cond and Options.DeclareGoalCond and Options.ImplicitTravel:
            #(and not hasattr(self.cond.until,'Order_adj') and not self.cond.until.execute(robot)):
            self.update('Travelling to goal description',self.cond.execute(robot),robot)
        if self.goal: logger.info('Declaring at Position %r', self.goal)
        #print 'Declare Goal explicitly?', Options.DeclareGoalForPosition
        if Options.DeclareGoalForPosition:
            raise StopIteration(self.update('declareGoal',robot.declareGoal(),robot))
        else: return 0,[]

def findNearestNeighbor(senseStrs,kb):
    senses = [sense.synset for list in senseStrs for sense in list if sense]
    for synsets in [senses,
                    [h for s in senses for h in tools.hypernyms(s)[1:]]]:
        for synset in synsets:
            if synset.offset in kb:
                return kb[synset.offset]

numerical_suffix = re.compile('(_\d+)?$')
def isCanonical(key): return not key[-1].isdigit()

def valCanonical(key):
    """
    >>> ' '.join(['%s:%s'%(key,valCanonical(key)) for key in ('Obj_n','Obj_n_1','Obj','Obj_1','Struct','Struct_1')])
    'Obj_n:Obj_n Obj_n_1:Obj_n Obj:Obj Obj_1:Obj Struct:Struct Struct_1:Struct'
    """
    if key[-1].isdigit(): return numerical_suffix.split(key)[0]
    else: return key
def valMatch(name,key):
    """
    >>> ' '.join(['%s:%s'%(key,bool(valMatch('Obj_n',key))) for key in ('Obj_n','Obj_n_1','Obj','Obj_1','Struct','Struct_1')])
    'Obj_n:True Obj_n_1:True Obj:False Obj_1:False Struct:False Struct_1:False'
    """
    return name == key or (key[-1].isdigit() and numerical_suffix.split(key)[0] == name)

def grab(sss,path,getname=False):
    if not isinstance(sss,SurfaceSemanticsStructure): return [],path
    if not path: return sss
    subsss = SurfaceSemanticsStructure()
    if '__iter__' in dir(path):
        name = path[0]
        if len(path) > 1:
            sss = grab(sss,name)
            if not sss : subsss = sss
            else:
                for key in sss.feature_names():
                    val = grab(sss[key],path[1:])
                    if val: subsss = val
            name = path[-1]
            if (not isinstance(subsss,SurfaceSemanticsStructure)
                or subsss and not subsss.feature_names()): subsss = []
            if getname: return subsss,name
            else: return subsss
    else: name = path
    try:
        for key in sss.feature_names():
            if valMatch(name,key):
                if (isinstance(sss[key], SurfaceSemanticsStructure)
                    and name in sss[key].feature_names()):
                    subsss[key] = grab(sss[key],name)
                else:
                    subsss[key] = sss[key]
    except (IndexError,KeyError,AttributeError),e: logger.error('grab %s',e)
    if (not isinstance(subsss,SurfaceSemanticsStructure)
        or subsss and not subsss.feature_names()): subsss = []
    if getname: return subsss,name
    else: return subsss

def uninterpretable(pred):
    return (any([pred.endswith(s) for s in ('Det','_p','_v',',','Punct','Cc','Then','Hyphen','Text_adj')]
                ,(True,))
            and not pred in ('Struct_v',))

def interpret(sss,path,values=None):
    subsss,name = grab(sss,path,getname=True)
    if (name in (('Ref',),'Ref') and Options.ReferenceResolution and Options.RawReferenceResolution):
        anaphor = Thing(subsss)
        logger.debug('interpret resolving raw reference %s = %r', name, anaphor)
        return SurfaceSemanticsStructure(Ref=anaphor)
    if not values or not subsss or isinstance(subsss[0], str): return subsss
    if not subsss.feature_names() or uninterpretable(name): return []
    #logger.debug('interpret 1 %s = %r', path,sss)
    #logger.debug('interpret 2 %s = %r', name,subsss)
    if (name == 'Side' and 'Side_p' in subsss['Side'].feature_names()
        and (('Agent' in subsss['Side'].feature_names()
              and 'MEAN' in subsss[('Side','Agent')].feature_names()
              and subsss[('Side','Agent','MEAN')] == 'YOU')
             # Handle "the stool ahead and the chair behind" (implicit you)
             or (not any(subsss['Side'].feature_names(),ObjectTypes)
                 and not 'Dir' in subsss['Side'].feature_names())
             ) # Should also handle "Behind the OBJ"...
        ):
        if 'Side_n' in subsss['Side'].feature_names(): side = [lookupMeaning(subsss[('Side','Side_n')],Directions)]
        else: side = [lookupMeaning(subsss[('Side','Side_p')],Directions)]
        #logger.debug('interpret 2.1 %s = %r','Side',Side)
        if side: return side
    meansss = SurfaceSemanticsStructure()
    for key,value in subsss.items():
        if not value or not isinstance(value,SurfaceSemanticsStructure): continue
        #logger.debug('interpret 2.1 %s = %r',key,value)
        if (name in ('Ref',) and Options.ReferenceResolution):# and value['MEAN'] == 'IT'
            meansss.add(name, copy.deepcopy(Thing.referenceCache.get(Thing,subsss)))
            logger.debug('interpret resolving reference %s = %r', name,meansss)
        elif isinstance(value,str) or 'MEAN' in value.feature_names():
            meansss.add(key, lookupMeaning(value,values))
        elif name in value.feature_names():
            meansss.add(name, meanings(value, key, values))
        elif name in ObjectTypes:
            temp = SSS()
            temp[name] = value
            meansss.add(key, Thing(temp,type=name))
        elif name in Verify.knownClauses:
            meansss.add(key, getHead(subsss)(value))
        else:
            for key2 in value.feature_names():
                if uninterpretable(key2): continue
                meansss.add(key2, meanings(value,key2,values))
        #logger.debug('interpret 2.2 %s = %r',key,meansss)
    if not meansss.feature_names():# or [v for v in meansss if not v]:
        logger.error("Can't interpret %s = %r", path, subsss)
        logger.error('in %r', sss)
        return subsss
    logger.debug('interpret %s = %r', name ,meansss)
    return meansss

def meanings(sss,path,values=None):
    """
    >>> meanings(SSS.parse("[Appear=[Appear=[MEAN='red_ADJ_1', TAG='JJ', TEXT='red'], Appear_8=[MEAN='brick_N_1', TAG='NN', TEXT='brick']]]"),'Appear',KB)
    [Brick, Brick]
    >>> meanings(SSS.parse("[Appear=[MEAN='blue_ADJ_1', TAG='JJ', TEXT='blue']]"),'Appear',KB)
    [BlueTile]
    >>> meanings(SSS.parse("[Appear=[MEAN='blue_ADJ_1', TAG='JJ', TEXT='blue'], Hyphen=[MEAN='-', TAG=':', TEXT='-'],Appear_11=[MEAN='tiled_ADJ_1', TAG='NN', TEXT='tiled']]"),'Appear',KB)
    [BlueTile, Flooring]
    >>> meanings(SSS.parse("[Obj=[Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='chair_N_1']], Obj_18=[Obj_n=[MEAN='stool_N_1']]]]"),'Obj',KB)
    [Thing(value=Barstool, type='Obj'), Thing(value=GenChair, type='Obj')]
    >>> meanings(SSS.parse("[Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='futon_N_1'], Obj_n_7=[MEAN='bench_N_1'], Punct=[MEAN='/']]]"),'Obj',KB)
    [Thing(value=Furniture, type='Obj')]
    >>> meanings(SSS.parse("[Side=[Agent=[MEAN='YOU'], Side_p=[MEAN='BEHIND']]]"),'Side',KB)
    [Back]
    >>> meanings(SSS.parse("[Side=[Side_p=[MEAN='BEHIND']]]"),'Side',KB)
    [Back]
    >>> meanings(SSS.parse("[Side=[Side_p=[MEAN='IN'], Side_n=[MEAN='front_N_2,3,10'], Rel_p=[MEAN='OF'], Agent=[MEAN='YOU']]]"),'Side',KB)
    [Front]
    >>> meanings(SSS.parse("[Obj=[Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='fish_N_1'], Side=[Side_p=[MEAN='ON'], Det=[MEAN='EITHER'], Side_n=[MEAN=\\"sides_N_('side', [1])\\"], Part=[Part_p=[MEAN='OF'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN=\\"walls_N_('wall', [1])\\"]]]]], Cc=[MEAN='AND'], Obj_21=[Det=[MEAN='AN'], Obj_n=[MEAN='easel_N_1'], Loc=[Loc_p=[MEAN='AT'], Struct=[Det=[MEAN='THE'], Struct_n=[MEAN='end_N_1']]]]]]"),'Obj',KB)
    [Thing(Loc=[Thing(value=End, type='Struct')], value=Easel, type='Obj'), Thing(value=Fish, side=[Sides], type='Obj')]
    >>> meanings(SSS.parse("[Is_v=[MEAN='be_V_1,2,3,4,5'], Loc=[Loc_p=[MEAN='AT'], Struct=[Det=[MEAN='THE'], Part=[Part_p=[MEAN='OF'], Path=[Det=[MEAN='THE'], Path_n=[MEAN='path_N_2,3,4']]], Struct_n=[MEAN='end_N_1']]], Loc_20=[Loc_p=[MEAN='WHERE'], S=[Struct=[Part=[Path=[Appear=[MEAN='pink_ADJ_1'], Det=[MEAN='THE'], Path_n=[MEAN='path_N_2,3,4']]], Part_20=[Part_p=[MEAN='BY'], Path=[Appear=[MEAN='blue_ADJ_1'], Det=[MEAN='A'], Path_n=[MEAN='path_N_2,3,4']]], Struct_v=[MEAN=\\"intersected_V_('intersect', [1])\\"]]]], Position=[P_name=[MEAN='4'], Pos_n=[MEAN='position_N_1']]]"),'Loc_20',KB)
    [Verify(desc=[Thing(Part=[Thing(Appear=[BlueTile], value=Path, type='Path'), Thing(Appear=[Rose], value=Path, type='Path')], value=Intersection, type='Struct')])]
    >>> meanings(SSS.parse("[Appear=[Appear=[Obj=[Obj_n=[MEAN='fish_N_1']]], Appear_9=[MEAN='walled_ADJ_1']]]"), 'Appear',KB)
    [Verify(desc=[Thing(value=Fish, type='Obj')])]
    """
    Meanings = interpret(sss,path,values)
    if not Meanings: return []
    if isinstance(Meanings[0],list) and Meanings[0]:
        meaningList = []
        for m in Meanings: 
            if not m: continue
            append_flat(meaningList,m)
    else:
        meaningList = [m for m in Meanings if m]
    verifyDescs = []
    for meaning in meaningList[:]:
        if isinstance(meaning, list):
            verifyDescs.extend(meaning[:])
            meaningList.remove(meaning)
    for meaning in meaningList[:]:
        if isinstance(meaning, Verify):
            verifyDescs.extend(meaning.desc)
            meaningList.remove(meaning)
    if verifyDescs: meaningList.append(Verify(desc=verifyDescs))
    if ('Not' in sss.feature_names() or 'Obj' in sss.feature_names() and 'Not' in sss['Obj'].feature_names()
        and Options.RecognizeNegativeCompound):
        for meaning in meaningList: meaning.negate = True
    meaningList.sort()
    logger.debug('meanings %r[%s] = %r', sss, path, meaningList)
    return meaningList

def lookupMeaning(key,values):
    meaning = []
    try:
        if isinstance(key,SurfaceSemanticsStructure):
            meaning = key['MEAN']
        else:
            meaning = key
        v = values[meaning]
        if hasattr(v, 'narrow'): FuzzyMeaning.setMeanings(Options.FuzzyMeanings)
        return v
    except KeyError:
        if '_' not in meaning: return meaning
        try:
            text,POS,senses = splitSurfaceSemantics(meaning)
            for sense in senses:
                key = printSurfaceSemantics(text,POS,sense)
                if key in values:
                    v = values[key]
                    if hasattr(v, 'narrow'): FuzzyMeaning.setMeanings(Options.FuzzyMeanings)
                    return v
                nn = (findNearestNeighbor([parseSurfaceSemantics(meaning)],SynSetKB)
                      or KB.get(meaning,[]))
                logger.debug("Nearest neighbor for %s: %s, found %r", meaning, parseSurfaceSemantics(meaning),nn)
                if nn: return nn
        except ValueError:
            logger.error("Can't get text,POS,senses from %s", meaning)
            return meaning
        return []

def getHead(sss,Framer=False):
    features = sss.feature_names()
    if not features or features == ['Punct'] or features == ['MEAN']: return []
    if (Framer and 'Struct_s' in features and grab(sss,('Struct_s','Agent'))
        and Options.RecognizeStructAgentFrame):
        sss = sss['Struct_s']
        features = sss.feature_names()
    if (Framer and 'Arrive' in features and Options.RecognizeStandaloneArrive):
        return Travel
    
    for Head in (Face,Turn,Travel,Verify,Find):
        heads = [h for h in features
                 if h in Head.knownClauses or h in Head.knownVerbs]
        if heads:
            if (Head == Verify and Framer and not any(features,('Position',)) and Options.FaceDeclaratives):
                return FaceVerify
            else: return Head
    Counts = {}
    Head = Verify
    if any(features, ('Dist', 'Along', 'Past', 'Until')):
        Head = Travel
    elif any(features,Turn.knownFacePhrases+('Dir','Turn')):
        Sides = meanings(sss,('Dir'),Directions)
        if Sides == [Front]: Head = Travel
        else: Head = Turn
    elif 'Loc' in features:
        Head = Find
    logger.error("Couldn't find head in %r, but heuristic is %s",sss,Head.__name__)
    return Head

def frame2action(frame, Actions=None, source=0, index=0):
    logger.info('frame2action %r', frame)
    head = getHead(frame,Framer=True)
    if not head: return
    if Actions is None: index = index
    else: index = len(Actions) + index
    if head == FaceVerify:
        action = Face(faced=Verify(frame,index=index,source=source), index=index, source=source)
    else:
        action = head(frame,index=index,source=source)
    if (Actions is not None and action and
        not (isinstance(action, Verify) and not action.desc and not action.goal)):
        Actions.append(action)
        logger.info('<%s> %r',index,action)
    return action

def model(frames,instructID=''):
    """
    >>> model([SSS.parse("[S=[Travel_v=[MEAN='move_N_4'], Until=[Until_p=[MEAN='TO'], Path=[Det=[MEAN='THE'], Appear=[MEAN='brick_ADJ_1'], Path_n=[MEAN='alley_N_1']]]], Then=[MEAN='AND'], S_8=[Turn_v=[MEAN='turn_N_1,2'], Dir=[MEAN='right_ADV_4']]]")])
    ([Travel(until=Verify(desc=[Thing(dist='0', Appear=[Brick], value=Path, type='Path', side=[At])])), Turn(direction=[Right])], None, '')
    >>> model([SSS.parse("[,=[MEAN=','], Cond=[Arrive=[Agent=[MEAN='YOU'], Arrive_v=[MEAN='reach_V_1'], Struct=[Det=[MEAN='AN'], Part=[Part_p=[MEAN='WITH'], Path=[Appear=[Appear=[MEAN='black_ADJ_1'], Appear_10=[Appear=[MEAN='normal_ADJ_1'], Appear_10=[MEAN='concrete_ADJ_2']], Cc=[MEAN='AND']]]], Struct_n=[MEAN='intersection_N_2']]], Cond_p=[MEAN='WHEN']], S=[Agent=[MEAN='YOU'], Is_v=[MEAN=\\"are_V_('be', [1, 2, 3, 4, 5])\\"], Loc=[Loc_p=[MEAN='AT'], Position=[P_name=[MEAN='6']]]]]")])
    ([Verify(goal=DeclareGoal(cond=Travel(until=Verify(desc=[Thing(dist='0', Part=[Thing(dist='0', Appear=[Cement, Stone], value=Path, type='Path')], value=Intersection, type='Struct', side=[At])])), goal=['6']))], None, '')
    >>> model([SSS.parse("[S=[Cond=[Cond_p=[MEAN='WITH'], Obj=[Det=[MEAN='YOUR'], Obj_n=[MEAN='back_N_1']], Against=[Against_p=[MEAN='TO'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='wall_N_1']]]], S=[Turn_v=[MEAN='turn_N_1,2'], Dir=[MEAN='left_ADV_1']]], Then=[MEAN='AND'], S_19=[Travel_v=[MEAN='move_N_4'], Dist=[Count=[MEAN='two_ADJ_1'], Dist_unit=[Path_n=[MEAN=\\"alleys_N_('alley', [1])\\"]]], Until=[Until_p=[MEAN='TO'], Path=[Det=[MEAN='THE'], Structural=[MEAN='side_N_1'], Path_n=[MEAN='alley_N_1'], Detail=[Detail_p=[MEAN='WITH'], Obj=[Appear=[MEAN='stone_ADJ_1'], Obj_n=[MEAN='flooring_N_1,2']]]]]]]")])
    ([Turn(direction=[Left], precond=Face(faced=Verify(desc=[Thing(dist='0', value=Wall, type='Obj', side=[Back])]))), Travel(distance=[Distance(count=2, distUnit=Verify(desc=[Thing(dist='0', value=Path, type='Path', side=[Sides])]))], until=Verify(desc=[Thing(dist='0', Detail=[Thing(dist='0', Appear=[Stone], value=Flooring, type='Obj')], value=Path, type='Path', side=[At])]))], None, '')
    >>> model([SSS.parse("[Cond=[Cond_p=[MEAN='WHEN'], Arrive=[Agent=[MEAN='YOU'], Arrive_v=[MEAN='come_V_1,2,3,4'], Arrive_p=[MEAN='TO'], Path=[Det=[MEAN='A'], Appear=[Appear=[MEAN='red_ADJ_1'], Appear_7=[MEAN='brick_N_1']], Path_n=[MEAN='path_N_2,3,4']]]], ,=[MEAN=','], S=[Travel_v=[MEAN='go_V_1'], Dir=[MEAN='left_ADV_1']]]")])
    ([Travel(distance=
    [Distance(count=1, distUnit=Verify(desc=[Thing(dist='0', value=TopoPlace, type='Struct', side=[At])]))], loc=Travel(until=Verify(desc=[Thing(dist='0', Appear=[Brick], value=Path, type='Path', side=[At])])), face=Face(faced=Verify(), direction=[Left]))], None, '')
    >>> model([SSS.parse("[,=[MEAN=','], Cond=[Cond_p=[MEAN='WHEN'], Struct_s=[Part=[Path=[Det=[MEAN='THE'], Path_n=[MEAN='hall_N_1']]], Struct_v=[MEAN=\\"ends_V_('end', [1])\\"]]], S=[Dir=[MEAN='right_ADV_4'], Turn_v=[MEAN='take_V_1,11,40']]]")])
    ([Turn(direction=[Right], precond=Travel(until=Verify(desc=[Thing(Struct_n=End, dist='0', type='Struct', Detail=[Thing(dist='0', value=Path, type='Path', side=[At])], value=End, side=[At])])))], None, '')
    >>> model([SSS.parse("[S=[Loc=[Loc_p=[MEAN='AT'], Struct=[Det=[MEAN='THE'], Struct_n=[MEAN='end_N_1'], Part=[Part_p=[MEAN='OF'], Path=[Det=[MEAN='THIS'], Path_n=[MEAN='hall_N_1']]]]], ,=[MEAN=','], Turn_v=[MEAN='take_V_1,11,40'], Dir=[MEAN='left_ADV_1'], Toward=[Toward_p=[MEAN='TOWARDS'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='lamp_N_2']]]], Then=[MEAN='AND'], S_22=[Cond=[Cond_p=[MEAN='ONCE'], Loc=[Loc_p=[MEAN='AT'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='lamp_N_2']]]], ,=[MEAN=','], S=[Turn_v=[MEAN='take_V_1,11,40'], Dir=[MEAN='right_ADV_4']]]]")])
    ([Turn(face=Face(faced=Verify(desc=[Thing(dist='1:', value=Lamp, type='Obj', side=[Front])]), direction=[Left]), location=Travel(until=Verify(desc=[Thing(dist='0', Part=[Thing(dist='0', value=Path, type='Path')], value=End, type='Struct', side=[At])]))), Turn(direction=[Right], precond=Face(faced=Verify(desc=[Thing(dist='0', value=Lamp, type='Obj', side=[Front])])))], None, '')
    >>> model([SSS.parse("[View=[Agent=[MEAN='YOU'], Orient_v=[MEAN='looking_ADJ_1'], Orient_p=[MEAN='AT'], Obj=[Det=[MEAN='AN'], Obj_n=[MEAN='easel_N_1']]]]")])
    ([Turn(face=Face(faced=Verify(desc=[Thing(value=Easel, type='Obj', side=[Front])])))], None, '')
    >>> model([SSS.parse("[Loc=[Loc_p=[MEAN='BY'], Obj=[Det=[MEAN='THE'], Obj_n=[MEAN='easel_N_1']]]]")])
    ([Find(until=Verify(desc=[Thing(dist='0', value=Easel, type='Obj', side=[At])]))], None, '')
    """
    plan = Plan(frames, instructID)
    return plan, plan.CaughtError,plan.CaughtErrorTxt

def applyConditions(parent,frame):
    for condition in ('Cond','Loc'):
        if parent and parent.has_key(condition):
            frame[condition] = grab(parent,condition)[condition]

class Plan(list):
    def __init__(self, frames, instructID='', index=0, subplan=False):
        self.CaughtError=None
        self.CaughtErrorTxt=''
        
        plan = []
        if not subplan:
            Thing.referenceCache = {}
            re.purge()
        
        for i,frame in enumerate(frames):
            j = 0
            try:
                subframes = grab(frame,'S')
                if not subframes:
                    frame2action(frame, plan, (i,j), index)
                    continue
                keys = [key for key in frame.feature_names()
                        if valCanonical(key) not in ('Then',',','Cond','Loc','Cc')]
                keys.sort()
                for key in keys:
                    subframe = frame[key]
                    applyConditions(frame, subframe)
                    #print i,j,key,repr(subframe)
                    plan.extend(Plan([subframe], index=len(plan)+index, subplan=True))
                    j += 1
            except (TypeError,AttributeError),e: #None
                self.CaughtError = e
                self.CaughtErrorTxt = "Can't interpret: "+`frame`+str(e)
                logger.error("%s",self.CaughtErrorTxt) 
        for p in plan:
            if p: p.plan = weakref.ref(self)
        list.__init__(self, plan)

        ### Check for TravelOnFinalTurn
        lastMotion = self.getNextMotion(-1,-1)
        if not (lastMotion and not subplan
                and (isinstance(lastMotion,Turn)
                     and (Options.TravelOnFinalView or hasattr(lastMotion,'view') and not lastMotion.view)
                     and hasattr(lastMotion,'postcond') and not lastMotion.postcond)
                and Options.TravelOnFinalTurn and Options.ImplicitTravel):
            return
        for action in self[lastMotion.index+1:]: action.index += 1
        i = lastMotion.index+1
        logger.info('<%s> %r # Adding travel after final turn.' ,i,Travel(index=i))
        self.insert(i,Travel(index=i,plan=weakref.ref(self)))
    
    def getUntilFromNextAction(self,currAction):
        try:
            nextAction = self[currAction.index+1]
            if hasattr(nextAction,'location') and nextAction.location and Options.LookAheadForTravelTermLoc:
                for desc in nextAction.location.until.desc:
                    if hasattr(desc,'Order_adj'): return
                currAction.until = nextAction.location.until
            elif hasattr(nextAction,'desc') and nextAction.desc and Options.LookAheadForTravelTermDesc:
                currAction.until = nextAction
                setSideDist(currAction.until.desc, [At],'0')
            elif (isinstance(nextAction,Travel)
                  or (isinstance(nextAction,DeclareGoal) and hasattr(nextAction,'cond') and nextAction.cond)
                  or (isinstance(nextAction,Verify) and hasattr(nextAction,'goal') and
                      hasattr(nextAction.goal,'cond') and nextAction.goal.cond)
                  and Options.LookAheadForTravelTermDesc):
                return
        except IndexError: pass
        if not currAction.until:
            if isinstance(currAction,Follow) or hasattr(currAction,'follow') and currAction.follow:
                if (currAction.distance and not currAction.distance[0].count
                    and hasattr(currAction,'follow') and currAction.follow):
                    currAction.follow.until = Verify(desc=[Thing(type='Boolean',value=False)])
                else:
                    currAction.until = currAction.end
                #currAction.until = Verify(desc=[Thing(dist='0', value=TopoPlace, type='Struct', side=[At])])
            elif Options.DistanceCount:
                 distUnit = Verify(desc=[Thing(dist='0', value=TopoPlace, type='Struct', side=[At])])
                 currAction.distance = [Distance(count=1, distUnit=distUnit)] #Place for deadend
        
    def getNextMotion(self,index,iterate=1):
        lastMotion = None
        try:
            for action in self[index::iterate]:
                if (isinstance(action,Travel) or isinstance(action,Turn)
                    or (isinstance(action,DeclareGoal) and action.cond)
                    or (isinstance(action,Verify) and action.goal and action.goal.cond)):
                    lastMotion = action
                    break
        except IndexError: pass
        return lastMotion

def outputCompoundActions():
    import ParseDirections, DirectionCorpus
    from Sense import saveFrame
    from SubjectLogs.SubjectGroups import Directors
    instRegexp = DirectionCorpus.constructItemRegexp(Directors, mapversions='[01]')
    Instructions = DirectionCorpus.DirectionCorpusReader(instRegexp).items('ContentFrames')
    for ri_file in Instructions:
        instructID = ri_file.split('-')[1]
        print instructID
        saveFrame(Plan(getSSS(instructID), instructID), instructID,
                  directory='Directions/CompoundActions/', prefix='CompoundAction-')

def _test(suite='All',verbose=False):
    import doctest
    if suite != 'All':
        doctest.run_docstring_examples(suite, globals(), verbose=verbose)
    else: doctest.testmod(verbose=verbose)

def _profile():
    import doctest, hotshot, hotshot.stats
    prof = hotshot.Profile("model.prof")
    for fn in ('Verify','Turn', 'Face', 'Travel', 'Distance', 'Find', 'model'):
        prof.run(doctest.testsource('CompoundAction',fn))
    prof.close()
    stats = hotshot.stats.load("model.prof")
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats(20)

if __name__ == '__main__':
    logger.initLogger('CompoundAction',LogDir='MarcoLogs')
    from Sense import getSSS, genCorrContentFrame
