## Automatically adapted for numpy.oldnumeric May 17, 2011 by -c

import copy
import operator

from Meanings import *
from ViewCache import ViewCache

from CompoundAction import Thing
from Options import Options
from Utility import append_flat, logger

class CompoundRecognizer(object):
    Recognizers = {}

    def countPathFragments(cls,viewCache): raise NotImplementedError
    def countGateways(cls,viewCache): raise NotImplementedError
    def recPathAppear(cls,view,desc,path,ignore=[]): raise NotImplementedError
    def recFwdPath(cls,desc,viewCache): raise NotImplementedError
    def recNeedTurn(cls,desc,viewCache): raise NotImplementedError

    @classmethod
    def rtn(cls,match,desc):
        if desc.negate: return not match
        return match

    def checkDesc(cls,desc,name):
        if desc.type == 'Boolean': return
        if isinstance(desc,str):
            raise ValueError(name+": desc is string:",desc)
        if not desc.value:
            raise ValueError(name+": No value for desc:",desc)
        if isinstance(desc.value,str):
            raise ValueError(name+": desc.value is string, not Meaning:",desc)
        if not desc.side:
            #raise ValueError(name+": No side for desc:",desc)
            desc.side = [desc.value.ViewPosition]
    checkDesc = classmethod(checkDesc)

    def decrDist(cls,dist):
        if not Options.PerspectiveTaking: return dist
        end = None
        if dist == '0': return '0'
        if not dist: dist = '0:'
        if ':' in dist:
            start,end = dist.split(':')
            if start == '0': dist = '0:'
            else: dist = str(int(start)-1)+':'
            if end: dist+str(int(end)-1)
        else: dist = str(int(dist)-1)
        return dist
    decrDist = classmethod(decrDist)

    @classmethod
    def countDesc(cls, desc, viewCache, recFn):
        """Count the matches of the desc to the front.
        
        >>> viewCache = ViewCache(obs=[(Cement, Empty, Wall, Butterfly, Brick, Butterfly), (Wall, Empty, Cement, Eiffel, Brick, Eiffel), (Cement, Sofa, Wall, Eiffel, Brick, Eiffel), (Wall, Empty, Wall, End, Wall, End)])
        >>> desc = Thing(dist='0', value=Intersection, type='Struct', side=[Front])
        >>> PomdpAntieSimRecognizer.countDesc(desc, viewCache, PomdpAntieSimRecognizer.recIntersection)
        3
        >>> desc.Part=[Thing(dist='0', Path_n=Path, value=Path, type='Path', side=[Left])]
        >>> PomdpAntieSimRecognizer.countDesc(desc, viewCache, PomdpAntieSimRecognizer.recIntersection)
        2
        >>> desc.Part[0].side=[Right]
        >>> PomdpAntieSimRecognizer.countDesc(desc, viewCache, PomdpAntieSimRecognizer.recIntersection)
        1
        >>> del desc.Part
        >>> desc.Struct_type=[Thing(Count=[4], dist='0:', value=PathDir, type='Pathdir')]
        >>> PomdpAntieSimRecognizer.countDesc(desc, viewCache, PomdpAntieSimRecognizer.recIntersection)
        0
        >>> desc.Struct_type=[Thing(Count=[3], dist='0:', value=PathDir, type='Pathdir')]
        >>> PomdpAntieSimRecognizer.countDesc(desc, viewCache, PomdpAntieSimRecognizer.recIntersection)
        2
        """
        if 'im_class' in dir(recFn):
            count = int(recFn(desc, viewCache))
        else:
            count = int(recFn(cls, desc, viewCache))
        if len(viewCache[Front]) == 1:
            return count
        else:
            return count + cls.countDesc(desc, viewCache.project(), recFn)

    def recDetails(cls,desc,viewCache,checkPaths=True):
        """
        >>> PomdpAntieSimRecognizer.recPath(Thing(Loc=[Thing(dist='0:', value=Corner, type='Struct')], dist='0', Appear=[Brick], value=Path, type='Path', side=[Front]), ViewCache([[(Cement, Empty, Wall, Butterfly, Brick, Butterfly), (Wall, Empty, Cement, Eiffel, Brick, Eiffel), (Cement, Sofa, Wall, Eiffel, Brick, Eiffel), (Wall, Empty, Wall, End, Wall, End)], [(Brick, Empty, Wall, End, Wall, End)], [(Wall, Empty, Cement, End, Wall, End)], [(Brick, Empty, Cement, Butterfly, Cement, Butterfly), (Wood, Easel, Wood, Eiffel, Cement, Eiffel), (Wall, Empty, BlueTile, End, Wall, End)]]))
        True
        >>> PomdpAntieSimRecognizer.recIntersection(Thing(dist='0:', Detail=[Thing(dist='0:', side=[Front], value=Eiffel, type='Obj'), Thing(dist='0:', side=[Back], value=Butterfly, type='Obj'), Thing(dist='0:', type='Obj', value=Empty)], value=Intersection, type='Struct', side=[Front]), ViewCache([[(Cement, Empty, Wall, Butterfly, Brick, Butterfly), (Wall, Empty, Cement, Eiffel, Brick, Eiffel), (Cement, Sofa, Wall, Eiffel, Brick, Eiffel), (Wall, Empty, Wall, End, Wall, End)], [(Brick, Empty, Wall, End, Wall, End)], [(Wall, Empty, Cement, End, Wall, End)], [(Brick, Empty, Cement, Butterfly, Cement, Butterfly), (Wood, Easel, Wood, Eiffel, Cement, Eiffel), (Wall, Empty, BlueTile, End, Wall, End)]]))
        True
        >>> PomdpAntieSimRecognizer.recIntersection(Thing(dist='0', Detail=[Thing(dist='0:', value=DeadEnd, side=[Front], type='Struct'), Thing(value=Intersection, Part=[Thing(dist='0', Path_n=Path, value=Path, type='Path', side=[Left])], dist='0:', type='Struct')], value=Intersection, type='Struct', side=[At]), ViewCache([[(Honeycomb, Empty, Wall, Fish, Rose, Fish), (Wall, Empty, Wall, End, Wall, End)], [(Rose, Empty, Wall, End, Wall, End)], [(Wall, Empty, Honeycomb, Fish, Rose, Fish), (Wall, Empty, Wall, Fish, Rose, Fish), (Brick, Hatrack, Brick, Eiffel, Rose, Eiffel), (BlueTile, Empty, BlueTile, End, Wall, End)], [(Rose, Empty, Wall, End, Honeycomb, End)]]))
        True
        """
        match = True
        details = []
        for attrib in ('Detail','Loc','On','Part','Type'):
            if hasattr(desc,attrib) and getattr(desc,attrib):
                val = getattr(desc,attrib)
                if (attrib == 'On' and
                    ((isinstance(val,list) and isinstance(val[0],Thing) and val[0].value == Wall)
                     or (isinstance(val,Thing) and val.value == Wall))):
                    continue
                if (not checkPaths and
                    ((isinstance(val,list) and isinstance(val[0],Thing) and val[0].value == Path)
                     or (isinstance(val,Thing) and val.value == Path))):
                    continue
                new_details = copy.deepcopy(val)
                if not isinstance(new_details,list): new_details = [new_details]
                for detail in new_details[:]:
                    if not isinstance(detail,Thing):
                        new_details.remove(detail)
                        continue
                    if attrib in ('Loc','On',):
                        detail.side = [At]
                    elif not detail.side or detail.side == [detail.value.ViewPosition]:
                        detail.side = desc.side
                    # Mirror direction if looking backwards
                    elif desc.side == [Back]:
                        for i,side in enumerate(detail.side[:]):
                            detail.side[i] = opposite(side)
                    if not hasattr(detail,'dist') or not detail.dist: detail.dist = desc.dist
                details.extend(new_details)
##        if details:
##            logger.debug('recDetails(%r, %r, checkPaths=%r): details %r',desc,viewCache,checkPaths,details)
        for detail in details:
            match = cls.Recognizers[detail.type](cls,detail,viewCache)
            if not match and detail.side == [Back]:
                detail.side = [Front]
                match = cls.Recognizers[detail.type](cls,detail,viewCache)
            if not match: return False # Prob
        if hasattr(desc,'between') and desc.between:
            objA,recA = desc.between[0], cls.Recognizers[desc.between[0].type]
            if len(desc.between) == 1:
                objB,recB = None, None
            elif len(desc.between) == 2:
                objB,recB = desc.between[1], cls.Recognizers[desc.between[1].type]
            else:
                objB,recB = desc.between[1], cls.Recognizers[desc.between[1].type]
                logger.error('recDetails(%r, %r): between, Too many objects: %r',desc,viewCache,desc.between)
            match = cls.recBetween(desc,viewCache, objA, recA, objB, recB)
        return match # Prob
    recDetails = classmethod(recDetails)

    def getPathDescs(cls,desc):
        """
        >>> desc = Thing(dist='0', Appear=[Brick], value=Intersection, type='Struct', side=[At])
        >>> CompoundRecognizer.getPathDescs(desc)
        [Thing(dist='0', Appear=[Brick], value=Path, type='Path')]
        """
        paths = []
        if hasattr(desc,'Path') and desc.Path:
            append_flat(paths,copy.deepcopy(desc.Path))
        if hasattr(desc,'Part') and desc.Part:
            append_flat(paths,copy.deepcopy(desc.Part))
        if hasattr(desc,'Detail') and desc.Detail:
            if hasattr(desc.Detail,'desc'):
                details = desc.Detail.desc
            else: details = desc.Detail
            for detail in details:
                if detail.value==Path:
                    paths.append(copy.deepcopy(detail))
        if not paths:
            paths=[Thing(value=Path,type='Path')]
        for path in paths:
            if not path: paths.remove(path)
            for attr in ('Appear','Detail'):
                if hasattr(desc,attr) and getattr(desc,attr):
                    setattr(path, attr, [])
                    for val in getattr(desc,attr):
                        if not isinstance(val,Thing) or val.value != Path:
                            getattr(path,attr).append(val)
            path.dist = desc.dist
        return paths
    getPathDescs = classmethod(getPathDescs)

    def recDesc(cls,desc,viewCache):
        """
        >>> R = PomdpAntieSimRecognizer
        >>> R.recStruct(Thing(dist='0', value=Corner, type='Struct', side=[At]), ViewCache(obs=[(Wall, Empty, Cement, End, Wall, End)]))
        False
        >>> R.recStruct(Thing(dist='0', value=End, type='Struct', side=[At]), ViewCache(obs=[(Wall, Empty, Cement, End, Wall, End)]))
        True
        >>> R.recStruct(Thing(dist='0:', Detail=[Thing(dist='0:', type='Obj', value=Barstool)], value=Intersection, type='Struct', side=[Front]), ViewCache(obs=[(Cement, Empty, Wall, End, Wall, End)]))
        False
        >>> R.recStruct(Thing(dist='0:', Detail=[Thing(dist='0:', type='Obj', value=Barstool)], value=Intersection, type='Struct', side=[Front]), ViewCache(obs=[(Wall, Empty, Wall, Fish, Cement, Fish), (Cement, Barstool, Wall, End, Wall, End)]))
        True
        >>> desc = Thing(dist='0', value=Intersection, type='Struct', side=[Front])
        >>> desc.Count=[3]
        >>> viewCache = ViewCache(obs=[(Cement, Empty, Wall, Butterfly, Brick, Butterfly), (Wall, Empty, Cement, Eiffel, Brick, Eiffel), (Cement, Sofa, Wall, Eiffel, Brick, Eiffel), (Wall, Empty, Wall, End, Wall, End)])
        >>> PomdpAntieSimRecognizer.recDesc(desc, viewCache)
        True
        >>> desc.Count=[2]
        >>> PomdpAntieSimRecognizer.recDesc(desc, viewCache)
        False
        >>> desc.Count_operator = operator.ge
        >>> PomdpAntieSimRecognizer.recDesc(desc, viewCache)
        True
        """
        #logger.debug('recDesc(%r, %r)',desc,viewCache)
        cls.checkDesc(desc,'recDesc')
        try:
            recognizer = cls.Recognizers[desc.value]
        except LookupError:
            try:
                recognizer = cls.Recognizers[desc.type]
            except LookupError:
                raise ValueError("recDesc: Can't recognize", desc)
        if recognizer == cls.Recognizers['Struct']:
            raise ValueError("recDesc: Can't recognize because of recursion without bound", desc)
        if hasattr(desc, 'Count') and desc.Count!=[] and Options.RecognizeCount:
            descCp = copy.deepcopy(desc)
            if descCp.dist:
                view_dist = int(descCp.dist[0])
            else: view_dist = 0
            if view_dist != 0 and len(viewCache[Front]) > view_dist:
                vc = viewCache.project()
            else: vc = viewCache
            descCp.dist = '0'
            del descCp.Count
            if descCp.side == [Front]: descCp.side = [At]
            if hasattr(desc, 'Count_operator'):
                comparison = desc.Count_operator
                del descCp.Count_operator
            else: comparison = operator.eq
            return comparison(cls.countDesc(descCp, vc, recognizer), desc.Count[0])
        return recognizer(cls,desc,viewCache) # Prob
    Recognizers['Struct'] = recDesc
    Recognizers[AreaStructure] = recDesc
    recDesc = classmethod(recDesc)
    recStruct = recDesc

    def recEnd(cls,desc,viewCache):
        """
        >>> R = PomdpAntieSimRecognizer
        >>> R.recEnd(Thing(dist='0:', Part=[Thing(dist='0:', Appear=[BlueTile], value=Path, type='Path')], value=End, type='Struct', side=[Front]), ViewCache(obs=[(BlueTile, Empty, BlueTile, End, Wall, End)]))
        False
        >>> vc = ViewCache([[(BlueTile, Empty, BlueTile, End, Wall, End)], [(Wall, Empty, Cement, Fish, BlueTile, Fish), (Cement, Chair, Cement, Eiffel, BlueTile, Eiffel), (Wall, Empty, Wall, Eiffel, BlueTile, Eiffel), (Wall, Empty, Rose, Eiffel, BlueTile, Eiffel), (Cement, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, BlueTile, Fish, Cement, Fish), (Cement, Empty, Wall, End, Wall, End)], [(Cement, Empty, Wall, Fish, BlueTile, Fish), (Honeycomb, Hatrack, Honeycomb, Fish, BlueTile, Fish), (Wall, Empty, Wall, End, Wall, End)]])
        >>> blueAtDesc = Thing(dist='0', Part=[Thing(dist='0:', value=Path, Appear=[BlueTile], type='Path')], value=End, type='Struct', side=[At])
        >>> cementAtDesc = Thing(dist='0', Part=[Thing(dist='0:', value=Path, Appear=[Cement], type='Path')], value=End, type='Struct', side=[At])
        >>> R.recEnd(cementAtDesc, vc) # West
        True
        >>> R.recEnd(blueAtDesc, vc) # West
        False
        >>> vc=vc.rotate(Left)
        >>> R.recEnd(blueAtDesc, vc) # South
        False
        >>> R.recEnd(cementAtDesc, vc) # South
        False
        >>> cementFrontDesc = Thing(dist='0:', Part=[Thing(dist='0:', Appear=[Cement], value=Path, type='Path')], value=End, type='Struct', side=[Front])
        >>> blueFrontDesc = Thing(dist='0:', Part=[Thing(dist='0:', Appear=[BlueTile], value=Path, type='Path')], value=End, type='Struct', side=[Front])
        >>> vc=vc.rotate(Left)
        >>> R.recEnd(cementFrontDesc, vc) # West
        True
        >>> R.recEnd(blueFrontDesc, vc) # West
        False
        >>> vc=vc.rotate(Left)
        >>> R.recEnd(blueFrontDesc, vc) # South
        True
        >>> R.recEnd(cementFrontDesc, vc) # South, There is an end of a cement path in front, but there's another closer...
        True
        >>> vc=vc.rotate(Back)
        >>> R.recEnd(blueFrontDesc, vc) # North
        True
        >>> R.recEnd(cementFrontDesc, vc) # North
        False
        >>> R.recEnd(Thing(dist='0:', Part=[Thing(dist='0:', Appear=[Brick], value=Intersection, type='Struct')], value=Corner, type='Struct', side=[Front]), ViewCache([[(BlueTile, Empty, Wall, Eiffel, Cement, Eiffel), (Wood, Easel, Wood, Butterfly, Cement, Butterfly), (Brick, Empty, Wall, End, Wall, End)], [(Wall, Empty, Rose, Eiffel, BlueTile, Eiffel), (Cement, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, BlueTile, Unknown, Rose, Unknown)], [(Rose, Empty, Wall, Eiffel, BlueTile, Eiffel), (Rose, Empty, Rose, Unknown, Cement, Unknown)]]))
        True
        >>> R.recEnd(Thing(dist='0:', Detail=[Thing(dist='0:', value=Hatrack, type='Obj')], value=End, type='Struct', side=[Back]), ViewCache([[(Wall, Sofa, Cement, Butterfly, Wood, Butterfly), (Brick, Empty, Brick, Butterfly, Wood, Butterfly), (Wall, Empty, Wall, End, Wall, End)], [(Wood, Sofa, Wood, Butterfly, Cement, Butterfly), (Wall, Empty, Wall, End, Wall, End)], [(Cement, Sofa, Wall, Butterfly, Wood, Butterfly), (Wall, Hatrack, Wall, End, Wall, End)], [(Wood, Sofa, Wood, End, Wall, End)]]))
        True
        >>> R.recEnd(Thing(dist='0:', Part=[Thing(dist='0:', Appear=[Honeycomb], value=Path, type='Path')], value=End, type='Struct', side=[Back]), ViewCache([[(Wall, Empty, Wall, Fish, Honeycomb, Fish), (Cement, Empty, Wall, Fish, Honeycomb, Fish), (BlueTile, Hatrack, BlueTile, Fish, Honeycomb, Fish), (Wall, Empty, Stone, End, Wall, End)], [(Honeycomb, Empty, Wall, End, Wall, End)], [(Wall, Empty, Wall, End, Wall, End)], [(Wall, Empty, Honeycomb, End, Wall, End)]]))
        True
        >>> R.recEnd(Thing(dist='0:', value=End, type='Struct', side=[Front]), ViewCache(obs=[(Honeycomb, Empty, Honeycomb, Fish, Stone, Fish), (Cement, Empty, Cement, Fish, Stone, Fish), (Brick, Empty, Brick, Eiffel, Stone, Eiffel), (BlueTile, Lamp, BlueTile, End, Wall, End)]))
        True
        >>> R.recStruct(Thing(dist='0:', value=End, type='Struct', side=[At]), ViewCache(obs=[(Honeycomb, Empty, Honeycomb, Fish, Stone, Fish), (Cement, Empty, Cement, Fish, Stone, Fish), (Brick, Empty, Brick, Eiffel, Stone, Eiffel), (BlueTile, Lamp, BlueTile, End, Wall, End)]))
        True
        >>> R.recEnd(Thing(Part=[Thing(dist='0:', Appear=[Brick], value=Path, type='Path')], dist='1:', type='Struct', side=[Front], value=End, Structural=['Long']), ViewCache([[(Cement, Empty, Wall, Butterfly, Brick, Butterfly), (Wall, Empty, Cement, Eiffel, Brick, Eiffel), (Cement, Sofa, Wall, Eiffel, Brick, Eiffel), (Wall, Empty, Wall, End, Wall, End)], [(Brick, Empty, Wall, End, Wall, End)], [(Wall, Empty, Cement, Eiffel, Brick, Eiffel), (Wall, Empty, Cement, End, Wall, End)], [(Brick, Empty, Cement, Butterfly, Cement, Butterfly), (Wood, Easel, Wood, Eiffel, Cement, Eiffel), (Wall, Empty, BlueTile, End, Wall, End)]]))
        True
        >>> R.recEnd(Thing(dist='1:', Part=[Thing(dist='0:', type='Path', value=Path)], value=End, Structural=['Long'], type='Struct', side=[Front]), ViewCache([[(Wall, Empty, Honeycomb, End, Wall, End)], [(Wall, Empty, Honeycomb, Fish, Honeycomb, Fish), (Cement, Empty, Wall, Fish, Honeycomb, Fish), (BlueTile, Hatrack, BlueTile, Fish, Honeycomb, Fish), (Wall, Empty, Stone, End, Wall, End)], [(Honeycomb, Empty, Wall, End, Wall, End)], [(Wall, Empty, Wall, End, Wall, End)]]))
        False
        """
        logger.debug('recEnd(%r, %r)',desc,viewCache)
        if not Options.IntersectionLandmarks: return True
        if hasattr(desc,'Structural'): return cls.recPath(desc,viewCache)
        cls.checkDesc(desc,'recEnd')
        view = viewCache[Front]
        for side in desc.side or [At]:
            match = False
            if side == Back and Options.PerspectiveTaking:
                match = viewCache.lookToSide(desc,side,cls.recEnd)
                if match: return match #Prob
            elif side == At:
                if '0' in desc.dist:
                    match = view.match(Wall,Front)
                    if not match and desc.value == Corner and Options.PerspectiveTaking:
                        match = viewCache.lookToSide(Thing(value=Path,type='Path',
                                                           side=[Front],dist=desc.dist),
                                                     Back,cls.recEnd)
                    if match and desc.value == Corner:
                        Lmatch = view.match(Wall,Left)
                        Rmatch = view.match(Wall,Right)
                        match = (Lmatch and not Rmatch) or (not Lmatch and Rmatch) # xor
                if not match and desc.dist != '0' and Options.PerspectiveTaking:
                    if desc.side == [Front] and desc.dist == cls.decrDist(desc.dist):
                        descFront = desc
                    else:
                        descFront = copy.deepcopy(desc)
                        descFront.side,descFront.dist = [Front],cls.decrDist(desc.dist)
                    match = cls.recEnd(descFront,viewCache)
            else: # Front,Left,Right
                if '0' in desc.dist: match = view.match(Wall,side)
                if not match and desc.dist != '0':
                    match = view.search(Wall,Front,desc.dist)
            if match:
                match = cls.recDetails(desc,viewCache,checkPaths=False)
                if not match: return match # Prob
                if (side in Opposites and ('0' in desc.dist and len(view)==1)): #flipPath
                    sides = [opposite(side) for side in (Left, Front, Right) if view.match(Wall,side)]
                    if viewCache[Back].match(Wall,Front): sides.append(Back)
                else: sides = desc.side
                match = cls.recPaths(desc,viewCache,sides)
                if not match: return match # Prob
            if match: return match # Prob
        return match #Prob
    Recognizers[End] = recEnd
    recEnd = classmethod(recEnd)

    def recCorner(cls,desc,viewCache):
        """
        >>> R = PomdpAntieSimRecognizer
        >>> desc = Thing(dist='0:', value=Corner, type='Struct', side=[Front])
        >>> viewCache = ViewCache(obs=[(Cement, Sofa, Wall, Butterfly, BlueTile, Butterfly), (Cement, Hatrack, Wall, Butterfly, BlueTile, Butterfly), (Cement, Empty, Wall, End, Wall, End)])
        >>> viewCache.update(Right,[(BlueTile, Hatrack, BlueTile, Butterfly, Cement, Butterfly), (Wall, Empty, Cement, End, Wall, End)])
        >>> R.recCorner(desc, viewCache) # True corner in the distance
        True
        >>> desc.dist = '0'
        >>> R.recCorner(desc, viewCache) # no immediate corner
        False
        >>> viewCache.update(Front, [(Wall, Empty, Cement, End, Wall, End)])
        >>> R.recCorner(desc, viewCache) # Immediate corner
        True
        
        Ignoring Corner probably means L corner.
        """
        return (cls.recIntersection(desc,viewCache) and cls.recEnd(desc,viewCache)) #Prob
    Recognizers[Corner] = recCorner
    recCorner = classmethod(recCorner)

    def recTopoPlace(cls,desc,viewCache):
        return (cls.recIntersection(desc,viewCache) or cls.recDeadEnd(desc,viewCache)) # Prob
    Recognizers[TopoPlace] = recTopoPlace
    recTopoPlace = classmethod(recTopoPlace)

    def recMiddle(cls,desc,viewCache):
        """
        >>> R = PomdpAntieSimRecognizer
        >>> viewCache = ViewCache([[(Wall, Empty, Cement, Butterfly, Brick, Butterfly), (Wood, Empty, Wood, Butterfly, Brick, Butterfly), (Wall, Empty, Wall, End, Wall, End)], [(Brick, Empty, Brick, Unknown, Wood, Unknown)], [(Cement, Empty, Wall, Butterfly, Brick, Butterfly), (Cement, Lamp, Wall, Eiffel, Brick, Eiffel), (Wall, Empty, Wall, Eiffel, Brick, Eiffel), (Rose, Hatrack, Rose, Eiffel, Brick, Eiffel), (Grass, Empty, Grass, Eiffel, Brick, Eiffel), (Stone, Empty, Stone, Eiffel, Brick, Eiffel), (Cement, Empty, Wall, End, Wall, End)], [(Brick, Empty, Brick, End, Wall, End)]])
        >>> R.recStruct(Thing(dist='0', type='Struct', side=[Front], Detail=[Thing(dist='0:', Appear=[Wood], value=Path, type='Path')], value=Middle), viewCache)
        False
        >>> R.recStruct(Thing(dist='0:', type='Struct', side=[Front], Detail=[Thing(dist='0:', Appear=[Wood], value=Path, type='Path')], value=Middle), viewCache)
        True
        >>> R.recStruct(Thing(dist='0', type='Struct', side=[Front], Detail=[Thing(dist='0:', Appear=[Brick], value=Path, type='Path')], value=Middle), viewCache)
        True
        >>> R.recStruct(Thing(dist='0:', type='Struct', side=[Front], Detail=[Thing(dist='0:', value=Path, type='Path')], value=Middle), viewCache)
        True
        >>> R.recStruct(Thing(dist='0:', type='Struct', side=[Front], Detail=[Thing(dist='0:', Appear=[Cement], value=Path, type='Path')], value=Middle), viewCache)
        False # But returns True because of not checking they are two different cement halls
        """
        if not Options.CausalLandmarks: return True
        match = False
        paths = cls.getPathDescs(desc)
        if not paths:
            pathA, pathB = Thing(dist='0:', value=Path, type='Path'), None
        elif len(paths) == 1:
            pathA, pathB = paths[0],None
        elif len(paths) == 2:
            pathA,pathB = paths
        else:
            pathA,pathB = paths[:2]
            logger.error('recMiddle(%r, %r): Too many paths: %r',desc,viewCache,paths)
        match = cls.recBetween(desc,viewCache, pathA,cls.Recognizers[Path], pathB,cls.Recognizers[Path])
        return match #Prob
    Recognizers[Middle] = recMiddle
    recMiddle = classmethod(recMiddle)

    def recBetween(cls,desc,viewCache,objA,recA,objB=None,recB=None):
        """
        >>> R = PomdpAntieSimRecognizer
        >>> desc=Thing(dist='0', value=Position, between=[Thing(dist='1:', Appear=[Stone], value=Path, type='Obj', side=[Between]), Thing(dist='1:', Appear=[Greenish], value=Path, type='Obj', side=[Between])], type='Position', side=[At])
        >>> R.recBetween(desc, ViewCache([[(Wall, Empty, Cement, Butterfly, Brick, Butterfly), (Wood, Empty, Wood, Butterfly, Brick, Butterfly), (Wall, Empty, Wall, End, Wall, End)], [(Brick, Empty, Brick, Unknown, Wood, Unknown)], [(Wood, Empty, Wood, Butterfly, Brick, Butterfly), (Cement, Empty, Wall, Butterfly, Brick, Butterfly), (Cement, Lamp, Wall, Eiffel, Brick, Eiffel), (Wall, Empty, Wall, Eiffel, Brick, Eiffel), (Rose, Hatrack, Rose, Eiffel, Brick, Eiffel), (Grass, Empty, Grass, Eiffel, Brick, Eiffel), (Stone, Empty, Stone, Eiffel, Brick, Eiffel), (Cement, Empty, Wall, End, Wall, End)], [(Brick, Empty, Brick, End, Wall, End)]]), desc.between[0], R.Recognizers[Path], desc.between[1], R.Recognizers[Path])
        False
        >>> desc=Thing(dist='0', value=Position, between=[Thing(dist='1:', Appear=[Wood], value=Path, type='Obj', side=[Between]), Thing(dist='1:', Appear=[Greenish], value=Path, type='Obj', side=[Between])], type='Position', side=[At])
        >>> R.recBetween(desc, ViewCache([[(Wall, Empty, Cement, Butterfly, Brick, Butterfly), (Wood, Empty, Wood, Butterfly, Brick, Butterfly), (Wall, Empty, Wall, End, Wall, End)], [(Brick, Empty, Brick, Unknown, Wood, Unknown)], [(Wood, Empty, Wood, Butterfly, Brick, Butterfly), (Cement, Empty, Wall, Butterfly, Brick, Butterfly), (Cement, Lamp, Wall, Eiffel, Brick, Eiffel), (Wall, Empty, Wall, Eiffel, Brick, Eiffel), (Rose, Hatrack, Rose, Eiffel, Brick, Eiffel), (Grass, Empty, Grass, Eiffel, Brick, Eiffel), (Stone, Empty, Stone, Eiffel, Brick, Eiffel), (Cement, Empty, Wall, End, Wall, End)], [(Brick, Empty, Brick, End, Wall, End)]]), desc.between[0], R.Recognizers[Path], desc.between[1], R.Recognizers[Path])
        True
        """
        if not (Options.ObjectLandmarks and Options.PerspectiveTaking): return True
        match = False
        objCpA = copy.deepcopy(objA)
        if objB:
            objCpB = copy.deepcopy(objB)
            SidePairs = ((Front,Back), (Back,Front), (Left,Right), (Right,Left))
        else:
            objCpB = copy.deepcopy(objA)
            recB = recA
            SidePairs = ((Front,Back), (Right,Left))
        if '0' in desc.dist:
            for sideA,sideB in SidePairs:
                objCpA.side,objCpB.side = [sideA],[sideB]
                match = recA(cls,objCpA,viewCache) and recB(cls,objCpB,viewCache) #Prob
                if match: break
        if '0' not in desc.dist or (':' in desc.dist and not match):
            objCpA.side = objCpB.side = [At]
            objCpA.dist = '0:'
            if desc.side[0] not in (At,Front):
                viewCache = viewCache.rotate(desc.side[0])
            match = recA(cls,objCpA,viewCache) and recB(cls,objCpB,viewCache) # Prob
            if not match:
                objCpB.dist = '0:'
                objCpA.dist = '1:'
                match = recA(cls,objCpA,viewCache) and recB(cls,objCpB,viewCache) #Prob
        logger.debug('recBetween(%r, %r, %r, %r): %r',desc,viewCache,objA,objB,match)
        return match # Prob
    Recognizers[Between] = recBetween
    recBetween = classmethod(recBetween)

    def recDeadEndImmediateFront(cls,view):
        """
        >>> R = PomdpAntieSimRecognizer
        >>> R.recDeadEndImmediateFront(ViewCache(obs=[(Wall, Empty, Wall, End, Wall, End)])[Front])
        True
        >>> R.recDeadEndImmediateFront(ViewCache(obs=[(Wall, Empty, Wall, End, BlueTile, End)])[Front])
        False
        """
        for side in Front,Left,Right:
            if not view.match(Wall,side): return False
        return True
    recDeadEndImmediateFront = classmethod(recDeadEndImmediateFront)

    def recDeadEnd(cls,desc,viewCache):
        """
        >>> R = PomdpAntieSimRecognizer
        >>> R.recDeadEnd(Thing(dist='0:', value=DeadEnd, type='Struct', side=[Front]), ViewCache([[(Wall, Empty, Wall, End, Wall, End)], [(Wall, Empty, Honeycomb, End, Wall, End)], [(Wall, Empty, Honeycomb, Fish, Honeycomb, Fish), (Cement, Empty, Wall, Fish, Honeycomb, Fish), (BlueTile, Hatrack, BlueTile, Fish, Honeycomb, Fish), (Wall, Empty, Stone, End, Wall, End)], [(Honeycomb, Empty, Wall, End, Wall, End)]]))
        True
        >>> R.recDeadEnd(Thing(dist='0:', value=DeadEnd, type='Struct', side=[Back]), ViewCache(obs=[(Cement, Empty, Wall, End, Wall, End)]))
        False
        >>> viewCache = ViewCache(obs=[(Wall, Barstool, Cement, Fish, Cement, Fish), (Wall, Empty, Wall, End, Wall, End)])
        >>> desc = Thing(dist='0', value=DeadEnd, type='Struct', side=[Front])
        >>> R.recDeadEnd(desc, viewCache) # No Immediate dead end
        False
        >>> desc.dist = '0:'
        >>> R.recDeadEnd(desc, viewCache) # Dead end in front down the hall
        True
        >>> R.recDeadEnd(Thing(dist='1:', type='Struct', side=[Front], value=TopoPlace), ViewCache([[(BlueTile, Chair, BlueTile, End, Wall, End)], [(Wall, Chair, BlueTile, Unknown, BlueTile, Unknown)], [(BlueTile, Chair, BlueTile, Fish, Wood, Fish), (Brick, Lamp, Brick, Eiffel, Wood, Eiffel), (Cement, Empty, Wall, Eiffel, Wood, Eiffel), (Honeycomb, Empty, Honeycomb, End, Wall, End)], [(Wall, Chair, BlueTile, Unknown, BlueTile, Unknown)]]))
        False
        >>> R.recDeadEnd(Thing(dist='0:', Part=[Thing(dist='0:', Appear=[Flooring, Honeycomb], Detail=[Thing(On=[Thing(dist='0:', type='Obj', value=Wall)], dist='0:', value=Fish, type='Obj')], value=Path, type='Path')], value=DeadEnd, type='Struct', side=[Front]), ViewCache([[(Wall, Empty, Wall, End, Wall, End)], [(Wall, Empty, Honeycomb, End, Wall, End)], [(Wall, Empty, Wall, Fish, Honeycomb, Fish), (Cement, Empty, Wall, Fish, Honeycomb, Fish), (BlueTile, Hatrack, BlueTile, Fish, Honeycomb, Fish), (Wall, Empty, Stone, End, Wall, End)], [(Honeycomb, Empty, Wall, End, Wall, End)]]))
        True
        >>> R.recDeadEnd(Thing(dist='0', Detail=[Thing(dist='0', Part=[Thing(dist='0', Appear=[Stone], value=Path, type='Path')], value=DeadEnd, type='Struct')], value=Path, type='Path', side=[At]), ViewCache([(Honeycomb, Empty, Honeycomb, Fish, Stone, Fish), (Wall, Easel, Wall, End, Wall, End)]))
        False
        >>> R.recEnd(Thing(dist='1:', type='Struct', side=[Front], value=DeadEnd), ViewCache([[(Honeycomb, Empty, Wall, Fish, Rose, Fish), (Wall, Empty, Wall, End, Wall, End)], [(Rose, Empty, Wall, End, Wall, End)], [(Wall, Empty, Honeycomb, Fish, Rose, Fish), (Wall, Empty, Wall, Eiffel, Rose, Eiffel), (Brick, Hatrack, Brick, Unknown, Rose, Unknown)], [(Rose, Empty, Wall, End, Honeycomb, End)]]))
        True
        """
        cls.checkDesc(desc,'recDeadEnd')
        logger.debug('recDeadEnd(%r, %r)',desc,viewCache)
        if not Options.IntersectionLandmarks: return True
        match = cls.recEnd(desc,viewCache)
        if not match: return match #Prob
        for side in desc.side:
            match = False
            if side in (At, Front): # Front,Left,Right
                match = cls.recDeadEndImmediateFront(viewCache[Front])
                if not match and desc.dist != '0' and len(viewCache[Front]) > 1 and Options.PerspectiveTaking:
                    if desc.dist == cls.decrDist(desc.dist):
                        descCp = desc
                    else:
                        descCp = copy.deepcopy(desc)
                        descCp.dist = cls.decrDist(desc.dist)
                    match = cls.recDeadEnd(descCp,viewCache.project())
            elif side == Back and Options.PerspectiveTaking:
                match = viewCache.lookToSide(desc,side,cls.recDeadEnd)
            elif side == At:
                if Options.PerspectiveTaking: Sides = (Front,Left,Right,Back)
                else: Sides = (Front,)
                for oside in Sides:
                    match = viewCache.lookToSide(desc,oside,cls.recDeadEnd)
                    if match: return match # Prob
            if match: return match # Prob
        return match # Prob
    Recognizers[DeadEnd] = recDeadEnd
    recDeadEnd = classmethod(recDeadEnd)

    def recIntersection(cls,desc,viewCache):
        """
        >>> R = PomdpAntieSimRecognizer
        >>> R.recIntersection(Thing(dist='1', Part=[Thing(dist='0', Appear=[Honeycomb, Stone], value=Path, type='Path')], value=Intersection, type='Struct', side=[At]), ViewCache([[(Cement, Empty, Cement, Fish, Stone, Fish), (Honeycomb, Empty, Honeycomb, Fish, Stone, Fish), (Wall, Easel, Wall, End, Wall, End)], [(Stone, Empty, Stone, Unknown, Cement, Unknown)], [(Cement, Empty, Cement, Fish, Stone, Fish), (Brick, Empty, Brick, Fish, Stone, Fish), (Brick, Empty, Brick, Unknown, Grass, Unknown)], [(Stone, Empty, Stone, Unknown, Cement, Unknown)]]))
        True
        >>> R.recIntersection(Thing(dist='0', Part=[Thing(dist='0', Appear=[Rose,Brick], value=Path, type='Path')], value=Intersection, type='Struct', side=[At]), ViewCache([[(Rose, Easel, Rose, Fish, Brick, Fish), (Grass, Hatrack, Grass, End, Wall, End)], [(Brick, Easel, Brick, Unknown, Unknown, Unknown)], [(Unknown, Easel, Rose, Unknown, Brick, Unknown)], [(Brick, Easel, Brick, Fish, Rose, Fish), (BlueTile, Empty, BlueTile, End, Wall, End)]]))
        True
        >>> R.recIntersection(Thing(dist='0', Part=[Thing(dist='0', Appear=[Rose,Grass], value=Path, type='Path', side=[Right])], value=Intersection, type='Struct', side=[At]), ViewCache([[(Rose, Easel, Rose, Fish, Brick, Fish), (Grass, Hatrack, Grass, End, Wall, End)], [(Brick, Easel, Brick, Unknown, Unknown, Unknown)], [(Unknown, Easel, Rose, Unknown, Brick, Unknown)], [(Brick, Easel, Brick, Fish, Rose, Fish), (BlueTile, Empty, BlueTile, End, Wall, End)]]))
        False
        >>> R.recIntersection(Thing(dist='0:', Part=[Thing(dist='0', Appear=[Rose,Grass], value=Path, type='Path', side=[Right])], value=Intersection, type='Struct', side=[At]), ViewCache([[(Rose, Easel, Rose, Fish, Brick, Fish), (Grass, Hatrack, Grass, End, Wall, End)], [(Brick, Easel, Brick, Unknown, Unknown, Unknown)], [(Unknown, Easel, Rose, Unknown, Brick, Unknown)], [(Brick, Easel, Brick, Fish, Rose, Fish), (BlueTile, Empty, BlueTile, End, Wall, End)]]))
        False #But Currently True
        >>> R.recIntersection(Thing(dist='0', Part=[Thing(dist='0:', Appear=[Honeycomb, Stone], value=Path, type='Path')], value=Intersection, type='Struct', side=[At]), ViewCache(obs=[(Brick, Empty, Brick, Fish, Stone, Fish), (Cement, Empty, Cement, Fish, Stone, Fish), (Honeycomb, Empty, Honeycomb, Fish, Stone, Fish), (Wall, Easel, Wall, End, Wall, End)]))
        False
        >>> R.recIntersection(Thing(dist='0', Part=[Thing(dist='0:', Appear=[Honeycomb, Stone], value=Path, type='Path')], value=Intersection, type='Struct', side=[At]), ViewCache(obs=[(Honeycomb, Empty, Honeycomb, Fish, Stone, Fish), (Wall, Easel, Wall, End, Wall, End)]))
        True
        >>> R.recIntersection(Thing(dist='0', Part=[Thing(dist='0', Appear=[Stone], value=Path, type='Path', side=[At]), Thing(dist='0:', Appear=[Honeycomb], value=Path, type='Path')], value=Intersection, type='Struct', side=[At]), ViewCache(obs=[(Honeycomb, Empty, Honeycomb, Fish, Stone, Fish), (Wall, Easel, Wall, End, Wall, End)]))
        True
        >>> R.recIntersection(Thing(dist='0', Part=[Thing(dist='0', Appear=[Stone], value=Path, type='Path', side=[At]), Thing(dist='0:', Appear=[Honeycomb], value=Path, type='Path')], value=Intersection, type='Struct', side=[At]), ViewCache(obs=[(Brick, Empty, Brick, Fish, Stone, Fish), (Cement, Empty, Cement, Fish, Stone, Fish), (Honeycomb, Empty, Honeycomb, Fish, Stone, Fish), (Wall, Easel, Wall, End, Wall, End)]))
        False
        >>> R.recIntersection(Thing(dist='0', Part=[Thing(dist='0:', Appear=[Flooring, Rose, BlueTile], value=Path, type='Path')], value=Intersection, type='Struct', side=[At]), ViewCache(obs=[(Cement, Sofa, Wall, Butterfly, Wood, Butterfly), (Wall, Hatrack, Wall, End, Wall, End)]))
        False
        >>> R.recIntersection(Thing(dist='0', Part=[Thing(dist='0:', Appear=[Flooring, Cement, Wood], value=Path, type='Path')], value=Intersection, type='Struct', side=[At]), ViewCache(obs=[(Cement, Sofa, Wall, Butterfly, Wood, Butterfly), (Wall, Hatrack, Wall, End, Wall, End)]))
        True
        >>> R.recIntersection(Thing(dist='0', value=Intersection, type='Struct', side=[At]), ViewCache([[(Wall, Empty, Wall, Eiffel, BlueTile, Eiffel), (Wall, Empty, Rose, Eiffel, BlueTile, Eiffel), (Cement, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, Cement, End, Wall, End)], [(Wall, Empty, Wall, Eiffel, BlueTile, Eiffel), (Cement, Chair, Cement, Fish, BlueTile, Fish), (Cement, Empty, Wall, Fish, BlueTile, Fish), (Honeycomb, Hatrack, Honeycomb, Fish, BlueTile, Fish), (Wall, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, Cement, End, Wall, End)]]))
        False
        >>> R.recIntersection(Thing(dist='1', value=Intersection, type='Struct', side=[At]), ViewCache([[(Wall, Empty, Wall, Eiffel, BlueTile, Eiffel), (Wall, Empty, Rose, Eiffel, BlueTile, Eiffel), (Cement, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, Cement, End, Wall, End)], [(Wall, Empty, Wall, Eiffel, BlueTile, Eiffel), (Cement, Chair, Cement, Fish, BlueTile, Fish), (Cement, Empty, Wall, Fish, BlueTile, Fish), (Honeycomb, Hatrack, Honeycomb, Fish, BlueTile, Fish), (Wall, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, Cement, End, Wall, End)]]))
        True
        >>> R.recIntersection(Thing(dist='1:', value=Intersection, type='Struct', side=[At]), ViewCache([[(Wall, Empty, Wall, Eiffel, BlueTile, Eiffel), (Wall, Empty, Rose, Eiffel, BlueTile, Eiffel), (Cement, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, Cement, End, Wall, End)], [(Wall, Empty, Wall, Eiffel, BlueTile, Eiffel), (Cement, Chair, Cement, Fish, BlueTile, Fish), (Cement, Empty, Wall, Fish, BlueTile, Fish), (Honeycomb, Hatrack, Honeycomb, Fish, BlueTile, Fish), (Wall, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, Cement, End, Wall, End)]]))
        True
        >>> R.recIntersection(Thing(dist='0', value=TopoPlace, between=[Thing(dist='1:', Appear=[BlueTile], value=Path, type='Path', side=[At]), Thing(dist='1:', Appear=[Brick], value=Path, type='Path', side=[At])], type='Struct', side=[At]), ViewCache([[(Cement, Easel, Cement, Butterfly, Wood, Butterfly), (Wall, Empty, Wall, Butterfly, Wood, Butterfly), (Cement, Empty, Wall, End, Wall, End)], [(Wood, Easel, Cement, Eiffel, Cement, Eiffel), (Wall, Empty, BlueTile, End, Wall, End)], [(Cement, Easel, Cement, Unknown, Wood, Unknown)], [(Wood, Easel, Cement, Butterfly, Cement, Butterfly), (Brick, Empty, Wall, End, Wall, End)]]))
        True
        >>> PomdpAntieSimRecognizer.recIntersection(Thing(dist='0:', Part=[Thing(dist='0:', Appear=[Rose], value=Path, type='Path')], value=Intersection, Struct_type=[T_Int], type='Struct', side=[Front]), ViewCache([[(Cement, Empty, Wall, Fish, Honeycomb, Fish), (Stone, Empty, Stone, Fish, Honeycomb, Fish), (Grass, Empty, Grass, Fish, Honeycomb, Fish), (Rose, Empty, Rose, End, Wall, End)], [(Honeycomb, Empty, Wall, End, Wall, End)], [(Wall, Empty, Cement, End, Wall, End)], [(Honeycomb, Empty, Cement, Fish, Cement, Fish), (Wall, Chair, Wall, Fish, Cement, Fish), (Wall, Barstool, Cement, End, Wall, End)]]))
        True
        >>> PomdpAntieSimRecognizer.recIntersection(Thing(dist='0:', Part=[Thing(dist='0:', Detail=[Thing(dist='0:', value=GenChair, type='Obj')], value=Path, type='Path')], value=Intersection, type='Struct', side=[Back], Structural=['Short']), ViewCache([[(Grass, Hatrack, Grass, End, Wall, End)], [(Wall, Hatrack, Brick, Unknown, Grass, Unknown)], [(Grass, Hatrack, Grass, Fish, Brick, Fish), (Rose, Easel, Rose, Fish, Brick, Fish), (Wood, Lamp, Wood, Eiffel, Brick, Eiffel), (Stone, Empty, Stone, Butterfly, Brick, Butterfly), (Cement, Empty, Cement, Unknown, Unknown, Unknown)], [(Brick, Hatrack, Wall, Unknown, Grass, Unknown)]]))
        False
        >>> PomdpAntieSimRecognizer.recIntersection(Thing(dist='0:', Detail=[Thing(dist='0:', Appear=[Rose], value=Path, side=[Left], type='Path')], value=Intersection, type='Struct', side=[Back]), ViewCache([[(Wall, Empty, Cement, Eiffel, BlueTile, Eiffel), (Rose, Empty, Wall, Eiffel, BlueTile, Eiffel), (Wall, Empty, Wall, Eiffel, BlueTile, Eiffel), (Cement, Chair, Cement, Fish, BlueTile, Fish), (Cement, Empty, Wall, Fish, BlueTile, Fish), (Honeycomb, Hatrack, Honeycomb, Fish, BlueTile, Fish), (Wall, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, Cement, Eiffel, Cement, Eiffel), (Wood, Easel, Wood, End, Wall, End)], [(Cement, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, Wall, End, Wall, End)]]))
        False
        >>> PomdpAntieSimRecognizer.recIntersection(Thing(dist='0:', Detail=[Thing(dist='0:', Appear=[Rose], value=Path, side=[Left], type='Path')], value=Intersection, type='Struct', side=[Back]), ViewCache([[(Wall, Empty, Cement, Eiffel, BlueTile, Eiffel), (Rose, Empty, Wall, Eiffel, BlueTile, Eiffel), (Wall, Empty, Wall, Eiffel, BlueTile, Eiffel), (Cement, Chair, Cement, Fish, BlueTile, Fish), (Cement, Empty, Wall, Fish, BlueTile, Fish), (Honeycomb, Hatrack, Honeycomb, Fish, BlueTile, Fish), (Wall, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, Cement, Eiffel, Cement, Eiffel), (Wood, Easel, Wood, End, Wall, End)], [(Cement, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, Wall, End, Wall, End)]]).project())
        True
        >>> PomdpAntieSimRecognizer.recIntersection(Thing(Part=[Thing(dist='0:', Appear=[Rose, Brick], value=Path, type='Path')], dist='0:', Struct_type=[Thing(Count=[4], dist='0:', value=PathDir, type='Pathdir')], type='Struct', Detail=[Thing(dist='0:', value=Easel, type='Obj')], value=Intersection, side=[At]), ViewCache([[(Cement, Empty, Cement, Butterfly, Brick, Butterfly), (Stone, Empty, Stone, Eiffel, Brick, Eiffel), (Wood, Lamp, Wood, Fish, Brick, Fish), (Rose, Easel, Rose, Fish, Brick, Fish), (Grass, Hatrack, Grass, End, Wall, End)], [(Brick, Empty, Cement, Unknown, Cement, Unknown)], [(Cement, Empty, Cement, Unknown, Unknown, Unknown)], [(Brick, Empty, Cement, Unknown, Cement, Unknown)]]))
        True
        recIntersection(Thing(dist='0:', Struct_type=[3, PathDir], value=Intersection, type='Struct', side=[At]), ViewCache([[(Cement, Chair, Cement, End, Wall, End)], [(Wall, Chair, Brick, Unknown, Cement, Unknown)], [(Cement, Chair, Cement, Butterfly, Brick, Butterfly), (Cement, Empty, Cement, Butterfly, Brick, Butterfly), (Stone, Empty, Stone, Eiffel, Brick, Eiffel), (Wood, Lamp, Wood, Fish, Brick, Fish), (Rose, Easel, Rose, Unknown, Unknown, Unknown)], [(Brick, Chair, Wall, Unknown, Cement, Unknown)]]))
        True
        """
        cls.checkDesc(desc,'recIntersection')
        logger.debug('recIntersection(%r, %r)',desc,viewCache)
        match = cls.recDetails(desc,viewCache,checkPaths=False)
        if not match or not Options.IntersectionLandmarks: return match # Prob
        match = False
        if '0' in desc.dist: match = cls.countPathFragments(viewCache) >= 2
        if not match:
            if not desc.dist: desc.dist = '0:'
            if ':' in desc.dist: match = viewCache[Front].search(Path,Sides,desc.dist)
            elif match != '0': match = cls.countPathFragments(viewCache.project(int(desc.dist))) >= 2
        if not match: return match # Prob
        if hasattr(desc,'Struct_type'):
            count = None
            for struct_type in desc.Struct_type:
                match = False
                if '0' in desc.dist:
                    if struct_type == T_Int: match = cls.countGateways(viewCache) == 3 #FIXME not Y
                    elif struct_type == Corner:
                        cornerDesc = copy.deepcopy(desc)
                        del cornerDesc.Struct_type
                        cornerDesc.value = Corner
                        match = cls.recCorner(cornerDesc, viewCache)
                    elif isinstance(struct_type, int):
                        count = struct_type
                    elif ((struct_type == PathDir or struct_type == Path or
                           (hasattr(struct_type,'value') and struct_type.value == PathDir or struct_type.value == Path))
                          and not count): # n-way intersection
                        if hasattr(struct_type,'Count'):
                            count = struct_type.Count[0]
                        elif not count and len(desc.Struct_type) > 1:
                            for st in desc.Struct_type:
                                if isinstance(st, int):
                                    count = st
                                    break
                        else:
                            logger.error('recIntersection(%r) :: no count in %r.', desc, struct_type)
                            count = 2
                    # n-way intersection
                    elif struct_type == PathDir or (hasattr(struct_type, 'value') and struct_type.value == PathDir):
                        match = (cls.countGateways(viewCache) == count)
                    # n path intersection
                    elif struct_type == Path or (hasattr(struct_type, 'value') and struct_type.value == Path):
                        match = (cls.countPathFragments(viewCache) == count)
                    else:
                        logger.error('recIntersection(%r) :: Unknown Struct_type %r.', desc, struct_type)
        if match and Options.PerspectiveTaking:
            if desc.side == [Back]: match = cls.recPaths(desc,viewCache.rotate(Back),[Back])
            else: match = cls.recPaths(desc,viewCache,[At])
        if (not match and desc.dist != '0' and Options.PerspectiveTaking
            and desc.side[0] in (Front,Left,Right,At) and len(viewCache[Front])>1):
            if desc.side == [At] and desc.dist == cls.decrDist(desc.dist):
                descCp = desc
            else:
                descCp = copy.deepcopy(desc)
                descCp.side,descCp.dist = [At],cls.decrDist(desc.dist)
            match = cls.recIntersection(descCp,viewCache.project())
        return match # Prob
    Recognizers[Intersection] = recIntersection
    recIntersection = classmethod(recIntersection)

    def recPosition(cls,desc,viewCache):
        logger.debug('recPosition(%r, %r)',desc,viewCache)
        cls.checkDesc(desc,'recPosition')
        match = cls.recDetails(desc,viewCache)
        if not match: return match # Prob
        if not desc.dist: desc.dist = '0:'
        return len(viewCache[Front]) > int(desc.dist[0]) #minDist  !! Generalize?  # Prob
    Recognizers['Position'] = recPosition
    Recognizers[Position] = recPosition
    Recognizers[Block] = recPosition
    recPosition = classmethod(recPosition)

    def recSidePath(cls, desc,viewCache):
        """
        >>> R = PomdpAntieSimRecognizer
        >>> viewCache = ViewCache(obs=[(Honeycomb, Empty, Wall, Fish, Grass, Fish), (Cement, Empty, Wall, Fish, Grass, Fish), (Brick, Hatrack, Wall, Fish, Grass, Fish), (BlueTile, Empty, Wall, End, Wall, End)])
        >>> desc = Thing(dist='0:', Appear=[Brick], value=Path, side=[Left])
        >>> R.recSidePath(desc, viewCache)
        True
        >>> desc.side = [Right]
        >>> R.recSidePath(desc, viewCache)
        False
        >>> desc.side = [Front]
        >>> R.recSidePath(desc, viewCache)
        True
        >>> desc.side = [Sides]
        >>> R.recSidePath(desc, viewCache)
        True
        >>> R.recPath(Thing(dist = '0', Appear = [Flooring, Rose], value = Path, type = 'Path', side = [At]), ViewCache(obs=[(Rose, Empty, Wall, Eiffel, BlueTile, Eiffel), (Wall, Empty, Wall, Eiffel, BlueTile, Eiffel), (Cement, Chair, Cement, Fish, BlueTile, Fish), (Cement, Empty, Wall, Fish, BlueTile, Fish), (Honeycomb, Empty, Honeycomb, Fish, BlueTile, Fish), (Wall, Hatrack, Wall, End, Wall, End)]))
        True
        >>> R.recPath(Thing(dist='0:', Appear=[Brick], value=Path, type='Path', side=[Front], Structural=['Long']), ViewCache([[(Rose, Easel, Rose, Fish, Brick, Fish), (Grass, Hatrack, Grass, End, Wall, End)], [(Brick, Easel, Brick, Unknown, Unknown, Unknown)], [(Unknown, Easel, Rose, Unknown, Brick, Unknown)], [(Brick, Easel, Brick, Fish, Rose, Fish), (BlueTile, Empty, BlueTile, End, Wall, End)]]))
        False
        >>> R.recPath(Thing(dist='0:', Appear=[Brick, Flooring], value=Block, type='Region', side=[At]), ViewCache([[(Unknown, Empty, Wall, End, Wall, End)], [(Wall, Empty, Wall, End, Wall, End)], [(Wall, Empty, Unknown, End, Wall, End)], [(Wall, Empty, Wall, Unknown, Unknown, Unknown)]]))
        False
        >>> R.recPath(Thing(dist='0:', Appear=[Flooring, Brick], value=Block, type='Region', side=[At]), ViewCache([[(Wall, Empty, Wall, End, Wall, End)], [(Wall, Empty, Wall, Unknown, Unknown, Unknown)], [(Wall, Empty, Wall, End, Wall, End)], [(Wall, Empty, Wall, End, Wall, End)]]))
        False
        """
        #if not Options.CausalLandmarks: return True
        match = False
        disjunct = False
        cls.checkDesc(desc,'recSidePath')
        descCp = copy.deepcopy(desc)
        sides = desc.side[:]
        if not sides or sides in ([Sides],):
            sides = [Left,Right]
            disjunct = True
        elif sides in ([At],):
            sides = [Front,Left,Right,Back]
            disjunct = True
        for side in sides:
            descCp.side = [side]
            if side in (Left,Right,Front) and desc.side != [At] and Options.PerspectiveTaking:
                match = viewCache.lookToSide(descCp,side,cls.recPath)
            elif side == Back:
                descCp.side = [Front]
                match = cls.recPathAppear(viewCache[Back],descCp,viewCache[Back][Front])
            if not match and side in (Left,Right,Front):
                path = viewCache[Front][side]
                if path == Unknown: return False
                match = cls.recPathAppear(viewCache[Front],descCp,path)
            if match and disjunct: break
        #logger.debug('recSidePath(%r, %r, %r, => %r)',viewCache[Front],desc,viewCache, match)
        return match # Prob
    recSidePath = classmethod(recSidePath)

    def recPath(cls,desc,viewCache):
        """
        >>> R = PomdpAntieSimRecognizer
        >>> desc = Thing(Path_n=Path, value=Path, type='Path', side=[Front],dist='0:')
        >>> R.recPath(desc,ViewCache(obs=[(Wall, Empty, Hall, End, Wall, End)])) # Want Path Front, Have Wall Front
        False
        >>> viewCache = ViewCache(obs=[(Wall, Empty, Hall, Fish, Honeycomb, Fish),(Hall, Easel, Hall, End, Wall, End)])
        >>> R.recPath(desc, viewCache) # Want Path Front, Have Path Front
        True
        >>> desc.Appear = [Honeycomb]
        >>> R.recPath(desc, viewCache) # Want Honey Path Front, Have Honey Path Front
        True
        >>> desc.Appear = [BlueTile]
        >>> R.recPath(desc, viewCache) # Want Blue Path Front, Have Honey Path Front
        False
        >>> desc.Appear = [Brick] # Want Brick Path Front, Have Honey Path Front, Brick intersect in Front
        >>> R.recPath(desc, ViewCache(obs=[(Wall, Empty, BlueTile, Fish, Honeycomb, Fish),(Brick, Easel, Brick, End, Wall, End)]))
        True
        >>> R.recPath(Thing(dist='0:', Detail=[Thing(dist='0:', value=Barstool, type='Obj')], value=Path, type='Path', side=[Right]), ViewCache(obs=[(Wall, Barstool, Cement, End, Wall, End)])) # Test Detail Obj in intersection is also on side paths
        True
        >>> R.recPath(Thing(dist='0', Path_n=Path, value=PathDir, type='Pathdir'), viewCache)
        True
        >>> R.recPath(Thing(dist='0', Path_n=Path, value=PathDir, type='Pathdir'), ViewCache(obs=[(Wall, Barstool, Cement, End, Wall, End)]))
        False
        >>> viewCache = ViewCache(obs=[(Rose, Easel, Rose, Fish, Brick, Fish), (Grass, Hatrack, Grass, End, Wall, End)])
        >>> PomdpAntieSimRecognizer.recPath(Thing(dist='0', Appear=[Rose], value=Path, type='Region', side=[At]), viewCache)
        True
        >>> PomdpAntieSimRecognizer.recPath(Thing(dist='0', Appear=[Rose], value=Path, type='Region', side=[At], negate=True), viewCache)
        False
        >>> PomdpAntieSimRecognizer.recPath(Thing(dist='0', Appear=[Brick], value=Path, type='Region', side=[At]), viewCache)
        True
        >>> PomdpAntieSimRecognizer.recPath(Thing(dist='0', Appear=[Brick], value=Path, type='Region', side=[At], negate=True), viewCache)
        False
        >>> PomdpAntieSimRecognizer.recPath(Thing(dist='0', Appear=[BlueTile], value=Path, type='Region', side=[At], negate=True), viewCache)
        True
        >>> PomdpAntieSimRecognizer.recPath(Thing(dist='0', Appear=[BlueTile], value=Path, type='Region', side=[At]), viewCache)
        False
        """
        #logger.debug('recPath(%r, %r)',desc,viewCache)
        cls.checkDesc(desc,'recPath')
        match = cls.recDetails(desc,viewCache)
        if not Options.CausalLandmarks: return match
        if not match: return cls.rtn(match,desc)
        if desc.side == [Front]:
            match = cls.recFwdPath(desc,viewCache) # Prob
            if match: return cls.rtn(match,desc)
        if desc.side != [Front]:
            match = cls.recSidePath(desc,viewCache) # Prob
        return cls.rtn(match,desc)
    Recognizers['Path'] = recPath
    Recognizers[Path] = recPath
    Recognizers['Region'] = recPath
    Recognizers[Region] = recPath
    Recognizers['Pathdir'] = recPath
    Recognizers[PathDir] = recPath
    Recognizers['Segment'] = recPath
    Recognizers[Segment] = recPath
    recPath = classmethod(recPath)

    def recPaths(cls,desc,viewCache,sides):
        """
        >>> PomdpAntieSimRecognizer.recIntersection(Thing(dist='0', Part=[Thing(dist='0:', Detail=[Thing(dist='0:', Appear=[Rose], value=Path, type='Obj')], value=Path, type='Path')], value=Intersection, type='Struct', side=[At]), ViewCache([[(Rose, Empty, Rose, Fish, Cement, Fish), (Grass, Empty, Grass, End, Wall, End)], [(Cement, Empty, Wood, Unknown, Rose, Unknown)], [(Rose, Empty, Rose, Fish, Cement, Fish), (Wood, Empty, Wood, Unknown, Unknown, Unknown)], [(Cement, Empty, Wood, Unknown, Rose, Unknown)]]))
        True
        """
        if not Options.CausalLandmarks: return True
        match = False
        for path in cls.getPathDescs(desc):
            for side in sides:
                match = False
                if not path.side or len(sides) > 1:# or path.side == [path.value.ViewPosition]:
                    path.side = [side]
                elif side == Back:
                    for i,side in enumerate(path.side[:]):
                        path.side[i] = opposite(side)
                if hasattr(desc,'Structural'): path.Structural = desc.Structural
                match = cls.recPath(path,viewCache)
                #Handle case of "intersection of red and yellow path(s)"
                if not match and hasattr(path,'Appear') and len(path.Appear)>1:
                    paths = []
                    for appear in path.Appear[:]:
                        paths.append(copy.deepcopy(path))
                        paths[-1].Appear = [appear]
                    for path in paths:
                        match = cls.recPath(path,viewCache)
                        if not match: break
                if match: break # sides disjunctive
            if not match: break # paths conjunctive
        return match # Prob
    recPaths = classmethod(recPaths)

    def recEval(cls,desc,viewCache): return eval(desc.value)
    Recognizers['Boolean'] = recEval
    recEval = classmethod(recEval)

    @classmethod
    def recDefault(cls,desc,viewCache): return True

def _test(verbose=False):
    import doctest
    doctest.testmod(verbose=verbose)

if __name__ == '__main__':
    from Recognizers_POMDP_Antie_Sim import *
    from CompoundAction import *
    import sys
    sys.path.append('..')
    logger.initLogger('Recognizer',LogDir='../MarcoLogs')
    #from SmallScaleStar import *
