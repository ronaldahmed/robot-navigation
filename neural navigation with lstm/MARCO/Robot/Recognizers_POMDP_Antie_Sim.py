import copy,sys
from Recognizers_Compound import *
from Utility import logger
from ViewCache import ViewCache

class PomdpAntieSimRecognizer(CompoundRecognizer):
    Recognizers = CompoundRecognizer.Recognizers.copy()
    ViewCache = ViewCache
    
    def recImmediatePicture(cls,desc,viewCache,side):
        match = False
        picture = desc.value
        if hasattr(desc,'Detail') and desc.Detail:
            if isinstance(desc.Detail,list): picture = desc.Detail[0]
            else: picture = desc.Detail
        if isinstance(picture,Thing): picture = picture.value
        if side in (FrontLeft,FrontRight,At,Front):
            match = viewCache[Front].match(picture,picture.ViewPosition)
            if not match: side = Front
        elif side in (Back,Left,Right):
            match = viewCache[side].match(picture,picture.ViewPosition)
        else:
            logger.error('recImmediatePicture(%r): Unknown side %r',desc,side)
        return match,side
    recImmediatePicture = classmethod(recImmediatePicture)

    def recPathAppear(cls,view,desc,path,ignore=[]):
        """
        >>> PomdpAntieSimRecognizer.recPath(Thing(dist='0', Appear=[Fish], value=Path, type='Region', side=[Front]), ViewCache(obs=[(Rose, Empty, Rose, Eiffel, Cement, Eiffel), (Wall, Empty, Wall, End, Wall, End)]))
        False
        >>> PomdpAntieSimRecognizer.recPath(Thing(dist='0', Appear=[Flooring, Thing(value=Rose, type='Obj')], value=Path, type='Path', side=[At]),ViewCache(obs=[(Brick,Hatrack,Brick, Eiffel,Rose,Eiffel), (BlueTile,Empty,BlueTile,End,Wall,End),]))
        True
        >>> PomdpAntieSimRecognizer.recPath(Thing(dist='0', Appear=[Flooring, Thing(value=Rose, type='Obj')], value=Path, type='Path', side=[At]),ViewCache(obs=[(Brick,Hatrack,Brick, Eiffel,Honeycomb,Eiffel), (BlueTile,Empty,BlueTile,End,Wall,End),]))
        False
        >>> PomdpAntieSimRecognizer.recPath(Thing(dist='0', Detail=[Thing(Appear=[Honeycomb, Stone], value=Path, type='Obj')], value=Path, type='Path', side=[Front]), ViewCache([(Stone, Easel, Stone, Eiffel, Honeycomb, Eiffel), (Wood, Empty, Wall, Fish, Honeycomb, Fish),  (Grass, Empty, Wall, End, Wall, End)]))
        True
        >>> PomdpAntieSimRecognizer.recPath(Thing(dist='0', Appear=[Flooring, Thing(value=Rose, type='Obj')], value=Path, type='Path', side=[At]),ViewCache(obs=[(Grass, Chair, Wall, Eiffel, BlueTile, Eiffel), (Rose, Empty, Wall, Eiffel, BlueTile, Eiffel), (Wall, Empty, Wall, Eiffel, BlueTile, Eiffel), (Cement, Sofa, Wall, End, Wall, End)]))
        False
        >>> PomdpAntieSimRecognizer.recPath(Thing(dist='0', Appear=[Flooring, Thing(value=Rose, type='Obj')], value=Path, type='Path', side=[At]),ViewCache(obs=[(Rose, Empty, Wall, Eiffel, BlueTile, Eiffel), (Wall, Empty, Wall, Eiffel, BlueTile, Eiffel), (Cement, Sofa, Wall, End, Wall, End)]))
        True
        """
        match = False
        if (hasattr(desc,'Appear') and 'Appear' not in ignore
            and isinstance(desc.Appear,list) and desc.Appear) and Options.AppearanceLandmarks:
            for attribDesc in desc.Appear:
                if not isinstance(attribDesc,Meaning) and not isinstance(attribDesc,Thing):
                    attribDesc = Path
                if isinstance(attribDesc, Texture):
                    if len(desc.Appear) > 1 and attribDesc == Stone:
                        continue # Handles "yellow stone" etc
                    match = path.match(attribDesc)
                elif desc.side == [Front] and attribDesc.name in Picture.Names:
                    match = view.match(attribDesc,attribDesc.ViewPosition)
                elif attribDesc == Path and Options.CausalLandmarks:
                    match = path.match(attribDesc) or path.match(Flooring)
                elif isinstance(attribDesc, Thing):
                    if isinstance(attribDesc.value, Texture):
                        match = path.match(attribDesc.value)
                    else:
                        match = cls.Recognizers[attribDesc.type](cls,attribDesc,ViewCache(obs=view.view))
                #else: logger.warning('Unknown Appear %r', attribDesc)
                # Look for path in the distance if we don't need to see the structure.
                if (not match and desc.dist != '0' and not (hasattr(desc,'Structural') and desc.Structural)
                    and Options.PerspectiveTaking):
                    if desc.side == [Front]: sides = [Sides]
                    else: sides =  desc.side
                    dist = cls.decrDist(desc.dist)
                    for side in sides:
                        if side == At: side = Sides
                        match = view[1:].search(attribDesc,side,dist)
                        if match: break
                if not match: break
        elif Options.CausalLandmarks:
            attribDesc = Path
            match = path.match(attribDesc) or path.match(Flooring)
        else: return True
        #logger.debug('recPathAppear(%r, %r, %r, %r) => %r',view,desc,path,ignore, match)
        return match # Prob
    recPathAppear = classmethod(recPathAppear)

    def recFwdPath(cls,desc,viewCache,ignore=[]):
        """
        >>> R = PomdpAntieSimRecognizer
        >>> R.recFwdPath(Thing(dist='0:', Appear=['OPEN'], value=Path, type='Path'), ViewCache(obs=[(Wall, Empty, Cement, End, Wall, End)]))
        False
        >>> R.recFwdPath(Thing(dist='0:', Appear=['OPEN'], value=Path, type='Path'), ViewCache(obs=[(Wall, Empty, Wall, Fish, Cement, Fish),(Wall, Empty, Cement, End, Wall, End)]))
        True
        >>> R.recFwdPath(Thing(dist='0:', Appear=[Brick], value=Path, type='Path', side=[Front], Structural=[Sides]), ViewCache(obs=[(Wall, Empty, Honeycomb, Fish, Rose, Fish), (Wall, Empty, Wall, Fish, Rose, Fish), (Brick, Hatrack, Brick, Eiffel, Rose, Eiffel), (BlueTile, Empty, BlueTile, End, Wall, End)]))
        True
        
        Test Longer end of the hallway.
        >>> desc = Thing(dist='0', Part=[Thing(Appear=[Brick], value=Path, type='Path')], value=End, type='Struct', side=[Front], Structural=['Long'])
        >>> viewCache = ViewCache(obs=[(Brick, Empty, Brick, End, Wall, End)])
        >>> R.recFwdPath(desc, viewCache)
        False
        >>> viewCache.update(Right, [(Wall, Empty, Cement, Butterfly, Brick, Butterfly), (Wall, Empty, Wall, End, Wall, End)])
        >>> R.recFwdPath(desc, viewCache)
        False
        >>> viewCache.update(Right, [(Brick, Empty, Brick, Butterfly, Cement, Butterfly), (Wall, Empty, Wall, End, Wall, End)])
        >>> R.recFwdPath(desc, viewCache)
        False
        >>> viewCache.update(Right, [(Wall, Empty, Cement, Butterfly, Brick, Butterfly), (Wood, Empty, Wood, Butterfly, Brick, Butterfly), (Wall, Empty, Wall, End, Wall, End)])
        >>> R.recFwdPath(desc, viewCache)
        True
        """
        #logger.debug('recFwdPath(%r, %r, %r, %r)',view,desc,viewCache,ignore)
        view = viewCache[Front]
        match = cls.recPathAppear(view,desc,view[Front],ignore)
        if (match and #desc.dist != '0' and 
            hasattr(desc,'Structural') and 'Structural' not in ignore and desc.Structural
            and True in [s in Structurals.values() for s in desc.Structural]):
            if not Options.PerspectiveTaking: return match
            length = len(view)
            match = False
            for side in (Back,): #(Left,Back,Right):
                v = viewCache[side]
                if not v or Unknown in v[0].view[0] or len(v)==1: continue
                sides = desc.side
                desc.side = [Sides]
                if cls.recFwdPath(desc,viewCache.rotate(side),ignore+['Structural']):
                    if desc.Structural == [Short]:
                        match = (view == v or length < len(v))
                    elif desc.Structural == [Long]:
                        match = (length > len(v))
                    elif desc.Structural == [Winding]:
                        match = True
                    else: logger.warning('Unknown Structural %r', desc.Structural)
                desc.side = sides
                if match: break
        if (not match and Options.IntersectionLandmarks
            and hasattr(desc,'Structural') and 'Structural' not in ignore and Sides in desc.Structural):
            descCp = copy.deepcopy(desc)
            del descCp.Structural
            descCp.side=[Sides]
            match = cls.recSidePath(descCp,viewCache)
        return match # Prob
    recFwdPath = classmethod(recFwdPath)

    def recObj(cls,desc,viewCache):
        """
        >>> R = PomdpAntieSimRecognizer
        >>> desc = Thing(value=Wall, dist='0:', Obj_n=Wall, type='Obj', side=[Back])
        >>> viewCache = ViewCache(obs=[(BlueTile, Empty, BlueTile, End, Wall, End)])
        >>> R.recObj(desc,viewCache) # Looking for Wall in Back : Wall is Front, ViewCache Too Short
        False
        >>> viewCache.update(Left,[(Wall, Empty, Cement, Butterfly, BlueTile, Butterfly), (Wall, Barstool, Stone, Butterfly, BlueTile, Butterfly), (Wall, Chair, Wood, Fish, BlueTile, Fish), (Wall, Empty, Rose, Fish, BlueTile, Fish), (Wall, Empty, Grass, End, Wall, End)])
        >>> R.recObj(desc,viewCache) # Looking for Wall in Back : Wall is Left, ViewCache Too Short
        False
        >>> viewCache.update(Left,[(BlueTile, Empty, BlueTile, Butterfly, Cement, Butterfly), (Brick, Empty, Brick, Eiffel, Cement, Eiffel), (Cement, Empty, Cement, End, Wall, End)])
        >>> R.recObj(desc,viewCache) # Looking for Wall in Back : Wall is Back
        True
        >>> desc.side = [Left]
        >>> R.recObj(desc,ViewCache(obs=[(Wall, Empty, Cement, Butterfly, BlueTile, Butterfly), (Wall, Barstool, Stone, Butterfly, BlueTile, Butterfly)]))
        True
        >>> desc.side = [Right, Back]
        >>> R.recObj(desc, ViewCache([[(Cement, Empty, Wall, Butterfly, Cement, Butterfly), (BlueTile, Hatrack, BlueTile, End, Wall, End)]]))
        False
        >>> R.recObj(Thing(dist='0:', value=Sofa, type='Obj', side=[Right]), ViewCache([[(Stone, Barstool, Stone, End, Wall, End)], [(Wall, Barstool, Cement, Eiffel, Stone, Eiffel), (Cement, Sofa, Wall, End, Wall, End)], [(Stone, Barstool, Stone, Eiffel, Cement, Eiffel), (BlueTile, Chair, BlueTile, Fish, Cement, Fish), (Wall, Empty, Cement, End, Wall, End)], [(Cement, Barstool, Wall, Fish, Stone, Fish), (Wall, Empty, Wall, Fish, Stone, Fish), (Honeycomb, Empty, Wall, End, Wall, End)]]))
        True
        >>> R.recObj(Thing(dist='1:', value=Chair, type='Obj', side=[Front]),ViewCache(obs=[(Wall, Empty, Cement, Fish, BlueTile, Fish), (Cement, Chair, Cement, Eiffel, BlueTile, Eiffel)]))
        True
        >>> R.recObj(Thing(dist='0', value=Wall, type='Obj', side=[Back]), ViewCache([[(Cement, Empty, Wall, Butterfly, BlueTile, Butterfly), (Cement, Sofa, Wall, Butterfly, BlueTile, Butterfly), (Cement, Hatrack, Wall, Butterfly, BlueTile, Butterfly), (Cement, Empty, Wall, End, Wall, End)], [(BlueTile, Empty, BlueTile, End, Wall, End)], [(Wall, Empty, Unknown, Unknown, BlueTile, Unknown)], [(BlueTile, Empty, BlueTile, Unknown, Unknown, Unknown)]]))
        False
        >>> R.recObj(Thing(dist = '1:', value = Sofa, type = 'Obj', side = [At]), ViewCache(obs=[(Cement, Chair, Cement, Fish, BlueTile, Fish), (Cement, Sofa, Wall, Fish, BlueTile, Fish), (Honeycomb, Empty, Honeycomb, Fish, BlueTile, Fish), (Wall, Hatrack, Wall, End, Wall, End)]))
        True
        >>> R.recObj(Thing(dist = '1:', value = GenChair, type = 'Obj', side = [At],), ViewCache(obs=[(Cement, Chair, Cement, Fish, BlueTile, Fish), (Cement, Sofa, Wall, Fish, BlueTile, Fish), (Honeycomb, Empty, Honeycomb, Fish, BlueTile, Fish), (Wall, Hatrack, Wall, End, Wall, End)]))
        True
        >>> R.recObj(Thing(dist = '1:', value = Chair, type = 'Obj', side = [At],), ViewCache(obs=[(Cement, Chair, Cement, Fish, BlueTile, Fish), (Cement, Sofa, Wall, Fish, BlueTile, Fish), (Honeycomb, Empty, Honeycomb, Fish, BlueTile, Fish), (Wall, Hatrack, Wall, End, Wall, End)]))
        False
        >>> R.recObj(Thing(dist='0', type='Struct', side=[Back], value=Intersection), ViewCache([[(BlueTile, Empty, BlueTile, Butterfly, Cement, Butterfly), (Brick, Empty, Brick, Eiffel, Cement, Eiffel), (Cement, Empty, Cement, End, Wall, End)], [(Cement, Empty, Wall, Unknown, Unknown, Unknown)], [(Unknown, Empty, BlueTile, End, Wall, End)], [(Wall, Empty, Cement, Butterfly, BlueTile, Butterfly), (Wall, Barstool, Stone, Butterfly, BlueTile, Butterfly), (Wall, Chair, Wood, Fish, BlueTile, Fish), (Wall, Empty, Rose, Fish, BlueTile, Fish), (Wall, Empty, Grass, End, Wall, End)]]))
        True
        >>> R.recObj(Thing(dist='1:', Appear=[Flooring, BlueTile], value=Flooring, type='Path', side=[Front]), ViewCache(obs=[(Brick, Empty, Brick, Eiffel, Cement, Eiffel), (Cement, Empty, Cement, End, Wall, End)]))
        False
        >>> R.recObj(Thing(dist='1:', Appear=[Flooring, BlueTile], value=Flooring, type='Path', side=[Front]), ViewCache(obs=[(Brick, Empty, Brick, Eiffel, Cement, Eiffel), (BlueTile, Empty, BlueTile, End, Wall, End)]))
        True
        >>> R.recObj(Thing(On=Thing(dist='0:', type='Obj', side=[Front], value=Wall), dist='0:', value=Butterfly, type='Obj'), ViewCache(obs=[(Stone, Empty, Stone, Eiffel, Brick, Eiffel), (Grass, Empty, Grass, Eiffel, Brick, Eiffel), (Rose, Hatrack, Rose, Eiffel, Brick, Eiffel), (Wall, Empty, Wall, Eiffel, Brick, Eiffel), (Wall, Empty, Wall, End, Wall, End)]))
        False
        >>> R.recObj(Thing(On=Thing(dist='0:', type='Obj', side=[Front], value=Wall), dist='0:', value=Butterfly, type='Obj'), ViewCache(obs=[(Stone, Empty, Stone, Eiffel, Brick, Eiffel), (Grass, Empty, Grass, Eiffel, Brick, Eiffel), (Rose, Hatrack, Rose, Eiffel, Brick, Eiffel), (Wall, Empty, Wall, Eiffel, Brick, Eiffel), (Wall, Lamp, Cement, Butterfly, Brick, Butterfly), (Wall, Empty, Cement, Butterfly, Brick, Butterfly), (Wood, Empty, Wood, Butterfly, Brick, Butterfly), (Wall, Empty, Wall, End, Wall, End)]))
        True
        >>> R.recObj(Thing(dist='0:', value=Sofa, type='Obj', side=[Back]), ViewCache(obs=[(Wall, Empty, Cement, Butterfly, BlueTile, Butterfly), (Wall, Barstool, Stone, Butterfly, BlueTile, Butterfly), (Wall, Chair, Wood, Fish, BlueTile, Fish), (Wall, Empty, Rose, Fish, BlueTile, Fish), (Wall, Empty, Grass, End, Wall, End)]))
        False
        >>> R.recObj(Thing(dist='0', type='Obj', side=[At], value=Hatrack), ViewCache(obs=[(Brick,Hatrack,Brick, Eiffel,Rose,Eiffel), (BlueTile,Empty,BlueTile,End,Wall,End),]))
        True
        >>> R.recObj(Thing(dist='0', type='Obj', side=[Front], value=Hatrack), ViewCache(obs=[(Brick,Hatrack,Brick, Eiffel,Rose,Eiffel), (BlueTile,Empty,BlueTile,End,Wall,End),]))
        True
        >>> PomdpAntieSimRecognizer.recObj(Thing(On=[Thing(dist='0:', Appear=[Stone], value=Path, type='Path')], dist='0:', value=Easel, side=[Left], type='Obj'), ViewCache([[(Stone, Empty, Stone, Fish, Honeycomb, Fish), (Grass, Empty, Grass, Fish, Honeycomb, Fish), (Rose, Empty, Rose, End, Wall, End)], [(Honeycomb, Empty, Stone, Fish, Stone, Fish), (Cement, Empty, Cement, Fish, Stone, Fish), (Brick, Empty, Brick, Eiffel, Stone, Eiffel), (BlueTile, Lamp, BlueTile, End, Wall, End)], [(Stone, Empty, Stone, Fish, Honeycomb, Fish), (Wall, Empty, Cement, End, Wall, End)], [(Honeycomb, Empty, Stone, Fish, Stone, Fish), (Wall, Easel, Wall, End, Wall, End)]]))
        True
        >>> R.recObj(Thing(dist='0', Detail=[Thing(value=Eiffel, type='Obj')], value=Wall, type='Obj', side=[Front]), ViewCache([(Brick, Empty, Brick, Eiffel, Cement, Eiffel), (Cement, Empty, Cement, End, Wall, End)]))
        True
        >>> R.recObj(Thing(dist='0', Detail=[Thing(Appear=[Brick], value=Path, side=[Sides], type='Obj')], value=Wall, type='Obj', side=[Front]),ViewCache(obs=[(Brick, Lamp, Brick, End, Wall, End)]))
        True
        """
        #logger.debug('recObj(%r, %r)',desc,viewCache)
        disjunct = False
        cls.checkDesc(desc,'recObj')
        match = cls.recDetails(desc,viewCache)
        if not match and not (hasattr(desc,'negate') and desc.negate): return match # Prob
        if desc.side == [Sides]:
            if isinstance(desc.value,Picture): desc.side = [FrontRight]
            else: desc.side =[Left,Right]
            disjunct = True
        elif not desc.side:
            desc.side = [desc.value.ViewPosition]
        elif hasattr(desc.side[0],'value') and desc.side[0].value == Wall: #Handle on the walls (Obsolete?)
            desc.side = [FrontLeft,FrontRight]
            disjunct = True
        view = viewCache[Front]
        for side in desc.side:
            match = False
            if (desc.value == Path):
                match = cls.recPath(desc,viewCache)
            elif isinstance(desc.value, Texture) or hasattr(desc,'Structural'):
                path = copy.deepcopy(desc)
                path.value,path.type = Path,'Path'
                if hasattr(path,'Appear'): path.Appear.append(desc.value)
                else: path.Appear = [desc.value]
                match = cls.recPath(path,viewCache)
            elif ((isinstance(desc.value, Picture) or desc.value == Picture
                   and ('0' in desc.dist or '1' in desc.dist))
                  or (desc.value == Wall and hasattr(desc,'Detail')
                      and desc.Detail and isinstance(desc.Detail[0].value,Picture))):
                if not Options.ObjectLandmarks: return True
                match,side = cls.recImmediatePicture(desc,viewCache,side)
            elif ((isinstance(desc.value, Object) or desc.value == Wall)
                  and side in (Left,Right,Front,At) and '0' in desc.dist):
                if not Options.ObjectLandmarks and desc.value != Wall: return True
                if not Options.CausalLandmarks and desc.value == Wall: return True
                if side == At and desc.value == Wall: side = Front
                if side == Front and desc.value != Wall: side = At
                match = view.match(desc.value,side)
            elif isinstance(desc.value,Structure) and not desc.value == Wall:
                match = cls.recStruct(desc,viewCache)
            
            if not match and Options.PerspectiveTaking:
                if side in (At, Front, FrontRight, Sides) and desc.dist != '0' and len(view)>1:
                    if desc.side == [desc.value.ViewPosition] and desc.dist == cls.decrDist(desc.dist):
                        descCp = desc
                    else:
                        descCp = copy.deepcopy(desc)
                        descCp.side, descCp.dist = [descCp.value.ViewPosition],cls.decrDist(desc.dist)
                    match = cls.recObj(descCp,viewCache.project())
                elif len(viewCache)>1 and side in (Left,Right,Back):
                    match = viewCache.lookToSide(desc,side,cls.recObj)
                elif not side in (Left,Right,Back,At,Front,FrontLeft,FrontRight,Sides):
                    print ValueError("recObj: Can't recognize position", side,'for', desc)
                    match = viewCache.lookToSide(desc,Front,cls.recObj)
                if not match and not disjunct:
                    return cls.rtn(match,desc) #If still no match, conjunction is impossible # Prob
        return cls.rtn(match,desc) # Prob
    Recognizers['Obj'] = recObj
    Recognizers['Thing'] = recObj
    recObj = classmethod(recObj)

    def recSide(cls,desc,viewCache):
        match = True
        return match
    Recognizers['Side'] = recSide
    Recognizers[Side] = recSide
    recSide = classmethod(recSide)

    def countGateways(cls,viewCache,back=0):
        view = viewCache[Front]
        return (sum((view.match(Flooring,Left),
                     view.match(Flooring,Right),
                     view.match(Flooring,Front),
                     viewCache[Back].match(Flooring,Front)))
                +back)
    countGateways = classmethod(countGateways)

    def countPathFragments(cls,viewCache,back=0):
        view = viewCache[Front]
        return (sum(((view.match(Flooring,Left) or view.match(Flooring,Right)),
                     (view.match(Flooring,Front) or viewCache[Back].match(Flooring,Front))))
                +back)
    countPathFragments = classmethod(countPathFragments)

    def recNeedTurn(cls,desc,viewCache):
        return (len(viewCache) < 4 and
                (desc.type == Path and desc.side in ([Sides],[Right],[Left]))
                )
    recNeedTurn = classmethod(recNeedTurn)

def _test(verbose=False):
    import doctest, Recognizers_Compound, CompoundAction, ViewCache
    doctest.testmod(verbose=verbose)
    d= Recognizers_Compound.__dict__.copy()
    d['PomdpAntieSimRecognizer'] = PomdpAntieSimRecognizer
    d['ViewCache'] = ViewCache.ViewCache
    d['logger'] = logger
    d.update(CompoundAction.__dict__)
    doctest.testmod(Recognizers_Compound, globs=d, verbose=verbose)

def _profile():
    import hotshot, hotshot.stats
    prof = hotshot.Profile("recognizer.prof")
    prof.run('_test()')
    prof.close()
    stats = hotshot.stats.load("recognizer.prof")
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats(40)

if __name__ == '__main__':
    logger.initLogger('Recognizer',LogDir='../MarcoLogs')
    from CompoundAction import *

