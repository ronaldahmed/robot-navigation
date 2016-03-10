import copy
from Utility import logger
from POMDP.MarkovLoc_Antie import observation
from Recognizers_Compound import *
from SmallScaleStar import *

class HsshRecognizer(CompoundRecognizer):
    """
    >>> sssViewCache = SmallScaleStarViewCache()
    >>> desc = Thing(dist='0', value=Corner, type='Struct', side=[Front])
    >>> sssViewCache.update(At,observation([(Cement, Sofa, Wall, Butterfly, BlueTile, Butterfly), (Cement, Hatrack, Wall, Butterfly, BlueTile, Butterfly), (Cement, Empty, Wall, End, Wall, End)]))
    >>> sssViewCache.update(Right,observation([(BlueTile, Hatrack, BlueTile, Butterfly, Cement, Butterfly), (Wall, Empty, Cement, End, Wall, End)]))
    >>> HsshRecognizer.recCorner(desc, sssViewCache)
    False
    >>> sssViewCache.update(Front,observation([(Wall, Empty, Cement, End, Wall, End)]))
    >>> HsshRecognizer.recCorner(desc, sssViewCache) # Immediate corner
    True
    >>> HsshRecognizer.recIntersection(Thing(dist='0', value=Intersection, type='Struct', side=[At], Struct_type=[Thing(Count=[4], dist='0:', value=PathDir, type='Pathdir')]), SmallScaleStarViewCache( *(SmallScaleStar( *([gateway_type( *(-1, 0, 0, 0.0, 'TwoWalled') ), gateway_type( *(0, 1, 0, 0.0, 'TwoWalled') ), gateway_type( *(1, 0, 0, 0.0, 'TwoWalled') ), gateway_type( *(0, -1, 0, 0.0, 'TwoWalled') )], [0, 1], [small_scale_star_tuple( *(0, 0, True) ), small_scale_star_tuple( *(1, 1, True) ), small_scale_star_tuple( *(2, 0, False) ), small_scale_star_tuple( *(3, 1, False) )], 1) ), {0: [], 1: [], 2: [], 3: []}) ))
    True
    >>> HsshRecognizer.recIntersection(Thing(dist='0', value=Intersection, type='Struct', side=[At], Struct_type=[Thing(Count=[4], dist='0:', value=PathDir, type='Pathdir')]), SmallScaleStarViewCache( *(SmallScaleStar( *([gateway_type( *(0, 1, 0, 0.0, 'TwoWalled') ), gateway_type( *(1, 0, 0, 0.0, 'TwoWalled') ), gateway_type( *(0, -1, 0, 0.0, 'TwoWalled') )], [0, 1], [small_scale_star_tuple( *(Wall, 0, True) ), small_scale_star_tuple( *(0, 1, True) ), small_scale_star_tuple( *(1, 0, False) ), small_scale_star_tuple( *(2, 1, False) )], 1) ), {0: [], 1: [], 2: []}) ))
    False
    """
    Recognizers = CompoundRecognizer.Recognizers.copy()
    ViewCache = SmallScaleStarViewCache
    
    def countPathFragments(cls,viewCache): return viewCache.local.countPathFragments()
    countPathFragments = classmethod(countPathFragments)
    
    def countGateways(cls,viewCache): return viewCache.local.countGateways()
    countGateways = classmethod(countGateways)
    
    def recPathAppear(cls,view,desc,path,ignore=[]): return path.match(Path)
    recPathAppear = classmethod(recPathAppear)
    
    def recFwdPath(cls,desc,viewCache): return viewCache.local.match(Path)
    recFwdPath = classmethod(recFwdPath)
    
    def recObj(cls,desc,viewCache):
        """
        >>> R = HsshRecognizer
        >>> desc = Thing(value=Wall, dist='0', Obj_n=Wall, type='Obj', side=[Back])
        >>> sssViewCache = SmallScaleStarViewCache(obs=observation([(BlueTile, Empty, BlueTile, End, Wall, End)]))
        >>> R.recObj(desc,sssViewCache) # Looking for Wall in Back : Wall is Front, ViewCache Too Short
        False
        >>> view = [(Wall, Empty, Cement, Butterfly, BlueTile, Butterfly), (Wall, Barstool, Stone, Butterfly, BlueTile, Butterfly), (Wall, Chair, Wood, Fish, BlueTile, Fish), (Wall, Empty, Rose, Fish, BlueTile, Fish), (Wall, Empty, Grass, End, Wall, End)]
        >>> sssViewCache.update(Left,view)
        >>> R.recObj(desc,sssViewCache) # Looking for Wall in Back : Wall is Left, ViewCache Too Short
        False
        >>> view = [(BlueTile, Empty, BlueTile, Butterfly, Cement, Butterfly), (Brick, Empty, Brick, Eiffel, Cement, Eiffel), (Cement, Empty, Cement, End, Wall, End)]
        >>> sssViewCache.update(Left,view)
        >>> R.recObj(desc,sssViewCache) # Looking for Wall in Back : Wall is Back
        True
        >>> desc.side = [Left]
        >>> R.recObj(desc,SmallScaleStarViewCache(obs=observation([(Wall, Empty, Cement, Butterfly, BlueTile, Butterfly), (Wall, Barstool, Stone, Butterfly, BlueTile, Butterfly)])))
        True
        >>> desc.side = [Right, Back]
        >>> R.recObj(desc, SmallScaleStarViewCache(obs=observation([(Cement, Empty, Wall, Butterfly, Cement, Butterfly), (BlueTile, Hatrack, BlueTile, End, Wall, End)])))
        False
        """
        logger.debug('recObj(%r, %r)',desc,viewCache)
        disjunct = False
        cls.checkDesc(desc,'recObj')
        match = cls.recDetails(desc,viewCache)
        if not match: return match
        for side in desc.side:
            match = False
            if desc.value == Wall and side in (Left,Right,Front,At,Back) and '0' in desc.dist:
                if side == At and desc.value == Wall: side = Front
                match = viewCache[Front].match(desc.value,side)
            if not match and not disjunct: return match #If still no match, conjunction is impossible
        return match
    Recognizers['Obj'] = recObj
    Recognizers['Thing'] = recObj
    recObj = classmethod(recObj)
    
    def recNeedTurn(cls,desc,viewCache): return False
    recNeedTurn = classmethod(recNeedTurn)

def _test(verbose=False):
    import doctest
    doctest.testmod(verbose=verbose)

if __name__ == '__main__':
    logger.initLogger('Recognizer')
    from ViewCache import ViewCache
