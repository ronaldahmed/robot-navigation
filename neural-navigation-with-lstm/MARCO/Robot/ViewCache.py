import copy
from Meanings import Back, Front, Left, Right, At, Wall, End, Unknown, opposite
try:
    from POMDP.MarkovLoc_Antie import observation
except:
    import sys
    sys.path.append('..')
    from POMDP.MarkovLoc_Antie import observation

class ViewCache:
    """
    ViewCache goes from left to right
    ViewCache[0] is always directly in front
    """
    def __init__(self,cache=None,obs=None):
        self._len = 4
        self.reset(obs,cache)
    
    ## Wrapper functions around the container
    def __repr__(self): return self.__class__.__name__+'('+repr(self._cache)+')'
    def __str__(self): return self.__class__.__name__+'('+str(self._cache)+')'
    def __contains__(self,item): return self._cache.__contains__(item)
    def __delitem__(self,item): return self._cache.__delitem__(item)
    directions = {Back: -2,
                  Left: -1,
                  Right: 1,
                  Front: 0}
    def __getitem__(self,index):
        if index in self.directions:
            return self._cache.__getitem__(self.directions[index])
        return self._cache.__getitem__(index)
    def __iter__(self): return self._cache.__iter__()
    def __len__(self): return self._cache.__len__()
    def __setitem__(self,item,value): return self._cache.__setitem__(item,value)
    def reset(self,obs=None,cache=None):
        if cache:
            self._cache = cache
            self._len = len(cache)
        elif obs:
            l,m,r,fl,p,fr = obs[0]
            if r == Wall: FL = FR = End
            else:  FL = FR = Unknown
            rightV = [(p,m,Unknown,FL,r,FR)]
            reverseV = [(r,m,l,Unknown,Unknown,Unknown)]
            if l == Wall: FL = FR = End
            else:  FL = FR = Unknown
            leftV = [(Unknown,m,p,FL,l,FR)]
            self._cache = [obs, rightV, reverseV, leftV]
        else:
            self._cache = [[tuple([Unknown]*6)]] * self._len
        self._cache = [observation(v) for v in self._cache]
    
    def update(self,direction,obs):
        obs = observation(obs)
        if direction == Left:
            # Rotate off right side, replace former leftmost
            self._cache = [obs] + self._cache[0:self._len-1]
        elif direction == Right:
            # Rotate off left side, replace former rightmost
            self._cache = [obs]+ self._cache[2:self._len] + [self._cache[0]]
        elif direction == Front:
            oldL,oldM,oldR,oldFL,oldP,oldFR = self._cache[0].view[0]
            l,m,r,fl,p,fr = obs[0].view[0]
            if r == Wall: FL = FR = End
            else: FL = FR = Unknown
            rightV = [(p,m,oldP,FL,r,FR)]
            backV = [(r,m,l,oldFR,oldP,oldFL)] + self._cache[-2].view[:]
            if l == Wall: FL = FR = End
            else: FL = FR = Unknown
            leftV = [(oldP,m,p,FL,l,FR)]
            self._cache = [obs, observation(rightV), observation(backV), observation(leftV)]
        elif direction == At:
            oldL,oldM,oldR,oldFL,oldP,oldFR = self._cache[0].view[0]
            l,m,r,fl,p,fr = obs[0].view[0]
            
            R_oldL,R_oldM,R_oldR,R_oldFL,R_oldP,R_oldFR = self._cache[1].view[0]
            if r == Wall:
                R_oldFL = R_oldFR = End
            rightV = [(p,m,R_oldP,R_oldFL,r,R_oldFR)] + self._cache[1].view[1:]
            
            B_oldL, B_oldM, B_oldR, B_oldFL, B_oldP, B_oldFR = self._cache[-2].view[0]
            backV = [(r,m,l,B_oldFR,B_oldP,B_oldFL)] + self._cache[-2].view[1:]
            
            L_oldL,L_oldM,L_oldR,L_oldFL,L_oldP,L_oldFR = self._cache[-1].view[0]
            if l == Wall:
                L_oldFL = L_oldFR = End
            leftV = [(p,m,L_oldP,L_oldFL,l,L_oldFR)] + self._cache[-1].view[1:]
            
            self._cache = [obs, observation(rightV), observation(backV), observation(leftV)]
        else: raise ValueError('Unknown turn direction %r', direction)
    
    def lookToSide(self,desc,side,recFn):
        try: tmpVC = self.rotate(side)
        except KeyError: raise ValueError('Unknown side %r', side)
        if tmpVC[0] == observation([tuple([Unknown]*6)]): return False
        tmpDesc = copy.deepcopy(desc)
        tmpDesc.side = [desc.value.ViewPosition]
        return recFn(tmpDesc,tmpVC)
    
    def rotate(self,direction):
        idx = self.directions[direction]
        return ViewCache(self._cache[idx:]+self._cache[:idx])
    
    def project(self,dist=1):
        vc = copy.deepcopy(self)
        vc.update(Front,vc[Front][dist:])
        return vc

class AmnesiacViewCache(ViewCache):
    def update(self,direction,obs):
        self.reset(obs)
