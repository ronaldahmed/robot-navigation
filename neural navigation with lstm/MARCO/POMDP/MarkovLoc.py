#!/usr/bin/env python
import sys
import Base

## === Actions === ##
class TurnLeft(Base.ConsistentCostAction):
    def __init__(self,n,cost):
        self.NumPoses = n
        self.cost = cost
    def act(self,(place,pose)):
        return [((place, (pose - 1) % self.NumPoses), self.cost, 1.0),]

class TurnRight(Base.ConsistentCostAction):
    def __init__(self,n,cost):
        self.NumPoses = n
        self.cost = cost
    def act(self,(place,pose)):
        return [((place, (pose + 1) % self.NumPoses), self.cost, 1.0),]

class TravelFwd(Base.Action):
    def __init__(self,Gateways,Costs):
        self.Gateways = Gateways
        self.HallTravel, self.WallTravel = Costs
    def act(self,(place,pose)):
        if (place,pose) in self.Gateways:
            result = self.Gateways[(place,pose)]
            if type(result) == list:
                prob = 1.0/len(result)
                return [(r,prob) for r in result]
            return [(result, self.HallTravel, 1.0),]
        else:
            return [((place, pose), self.WallTravel, 1.0),]

class DeclareGoal(Base.Action):
    def __init__(self,Dest,Costs):
        self.DestinationPlace,self.DestinationPose = Dest
        self.CorrectDestination,self.IncorrectDestination = Costs
    def act(self,(place,pose)):
        if (place == self.DestinationPlace
            and (not self.DestinationPose or pose == self.DestinationPose)):
            return [((place,pose),self.CorrectDestination,1.0),]
        else:
            return [((place,pose),self.IncorrectDestination,1.0),]
        return [((place, pose), 1.0),]

class POMDP(Base.POMDP):
    def __init__(self,name):
        Base.POMDP.__init__(self,name)
    
    def generateStates(self):
        for place in xrange(self.NumPlaces):
            for pose in xrange(self.NumPoses):
                self.States[(place,pose)] = 'Pl_%(place)i_%(pose)i' % locals()
                yield self.States[(place,pose)]
    
    def generateStart(self):
        for (place,pose) in self.States:
            if (place == self.StartPlace):
                if not self.StartPose: yield '%1.3f' % (1.0/self.NumPoses)
                elif pose == self.StartPose: yield '1.0'
                else: yield '0.0'
            else: yield '0.0'

    def oppositeGW(self,place,gw): return (gw+self.NumPoses/2)%self.NumPoses

    def invertGateways(self):
        for (plFrom,gwFrom),endstates in self.Gateways.items():
            if type(endstates) is tuple: endstates = [endstates]
            for (plTo,gwTo) in endstates:
                gw = self.oppositeGW(plTo,gwTo)
                self.Gateways[(plTo,gw)] = (plFrom,gw)
