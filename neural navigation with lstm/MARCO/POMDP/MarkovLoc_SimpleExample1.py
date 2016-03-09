#!/usr/bin/env python
from MarkovLoc import TurnLeft,TurnRight,TravelFwd,DeclareGoal
import MarkovLoc_Simple

pomdp = MarkovLoc_Simple.POMDP('MarkovLoc_SimpleExample1')

# (Place,Pose)
pomdp.NumPlaces = 10
pomdp.NumPoses = 4
pomdp.StartPlace,pomdp.StartPose = (0,None)
Dest = (8,None)

pomdp.Gateways = {
    (0,1) : (1,1),
    (0,2) : (3,2),
    (2,1) : (3,1),
    (3,2) : (4,2),
    (4,1) : (5,1),
    (4,2) : (7,2),
    (6,1) : (7,1),
    (7,1) : (8,1),
    (7,2) : (9,2),
    }
pomdp.invertGateways()

pomdp.PeripheralViews = ['H','W']
pomdp.ForwardViews = ['W','Y','G','B'] # Yellow, Green, Blue

pomdp.ViewsFwd = {
    (0,1) : 'G', (1,3) : 'G',
    (0,2) : 'Y', (3,0) : 'Y',
    (2,1) : 'B', (3,3) : 'B',
    (3,2) : 'Y', (4,0) : 'Y',
    (4,1) : 'G', (5,3) : 'G',
    (4,2) : 'Y', (7,0) : 'Y',
    (6,1) : 'B', (7,3) : 'B',
    (7,1) : 'B', (8,3) : 'B',
    (7,2) : 'Y', (9,0) : 'Y',
    }

pomdp.Actions = {
    'TurnLeft' : TurnLeft(pomdp.NumPoses,1),
    'TurnRight' : TurnRight(pomdp.NumPoses,1),
    'TravelFwd' : TravelFwd(pomdp.Gateways,1),
    'DeclareGoal' : DeclareGoal(Dest,(-100,100)),
    }

pomdp.writefile()
