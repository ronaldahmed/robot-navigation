#!/usr/bin/env python
import InferredMap
from MarkovLoc import TurnLeft,TurnRight,TravelFwd,DeclareGoal

ViewDescs = {
    (0,0) : '.,W,.',
    (0,1) : 'W,.,H',
    (0,2) : '.,Y,.',
    (0,3) : 'H,.,W',
    
    (1,0) : '.,Y,.',
    (1,1) : 'H,(?!B).,H',
    (1,2) : '.,Y,.',
    (1,3) : 'H,.,H',
    
    (2,0) : '.,Y,H',
    (2,1) : 'H,B,.',
    (2,2) : 'H,(Y|W),.',
    (2,3) : '.,(B|W),H',
    
    (3,0) : 'H,.,H',
    (3,1) : '.,B,.',
    (3,2) : 'H,.,H',
    (3,3) : '.,B,.',
    
    (4,0) : 'H,.,W',
    (4,1) : '.,W,.',
    (4,2) : 'W,.,H',
    (4,3) : '.,B,.',
    }

pomdp = InferredMap.POMDP('InferredMap_SimpleExample1',ViewDescs)
pomdp.NumPlaces=5
pomdp.NumPoses=4

pomdp.StartPlace = 0
pomdp.StartPose = None

Dest = (4, None)

pomdp.Gateways = {
    (0,2) : [(1,2),(2,2)],
    (1,2) : [(1,2),(2,2)],
    (2,1) : [(3,1),(4,1)],
    (3,1) : [(3,1),(4,1)],
    }
pomdp.invertGateways()

Colors = ['Y','G','B'] # Yellow, Green, Blue
pomdp.PeripheralViews = ['H','W']
pomdp.ForwardViews = ['W']+Colors

pomdp.Actions = {
    'TurnLeft' : TurnLeft(pomdp.NumPoses,1),
    'TurnRight' : TurnRight(pomdp.NumPoses,1),
    'TravelFwd' : TravelFwd(pomdp.Gateways,1),
    'DeclareGoal' : DeclareGoal(Dest,(100,-100)),
    }

pomdp.writefile()
