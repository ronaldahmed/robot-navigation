import random
from Meanings import Left, Right, Front, At
from Print import PrintRobot
from Utility import logger

class Robot_POMDP_Sim_Min(PrintRobot):
    name = 'Robot_POMDP_Sim_Min'
    def __init__(self,pomdp,mapname):
        self.pomdp=pomdp
        self.mapname = mapname
        self.completed = False
        self.path=None
        self.currentState=None
        self.userInterfaceMode="AutoNav"
        
    def __repr__(self):
        return (self.name+'('+
                ', '.join([`s` for s in [self.mapname]])
                +')')

    def setRoute(self,Start,Dest):
        PrintRobot.setRoute(self,Start,Dest)
        self.pomdp.setRoute(Start,Dest)
        self.teleport(Start)

    #update current state occording to action taken
    def updateState(self,action):
	if self.currentState:
	        (loc, direction) = self.currentState
        
        	if action == "TravelFwd":
	            try:
        	        (loc,direction) = self.pomdp.Gateways[(loc,direction)]
	            except KeyError:
        	        #print "Illegal move, traveled into WALL!"
                	# keep same loc & direction
	                pass
        	elif action == "TurnLeft":
	            direction -= 1
    
        	elif action == "TurnRight":
	            direction += 1

        	direction = direction % 4
        
	        self.currentState = (loc, direction)
        
    def teleport(self,pose):
        PrintRobot.teleport(self,pose)
        self.pomdp.observed = None
        self.pomdp.set(pose)

    def declareGoal(self):
        PrintRobot.declareGoal(self)
        cost,obs = self.pomdp.perform('DeclareGoal')
        self.completed = (cost < 0)
        return cost,obs

    def perform(self,action):
        if action == 'DeclareGoal': return self.declareGoal() #for self.completed
        self.updateState(action)
        return self.pomdp.perform(action)

    #flush the actionQ, punt to pomdp function
    def flushActions(self):
        return self.pomdp.flushActionQ()

