import random
from Meanings import Left, Right, Front, At
from Print import PrintRobot
from Utility import logger

class Robot_POMDP_Sim(PrintRobot):
    name = 'Robot_POMDP_Sim'
    def __init__(self,pomdp,mapname,recognizer,NLUQueue=None):
        self.pomdp=pomdp
        self.mapname = mapname
        self.recognizer = recognizer
        self.viewCache = self.recognizer.ViewCache()
        self.completed = False
        self.NLUQueue = NLUQueue
    
    def __repr__(self):
        return (self.name+'('+
                ', '.join([`s` for s in [self.mapname]])
                +')')

    def setRoute(self,Start,Dest):
        PrintRobot.setRoute(self,Start,Dest)
        self.pomdp.setRoute(Start,Dest)
        self.teleport(Start)

    def teleport(self,pose):
        PrintRobot.teleport(self,pose)
        self.pomdp.observed = None
        self.pomdp.set(pose)
        self.viewCache.reset()

    Directions = {Left : 'TurnLeft', Right : 'TurnRight'}
    def turn(self,direction):
        if direction not in self.Directions: direction = random.choice(self.Directions)
        self.pomdp.observed = None
        PrintRobot.turn(self,direction)
        cost,obs = self.pomdp.perform(self.Directions[direction])
        self.viewCache.update(direction,obs)
        return cost,obs

    def travel(self):
        self.pomdp.observed = None
        PrintRobot.travel(self)
        if not self.pomdp.observed:
            logger.info('travel observing...')
            self.pomdp.observe()
        self.viewCache.update(Front,self.pomdp.observed)
        logger.info("post travel obs: %r; viewCache: %r", self.pomdp.observed, self.viewCache)
        return self.pomdp.perform('TravelFwd')

    def declareGoal(self):
        PrintRobot.declareGoal(self)
        cost,obs = self.pomdp.perform('DeclareGoal')
        self.completed = (cost < 0)
        return cost,obs

    def getLookTurn(self,desc):
        turnDir = None
        descSide = desc.side
        if self.recognizer.recNeedTurn(desc,self.viewCache):
            if self.recognizer.recObj(Struct(value=Wall,side=Left), self.viewCache):
                turnDir = Right
            else: turnDir = Left
            self.turn(turnDir)
            desc.side = Front
        return turnDir,descSide

    def turnBack(self,turnDir,descSide):
        if turnDir:
            if turnDir == Left: self.turn(Right)
            else: self.turn(Left)
            desc.side = descSide
            
    def perform(self,action):
        if action == 'DeclareGoal': return self.declareGoal() #for self.completed
        return self.pomdp.perform(action)
        
    def recognize(self,description):
        PrintRobot.recognize(self,description)
        if not self.pomdp.observed:
            logger.info('recognize observing...')
            self.pomdp.observe()
            logger.info('Observed %r',self.pomdp.observed)
        self.viewCache.update(At,self.pomdp.observed)
        for desc in description:
            logger.info('.....Examining %s\n.....for %r', self.pomdp.observed, desc)
            if self.NLUQueue: self.NLUQueue.put(('Recognize', repr(desc)))
            turnDir,descSide = self.getLookTurn(desc)
            match = self.recognizer.recDesc(desc, self.viewCache)
            self.turnBack(turnDir,descSide)
            if not match:
                logger.info('.....Did NOT recognize %r',desc)
                return False
            logger.info('.....Recognized %r',desc)
        logger.info('.....Recognized entire description!')
        return True

