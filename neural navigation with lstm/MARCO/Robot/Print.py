import random
from Utility import logger
from Robot import RobotI

class PrintRobot(RobotI):
    def setRoute(self,Start,Dest):
        logger.info('Initializing route from %r to %r', Start,Dest)
        self.completed = False

    def teleport(self,pose):
        logger.info('Teleporting to %r.', pose)

    def turn(self,direction):
        logger.info('Turning %r.', direction)

    def travel(self):
        logger.info('Traveling forward.')

    def declareGoal(self):
        logger.info('Declaring at the goal.')

    def recognize(self,description,sim = False):
        logger.info('Recognizing %r.', description)
        if sim:
            match = random.choice((True,False))
            logger.info('%r was%r recognized', match and ' ' or ' not')
            return match

    def pose(self,pose):
        logger.info('At pose %r.', pose)
