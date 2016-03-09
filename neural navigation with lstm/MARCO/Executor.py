from Utility import logger

class Executor:
    def __init__(self):
        self.NLUQueue = None
    
    def initialize(self,instructID):
        director,env,start,dest,txt,set = instructID.split('_')
        if self.NLUQueue:
            self.NLUQueue.put(('Director',director))
            self.NLUQueue.put(('Start Place',start))
            self.NLUQueue.put(('Destination',dest))
            self.NLUQueue.put(('Set',set))
        self.robot.setRoute(start,dest)

    def setRobot(self,Robot):
        self.robot = Robot

    def _execute(self,Model): pass
    
    def execute(self,Model,instructID):
        CaughtError = None
        self.initialize(instructID)
        try:
            self._execute(Model)
        except (KeyError,LookupError,ValueError,AttributeError,RuntimeError,TypeError),e:
            logger.error('%s on %s', e.__class__.__name__, e)
            CaughtError = e
        if not self.robot.completed:
            reward,obs = self.robot.declareGoal()
            logger.stageComplete('Declared Goal to complete instruction execution %s',(reward,obs))
        ResultTxt = self.robot.completed and 'Success' or 'Failure'
        logger.runComplete("%s in direction following.", ResultTxt)
        return self.robot.completed, CaughtError, ResultTxt

class InstructionQueueExecutor(Executor):
    def _execute(self,Actions):
        for i,action in enumerate(Actions):
            logger.info('<%d> %s',i,action)
            if self.NLUQueue: self.NLUQueue.put(('Executing',i))
            try:
                results = action.execute(self.robot)
                logger.info('<%d> %s => %r', i,action,results)
            except Warning,e:
                logger.warning('<%d> %s => %s', i,action,e)
            except StopIteration,e:
                results = e    
                logger.info('<%d> %s => %r', i,action,results)
                logger.info('End of Instruction Execution after <%d>', i)

def test():
    import doctest
    doctest.testmod()
