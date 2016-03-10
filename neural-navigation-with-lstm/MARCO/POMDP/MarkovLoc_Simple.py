import MarkovLoc

class observation:
    def __init__(self,left,fwd,right):
        self.left=left
        self.fwd=fwd
        self.right=right
    def __str__(self):
        return '_'.join([self.left,self.fwd,self.right])
    code = __str__

class POMDP(MarkovLoc.POMDP):
    def __init__(self,name):
        MarkovLoc.POMDP.__init__(self,name)
        self.ObservationGenerators['*']=self.getView
    
    def getView(self,(place,pose)):
        """Looks up a view visible from a pose.
        
        Agent can see the presence (W) or absence (H) of walls to the sides.
        Agent can see the color of the hallway directly in front for one unit.
        Views are of the form (Left Hall|Wall)_(Front Wall| Color)_(Right Hall|Wall)
        """
        if (place,pose) in self.ViewsFwd:
            fwd = self.ViewsFwd[(place,pose)]
        else: fwd = 'W'
        if (place, (pose-1)%self.NumPoses) in self.Gateways:
            left = 'H'
        else: left = 'W'
        if (place, (pose+1)%self.NumPoses) in self.Gateways:
            right = 'H'
        else: right = 'W'
        return [(observation(left,fwd,right),1.0),]
    
    def generateObservations(self):
        """Generates the set of osbservations.
        """
        for left in self.PeripheralViews:
            for fwd in self.ForwardViews:
                for right in self.PeripheralViews:
                    yield observation(left,fwd,right).code()
