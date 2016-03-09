class RobotI:
    def setRoute(self,Start,Dest):
        """Initialize route"""
        raise NotImplementedError

    def teleport(self,pose):
        """Teleport to pose."""
        raise NotImplementedError

    def turn(self,direction):
        """Turn in place."""
        raise NotImplementedError

    def travel(self):
        """Travel from place to place."""
        raise NotImplementedError

    def declareGoal(self):
        """Declare navigation finished."""
        raise NotImplementedError

    def recognize(self,description):
        """Check if description matches observation."""
        raise NotImplementedError

