import MarkovLoc, ViewDescription

class POMDP(ViewDescription.ViewDescriptionObservations,
            POMDP_MarkovLoc.POMDP):
    def __init__(self,name,ViewDescs):
        MarkovLoc.POMDP.__init__(self,name)
        ViewDescription.ViewDescriptionObservations.__init__(self,ViewDescs)
