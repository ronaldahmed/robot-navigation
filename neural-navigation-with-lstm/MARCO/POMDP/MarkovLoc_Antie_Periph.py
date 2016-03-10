import MarkovLoc_Antie
from Robot import Meanings

class POMDP(MarkovLoc_Antie.POMDP):
    PeripheralViews = [Meanings.Honeycomb, Meanings.Brick, Meanings.Stone,
                       Meanings.Wood, Meanings.Rose, Meanings.BlueTile, Meanings.Grass,
                       Meanings.Wall]

    def getPeripheralView(self,(place,pose)):
        if (place, pose%self.NumPoses) in self.TextureLocs:
            return Meanings.Texture.Abbrevs[self.TextureLocs[(place,pose%self.NumPoses)]]
        else: return Meanings.Wall

