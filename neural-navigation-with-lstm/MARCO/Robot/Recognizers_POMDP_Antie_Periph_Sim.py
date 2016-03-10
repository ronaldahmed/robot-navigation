from Recognizers_POMDP_Antie_Sim import *

class PomdpAntiePeriphSimRecognizer(PomdpAntieSimRecognizer):
    def recNeedTurn(cls,desc,viewCache):
        return (len(viewCache)<4 and
                False
                )
    recNeedTurn = classmethod(recNeedTurn)

