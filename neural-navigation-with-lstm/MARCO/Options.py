from plastk import params

Counts = {}
class CountedBoolean(params.Boolean):

    def __get__(self,obj,type=None):
        if obj:
            name = self.get_name(obj).split('_')[1]
        else:
            name = str(self)
        #print 'gettting option', name, Counts
        try:
            Counts[name] += 1
        except KeyError, e:
            Counts.setdefault(name,1)
        return super(CountedBoolean,self).__get__(obj,type)
    
    @classmethod
    def reset(cls):
        Counts.clear()

Boolean = CountedBoolean # params.Boolean # 

Documentation = {}
Examples = {}

class OptionGroup(object): pass

class Fundamentals(OptionGroup):
    """Fundamental abilities:  Skills that are explicitly used in instructions, e.g. counting, traveling until."""
    
    DistanceCount = Boolean(default=True)
    Documentation['DistanceCount'] = "Move to distance count."
    Examples['DistanceCount'] = [('WLH_L0_7_2_Dirs_1', 'Move forward to the second alley.'),
                                 ('EDA_Grid0_1_5_Dirs_1','Move forward once.')]
    
    TravelUntil = Boolean(default=True)
    Documentation['TravelUntil'] = "Travel until the described view."
    Examples['TravelUntil'] = [('KXP_L0_7_4_Dirs_1','Go to the end of this hallway.'),
                               ('LEN_Grid1_7_3_Dirs_1','When you get to the red brick hall, \ldots')]

    TurnDirection = Boolean(default=True)
    Documentation['TurnDirection'] = "Turn towards a specified direction."
    Examples['TurnDirection'] = [('BKW_L1_2_5_Dirs_1','Turn left.'),
                                 ('KLS_Jelly0_7_3_Dirs_1','Take a right.')]

    FaceDescription = Boolean(default=True)
    Documentation['FaceDescription'] = "Turn until a view description is met."
    Examples['FaceDescription'] = [('WLH_Grid0_6_5_Dirs_1','Face the chair.'),
                                   ('LEN_Grid1_7_1_Dirs_1','With your back to the blank wall, \ldots.')]

    UseFind = Boolean(default=True)
    Documentation['UseFind'] = "Use the find behavior."
    Examples['UseFind'] = [('TJS_Grid0_6_4_Dirs_1',"it's at the corner of the yellow floor and the grassy floor."),
                           ('MPO_Jelly1_2_1_Dirs_1',"Find the blue road.")]

    UseFollow = Boolean(default=True)
    Documentation['UseFollow'] = "Use the follow behavior."
    Examples['UseFollow'] = [('WLH_Jelly0_2_6_Dirs',"Follow the alley around \ldots."),
                             ('TJS_L0_3_2_Dirs_1',"go all the way down the winding hall")]

    ViewMemory = Boolean(default=True)
    Documentation['ViewMemory'] = "Remember views even after turning or traveling."
    Examples['ViewMemory'] = [('KXP_L0_1_6_Dirs_1',"go down toward the longer end of the hallway."),
                               ('QNL_L1_6_4_Dirs_1',"Going away from the coat hanger")]

    PerspectiveTaking = Boolean(default=True)
    Documentation['PerspectiveTaking'] = "Project a different perspective from current view."
    Examples['PerspectiveTaking'] = [('WLH_L0_5_6_Dirs_1',"to your left is a chair."),
                                     ('BLO_Grid0_2_6_Dirs_1',"Go to the right of the stand")]
    
class Conditionals(OptionGroup):
    """Conditional abilities:  Skills that are explicit in instructions, but with conditionals."""
    
    DistanceBefore = Boolean(default=True)
    Documentation['DistanceBefore'] = "Move to distance targets of X before Y."
    Examples['DistanceBefore'] = [('TMH_Grid0_6_3_Dirs_1',"Stop at last intersection before the bench"),
                                  ('KLS_Jelly0_2_3_Dirs_1',"one intersection before the coat rack will be 3")]
    
    DistancePast = Boolean(default=True)
    Documentation['DistancePast'] = "Move to distance targets of X past Y."
    Examples['DistancePast'] = [('EMWC_Jelly0_2_3_Dirs_1',"One segment past the chair is Position 3."),
                                ('KLS_L0_4_1_Dirs_1',"\ldots until two sections past the lamp")]
    
    FaceAway = Boolean(default=True)
    Documentation['FaceAway'] = "Face the away condition of Turn."
    Examples['FaceAway'] = [('QNL_L1_6_3_Dirs_1',"Go away from the clothes hanger."),
                            ('EMWC_L0_3_2_Dirs_1',"Face away from the dead end.")]

    FaceExplicit = Boolean(default=True)
    Documentation['FaceExplicit'] = "Face the explicit argument of Face."
    Examples['FaceExplicit'] = [('BKW_L1_6_4_Dirs_1',"Face the wall \ldots."),
                                ('JXF_Grid1_1_4_Dirs_1',"face the easel and dark gray stone floor.")]

    FaceOnto = Boolean(default=True)
    Documentation['FaceOnto'] = "Face the onto condition of Turn."
    Examples['FaceOnto'] = [('KLS_Grid0_4_5_Dirs_1',"take a left onto the blue path."),
                            ('WLH_L0_4_6_Dirs_1',"turn right onto the stone")]

    FacePurpose = Boolean(default=True)
    Documentation['FacePurpose'] = "Face the purpose condition of Turn."
    Examples['FacePurpose'] = [('EDA_Grid0_1_2_Dirs_1',"turn to face the long red hallway."),
                               ('TXG_Jelly1_7_1_Dirs_1',"turn left so that you face another bench")]

    FaceToward = Boolean(default=True)
    Documentation['FaceToward'] = "Face the toward condition of Travel."
    Examples['FaceToward'] = [('BLO_Grid0_4_1_Dirs_1',"go towards the red brick floor."),
                              ('TXG_Jelly1_6_2_Dirs_1',"move toward the bench.")]

    FaceView = Boolean(default=True)
    Documentation['FaceView'] = "Face a stanfalone description of  a view."
    Examples['FaceView'] = [('EDA_L0_4_2_Dirs_1',"you should have butterfly pictures in front of you"),
                            ('MJB_Jelly1_6_7_Dirs_1.',"you see a little bit of yellow on the floor")]

    TravelPast = Boolean(default=True)
    Documentation['TravelPast'] = "Travel past conditions."
    Examples['TravelPast'] = [('JNN_Jelly0_5_1_Dirs_1',"move one place past the brick floor tiling"),
                              ('LEN_Grid1_4_1_Dirs_1',"pass the easle that is sitting in the hall")]

    TravelPrecond = Boolean(default=True)
    Documentation['TravelPrecond'] = "Turn to syntactically marked precondition of a Travel."
    Examples['TravelPrecond'] = [('JXF_Grid1_3_2_Dirs_1',"facing the easel, go forward."),
                                 ('WLH_L0_7_6_Dirs_1',"with your back to the wall move along the green \ldots")]

    TurnPrecond = Boolean(default=True)
    Documentation['TurnPrecond'] = "Turn to precondition precondition pose of a Turn."
    Examples['TurnPrecond'] = [('BKW_L1_7_4_Dirs_1',"With your back facing the wall turn right"),
                               ('WLH_Grid0_7_1_Dirs_1',"facing the long aisle turn left")]

    TurnLocation = Boolean(default=True)
    Documentation['TurnLocation'] = "Move to syntactically marked precondition location of a Turn."
    Examples['TurnLocation'] = [('JNN_Jelly0_2_3_Dirs_1', "once you hit the dining chair, turn right"),
                                ('JXL_Grid1_5_6_Dirs_1',"At the brick hallway, turn towards \ldots")]

    TravelLocation = Boolean(default=True)
    Documentation['TravelLocation'] = "Move to syntactically marked precondition location of a Travel."
    Examples['TravelLocation'] = [('KLS_Grid0_1_6_Dirs_1',"at the wood path intersection, take the wood path "),
                                  ('WLH_Jelly0_3_5_Dirs_2',"from the easel, move one block")]

    TurnPostcond = Boolean(default=True)
    Documentation['TurnPostcond'] = "Travel to syntactically marked postcondition pose of Turn."
    Examples['TurnPostcond'] = [('KLS_Jelly0_3_2_Dirs_1',"take a right to the end of the hall"),
                                ('KLS_Grid0_1_4_Dirs_1',"take a right onto the green path all the way to the end of the hall")]

    FaceTravelArgs = Boolean(default=True)
    Documentation['FaceTravelArgs'] = "Face Travel arguments."
    Examples['FaceTravelArgs'] = [('KLS_Grid0_2_6_Dirs_1',"take the yellow path to the wood path intersection"),
                                  ('MJB_Jelly1_2_5_Dirs_1',"Go down the short hall ")]
    
    DeclareGoalCond = Boolean(default=True)
    Documentation['DeclareGoalCond'] = "Move to satisfy condition on DeclareGoal."
    Examples['DeclareGoalCond'] = [('KLS_Grid0_7_6_Dirs_1',"when you come to the intersection with \ldots, you are at position 6"),
                                   ('MJB_Jelly1_1_5_Dirs_1', "once in the L you're in position 5.")]

    StopCond = Boolean(default=True)
    Documentation['StopCond'] = "Intrepret 'stop when COND' as 'go until COND.'"
    Examples['StopCond'] = [('TJS_L0_3_2_Dirs_1',"stop at the first intersection"),
                            ('KXP_L0_4_7_Dirs_1'," stop when the first fish hallway to your left appears")]

class Heuristics(OptionGroup):
    """Heuristics Hints:  Strategies that enforce simple implicit inferences, e.g. face an open path before traveling."""
    
    FaceDistance = Boolean(default=True)
    Documentation['FaceDistance'] = "Face at least one of the distance units (e.g. intersections, streets, movements) of Travels."

    FaceUntil = Boolean(default=True)
    Documentation['FaceUntil'] = "Face the termination condition of Travels."

    FaceUntilPostDist = Boolean(default=True)
    Documentation['FaceUntilPostDist'] = "Re-face the termination condition of Travels after going distance, to catch over-estimates."

    FacePast = Boolean(default=False)
    Documentation['FacePast'] = "Face objects the Travel will pass."

    TurnExplicit = Boolean(default=True)
    Documentation['TurnExplicit'] = "Turn towards an explicitly mentioned direction before checking condition."

    TurnTowardPath = Boolean(default=True)
    Documentation['TurnTowardPath'] = "If no explicit turn direction, turn towards a visible path instead of a wall."

    ReverseTurn = Boolean(default=True)
    Documentation['ReverseTurn'] = "If last action was a turn, but now facing a wall, turn around."

    TurnPreResetCache = Boolean(default=False)
    Documentation['TurnPreResetCache'] = "Reset anaphora cache before interpreting turn."

    TurnTermResetCache = Boolean(default=True)
    Documentation['TurnTermResetCache'] = "Reset anaphora cache before interpreting turn termination."

    TurnPostResetCache = Boolean(default=False)
    Documentation['TurnPostResetCache'] = "Reset anaphora cache after interpreting turn."

    LookAheadForTravelTerm = Boolean(default=True)
    Documentation['LookAheadForTravelTerm'] = "Look ahead in instuctions for termination for travel actions."

    LookAheadForTravelTermLoc = Boolean(default=True)
    Documentation['LookAheadForTravelTermLoc'] = "Look ahead in instuctions for termination for travel actions in location phrases."
    Examples['LookAheadForTravelTermLoc'] = [('JNN_Jelly0_4_5_Dirs_1',"go towards the easle but stop at the closest concrete square closest to the easle"),
                                             ('KLS_Jelly0_6_7_Dirs_1',"take a left onto the pink path. at the next intersection,  \ldots")]

    LookAheadForTravelTermDesc = Boolean(default=True)
    Documentation['LookAheadForTravelTermDesc'] = "Look ahead in instuctions for termination for travel actions in descriptions."
    Examples['LookAheadForTravelTermDesc'] = [('JXF_Grid1_1_6_Dirs_1',"Go foward.  Look for butterflies"),
                                              ('JXL_Grid1_7_4_Dirs_1',"Walk towards the brick hallway. At one end of the brick hallway, there is a chair.")]
    
    TravelOnFinalTurn = Boolean(default=True)
    Documentation['TravelOnFinalTurn'] = "Make a final travel forward when the last action was a turn."
    Examples['TravelOnFinalTurn'] = [('WLH_L0_1_6_Dirs_1',"with your back to the wall turn left and move one block. turn right."),
                                   ('KXP_Grid0_7_5_Dirs_1',"go down the butterfly walled/blue floored hallway. make a left at the hatrack.")]

    TravelOnFinalView = Boolean(default=False)
    Documentation['TravelOnFinalView'] = "Make a final travel forward when the last action was a view."
    Examples['TravelOnFinalView'] = [('JNN_Jelly0_3_7_Dirs_1',"turn left and you should see a barstool. this is position 7."),
                                     ('EMWC_Jelly0_4_1_Dirs_1',"You should be able to see the grassy hall to your right. This is Position 1.")]

    TravelBetweenTurns = Boolean(default=True)
    Documentation['TravelBetweenTurns'] = "Make a travel between consecutive turns."
    Examples['TravelBetweenTurns'] = [('MXM_Jelly0_4_2_Dirs_1',"take a left then a right."),
                                      ('KXP_Jelly0_6_4_Dirs_1',"left. left.")]

    TurnBetweenTravels = Boolean(default=True)
    Documentation['TurnBetweenTravels'] = "Make a turn between consecutive travels."
    Examples['TurnBetweenTravels'] = [('EMWC_Jelly0_2_1_Dirs_1',"Go forward down the hall until a hall opens to your left. Go forward one segment."),
                                      ('TJS_Grid0_7_6_Dirs_1',"yellow hall to wooden hall. then one space forward.")]

    TravelToNext = Boolean(default=True)
    Documentation['TravelToNext'] = "Travel to next match when last action was turn."
    Examples['TravelToNext'] = [('BKW_L1_5_1_Dirs_1',"\ldots take a left. Go down to the corner."),
                                ('KLS_L0_3_2_Dirs_1',"go until hall ends. take a left. go until hall ends.")]
    
    PropagateContextInfo = Boolean(default=True)
    Documentation['PropagateContextInfo'] = "Propagate the context information to embedded compound action specifications."

class Recoveries(OptionGroup):
    """Error recovery: Strategies that attempt to recover from a failed action."""
    
    TravelToDistantView = Boolean(default=True)
    Documentation['TravelToDistantView'] = "Travel when Facing a satisfying pose distantly visible from this place."
    Examples['TravelToDistantView'] = [('KLS_Jelly0_3_2_Dirs_1',"at the next corner, take a right at the lamp."),
                                       ('TJS_Grid0_5_1_Dirs_1',"go forward. then left all the way down the blue hall")]

    FindFace = Boolean(default=True)
    Documentation['FindFace'] = "Find when Face does not find a satisfying pose visible from this place."

    FindFaceTravel = Boolean(default=True)
    Documentation['FindFaceTravel'] = "Use Find as fallback for Face even if not ImplicitTravel"

    CheckAfterTurnFind = Boolean(default=True)
    Documentation['CheckAfterTurnFind'] = "Check the until condition of the find between the turn and travel."

class Tweaks(OptionGroup):
    """Tweaks: parameters that have small effects on how a behavior is executed."""
    
    FaceMaxTurns = params.PositiveInt(default=8)
    Documentation['FaceMaxTurns'] = "How many times to turn to ensure all views seen and preferred view in front."

    TravelNoTermination = Boolean(default=True)
    Documentation['TravelNoTermination'] = "Travel forward without termination conditions."

    TravelEmpty = Boolean(default=True)
    Documentation['TravelEmpty'] = "Travel forward without arguments."

    DeclareGoalForPosition = Boolean(default=False)
    Documentation['DeclareGoalForPosition'] = "DeclareGoal whenever a position is mentioned."

class Linguistics(OptionGroup):
    """Linguistic parameters:  parameters that affect the interpretation of text."""
    
    Spellcheck = Boolean(default=False)
    Documentation['Spellcheck'] = "Do spellchecking and replace unknown words with known."
    
    FuzzyMeanings = Boolean(default=True)
    Documentation['FuzzyMeanings'] = "Use broader definitions of concepts."
    Examples['FuzzyMeanings'] = [('chair, bench, or stool',"chair"),
                                 ('cement or stone flooring',"gray")]
    
    ReferenceResolution = Boolean(default=True)
    Documentation['ReferenceResolution'] = "Fill a reference phrase or pronoun with the corresponding noun phrase."
    Examples['ReferenceResolution'] = [('WLH_Jelly0_5_6_Dirs_1',"you should see an alley to your left. take it."),
                                       ('KLS_L0_3_4_Dirs_1',"you will be looking for a pink flowered path.  When you reach this path, \ldots")]
    
    RawReferenceResolution = Boolean(default=False)
    Documentation['RawReferenceResolution'] = "Fill a reference phrase or pronoun with the corresponding noun phrase, even if not limited by syntax."
    Examples['RawReferenceResolution'] = [('EMWC_Grid0_5_2_Dirs_1',"Face toward the hatrack. Walk the one segment to it."),
                                          ('WLH_Jelly0_5_6_Dirs_1',"there should be blue carpet on the first alley. walk to that and turn right.")]
    
    RecognizeDistalDeterminers = Boolean(default=True)
    Documentation['RecognizeDistalDeterminers'] = "Recognize determiners like 'that' and 'other' as marking distant entities."
    Examples['RecognizeDistalDeterminers'] = [('TJS_Grid0_1_3_Dirs_1',"go down the red hall until you see the blue hall. at that intersection stop."),
                                              ('TXG_Jelly1_3_2_Dirs_1',"then turn left until you see another  bench and move to it.")]
    
    DeclareGoalIdiom = Boolean(default=True)
    Documentation['DeclareGoalIdiom'] = "Treat 'This is (it|Position X) as idiom."
    Examples['DeclareGoalIdiom'] = [('KLS_Jelly0_1_6_Dirs_1',"this is 6."),
                                   ('LEN_Grid1_4_6_Dirs_1',"Position 6 is the next intersection as you follow the red-brick hallway.")]
    
    RecognizeFictiveTurnIntersections = Boolean(default=True)
    Documentation['RecognizeFictiveTurnIntersections'] = "Handle 'where you (can) turn|travel to the right...'"
    Examples['RecognizeFictiveTurnIntersections'] = [('TXG_Jelly1_5_6_Dirs_1',"move forward until you can turn left again."),
                                                     ('KLS_L0_2_7_Dirs_1',"go straight until you can either go left or towards a dead end.")]
    
    RecognizeDirTurn = Boolean(default=True)
    Documentation['RecognizeDirTurn'] = "Handle e.g.'(until) the last right.'"
    Examples['RecognizeDirTurn'] = [('EDA_L0_2_3_Dirs_1',"there should be only once choice, to turn right."),
                                   ('JJL_Grid0_5_4_Dirs_1',"this left turn should have a yellow floor.")]
    
    RecognizePathdir = Boolean(default=True)
    Documentation['RecognizePathdir'] = "Handle 'Only one way to go.'"
    Examples['RecognizePathdir'] = [('EDA_Jelly0_4_1_Dirs_1',"only one way to go"),
                                   ('JXF_Grid1_1_6_Dirs_1',"face the direction with the easel.")]
    
    RecognizeTakeTurnFrame = Boolean(default=True)
    Documentation['RecognizeTakeTurnFrame'] = "Handle 'Take ((the nth)|a) (right|left) [turn].'"
    Examples['RecognizeTakeTurnFrame'] = [('MXM_Jelly0_2_3_Dirs_1',"make a left."),
                                          ('MJB_Jelly1_7_6_Dirs_1',"take your second right.")]
    
    RecognizeArriveFrame = Boolean(default=True)
    Documentation['RecognizeArriveFrame'] = "Handle arrive frames as travel until."
    Examples['RecognizeArriveFrame'] = [('TXG_Jelly1_6_2_Dirs_1',"turn right when you reach the end."),
                                        ('LEN_Grid1_1_3_Dirs_1',"Once you get to the floral carpeted hallway, look for the easle.")]
    
    RecognizeStructFrame = Boolean(default=True)
    Documentation['RecognizeStructFrame'] = "Handle frames with structural descriptions as verbs, like 'where the paths cross.'."
    Examples['RecognizeStructFrame'] = [('KLS_L0_4_3_Dirs_1',"when road ends, go right."),
                                        ('TJS_Grid0_3_1_Dirs_1',"then a left down the red hall until it intersects with the rose hall")]
    
    RecognizeStructural = Boolean(default=True)
    Documentation['RecognizeStructural'] = "Handle structural adjectives, especially long and short."
    Examples['RecognizeStructural'] = [('TXG_Jelly1_4_3_Dirs_1',"turn until you face the short end of the hallway with blue flooring."),
                                   ('MJB_Jelly1_1_5_Dirs_1',"go down the longer part of the hall with blue rectangles.")]
    
    RecognizeStructAgentFrame = Boolean(default=True)
    Documentation['RecognizeStructAgentFrame'] = "Handle sentences like 'You will have to take a right. '"
    Examples['RecognizeStructAgentFrame'] = [('MJB_Jelly1_2_1_Dirs_1',"You will have to take a right onto a floor that is black."),]
    
    RecognizeStandaloneArrive = Boolean(default=True)
    Documentation['RecognizeStandaloneArrive'] = "Handle sentences like 'You will hit an l."
    Examples['RecognizeStandaloneArrive'] = [('KLS_Jelly0_1_5_Dirs_1',"the very first section you come to, you will be at 5"),
                                             ('MXP_L0_1_6_Dirs_1',"you will hit an intersection with black stone floors.")]
    
    RecognizeUntilView = Boolean(default=True)
    Documentation['RecognizeUntilView'] = "Handle Turn until VIEW."
    Examples['RecognizeUntilView'] = [('TXG_Jelly1_6_5_Dirs_1',"turn until you face the hallwya with the green floor"),
                                      ('WLH_Grid0_1_3_Dirs_1',"from one turn until you see a corner of blue carpet in a side alley.")]

    RecognizeUntilLocDist = Boolean(default=True)
    Documentation['RecognizeUntilLocDist'] = "Handle Travel UNTIL LOC DIST."
##    Examples['RecognizeUntilLoc'] = [('TXG_Jelly1_6_5_Dirs_1',"turn until you face the hallwya with the green floor"),
##                                      ('WLH_Grid0_1_3_Dirs_1',"from one turn until you see a corner of blue carpet in a side alley.")]
    
    RecognizeComplexExpressions = Boolean(default=True)
    RecognizeComplexExpressionsList = ['RecognizeFictiveTurnIntersections','RecognizeStructFrame','RecognizePassFrame',
                                       'RecognizeDirTurn', 'RecognizeDistalDeterminers', 'RecognizeLast', 'RecognizeNegative',
                                       'RecognizeNegativeCompound', 'RecognizeStandaloneArrive',
                                       'RecognizeStructAgentFrame', 'RecognizeUntilLocDist', 'RecognizeUntilView',]
    Documentation['RecognizeComplexExpressions'] = "Recognize complex expressions."
    
    FaceDeclaratives = Boolean(default=False)
    Documentation['FaceDeclaratives'] = 'Enforce declaratives with Face statements.'
    Examples['FaceDeclaratives'] = [('WLH_L0_7_5_Dirs_1',"to your right you should see an alley with grey carpet."),
                                   ('TJS_L0_2_1_Dirs_1',"there is a bench there.")]
    
    RecognizeNounNounCompound = Boolean(default=True)
    Documentation['RecognizeNounNounCompound'] = "Recognize noun noun-compounds."
    Examples['RecognizeNounNounCompound'] = [('KXP_Grid0_2_5_Dirs_1',"down the long butterfly hallway, with blue walls."),
                                             ('EMWC_Grid0_1_5_Dirs_1',"Face the pink-flowered carpet hall")]
    
    RecognizeNegative = Boolean(default=True)
    Documentation['RecognizeNegative'] = "Recognize negative phrases."
    Examples['RecognizeNegative'] = [('MXM_Jelly0_2_7_Dirs_1',"when there is not a wall to your left, go straight."),
                                     ('EMWC_Grid0_3_1_Dirs_1',"\ldots toward the pictures of not-butterflies")]

    RecognizeNegativeCompound = Boolean(default=True)
    Documentation['RecognizeNegativeCompound'] = "Recognize negative compounds phrases."
    Examples['RecognizeNegativeCompound'] = [('JXF_Grid1_3_5_Dirs_1',"not a bench or a stool."),]

    RecognizeLast = Boolean(default=True)
    Documentation['RecognizeLast'] = "Recognize last as order adj."
    Examples['RecognizeLast'] = [('LEN_Grid1_2_7_Dirs_1',"until you come to the last empty intersection before the easle"),
                                 ('JJL_Grid0_7_6_Dirs_1',"Follow this down untill you come to the second to last left.")]
    
    RecognizeCount = Boolean(default=True)
    Documentation['RecognizeCount'] = "Recognize counts in noun phrases."
    Examples['RecognizeCount'] = [('WLH_Grid0_4_5_Dirs_1',"walk past two chairs and to the lamp."),
                                   ('LEN_Grid1_2_7_Dirs_1',"This empty intersection is Position 7.")]

    RecognizeAcross = Boolean(default=True)
    Documentation['RecognizeAcross'] = "Recognize across phrases."
    Examples['RecognizeAcross'] = [('QNL_L1_1_2_Dirs_1',"move across one yellow panel."),
                                   ('QNL_L1_1_5_Dirs_1',"turn right across 2 black stone floors.")]

class Implicits(OptionGroup):
    """."""
    
    ImplicitTravel = Boolean(default=True)
    Documentation['ImplicitTravel'] = "Infer and perform implicit travels."
    
    ImplicitTurn = Boolean(default=True)
    Documentation['ImplicitTurn'] = "Infer and perform implicit turns."
    
    ImplicitActions = Boolean(default=True)
    Documentation['ImplicitActions'] = "Infer and perform implicit procedures."
    ImplicitActionsList = ['ImplicitTravel', 'ImplicitTurn']
    
    ImplicitSyntactic = Boolean(default=True)
    Documentation['ImplicitSyntactic'] = "Prepositional phrase markings."
    ImplicitSyntacticList = ['FacePurpose','TravelPrecond','TurnPrecond','TurnPostcond',
                             'DeclareGoalCond','TurnLocation','TravelLocation','StopCond',
                             'DeclareGoalForPosition', ]
    
    ImplicitSemantic = Boolean(default=True)
    Documentation['ImplicitSemantic'] = "Complex action frames."
    ImplicitSemanticList = ['RecognizeTakeTurnFrame', 'RecognizeArriveFrame',
                            'FaceTravelArgs','FaceDistance','FaceUntil','FacePast','TurnTowardPath',]
    
    ImplicitPragmaticCrossUtterance = Boolean(default=True)
    Documentation['ImplicitPragmaticCrossUtterance'] = "Pragmatics across utterances."
    ImplicitPragmaticCrossUtteranceList = ['LookAheadForTravelTerm', 'TravelOnFinalTurn', 
                                           'TravelBetweenTurns', 'TurnBetweenTravels', 'TravelToNext',
                                           'TravelOnFinalView', 'PropagateContextInfo',]
    
    ImplicitPragmaticPerUtterance = Boolean(default=True)
    Documentation['ImplicitPragmaticPerUtterance'] = "Pragmatics per utterance."
    ImplicitPragmaticPerUtteranceList = ['DeclareGoalIdiom','RecognizePathdir',]
    
    ImplicitPragmatic = Boolean(default=True)
    Documentation['ImplicitPragmatic'] = "Discourse and idiomatic cues."
    ImplicitPragmaticList = ImplicitPragmaticPerUtteranceList+ImplicitPragmaticCrossUtteranceList
    
    ImplicitExploration = Boolean(default=True)
    Documentation['ImplicitExploration'] = "Active resolution of referring phrases."
    ImplicitExplorationList = ['TravelToDistantView','FindFace','RecognizeStructural']

class Landmarks(OptionGroup):
    """Recognize different sorts of landmarks."""
    
    AppearanceLandmarks = Boolean(default=True)
    Documentation['AppearanceLandmarks'] = "Recognizing simple perceptual attributes, e.g. color, texture."
    Examples['AppearanceLandmarks'] = [('TJS_Grid0_7_3_Dirs_1',"take one movement towards the blue corridor"),
                                       ('LEN_Grid1_7_2_Dirs_1',"Go to the hallway that has the blue tiles and the orange butterflies on the wall.")]
    
    ObjectLandmarks = Boolean(default=True)
    Documentation['ObjectLandmarks'] = "Recognizing distinct objects, e.g. furniture and pictures."
    Examples['ObjectLandmarks'] = [('LEN_Grid1_7_4_Dirs_1',"The intersection with the chair is Position 4."),
                                   ('KLS_Jelly0_3_4_Dirs_1',"Go towards the coat rack and take a left at the coat rack.")]
    
    CausalLandmarks = Boolean(default=True)
    Documentation['CausalLandmarks'] = "Recognizing simple structural landmarks, e.g. paths, walls, positions."
    Examples['CausalLandmarks'] = [('WLH_Grid0_3_5_Dirs_1',"with your back to the wall turn right."),
                                   ('KLS_L0_7_6_Dirs_1',"take a left onto the black path")]
    
    IntersectionLandmarks = Boolean(default=True)
    Documentation['IntersectionLandmarks'] = "Recognizing intersection structural landmarks, e.g. dead ends, T intersections, corners."
    Examples['IntersectionLandmarks'] = [('JNN_Jelly0_3_4_Dirs_1',"the dead end is position 4."),
                                   ('KLS_Grid0_2_3_Dirs_1',"at the first intersection after the lamp,")]
    
    StructuralLandmarks = Boolean(default=True)
    Documentation['StructuralLandmarks'] = "Recognizing simple structural landmarks."
    StructuralLandmarksList = ['CausalLandmarks', 'IntersectionLandmarks']

class HSSH(OptionGroup):
    """Divide methods by what level of the SSH they require."""
    
    OpenLoopCausal = Boolean(default=True)
    Documentation['OpenLoopCausal'] = "Open loop causal control laws."
    OpenLoopCausalList = ['DistanceCount','TurnDirection']
    
    ClosedLoopCausal = Boolean(default=True)
    Documentation['ClosedLoopCausal'] = "Closed loop causal control laws."
    ClosedLoopCausalList = ['TravelUntil','FaceDescription','ObjectLandmarks']
    
    Causal = Boolean(default=True)
    Documentation['Causal'] = "All Causal actions."
    CausalList = OpenLoopCausalList + ClosedLoopCausalList + ['CausalLandmarks',]
    
    LocalMetrical =  Boolean(default=True)
    Documentation['LocalMetrical'] = "Local Metrical."
    LocalMetricalList = ['TravelToDistantView','ViewMemory','PerspectiveTaking', 'FaceDistance']
    
    LocalTopological = Boolean(default=True)
    Documentation['LocalTopological'] = "Local Topological."
    LocalTopologicalList = ['RecognizeStructural','IntersectionLandmarks', 'TurnTowardPath', 'ReverseTurn',
                            'LookAheadForTravelTerm', 'RecognizeCount', 'RecognizeDirTurn', 'RecognizePathdir',
                            'FaceUntil', 'FaceUntilPostDist']
    
    Topological =  Boolean(default=True)
    Documentation['Topological'] = "Topological."
    TopologicalList = ['UseFollow', 'UseFind', 'TravelToNext', 'TravelBetweenTurns',
                       'TurnBetweenTravels', 'TravelOnFinalTurn']

class Comparison(OptionGroup):
    """Comparisons to different agents or development baselines."""
    
    Corpus2Options = Boolean(default=True)
    Documentation['Corpus2Options'] = "Options added when examining corpus 2."
    Corpus2OptionsList = ['DeclareGoalForPosition', 'PropagateContextInfo', 'RawReferenceResolution',
                          'RecognizeAcross', 'RecognizeCount', 'RecognizeDirTurn',
                          'RecognizeDistalDeterminers', 'RecognizeLast', 'RecognizeNegative',
                          'RecognizeNegativeCompound', 'RecognizeStandaloneArrive',
                          'RecognizeStructAgentFrame', 'RecognizeUntilLocDist', 'RecognizeUntilView',
                          'ReverseTurn', 'Spellcheck', 'TravelOnFinalView']
    
    IBLOptions = Boolean(default=True)
    Documentation['IBLOptions'] = "Options that distinguish Marco from the IBL agent."
    IBLOptionsList = ['ReverseTurn', 'TurnTowardPath', 'FaceDistance', 'ImplicitPragmatic']

import plastk.base
class Options(plastk.base.BaseObject,
    Fundamentals,Conditionals,Heuristics,Recoveries,Tweaks,Linguistics,Implicits,Landmarks,HSSH,Comparison):
    """Options to control the behavior of the execution."""
    Statistics =  Boolean(default=False)
    
    def stats(self): return Counts
    def reset(self): CountedBoolean.reset()
