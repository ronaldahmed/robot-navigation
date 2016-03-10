def str2meaning(s):
    """Helper function to eval a string in the Meanings namespace."""
    return eval(s)

KB = {}
ObjectTypes = ('Path','Pathdir','Obj','Struct','Region','Segment')
class Meaning:
    gAbbrevs={}
    gNames={}
    gMeanings={}
    gFuzzy={}
    def __init__(self,n,a,m,cAbbrevs,cNames,cMeanings):
        self.name = n
        if type(a) is list:
            self.abbr = ''.join([str(s) for s in a])
        else:
            self.abbr = a
        self.meaning = m
        self.hash  = hash(self.abbr)
        if self.abbr in self.gAbbrevs:
            raise ValueError
        self.gAbbrevs[self.abbr] = self
        if n in self.gNames:
            raise ValueError
        self.gNames[n] = self
        if m in self.gMeanings:
            raise ValueError
        self.gMeanings[m] = self
        if self.abbr in cAbbrevs:
            raise ValueError
        cAbbrevs[self.abbr] = self
        if n in cNames:
            raise ValueError
        cNames[n] = self
        if m in cMeanings:
            raise ValueError
        cMeanings[m] = self
    def __str__(self): return self.abbr
    def __repr__(self): return self.name
    def __hash__(self): return self.hash
    def __add__(self,s): return self.name+s
    def __eq__(self,s): return hasattr(s,'abbr') and self.abbr == s.abbr
    def __ne__(self,s): return not self==s
    def __lt__(self,s): return self.name < repr(s)
    def match(self,s): return (hasattr(s,'abbr')
                               and (self.abbr == s.abbr
                                    or self.abbr in s.abbr
                                    or s.abbr in self.abbr))

class FuzzyMeaning(Meaning):
    gFuzzy={}
    state = 'widen' # 'narrow', 'mixed'
    def __init__(self,n,a,m,cAbbrevs,cNames,cMeanings,narrow=None):
        Meaning.__init__(self,n,a,m,cAbbrevs,cNames,cMeanings)
        if narrow:
            self.narrow = narrow.abbr
            self.wide = self.abbr
            self.gFuzzy[m] = self
    def narrowMeaning(self): self.abbr = self.narrow
    def widenMeaning(self): self.abbr = self.wide
    def swapMeaning(self):
        if self.abbr == self.wide: self.abbr = self.narrow
        else: self.abbr = self.wide
    def alterMeanings(cls,alteration='swap'):
        for meaning in cls.gFuzzy.values():
            if alteration == 'swap': meaning.swapMeaning()
            elif alteration == 'narrow': meaning.narrowMeaning()
            elif alteration == 'widen': meaning.widenMeaning()
            else: raise ValueError('Unknown alteration:', alteration)
    alterMeanings = classmethod(alterMeanings)
    
    def setMeanings(cls,wide=True):
        if wide: alteration = 'widen'
        else: alteration = 'narrow'
        if alteration == cls.state:
            return
        cls.alterMeanings(alteration=alteration)
        cls.state = alteration
    setMeanings = classmethod(setMeanings)

Unknown = Meaning('Unknown','\?','Unknown',{},{},{})

class Side(Meaning):
    Abbrevs = {}
    Names = {}
    Meanings = {}
    def __init__(self,n,a,m): Meaning.__init__(self,n,a,m,self.Abbrevs,self.Names,self.Meanings)

Left = Side('Left','<','Left')
Right = Side('Right','>','Right')
Back = Side('Back','[','Back')
Front = Side('Front',']','Front')
At = Side('At','@', 'At')
Sides = Side('Sides','=','Sides')
FrontLeft = Side('FrontLeft','\\','FrontLeft')
FrontRight = Side('FrontRight','/','FrontRight')
Between = Side('Between',':', 'Between')

Opposites = {
    Left : Right,
    Right : Left,
    Back : Front,
    Front : Back,
    At : At,
    }
def opposite(side): return Opposites[side]

Directions = {
    'RT' : Right,
    'right_ADV_4' : Right,
    'right_N_3' : Right,
    'LFT' : Left,
    'LT' : Left,
    'left_ADV_1' : Left,
    'left_V_1' : Left,
    'BACK' : Back,
    'BEHIND' : Back,
    'around_ADV_6' : Back,
    'back_ADV_1' : Back,
    'back_N_1' : Back,
    'behind_ADV_1' : Back,
    'AHEAD' : Front,
    'DOWN' : Front,
    'FOWARD' : Front,
    'FWD' : Front,
    'IN' : Front,
    'OUT' : Front,
    'UP' : Front,
    'forward_ADV_1' : Front,
    'front_ADJ_1' : Front,
    'front_ADV_1' : Front,
    'front_N_2' : Front,
    'straight_ADJ_2' : Front,
    'straight_ADV_1' : Front,
    'side_N_1' : Sides,
    'side_ADJ_1' : Sides,
}
KB.update(Directions)

Counts = {
    '1_ADJ_1' : 1,
    '1_N_1' : 1,
    '1st_ADJ_1' : 1,
    'first_ADJ_1' : 1,
    'next_ADJ_3' : 1,
    'once_ADJ_1' : 1,
    'once_ADV_1' : 1,
    'one_ADJ_1' : 1,
    'one_N_1' : 1,
    '2_ADJ_1' : 2,
    '2_N_1' : 2,
    'second_ADJ_1' : 2,
    'twice_ADJ_1' : 2,
    'twice_ADV_1' : 2,
    'two_ADJ_1' : 2,
    'two_N_1' : 2,
    '3_ADJ_1' : 3,
    '3_N_1' : 3,
    'third_ADJ_1' : 3,
    'three_ADJ_1' : 3,
    'three_N_1' : 3,
    '4_ADJ_1' : 4,
    '4_N_1' : 4,
    'four_ADJ_1' : 4,
    'four_N_1' : 4,
    'fourth_ADJ_1' : 4,
    '5_ADJ_1' : 5,
    '5_N_1' : 5,
    'five_ADJ_1' : 5,
    'five_N_1' : 5,
    '6_ADJ_1' : 6,
    '6_N_1' : 6,
    'six_ADJ_1' : 6,
    'six_N_1' : 6,
    '7_ADJ_1' : 7,
    '7_N_1' : 7,
    'seven_ADJ_1' : 7,
    'seven_N_1' : 7,
    '8_ADJ_1' : 8,
    '8_N_1' : 8,
    'eight_ADJ_1' : 8,
    'eight_N_1' : 8,
    '9_ADJ_1' : 9,
    '9_N_1' : 9,
    'nine_ADJ_1' : 9,
    'nine_N_1' : 9,
    
    'few_ADJ_1' : 3,
    'last_ADJ_2' : -1,
    }
KB.update(Counts)

class Texture(FuzzyMeaning):
    Abbrevs = {}
    Names = {}
    Meanings = {}
    ViewPosition = Front
    def __init__(self,n,a,m,narrow=None):
        FuzzyMeaning.__init__(self,n,a,m,self.Abbrevs,self.Names,self.Meanings,narrow)

Rose = Texture('Rose', 'r', 'rose_ADJ_1')
Wood = Texture('Wood', 'w', 'wooden_ADJ_1')
Grass = Texture('Grass', 'g', 'grassy_ADJ_1')
Cement = Texture('Cement', 'c', 'cement_N_1')
BlueTile = Texture('BlueTile', 't', 'blue_ADJ_1')
Brick = Texture('Brick', 'b', 'brick_N_1')
Stone = Texture('Stone','s', 'stone_ADJ_1')
Honeycomb = Texture('Honeycomb', 'h', 'honeycomb_N_1')
Flooring = Texture('Flooring', Texture.Abbrevs.values(), 'flooring_N_2')
Gray = Texture('Gray', [Cement,Stone], 'gray_ADJ_1', Cement)
Greenish = Texture('Greenish', [Grass,Honeycomb], 'green_ADJ_1', Grass)
Brown = Texture('Brown', [Brick,Wood], 'brown_ADJ_1', Wood)
Dark = Texture('Dark', [Stone,BlueTile,Wood,Brick], 'dark_ADJ_2', Stone)

Textures = {
    'flower_N_1' : Rose,
    'flower_N_2' : Rose,
    'flowered_ADJ_1' : Rose,
    'pink_ADJ_1' : Rose,
    'pink_N_1' : Rose,
    'rose_ADJ_1' : Rose,
    'orange_ADJ_1' : Wood,
    'wood_ADJ_1' : Wood,
    'wood_N_1' : Wood,
    'wooden_ADJ_1' : Wood,
    'wooden_N_1' : Wood,
    'brown_ADJ_1' : Brown,
    'green_ADJ_1' : Greenish,
    'grass_ADJ_1' : Grass,
    'grass_N_1' : Grass,
    'grassy_ADJ_1' : Grass,
    'bare_ADJ_4' : Cement,
    'cement_ADJ_1' : Cement,
    'cement_N_1' : Cement,
    'concrete_ADJ_2' : Cement,
    'plain_ADJ_3' : Cement,
    'white_ADJ_4' : Cement,
    'gray_ADJ_1' : Gray,
    'grey_N_1' : Gray,
    'blue_ADJ_1' : BlueTile,
    'brick_ADJ_1' : Brick,
    'brick_N_1' : Brick,
    'red_ADJ_1' : Brick,
    'black_ADJ_1' : Stone,
    'black_N_1' : Stone,
    'rock_ADJ_2' : Stone,
    'stone_ADJ_1' : Stone,
    'stone_N_2' : Stone,
    'hexagon_N_1' : Honeycomb,
    'hexagonal_ADJ_1' : Honeycomb,
    'honeycomb_N_1' : Honeycomb,
    'octagon_ADJ_1' : Honeycomb,
    'octagon_N_1' : Honeycomb,
    'olive_ADJ_1' : Honeycomb,
    'yellow_ADJ_1' : Honeycomb,
    'carpet_ADJ_1' : Flooring,
    'carpet_N_1' : Flooring,
    'carpeted_ADJ_1' : Flooring,
    'floor_N_1' : Flooring,
    'floored_ADJ_1' : Flooring,
    'flooring_N_1' : Flooring,
    'flooring_N_1' : Flooring,
    'tile_N_1' : Flooring,
    'tiled_ADJ_1' : Flooring,
    'dark_ADJ_2' : Dark,
    }
KB.update(Textures)

class Picture(Meaning):
    Abbrevs = {}
    Names = {}
    Meanings = {}
    ViewPosition = FrontRight
    def __init__(self,n,a,m): Meaning.__init__(self,n,a,m,self.Abbrevs,self.Names,self.Meanings)

Butterfly = Picture('Butterfly', '8', 'butterfly_N_1')
Eiffel = Picture('Eiffel', '7', 'eiffel_N_1')
Fish = Picture('Fish', '6', 'fish_N_1')
Pic = Picture('Pic', Picture.Abbrevs.values(), 'picture_N_2')

Pictures = {
    'butterfly_ADJ_1' : Butterfly,
    'butterfly_N_1' : Butterfly,
    'eiffel_ADJ_1' : Eiffel,
    'eiffel_N_1' : Eiffel,
    'tower_ADJ_1' : Eiffel,
    'tower_N_1' : Eiffel,
    'eiffel tower_N_1' : Eiffel,
    'fish_ADJ_1' : Fish,
    'fish_N_1' : Fish,
    'pic_N_2' : Pic,
    'picture_N_2' : Pic,
    'hanging_N_1' : Pic,
    }
KB.update(Pictures)

class Object(FuzzyMeaning):
    Abbrevs = {}
    Names = {}
    Meanings = {}
    ViewPosition = At
    def __init__(self,n,a,m,narrow=None):
        FuzzyMeaning.__init__(self,n,a,m,self.Abbrevs,self.Names,self.Meanings,narrow)

Chair = Object('Chair', 'C', 'straight chair_N_1')
Sofa = Object('Sofa', 'S', 'sofa_N_1')
Barstool = Object('Barstool', 'B', 'stool_N_1')
Hatrack = Object('Hatrack', 'H', 'hatrack_N_1')
Easel = Object('Easel', 'E', 'easel_N_1')
Lamp = Object('Lamp', 'L', 'lamp_N_1')
Furniture = Object('Furniture', Object.Abbrevs.values(), 'furniture_N_1')
Seat = Object('Seat', [Chair,Barstool,Sofa], 'seat_N_3')
GenChair = Object('GenChair', [Chair,Sofa], 'chair_N_1', Chair)
Empty = Object('Empty', 'O', 'empty_ADJ_1')

Objects = {
    'straight chair_N_1' : Chair,
    'chair_N_1' : GenChair,
    'stool_N_1' : Barstool,
    'bench_N_1' : Sofa,
    'sofa_N_1' : Sofa,
    'hatrack_N_1' : Hatrack,
    'coatrack_N_1' : Hatrack,
    'coat_N_1' : Hatrack,
    'rack_N_1' : Hatrack,
    'easel_N_1' : Easel,
    'lamp_N_1' : Lamp,
    'lamp_N_2' : Lamp,
    'pole_N_1' : Lamp,
    'furniture_N_1' : Furniture,
    'object_N_1' : Furniture,
    'empty_ADJ_1' : Empty,
    'vacant_ADJ_2' : Empty,
    'nothing_N_1' : Empty,
    }
KB.update(Objects)

class Structure(Meaning):
    Abbrevs = {}
    Names = {}
    Meanings = {}
    def __init__(self,n,a,m): Meaning.__init__(self,n,a,m,self.Abbrevs,self.Names,self.Meanings)

class LinearStructure(Structure):
    ViewPosition = Front

Wall = LinearStructure('Wall', 'q', 'wall_N_1')
Hall = LinearStructure('Hall', 'z', 'hall_N_1')
End = LinearStructure('End', 'x', 'end_N_1')
Open = LinearStructure('Open', '_', 'open_N_1')
Path = LinearStructure('Path', Flooring.abbr+'\|', 'path_N_2')
PathDir = LinearStructure('PathDir', '\^', 'way_N_6')
Segment = LinearStructure('Segment', '!', 'segment_N_1')

class AreaStructure(Structure):
    ViewPosition = At

Intersection = AreaStructure('Intersection', '\+', 'intersection_N_2')
DeadEnd = AreaStructure('DeadEnd', '\!', 'dead end_N_1')
Block = AreaStructure('Block', '\:', 'block_N_2')
Position = AreaStructure('Position', '\.', 'position_N_1')
T_Int = AreaStructure('T_Int','T','t_N_5')
Corner = AreaStructure('Corner', '^', 'corner_N_4')
Middle = AreaStructure('Middle', 'm', 'middle_N_1')
TopoPlace = AreaStructure('TopoPlace', 'p', 'place_N_1')
Structures = {
    'back_ADV_1' : Back,
    'block_N_2' : Block,
    'room_N_1' : Block,
    'segment_N_1' : Segment,
    'corner_N_4' : Corner,
    'l_N_5' : Corner,
    'dead end_N_1' : DeadEnd,
    'end_N_1' : End,
    'end_V_1' : End,
    'branch_V_3' : Intersection,
    'cross_V_1' : Intersection,
    'cross_N_4' : Intersection,
    'intersect_V_1' : Intersection,
    'intersection_N_2' : Intersection,
    'open_V_8' : Intersection,
    'opening_N_1' : Intersection,
    'meet_V_3' : Intersection,
    'middle_N_1' : Middle,
    'alley_N_1' : Path,
    'carpet_N_1' : Path,
    'corridor_N_1' : Path,
    'exit_N_1' : Path,
    'floor_N_1' : Path,
    'path_N_3' : Path,
    'section_N_4' : Path,
    'direction_N_1' : PathDir,
    'portion_N_1' : PathDir,
    'way_N_6' : PathDir,
    'place_N_1' : TopoPlace,
    'position_N_1' : Position,
    't_N_5' : T_Int,
    'wall_N_1' : Wall,
    }
KB.update(Structures)
Region = 'Region'

Short = 'Short'
Long = 'Long'
Winding  = 'Winding'
Structurals = {
    'short_ADJ_2' : Short,
    'shorter_ADJ_1' : Short,
    'long_ADJ_2' : Long,
    'longer_ADJ_1' : Long,
    'winding_ADJ_1' : Winding,
    }
KB.update(Structurals)

Near = '0:2'
Far = '1:'
Immediate = '0'
Reldists = {
    'immmediate_ADJ_1' : Immediate,
    'near_ADJ_1' : Near,
    'far_ADJ_1' : Far,
    'other_ADJ_1' : Far,
    'opposite_ADJ_1' : Far,
    'other_ADJ_1' : Far,
    'farther_ADJ_1' : Far,
    'farthest_ADJ_1' : Far,
    }
KB.update(Reldists)

Defaults = {
    'Path' : Path,
    'Obj' : Furniture,
    'Picture' : Pic,
    'Struct' : Intersection,
    'Region' : Path,
    'Pathdir' : PathDir,
    'Boolean' : True,
    }
