#!/usr/bin/env python
import os, string

from nltk.probability import ConditionalFreqDist
from nltk.tagger import BackoffTagger,DefaultTagger,UnigramTagger,tagger_accuracy
#,TaggerI,SequentialTagger,NthOrderTagger,RegexpTagger
from nltk.tagger.brill import BrillTagger, FastBrillTaggerTrainer
from nltk.tagger.brill import SymmetricProximateTokensTemplate,ProximateTokensTemplate,ProximateTokensRule
from nltk.token import Token
from nltk.tree import Tree
from nltk.stemmer.porter import PorterStemmer
from nltk.featurestructure import FeatureStructure
from nltk_contrib import pywordnet

import enchant

from Senses import Senses, Senses2, Senses3
from DirectionCorpus import printDirs,constructItemRegexp,DirectionCorpusReader,saveParse
from Options import Options
from Utility import logger, lstail

pywordnet.setCacheCapacity(100)
Lexicon = 'Directions/Lexicon2.lex'
Corpus2 = True
if Corpus2: Senses.update(Senses2)
Corpus3 = True
if Corpus3: Senses.update(Senses3)

class ProximateSensesRule(ProximateTokensRule):
    PROPERTY_NAME = 'sense' # for printing.
    TAG='SENSE'
    def extract_property(token): # [staticmethod]
        """@return: The given token's C{SENSE} property."""
        return token['SENSE']
    extract_property = staticmethod(extract_property)

class ProximateStemsRule(ProximateTokensRule):
    PROPERTY_NAME = 'stem' # for printing.
    TAG='STEM'
    def extract_property(token): # [staticmethod]
        """@return: The given token's C{STEM} property."""
        return token['STEM']
    extract_property = staticmethod(extract_property)

class ProximateSensesTemplate(ProximateTokensTemplate):
    TAG='SENSE'

class SymmetricProximateSensesTemplate(SymmetricProximateTokensTemplate):
    def __init__(self, rule_class, *boundaries):
        self._ptt1 = ProximateSensesTemplate(rule_class, *boundaries)
        reversed = [(-e,-s) for (s,e) in boundaries]
        self._ptt2 = ProximateSensesTemplate(rule_class, *reversed)

class SurfaceSemanticsStructure(FeatureStructure):
    """
    A class of C{FeatureStructure} to represent surface semantics.
    """
    def __setitem__(self, name, value):
        if type(name) == str:
            self._features[name] = value
        elif type(name) == tuple:
            if len(name) == 1:
                self._features[name[0]] = value
            else:
                self._features[name[0]][name[1:]] = value
        else: raise TypeError

    def _repr(self, reentrances, reentrance_ids):
        if 'MEAN' in self._features: return "[MEAN="+repr(self['MEAN'])+"]"
        else: return FeatureStructure._repr(self, reentrances, reentrance_ids)

    def add(self,key,value):
        if key in self.feature_names():
            val = self[key]
            if isinstance(val,list) and isinstance(value,list):
                val.extend(value)
            elif isinstance(val,list):
                val.append(value)
            elif isinstance(value,list):
                value.append(val)
                self[key] = value
            else: self[key] = [val, value]
        else:
            self[key] = value

    def has_key(self,key): return self._features.has_key(key)

    def getindex(self,name):
        if not isinstance(self._features[name],FeatureStructure):
            return None
        childFeatures = self._features[name]._features
        if 'INDEX' in childFeatures:
            return self[name]['INDEX']
        for feature in childFeatures:
            index = self._features[name].getindex(feature)
            if index is not None: return index
        if '_' in name and name[-1] in string.digits:
            return int(name.split('_')[-1])
        return None

    def sorted_features(self):
        indices = [(self.getindex(name),name) for name in self._features if name != 'INDEX']
        indices.sort()
        return [name for (index,name) in indices]

    def copydict(self,d):
        self._features.update(d)

    def deepcopy(self, memo=None):
        """
        @return: a new copy of this surface semantics structure.
        @param memo: The memoization dictionary, which should
            typically be left unspecified.
        """
        # Check the memoization dictionary.
        if memo is None: memo = {}
        memo_copy = memo.get(id(self))
        if memo_copy is not None: return memo_copy

        # Create a new copy.  Do this *before* we fill out its
        # features, in case of cycles.
        newcopy = SurfaceSemanticsStructure()
        memo[id(self)] = newcopy
        features = newcopy._features

        # Fill out the features.
        for (fname, fval) in self._features.items():
            if isinstance(fval, FeatureStructure):
                features[fname] = fval.deepcopy(memo)
            else:
                features[fname] = fval

        return newcopy

    def parse(cls,s):
        """
        Same as FeatureStructure.parse, but a classmethod,
        so it will return the subclass.
        
        Convert a string representation of a feature structure (as
        displayed by repr) into a C{FeatureStructure}.  This parse
        imposes the following restrictions on the string
        representation:
          - Feature names cannot contain any of the following:
            whitespace, parentheses, quote marks, equals signs,
            dashes, and square brackets.
          - Only the following basic feature value are supported:
            strings, integers, variables, C{None}, and unquoted
            alphanumeric strings.
          - For reentrant values, the first mention must specify
            a reentrance identifier and a value; and any subsequent
            mentions must use arrows (C{'->'}) to reference the
            reentrance identifier.
        """
        try:
            value, position = cls._parse(s, 0, {})
        except ValueError, e:
            estr = ('Error parsing field structure\n\n    ' +
                    s + '\n    ' + ' '*e.args[1] + '^ ' +
                    'Expected %s\n' % e.args[0])
            raise ValueError, estr
        if position != len(s): raise ValueError()
        return value

    def _parse(cls, s, position=0, reentrances=None):
        """
        Same as FeatureStructure._parse, but a classmethod,
        so it will return the subclass.
        
        Helper function that parses a feature structure.
        @param s: The string to parse.
        @param position: The position in the string to start parsing.
        @param reentrances: A dictionary from reentrance ids to values.
        @return: A tuple (val, pos) of the feature structure created
            by parsing and the position where the parsed feature
            structure ends.
        """
        # A set of useful regular expressions (precompiled)
        _PARSE_RE = cls._PARSE_RE

        # Check that the string starts with an open bracket.
        if s[position] != '[': raise ValueError('open bracket', position)
        position += 1

        # If it's immediately followed by a close bracket, then just
        # return an empty feature structure.
        match = _PARSE_RE['bracket'].match(s, position)
        if match is not None: return cls(), match.end()

        # Build a list of the features defined by the structure.
        # Each feature has one of the three following forms:
        #     name = value
        #     name (id) = value
        #     name -> (target)
        features = {}
        while position < len(s):
            # Use these variables to hold info about the feature:
            name = id = target = val = None
            
            # Find the next feature's name.
            match = _PARSE_RE['name'].match(s, position)
            if match is None: raise ValueError('feature name', position)
            name = match.group(1)
            position = match.end()

            # Check for a reentrance link ("-> (target)")
            match = _PARSE_RE['reentrance'].match(s, position)
            if match is not None:
                position = match.end()
                match = _PARSE_RE['ident'].match(s, position)
                if match is None: raise ValueError('identifier', position)
                target = match.group(1)
                position = match.end()
                try: features[name] = reentrances[target]
                except: raise ValueError('bound identifier', position)

            # If it's not a reentrance link, it must be an assignment.
            else:
                match = _PARSE_RE['assign'].match(s, position)
                if match is None: raise ValueError('equals sign', position)
                position = match.end()

                # Find the feature's id (if specified)
                match = _PARSE_RE['ident'].match(s, position)
                if match is not None:
                    id = match.group(1)
                    if reentrances.has_key(id):
                        raise ValueError('new identifier', position+1)
                    position = match.end()
                
                val, position = cls._parseval(s, position, reentrances)
                features[name] = val
                if id is not None:
                    reentrances[id] = val

            # Check for a close bracket
            match = _PARSE_RE['bracket'].match(s, position)
            if match is not None:
                return cls(**features), match.end()

            # Otherwise, there should be a comma
            match = _PARSE_RE['comma'].match(s, position)
            if match is None: raise ValueError('comma', position)
            position = match.end()

        # We never saw a close bracket.
        raise ValueError('close bracket', position)

    def _parseval(cls, s, position, reentrances):
        """
        Same as FeatureStructure._parseval, but a classmethod,
        so it will return the subclass.
        
        Helper function that parses a feature value.  Currently
        supports: None, integers, variables, strings, nested feature
        structures.
        @param s: The string to parse.
        @param position: The position in the string to start parsing.
        @param reentrances: A dictionary from reentrance ids to values.
        @return: A tuple (val, pos) of the value created by parsing
            and the position where the parsed value ends.
        """
        # A set of useful regular expressions (precompiled)
        _PARSE_RE = cls._PARSE_RE

        # End of string (error)
        if position == len(s): raise ValueError('value', position)
        
        # String value
        if s[position] in "'\"":
            start = position
            quotemark = s[position:position+1]
            position += 1
            while 1:
                match = _PARSE_RE['stringmarker'].search(s, position)
                if not match: raise ValueError('close quote', position)
                position = match.end()
                if match.group() == '\\': position += 1
                elif match.group() == quotemark:
                    return eval(s[start:position]), position

        # Nested feature structure
        if s[position] == '[':
            return cls._parse(s, position, reentrances)

        # Variable
        match = _PARSE_RE['var'].match(s, position)
        if match is not None:
            return FeatureVariable.parse(match.group()), match.end()

        # None
        match = _PARSE_RE['none'].match(s, position)
        if match is not None:
            return None, match.end()

        # Integer value
        match = _PARSE_RE['int'].match(s, position)
        if match is not None:
            return int(match.group()), match.end()

        # Alphanumeric symbol (must be checked after integer)
        match = _PARSE_RE['symbol'].match(s, position)
        if match is not None:
            return match.group(), match.end()

        # We don't know how to parse this value.
        raise ValueError('value', position)

    _parseval=classmethod(_parseval)
    _parse=classmethod(_parse)
    parse=classmethod(parse)

class SSS(SurfaceSemanticsStructure):
    """Alias for SurfaceSemanticsStructure"""

def tree2frame(Dirs, index = 0, parent = ''):
    """
    @return: content frame representation of the surface semantics of the parse tree.
    @rtype: C{SurfaceSemanticsStructure}
    
    @return proposition name
    @rtype: C{str}
    
    @return index
    @rtype: C{int}
    """
    Frame = SurfaceSemanticsStructure()
    if isinstance(Dirs,Tree):
        Prop = Dirs.node.capitalize()
        hasSubTree = True in [isinstance(child,Tree) for child in Dirs]
    else: Prop = None
    if isinstance(Dirs,Tree) and hasSubTree:
        for i,child in enumerate(Dirs):
            value,prop,index = tree2frame(child,index+1,Dirs.node.capitalize())
            filed = False # Account for children with the same names
            if value and prop:
                prop_name = prop
                while not filed:
                    if not Frame.has_key(prop):
                        Frame[prop] = value
                        filed = True
                    else:
                        prop= prop_name+'_'+str(i)
            elif value:
                Frame1 = Frame.unify(value)
                if Frame1: Frame = Frame1
                else:
                    while not filed:
                        if not Frame.has_key('SubFrame'+'_'+str(index)):
                            Frame['SubFrame'+'_'+str(index)] = value
                            filed = True
    elif ((isinstance(Dirs,Tree) and not hasSubTree and Dirs)
          or isinstance(Dirs,Token)):
        index += 1
        if isinstance(Dirs,Token): token = Dirs
        if isinstance(Dirs,Tree):
            token = Token(TEXT=' '.join([child['TEXT'] for child in Dirs]))
            parent = Dirs.node.capitalize()
        Frame['TEXT'] = token['TEXT']
        Frame['MEAN'] = extractSurfaceSemantics(token,parent)
        Frame['INDEX']=index
    return Frame,Prop,index

def trees2frames(Trees):
    return [tree2frame(tree)[0] for tree in Trees]

def saveFrame(frame_list,filename,directory='Directions/ContentFrames/',prefix='ContentFrame-'):
    """
    @param parse_list: List of content frames
    @type parse_list: C{list}
    
    @param directory: name of the directory to save the parses into
    @type parse_list: C{str}
    """
    filename = prefix+filename.split('/')[-1].split('-')[-1]
    fullpath = os.path.join(directory,filename)
    if os.path.isfile(fullpath): os.rename(fullpath,fullpath+'~')
    file = open(fullpath,'w')
    file.write('\n'.join([repr(d) for d in frame_list]))
    file.write('\n')
    file.close()

def getPartOfSpeech(token,parent):
    POS = ''
    if token.has_key('SENSE') or parent: # My tags
        if token.has_key('SENSE'): Sense = token['SENSE']
        else: Sense = parent
        if Sense.endswith('_n') or Sense in ('Dist_unit', 'Struct_type'): POS='N'
        elif Sense.endswith('_v') and Sense != 'Aux_v': POS='V'
        elif Sense.endswith('_p'): POS='P'
        elif Sense in ('Appear','Count','Reldist','Structural','Order_adj', 'Obj_adj'): POS='ADJ'
        elif Sense in ('Dir','Adv'): POS='ADV'
        else: return Sense
    elif token.has_key('TAG'): # Penn Treebank
        if token['TAG'].startswith('NN'): POS='N'
        elif token['TAG'].startswith('VB'): POS='V'
        elif token['TAG'].startswith('JJ') or token['TAG'].startswith('CD'): POS='ADJ'
        elif token['TAG'].startswith('RB'): POS='ADV'
        elif token['TAG'].startswith('IN'): POS='P'
        else: return token['TAG']
    return POS

def findMissingSenses():
    for k,v in Senses.items():
        for pos,senseList in v.items():
            for s in senseList:
                try:
                    if pos!='see': pywordnet.getSense(k,pos,s-1)
                except (KeyError,TypeError),err: # Inflected form
                    logger.errror('Trying inflected form b/c of Error %s',err)
                    logger.error('%s',pywordnet.getSense(s[0],pos,s[1][0]-1))
                except: logger.error('Cannot find %s, %s, %s', k,pos,s)

def printSurfaceSemantics(text,POS,senses):
    if isinstance(senses,str): return '_'.join([text,POS,senses])
    return  '_'.join([text,POS,','.join([str(i) for i in senses])])

def splitSurfaceSemantics(sss_str):
    if '_' not in sss_str: return []
    sss_str = sss_str.replace('[','')
    sss_str = sss_str.replace(']','')
    text,POS,senses = sss_str.split('_')
    if '(' in senses: # Handle 'see' redirection
        text,senses = senses[2:-1].split('\', ')
    senses = senses.split(',')
    return text,POS,senses

def parseSurfaceSemantics(sss_str):
    if '_' not in sss_str: return []
    text,POS,senses = splitSurfaceSemantics(sss_str)
    try:
        return [pywordnet.getWord(text,POS).getSenses()[int(s)-1] for s in senses]
    except (IndexError,KeyError):
        sense = None
        for altPOS in ('N','V','ADJ','ADV'):
            if altPOS == POS: continue
            try:
                return [pywordnet.getWord(text,POS).getSenses()[int(s)-1] for s in senses]
            except (IndexError,KeyError): pass
        return []

def extractSurfaceSemantics(token,parent):
    global Senses
    POS=getPartOfSpeech(token,parent)
    tokenSenses = {}
    text = token['TEXT'].lower()
    default = token['TEXT'].upper()
    if POS in ['N', 'V', 'ADV', 'ADJ']:
        try: #Redo as test = foo while not tokenSensesword: try: foo ; except KeyError: foo = next foo
            tokenSenses = Senses[text]
        except KeyError:
            logger.warning('extractSurfaceSemantics : Text not in tagged senses: %s', text)
            try: 
                #logger.warning('extractSurfaceSemantics : Previously unseen word but in WordNet?: %s', text)
                # stringified range of possible senses without spaces
                tokenSenses = {POS : range(1,len(pywordnet.getWord(text,POS).getSenses())+1)}
            except KeyError:
                try:
                    logger.warning('extractSurfaceSemantics : Inflected version of WordNet word? %s', text)
                    if text.endswith('s'):
                        text = text[:-1]
                        tokenSenses = Senses[text]
                    else:
                        stemmer = PorterStemmer() # Update WordNetStemmer to NLTK 1.4 API
                        stemmer.stem(token)
                        text = token['STEM']
                        tokenSenses = Senses[text]
                except KeyError:
                    text = token['TEXT'].lower()
                    try:
                        logger.warning('extractSurfaceSemantics : Misspelling / typo of WordNet word? %s', text)
                        spellchecker = enchant.DictWithPWL('en_US', Lexicon)
                        s = ''
                        for s in spellchecker.suggest(text):
                            if s in Senses:
                                tokenSenses = Senses[s]
                                break
                        if not tokenSenses and spellchecker.suggest(text):
                            s = spellchecker.suggest(text)[0]
                            tokenSenses = {POS : range(1,len(pywordnet.getWord(s,POS).getSenses())+1)}
                        if s and Options.Spellcheck:
                            logger.warning('extractSurfaceSemantics : Found spelling correction %s for %s', s,text)
                            text = s
                        #logger.debug('*** extractSurfaceSemantics : Implement spelling correction. *** ')
                        #raise KeyError
                    except KeyError:
                        logger.error('extractSurfaceSemantics : Unknown token: %s', text)
                        return default
        # Handle experienced typos.
        if 'see' in tokenSenses:
            ### FIXME adding to dict for typos that are other words
            text = tokenSenses['see']
            try:
                tokenSenses = Senses[text]
            except: return default
        # Handle morphology variants that wordnet understands.
        elif isinstance(tokenSenses, tuple):
            text,tokenSenses[POS] = tokenSenses[POS]
        try:
            return '_'.join([text,POS,','.join([str(i) for i in tokenSenses[POS]])])
        except KeyError:
            #logger.warning('extractSurfaceSemantics : Expected POS %s for token %s, Got %s, Using %s',
            #            POS, token, tokenSenses.keys(), tokenSenses.keys()[0])
            if tokenSenses.keys():
                POS = token['POS'] = tokenSenses.keys()[0]
                return '_'.join([text,POS,','.join([str(i) for i in tokenSenses.values()[0]])])
        except Exception,e:
            logger.error('extractSurfaceSemantics: %s: Could not find sense %s for token %s',
                      e, POS, token) #tokenSenses, text
    return default

def invertConditionalFreqDist(CFDist):
    iCFDist =  ConditionalFreqDist()
    Stemmer=PorterStemmer()
    for cond in CFDist.conditions():
        for val in CFDist[cond].samples():
            sense = cond.split('_')[0] #Cut off any POS
            for tok in val:
                if type(tok) == str:
                    iCFDist[Stemmer.raw_stem(tok)].inc(sense,CFDist[cond].count(val))
    return iCFDist

def TrainSenseTagger(Pcfg,CFDist):
    logger.info("Training unigram tagger:")
    SenseUnigramTagger = UnigramTagger(TAG='SENSE',TEXT='STEM')
    #SenseUnigramTagger.train(taggedData)
    SenseUnigramTagger._freqdist = invertConditionalFreqDist(CFDist)
    SenseDefaultTagger = DefaultTagger('APPEAR', TAG='SENSE',TEXT='STEM')
    backoff = BackoffTagger([SenseUnigramTagger,SenseDefaultTagger], TAG='SENSE',TEXT='STEM')
    return backoff

#     # Brill tagger

#     templates = [
#         SymmetricProximateSensesTemplate(ProximateSensesRule, (1,1)),
#         SymmetricProximateSensesTemplate(ProximateSensesRule, (2,2)),
#         SymmetricProximateSensesTemplate(ProximateSensesRule, (1,2)),
#         SymmetricProximateSensesTemplate(ProximateSensesRule, (1,3)),
#         SymmetricProximateSensesTemplate(ProximateStemsRule, (1,1)),
#         SymmetricProximateSensesTemplate(ProximateStemsRule, (2,2)),
#         SymmetricProximateSensesTemplate(ProximateStemsRule, (1,2)),
#         SymmetricProximateSensesTemplate(ProximateStemsRule, (1,3)),
#         ProximateSensesTemplate(ProximateSensesRule, (-1, -1), (1,1)),
#         ProximateSensesTemplate(ProximateStemsRule, (-1, -1), (1,1)),
#         ]
#     trace = 3
#     trainer = FastBrillTaggerTrainer(backoff, templates, trace, TAG='SENSE')
#     #trainer = BrillTaggerTrainer(backoff, templates, trace, TAG='SENSE')
#     b = trainer.train(trainingData, max_rules=100, min_score=2)

def readCorrFrame(parses,instructID):
    CaughtError=None
    CaughtErrorTxt=''
    frames=[]
    for frame in open('Directions/ContentFrames/ContentFrame-'+instructID).readlines():
        frame = str(frame) #Escape
        if not frame or frame == '\n\n':
            return [],Failed,EOFError,'Empty instruction file'
        try:
            frames.append(SurfaceSemanticsStructure.parse(frame))
        except ValueError,e:
            CaughtErrorTxt = "Can't parse: " + str(e)
            logger.error("%s.",CaughtErrorTxt)
            if str(e).startswith("Error parsing field structure"):
                CaughtError = 'EOFError'
            else:
                CaughtError = 'ValueError'
    return frames,CaughtError,CaughtErrorTxt

def getSSS(instructID):
    if not instructID.endswith('txt'): instructID += '.txt'
    return readCorrFrame([],instructID)[0]

if __name__ == '__main__':
    logger.initLogger('Sense',LogDir='MarcoLogs')
    Directors = ['EDA','EMWC','KLS','KXP','TJS','WLH']
    Maps = ['Jelly','L','Grid']
    Corpus = DirectionCorpusReader(constructItemRegexp(Directors,Maps))
else: Corpus = None

def genCorrContentFrame(filename, Corpus=Corpus, TreePath='CorrFullTrees/'):
    if '-' in filename: instructionID = filename.split('-')[1]
    else: instructionID = filename
    print '\n',instructionID
    if not Corpus:
        Directors = ['EDA','EMWC','KLS','KXP','TJS','WLH']
        Maps = ['Jelly','L','Grid']
        Corpus = DirectionCorpusReader(constructItemRegexp(Directors,Maps))
    Trees=[tree['TREE'] for tree in Corpus.read(TreePath+'/FullTree-'+instructionID)]
    Frames = trees2frames(Trees)
    saveParse(Trees,instructionID,directory='Directions/'+TreePath)
    saveFrame(Frames,instructionID)
    for frame in Frames: print `frame`
    #for frame in readCorrFrame('',instructionID): print `frame`

def genUncorrContentFrames(Directors):
    import re
    Corpus = DirectionCorpusReader(constructItemRegexp(Directors, mapversions='[01]'))
    for filename in lstail('Directions/FullTrees', re.compile('^FullTree-.*.txt$')):
        try:
            genCorrContentFrame(filename, TreePath='FullTrees/')
        except ValueError:
            pass

def output_pwl(filename=Lexicon):
    spellchecker = enchant.DictWithPWL('en_US',filename)
    Valid = []
    Unknown = []
    for word in Senses:
        if spellchecker.check(word): Valid.append(word)
        else: Unknown.append(word)
    print 'Found:', Valid
    print 'Unknown:'
    Unmatched = []
    Matched = []
    for w in Unknown:
        suggestions = spellchecker.suggest(w)
        match = ''
        for s in suggestions:
            if spellchecker.pwl.check(s):
                match = s
                break
        print ' ', w, 'Match: "'+match+'"', suggestions
        if match: Matched.append((w,match))
        else: Unmatched.append(w)
    Matched.sort()
    Unmatched.sort()
    print 'Matched:'
    for M in Matched: print M
    print 'Unmatched', Unmatched
    WordList = Valid+Unmatched
    WordList.sort()
    return WordList

if __name__ == '__main__':
    for filename in file('Directions/CorrFullTrees/new.txt'): genCorrContentFrame(filename[:-1])
    filelist = file('Directions/CorrFullTrees/new.txt','a')
    filelist.write('\n')
    filelist.close()
    #for filename in os.listdir('Directions/CorrFullTrees/'):
    #    if filename.startswith("FullTree-") and filename.endswith(".txt"): genCorrContentFrame(filename)
    #pass
