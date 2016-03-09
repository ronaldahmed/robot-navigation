#!/usr/bin/env python
import random,os,sys,time,cPickle

from nltk.parser import AbstractParser
from nltk.parser.chunk import RegexpChunkParser,RegexpChunkParserRule,ChunkRule,ChinkRule
from nltk.parser.probabilistic import ViterbiPCFGParser
from nltk.token import Token
from nltk.tree import Tree
import nltk.corpus, nltk.tagger.brill

from Sense import TrainSenseTagger
from HashCFG import cvTrainPCFG
from DirectionCorpus import DirectionCorpusReader, constructItemRegexp, constructSetOrRegexp
from ParseStatistics import parseTestSet,parseStats,parse_stats,print_parse_summary
from Utility import logger
from SubjectLogs.SubjectGroups import *

__modulepath__ = os.path.abspath(os.path.dirname(__file__))

#try:
#    import pychecker.checker
#except ImportError: pass

class DirectionParser(AbstractParser):
    #line_rule = ChunkRule('<.+>+\n','Chunk lines')
    chunkall_rule = ChunkRule('<.+>+','Chunk everything')
    sentence_rule = ChinkRule(r'<\.+>','Chink on sentence-ending punctuation') # <\.>
    punct_rule = ChinkRule(r'<[\.\,\?\!\;]+>','Chink on all extraword punctuation') # <[\.\,\?\!\;]>
    punct_chunker = RegexpChunkParser([chunkall_rule,punct_rule], chunk_node='S', top_node='DIRECTIONS', SUBTOKENS='WORDS')
    period_chunker = RegexpChunkParser([chunkall_rule,sentence_rule], chunk_node='S', top_node='DIRECTIONS', SUBTOKENS='WORDS')
    null_chunker = RegexpChunkParser([chunkall_rule], chunk_node='S', top_node='DIRECTIONS', SUBTOKENS='WORDS')
    
    def __init__(self, Parsers, SenseTagger = None, POSTagger = None, collectStats=True):
        self.stats = collectStats
        self.Parsers = Parsers
        self.SenseTagger = SenseTagger
        self.POSTagger = POSTagger

    def tagPOS(self,text):
        if 'SUBTOKENS' not in text: text['SUBTOKENS'] = text['WORDS']
        if 'WORDS' not in text: text['WORDS'] = text['SUBTOKENS']
        self.POSTagger.tag(text)
        for tok in text['SUBTOKENS']:
            tok['TAG'] = tok['POS']
            del tok['POS']

    def parse(self,Text,
              interactive=0, trace=1, draw=0, print_parses=1, cumStats=None, stats=True,
              chunker=period_chunker, trueTree=None):
        if type(Text) is str:
            Text = DirectionCorpusReader._ws_reader.read_token(Text.lower())
        if self.POSTagger: self.tagPOS(Text)
        results = self.parseToken(Text,interactive,trace,draw,print_parses,cumStats,chunker,trueTree)
        if not stats and self.stats: return results.parse_list
        else: return results

    def unigramTag(self, dirStats, tagged):
        from nltk.stemmer.porter import PorterStemmer
        for t in tagged['SUBTOKENS']: PorterStemmer().stem(t)
        self.SenseTagger.tag(tagged)
        print 'unigramTag: tagged --',tagged
        TagString = '[???'
        for t in tagged['SUBTOKENS']:
            try:
                TagString += ' ['+t['SENSE']+' '+t['TEXT']+']'
            except KeyError:
                print "Couldn't find Sense tag for", t, 'in', tagged
                TagString += '[??? ' + +t['TEXT']+']'
        TagString += ']'
        dirStats.parse_list[-1] = TagString
        print 'Tagged:', TagString
        return None

    def parseToken(self,text,
                   interactive=0,trace=1,draw=0,print_parses=1,cumStats=None,chunker=None,trueTree=None):       
        if chunker == None: chunker = self.period_chunker
        if self.stats:
            dirStats = parseStats()
        else:
            Parses = []
        chunker.parse(text)
        for sent in text['TREE']:
            if not isinstance(sent,Tree): continue
            sentToken = Token(WORDS=sent.leaves(),SUBTOKENS=sent.leaves())
            print 'parsing',sentToken
            if self.stats:
                parse_stats(self.Parsers, sentToken, dirStats, trace, trueTree)
                print_parse_summary(self.Parsers, dirStats, interactive, draw, print_parses)
                # Check for empty parse
                if (dirStats.parse_list == [] or not isinstance(dirStats.parse_list[-1][0],Tree)):
                    chunker2 = None
                    if chunker != self.period_chunker and sent.count('.')>1: chunker2 = self.period_chunker
                    elif chunker != self.punct_chunker and sent.count('.')<=1: chunker2 = self.punct_chunker
                    else: chunker2 = self.unigramTag(dirStats, sentToken)
                    if chunker2:
                        if __debug__: print 'No parse, retry',chunker2
                        dirStats = self.parseToken(sentToken,interactive,trace,draw,print_parses,dirStats,chunker2,trueTree)
                 ###elif __debug__: print 'Good parse, no retry',dirStats.parse_list
            else:
                parse = self.Parsers[0].get_parse_list(sentToken)
                if parse: Parses.append(parse[0])
        if self.stats:
            if cumStats:
                cumStats += dirStats.sum()
                return cumStats
            else: return dirStats.sum()
        else:
            return Parses

def testParses(Parser):
    t = time.time()
    for TestString in [
        'go forward',
        'go forward four segments',
        'go forward four xahsdas',
        'turn left at the lamp',
        'Go forward three spaces, past the chair.  This is position 6',
        'Go down the hall to the end',
        'Turn left',
        'The perpendicualr hall will have bvlue floring',
        #'You should see a chair in front of you',
        #'Go three spaces past the chair',
        #'This is position 6',
        #'Turn left.  You should see a chair in front of you',
        #'Turn left.  You should see a chair in front of you.  Go three spaces past the chair',
        'Turn left.  You should see a chair in front of you.  Go forward three spaces, past the chair.  This is position 6',
        #'You should see a chair in front of you.  Go three spaces past the chair.  This is position 6',
        'Go down the blue-tiled hallway to the corner. turn left',
        'go forward to the easel',
        'turn left and go forward to the easel'
        'Go down the blue-tiled hallway to the corner. turn left. go forward to the easel',
        ]:
        Parser.parse(TestString)
    print 'All parses took',time.time()-t,'seconds.'

def timeSort(files,prefix='CleanDirs/Cleaned-'):
    import os.path
    files.sort()
    #lambda a,b: cmp(os.path.getmtime('Directions/FullTrees/FullTree-'+a[len(prefix):]),
    #                os.path.getmtime('Directions/FullTrees/FullTree-'+b[len(prefix):])))
    return [os.path.basename(file).split('-')[1] for file in files]

def analyze():
    import pstats
    stats = pstats.Stats("parse.prof")
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats(20)
    stats.sort_stats('cumulative').print_stats(20)
    stats.print_callers()

def nullParse(text,instructID): return (text,None,'')

def readCorrParse(text, instructID, path='Directions/CorrFullTrees/'):
    parses = open(path+'FullTree-'+instructID).read()
    logger.info("%r",parses)
    return parses,None,''

def readUncorrParse(text, instructID):
    return readCorrParse(text, instructID, 'Directions/FullTrees/')

def parseInstruction(instructID):
    return DirectionCorpusReader(constructItemRegexp(Directors,Maps)).parseInstruction(
        getDirParser(Directors, Maps, collectStats=False), instructID, saveParses=True, frames=True)


PcfgFileName = os.path.join(__modulepath__, 'pcfg.pyc')
SenseTaggerFileName = os.path.join(__modulepath__, 'sensetagger.pyc')
POSTaggerFileName = os.path.join(__modulepath__, 'tagger.pyc')
usePOSTagger = True
collectStats = False

def getPOSTagger(usePOSTagger=True, POSTaggerFile=POSTaggerFileName):
    if not usePOSTagger: return None
    if POSTaggerFile:
        try:
            return cPickle.load(open(POSTaggerFile))
        except KeyboardInterrupt:
            raise
        except:
            print >>sys.stderr, "Can't load POS Tagger, training from scratch."
    POSTagger = nltk.tagger.brill.test(numFiles=300,max_rules=100,min_score=2)
    cPickle.dump(POSTagger,open(POSTaggerFile,'w'))
    return POSTagger

def trainPCFG(Directors, Maps, PcfgFile=PcfgFileName, SenseTaggerFile=SenseTaggerFileName,
              cv=0.1, Starts=[], mapversions='[01]', Lexicon=''):
    corpus_regexp = constructItemRegexp(Directors, Maps, starts=Starts, mapversions=mapversions)
    if Lexicon: corpus_regexp = constructSetOrRegexp([corpus_regexp, Lexicon])
    Directions = DirectionCorpusReader(corpus_regexp)
    Pcfg,SenseTagger,TestSet = cvTrainPCFG(Directions, saveParses=0, StartSymbol = 'S',
                                           Group='CorrFullDirTrees', cv=cv, parseTest=doParses)
    if __debug__: print Pcfg
    if PcfgFile:
        cPickle.dump(Pcfg,open(PcfgFile,'w'))
        cPickle.dump(SenseTagger,open(SenseTaggerFile,'w'))
    return Pcfg,SenseTagger,TestSet

def getPCFG(Directors, Maps, PcfgFile=PcfgFileName, SenseTaggerFile=SenseTaggerFileName, cv=0.0, Starts=[]):
    if PcfgFile and not cv:
        try:
            Pcfg = cPickle.load(open(PcfgFile))
            SenseTagger = cPickle.load(open(SenseTaggerFile))
            return Pcfg,SenseTagger,[]
        except KeyboardInterrupt:
            raise
        except:
            print >>sys.stderr, "Can't load Grammar, training from scratch."
    return trainPCFG(Directors,Maps,PcfgFile, SenseTaggerFile, cv, Starts)

def getDirParser(Directors, Maps,
                 usePOSTagger=usePOSTagger, POSTaggerFile=POSTaggerFileName,
                 PcfgFile=PcfgFileName, SenseTaggerFile=SenseTaggerFileName,
                 collectStats=collectStats, cv=0.0, Starts=[],
                 spellchecker=None):
    POSTagger = getPOSTagger(usePOSTagger, POSTaggerFile)
    Pcfg,SenseTagger,TestSet = getPCFG(Directors, Maps, PcfgFile, SenseTaggerFile, cv, Starts)
    Parsers = [ViterbiPCFGParser(Pcfg, guessUnknown=1, LEAF='TEXT', spellchecker=spellchecker)]
    DirParser = DirectionParser(Parsers,SenseTagger,POSTagger,collectStats)
    try:
        import psyco
        psyco.full()
    except: print 'Psyco optimizer not installed'
    if cv: return DirParser, TestSet
    else: return DirParser

#doParses = 'Profile'
#doParses = 'TestSet'
doParses = 'CommandLine'
#doParses = None

def parse3From12():
    PcfgFileName = 'Corpus1+Corpus2-12-Corrected.pcfg'
    Directors= Directors1+Directors2

    try: nltk.corpus.set_basedir(system_corpora)
    except: system_corpora=nltk.corpus.get_basedir()
    
    logger.initLogger('ParseDirections',LogDir='MarcoLogs')
    import enchant
    from Sense import Lexicon
    spellchecker = enchant.DictWithPWL('en_US', Lexicon)
    DirParser = getDirParser(Directors, Maps, usePOSTagger, POSTaggerFileName,
                             PcfgFileName, SenseTaggerFileName, collectStats,
                             spellchecker=spellchecker)
    Directions = DirectionCorpusReader(constructItemRegexp(Directors3,Maps,mapversions='[01]'))
    parseTestSet(DirParser, Directions, list(Directions.items('CleanDirs')), 1)

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]:
        doParses = 'CommandLine'

    try: nltk.corpus.set_basedir(system_corpora)
    except: system_corpora=nltk.corpus.get_basedir()
    
    logger.initLogger('ParseDirections',LogDir='MarcoLogs')
    import enchant
    from Sense import Lexicon
    spellchecker = enchant.DictWithPWL('en_US', Lexicon)

    DirParser = getDirParser(Directors, Maps, usePOSTagger, POSTaggerFileName,
                             PcfgFileName, SenseTaggerFileName, collectStats,
                             spellchecker=spellchecker)
    
    if doParses == 'Profile':
        import profile
        profile.run('testParses(DirParser)','parse.prof')
    elif doParses == 'TestSet':
        Directions = DirectionCorpusReader(constructItemRegexp(Directors,Maps))
        parseTestSet(DirParser,Directions,TestSet,1)
    elif doParses == 'CommandLine':
        if len(sys.argv)>1 and sys.argv[1]:
            Files = sys.argv[1]
        else:
            Files = constructItemRegexp(Directors,Maps,mapversions='[01]')
        print 'Parsing', Files
        Directions = DirectionCorpusReader(Files)
        TestSet=list(Directions.items('CleanDirs'))
        #random.shuffle(TestSet);
        #timeSort(TestSet)
        parseTestSet(DirParser,Directions,TestSet,1)
