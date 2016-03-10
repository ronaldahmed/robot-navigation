## Automatically adapted for numpy.oldnumeric May 17, 2011 by -c

import os, sys, re, types
from nltk.chktype import chktype as _chktype
from nltk.chktype import classeq as _classeq
from nltk.corpus import SimpleCorpusReader,set_basedir
from nltk.token import Token
from nltk.tokenizer import TokenizerI, RegexpTokenizer
from nltk.tokenreader import TokenizerBasedTokenReader
from nltk.tokenreader.tokenizerbased import NewlineSeparatedTokenReader
from nltk.tokenreader.treebank import TreebankTaggedTokenReader, TreebankTokenReader
from nltk.tree import Tree

from Utility import logger, lstail

def constructSetOrRegexp(List):
    if List: return '('+'|'.join(List)+')'
    else: return '.*'

def constructItemRegexp(directors=[],maps=[],mapversions='0',starts=[],dests=[],instances='\d'):
    return '.*'+'_'.join([
        constructSetOrRegexp(directors),
        constructSetOrRegexp(maps)+mapversions,
        constructSetOrRegexp(starts),
        constructSetOrRegexp(dests),
        'Dirs_'+instances+'.txt$'
        ])

class RegexpTokenReader(TokenizerBasedTokenReader):
    """
    A token reader that reads in tokens according to a provided regular
    expression.

        >>> reader = RegexpTokenReader(SUBTOKENS='WORDS')
        >>> print reader.read_tokens('tokens separated, by spaces & punctuation.')
        [<tokens>, <separated>, <,>,<by>, <spaces>,<.>]
    """
    def __init__(self, regexp,**property_names):
        tokenizer = RegexpTokenizer(regexp,**property_names)
        TokenizerBasedTokenReader.__init__(self, tokenizer)

class DirectionCorpusReader(SimpleCorpusReader):
    """
    A corpus reader implementation for the Route Direction Treebank.
    """
    def __init__(self, item_regexp='.*\.txt$'):
        print 'Creating direction corpus for', item_regexp
        if not item_regexp.startswith('.*'): item_regexp = '.*'+item_regexp
        groups = [
            ('RawDirs', r'^'+item_regexp[2:]),
            ('CleanDirs', r'CleanDirs/Cleaned-'+item_regexp),
            ('DirTrees', r'Trees/Tree-'+item_regexp),
            ('FullDirTrees', r'FullTrees/FullTree-'+item_regexp),
            ('CorrFullDirTrees', r'CorrFullTrees/FullTree-'+item_regexp),
            ('ContentFrames', r'ContentFrames/ContentFrame-'+item_regexp),
            ]
        SimpleCorpusReader.__init__(self, 'Directions', 'Directions/',
                                    item_regexp,groups,
                                    token_reader=self._token_reader)
        set_basedir(os.getcwd())
        self._tag_reader.parse_iter = self.parse_iter
    
    _tag_reader = TreebankTokenReader(SUBTOKENS='WORDS')
    _line_reader = NewlineSeparatedTokenReader(SUBTOKENS='S')
    _ws_reader = RegexpTokenReader(r'[a-zA-Z0-9]+|[^a-zA-Z0-9\s]+',SUBTOKENS='WORDS')
    _sss_reader = None
    #WhitespaceSeparatedTokenReader(SUBTOKENS='WORDS')
    
    #////////////////////////////////////////////////////////////
    #// Data access (items)
    #///////////////////////////////////////////////////////////
    def parseInstruction(self, parser, instructID, saveParses=0, frames=False, trueTree=None):
        trees = parser.parse(self.read('CleanDirs/Cleaned-'+instructID), cumStats=None, stats=False, trueTree=trueTree)
        if saveParses:
            print 'parseTestSet',trees
            saveParse(trees,instructID)
            if frames:
                from Sense import saveFrame,trees2frames
                saveFrame(trees2frames(trees),instructID)
        return trees
    
    def readDirectionTree(self,item,saveParses=0,verbose=0,latex=0):
        if verbose: sys.stdout.write(item+'\n')
        else: sys.stdout.write('.')
        sys.stdout.flush()
        DirTrees = self.read(item)
        if saveParses: saveParse(DirTrees[0],item,latex=latex)
        return DirTrees
    
    def read(self, item, *reader_args, **reader_kwargs):
        source = '%s/%s' % (self._name, item)
        text = self.raw_read(item)
        reader = self._token_reader(item)
        if item.startswith('CleanDirs') or '/' not in item: text = text.lower()
        if 'FullTree' in item:
            return reader.read_tokens(text, source=source,
                                      *reader_args, **reader_kwargs)
        return reader.read_token(text, source=source,
                                 *reader_args, **reader_kwargs)
    
    def _token_reader(self, item):
        if '/' not in item: #('RawDirs')
            return self._ws_reader
        elif item.startswith('CleanDirs'):
            return self._ws_reader
        elif item.startswith('Trees'):
            return self._tag_reader
        elif item.startswith('FullTrees'):
            return self._tag_reader
        elif item.startswith('CorrFullTrees'):
            return self._tag_reader
        elif item.startswith('ContentFrames'):
            return self._sss_reader
        else:
            raise ValueError, 'Unknown item %r' % (item,)
        
    
    def parse_iter(s, leafparser=None):
        pos = 0
        parens='[]'
        _l = parens[0]
        _r = parens[1]
        _ql = '\\'+parens[0]
        _qr = '\\'+parens[1]
        SPACE = re.compile(r'\s*')
        WORD = re.compile(r'\s*([^\s'+_ql+_qr+']*)\s*')
        
        while pos < len(s):
            try:
                treetok, pos = DirectionCorpusReader._parse(s, pos, leafparser,SPACE,WORD,_l,_r)
                yield treetok
            except:
                raise
                raise ValueError('Bad treebank tree')
        # Check that we made it to the end of the string.
        if pos != len(s): raise ValueError('Bad treebank tree')
    parse_iter = staticmethod(parse_iter)
    
    def _parse(s, pos, leafparser,SPACE,WORD,_l,_r):
        # Skip any initial whitespace
        pos = SPACE.match(s, pos).end()
        
        stack = []
        while pos < len(s):
            # Beginning of a tree/subtree.
            if s[pos] == _l:
                match = WORD.match(s, pos+1)
                stack.append(Tree(match.group(1), []))
                pos = match.end()
            # End of a tree/subtree.
            elif s[pos] == _r:
                pos = SPACE.match(s, pos+1).end()
                if len(stack) == 1: return stack[0], pos
                stack[-2].append(stack[-1])
                stack.pop()
            # Leaf token.
            else:
                match = WORD.match(s, pos)
                if leafparser is None: leaf = match.group(1)
                else: leaf = leafparser(match.group(1), (pos, match.end(1)))
                stack[-1].append(leaf)
                pos = match.end()
        raise ValueError, 'mismatched parens'
    _parse = staticmethod(_parse)

def reprTree(tree):
    str = '['+tree.node
    for child in tree:
        str += ' '+reprDirs(child)
    return str+']'

def reprDirs(Dirs):
    """
    @return: A concise string representation of this tree.
    @rtype: C{string}
    """
    if isinstance(Dirs,Token):
        if Dirs.has_key('TREE'):
            return reprTree(Dirs['TREE'])
        else: return Dirs['TEXT']
    elif isinstance(Dirs,Tree):
        return reprTree(Dirs)
    elif isinstance(Dirs,list): return '[??? '+(' '.join(Dirs))+']\n'
    else: return repr(Dirs)

def printDirs(Dirs, margin=70, indent=0):
    """
    @return: A pretty-printed string representation of this tree.
    @rtype: C{string}
    @param margin: The right margin at which to do line-wrapping.
    @type margin: C{int}
    @param indent: The indentation level at which printing
        begins.  This number is used to decide how far to indent
        subsequent lines.
    @type indent: C{int}
    """
    assert _chktype(1, margin, types.IntType)
    assert _chktype(2, indent, types.IntType)
    #    return repr(Dirs)
    if (isinstance(Dirs,str) or isinstance(Dirs,tuple)
        or (isinstance(Dirs,list) and not isinstance(Dirs,Tree))):
        return '\n%s%s\n' % (' '*(indent),Dirs)
    rep = reprDirs(Dirs)
    if len(rep)+indent < margin:
        if indent: return rep
        else: return rep+'\n'
    
    if isinstance(Dirs,Token) and Dirs.has_key('TREE'): tree=Dirs['TREE']
    else: tree = Dirs
    s = ['[',tree.node]
    for child in tree:
        if isinstance(child, Tree):
            s.extend(['\n',' '*(indent+2),printDirs(child,margin, indent+2)])
        elif isinstance(child, Token):
            s.extend([' ',child['TEXT']])
        else: s.extend(['\n',' '*(indent),str(child)])
    s.append(']')
    if indent == 0: s.append('\n')
    return ''.join(s)

def saveParse(trees,filename,directory='Directions/FullTrees/',prefix='FullTree-',latex=0):
    """
    """
    ## Split off filename and replace prefix, if any
    filename = prefix + filename.split('/')[-1].split('-')[-1]
    fullpath = os.path.join(directory,filename)
    try: os.rename(fullpath,fullpath+'~')
    except OSError: pass
    parsefile = file(fullpath,'w')
    parsefile.write('\n'.join([printDirs(tree) for tree in trees]))
    parsefile.write('\n')
    parsefile.close()
    if latex:
        latexfile = file(fullpath+'.tex','w')
        latexfile.write('\n'.join([tree.latex_qtree() for tree in trees]))
        latexfile.write('\n')
        latexfile.close()

def generateLatex(Corpus):
    for item in Corpus.items('CorrFullDirTrees'):
        for DirTree in Corpus.readDirectionTree(item,saveParses=1,latex=0):
            pass

def correlateRatingsTags(Ratings,Group='FullDirTrees'):
    import numpy.oldnumeric.mlab as MLab
    DirectionGivers = '(EDA|EMWC|KLS|KXP|TJS|WLH)'
    Routes = '.*'
    Envs = '.*'
    Suffix = 'Dirs_\d.txt$'
    Directions = DirectionCorpusReader('_'.join([DirectionGivers,Envs,Routes,Suffix]))

    CFDist = LearnCondDist(Directions, list(Directions.items(Group)), Start='DIRECTIONS',verbose=1)
    TagOrder = [tag.symbol() for tag in CFDist.conditions()
                if not (tag.symbol().endswith('_P') or tag.symbol().endswith('_N') or tag.symbol().endswith('_V'))]
    TagOrder.sort()
    results = {}
    for item in Directions.items(Group):
        dirID = item.split('-')[1]
        if not Ratings.has_key(dirID):
            print 'Skipping', dirID;
            continue
        TagCounts = {}
        DirModel = LearnCondDist(Directions, [item], Start='DIRECTIONS')
        for nonterm in DirModel.conditions():
            tag = nonterm.symbol()
            #print nonterm, [DirModel[nonterm].count(s) for s in DirModel[nonterm].samples()]
            if tag in TagOrder:
                TagCounts[tag] = MLab.sum([DirModel[nonterm].count(s) for s in DirModel[nonterm].samples()])
        TraitList = [Ratings[dirID][2], Ratings[dirID][3], Ratings[dirID][4],
                     Ratings[dirID][5], Ratings[dirID][6]+Ratings[dirID][7]]
        for TagName in TagOrder:
            if TagCounts.has_key(TagName):
                TraitList.append(TagCounts[TagName])
            else: TraitList.append(0)
        results[dirID] = TraitList
        #print dirID, TraitList
    for k in Ratings.keys():
        if not results.has_key(k):
            results[k] = [0] * (len(TagOrder)+5) # Number of ratings used
    return results, TagOrder

def readCleaned(instructions,instructID):
    try:
        text = open('Directions/CleanDirs/Cleaned-'+instructID).read()
    except: text = ''
    if not text.strip(): # Does it have non-whitespace characters?
        return text,'EOFError',EOFError('Empty Instruction Set')
    for sent in text.split('.'): logger.info(sent)
    return text,None,''

def get_option_matches(option, logfiles):
    pipe = os.popen('grep -l %s %s' %(option, logfiles))
    matches = [match.split('-')[-1] for match in pipe]
    pipe.close()
    return matches

def print_option_matches(option, logfiles, print_text=True, print_matches=True):
    """
         >>> for option in [o for o in dir(Options.Options) if type(getattr(Options.Options,o)) == bool]:
         ...		print_option_matches(option, 'MarcoLogs/Statistics/Statistics-2007-03-18-02-49*.txt', False)
         """
    n = 4
    DefaultFiles =  lstail('MarcoLogs', re.compile('Follower-Results-%s-200.*-Col.txt'%'Default'), n)
    OptionFiles =  lstail('MarcoLogs', re.compile('Follower-Results-%s-200.*-Col.txt'%option), n)
    Sums = {'With' : 0, 'Without' : 0, 'All' : 0, 'Gains':0, 'Losses':0}
    Last = {}
    for instructID in get_option_matches(option, logfiles):
        instructID = instructID.strip()
        if print_matches: print instructID,
        Sums['All'] += 1
        for label, files in [('With',DefaultFiles),('Without',OptionFiles)]:
            pipe = os.popen('grep -c -h "%s\tSuccess" %s' %
                            (instructID, ' '.join(['MarcoLogs/'+f for f in files])))
            matches = pipe.read()
            pipe.close()
            successes = sum([int(m) for m in matches.split('\n') if m])
            Sums[label] += successes
            Last[label] = successes
            if print_matches: print label, option, successes, '\t',
        if print_matches: print
        if Last['With'] > Last['Without']: Sums['Gains'] +=1
        elif Last['With'] < Last['Without']: Sums['Losses'] +=1
        if print_text:
            match_file = open('Directions/CleanDirs/Cleaned-'+instructID)
            print match_file.read()
            match_file.close()
    n = float(n)
    print '\tTotals:', option, '\t with', Sums['With']/n, '\t without', Sums['Without']/n, 'of', Sums['All'],
    print '\t+'+str(Sums['Gains']),'-'+str(Sums['Losses'])
