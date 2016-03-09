import random
from nltk.chktype import chktype as _chktype
from nltk.chktype import classeq as _classeq
from nltk.cfg import CFG,PCFG,Nonterminal,CFGProduction, PCFGProduction
from nltk.probability import ConditionalFreqDist,ConditionalProbDist,FreqDist,MLEProbDist#,WittenBellProbDist
from nltk.token import Token
from nltk.tree import Tree
from Sense import TrainSenseTagger

def CountSubTree(CFDist,token):
    Output = []
    for child in token:
        if isinstance(child,Tree):
            Output.append(Nonterminal(child.node))
            CountSubTree(CFDist,child)
        elif isinstance(child,Token):
            text_token = child['TEXT'].lower()
            Output.append(text_token)
        else:
            print 'Unmatched token', token, child
    CFDist[token.node].inc(tuple(Output))

def LearnCondDist(Corpus,TrainSet,CFDist=None, saveParses=0, verbose=0):
    if not CFDist: CFDist = ConditionalFreqDist()
    for item in TrainSet:
        for sent in Corpus.readDirectionTree(item,saveParses,verbose):
            CountSubTree(CFDist,sent['TREE'])
    # Make 'S' the top of the tree
##    for cat in ('TRAVEL','TURN','DESC','NAME'):
##        for s in CFDist[cat].samples(): CFDist['S'].inc(s,CFDist[cat].count(s))
    print '\nRead in training tokens for Conditional Frequency:', CFDist
    return CFDist

class HashCFG(CFG):
    def __init__(self, start, productions):
        """
        Create a new context-free grammar, from the given start state
        and set of C{CFGProduction}s.
        
        @param start: The start symbol
        @type start: L{Nonterminal}
        @param productions: The list of productions that defines the grammar
        @type productions: C{list} of L{CFGProduction}
        """
        assert _chktype(1, start, Nonterminal)
        assert _chktype(2, productions, (CFGProduction,), [CFGProduction])
        self._start = start
        self._productions = tuple(productions)
        # Index of lhs nonterminals to rules
        self._index = {}
        # Reverse index of rhs tokens to rules
        self._rindex = {}
        # List of productions that have some terminals in the rhs
        self._lexicon_grammar = []
        # List of productions that have no terminals in the rhs
        self._nt_grammar = []
        for production,n in zip(self._productions,range(len(self._productions))):
            self._index.setdefault(production.lhs(),[])
            self._index[production.lhs()].append(n)
            nonterminals = 1
            for token in production.rhs():
                nonterminals = nonterminals and isinstance(token,Nonterminal)
                if self._rindex.has_key(token): self._rindex[token].append(n)
                else: self._rindex[token] = [n]
            if nonterminals: self._nt_grammar.append(n)
            else: self._lexicon_grammar.append(n)

    def productions(self):
        return self._productions

    def rlookup(self,token):
        try: return [self._productions[n] for n in self._rindex.get(token)]
        except: return []

    def lexicon_grammar(self):
        """Return all productions with rhs "tainted" with terminals"""
        return [self._productions[n] for n in self._lexicon_grammar]

    def nt_grammar(self):
        """Return all productions with pure nonterminal rhs"""
        return [self._productions[n] for n in self._nt_grammar]

    def print_lhs_counts(self,lhs=None):
        if isinstance(lhs,list): LHSs = lhs
        elif lhs: LHSs = [lhs]
        else: LHSs = self._index.keys()
        counts = []
        for lhs in LHSs:
            counts.append((len(self.get_all_rhs(lhs)),lhs))
        counts.sort()
        for (n,l) in counts: print n,l

    def keys(self):
        return self._index.keys()

    def values(self):
        return list(self._productions)

    def get_all_rhs(self,lhs=''):
        if not lhs: return self._productions
        if not isinstance(lhs,Nonterminal): lhs = Nonterminal(lhs.upper())
        return [self._productions[n] for n in self._index[lhs]]

    def get_prob_rhs(self,lhs,prob):
        if not isinstance(lhs,Nonterminal): lhs = Nonterminal(lhs.upper())
        rhs = []
        if prob > 1 and prob < 100: prob /= 100
        p = 0.0
        for r in self._index[lhs]:
            rule = self._productions[r]
            p += rule.prob()
            rhs.append(rule)
            if p > prob: break
        return rhs

    def print_all_rhs(self,lhs=''):
        for i,p in enumerate(self.get_all_rhs(lhs)): print p
        if not lhs:
            print len(self._lexicon_grammar), 'Lexicon Rules;',
            print len(self._nt_grammar), 'Nonterminal Rules;',
        else:
            print len([i for i in self._index[Nonterminal(lhs)] if i in self._lexicon_grammar]), 'Lexicon Rules;',
            print len([i for i in self._index[Nonterminal(lhs)] if i in self._nt_grammar]), 'Nonterminal Rules;',
        print i, 'Rules.'

    def get_rand_rhs(self,lhs):
        if not isinstance(lhs,Nonterminal): lhs = Nonterminal(lhs.upper())
        return self._productions[random.choice(self._index[lhs])]

    def __str__(self):
        s = 'CFG with %d productions' % len(self._index)
        s = '%s (start state = %s)' % (s, self._start)
        return s+'\n    '.join([str(p) for p in self._productions])

    def generate(self,text = '',lhs=None):
        if not lhs: lhs=self.start()
        prod = self.get_rand_rhs(lhs)
        #print 'CFG Generate:\"',text,'\"',lhs,'=>', prod.rhs()[0]
        for token in prod.rhs():
            #print 'CFG Generate:\"',text,'\"',token
            if isinstance(token,str): text += token+' '
            elif isinstance(token,Nonterminal): text = self.generate(text,token)
            else: print 'Unknown token', token
        return text
    # print re.sub('[\.,]','\g<0>\n',Pcfg.generate())

class HashPCFG(PCFG,HashCFG):
    """
    Subclass to implement efficient key-based access to productions.
    """    
    def __init__(self, start, productions):
        """
        Create a new context-free grammar, from the given start state
        and set of C{CFGProduction}s.
        
        @param start: The start symbol
        @type start: L{Nonterminal}
        @param productions: The list of productions that defines the grammar
        @type productions: C{list} of C{PCFGProduction}
        @raise ValueError: if the set of productions with any left-hand-side
            do not have probabilities that sum to a value within
            PCFG.EPSILON of 1.
        """
        assert _chktype(1, start, Nonterminal)
        assert _chktype(2, productions, (PCFGProduction,), [PCFGProduction])
        HashCFG.__init__(self, start, productions)

        # Make sure that the probabilities sum to one.
        probs = {}
        for production in productions:
            probs[production.lhs()] = (probs.get(production.lhs(), 0) +
                                       production.prob())
        for (lhs, p) in probs.items():
            if not ((1-PCFG.EPSILON) < p < (1+PCFG.EPSILON)):
                raise ValueError("CFGProductions for %r do not sum to 1" % lhs)
        for lhs in self._index:
            self._index[lhs].sort(lambda x,y: cmp(self._productions[y].prob(),
                                                  self._productions[x].prob()))

    def __str__(self):
        s = 'PCFG with %d productions' % len(self._index)
        s = '%s (start state = %s)' % (s, self._start)
        return s+'\n    '.join([str(p) for p in self._productions])

    def get_max_rhs(self,lhs):
        if not isinstance(lhs,Nonterminal): lhs = Nonterminal(lhs.upper())
        return self._productions[self._index[lhs][0]]

    def get_rand_rhs(self,lhs):
        if not isinstance(lhs,Nonterminal): lhs = Nonterminal(lhs.upper())
        r = random.random()
        s = 0
        for i in self._index[lhs]:
            p = self._productions[i]
            s += p.prob()
            if s > r: return p
        return None

def TrainPCFG(CPDist,Start='S'):
    PcfgProds = []
    for node in CPDist.conditions(): #Populate PCFG
        for sample in CPDist[node].samples():
            RHS = []
            #try:
            if isinstance(sample,str): RHS = [sample]
            elif isinstance(sample,Nonterminal): RHS.append(sample)
            elif isinstance(sample,Tree): RHS.append(Nonterminal(sample.node))
            elif isinstance(sample,Token): RHS.append(sample)
            elif isinstance(sample,list) or isinstance(sample,tuple):
                for token in sample:
                    if isinstance(token,Tree): RHS.append(Nonterminal(token.node))
                    elif isinstance(token,Nonterminal): RHS.append(token)
                    elif isinstance(token,Token): RHS.append(token)
                    else: RHS.append(token)
            else:
                RHS.append(token.node)
                print 'Unknown ', node,sample
            PcfgProds.append(PCFGProduction(Nonterminal(node), RHS, prob=CPDist[node].prob(sample)))
            #except: print 'missed on', node,sample
    return HashPCFG(Nonterminal(Start), PcfgProds)

def cvTrainPCFG(Corpus, num_files=10000, saveParses=0, Group='Raw', cv=0.0, parseTest=0, StartSymbol='S'):
    """
    A simple demonstration function for the C{Parser} classes.
    It trains and tests the parser using the corpus.

    @type num_files: C{int}
    @param num_files: The number of files that should be used for
        training and for testing.  Two thirds of these files will be
        used for training.  All files are randomly selected
        (I{without} replacement) from the corpus.  If
        C{num_files>=500}, then all 500 files will be used.

    @type saveParses: C{int}
    @param saveParses: Whether or not to write parsed trees to disk

    @return: Trained Probabilistic Context-Free Grammar
    @rtype: C{HashPCFG}
    """
    CFDist = ConditionalFreqDist()
    items = list(Corpus.items(Group))
    print 'Got',len(items),'items'
    num_files = max(min(num_files, len(items)), 3)
    random.shuffle(items)
    cv_split = int(num_files*(1-cv))
    TrainSet = items[:cv_split]
    TestSet = items[cv_split:]
    LearnCondDist(Corpus, TrainSet, CFDist, saveParses,1)
    Pcfg = TrainPCFG(ConditionalProbDist(CFDist, WittenBellProbDist), StartSymbol)
    SenseTagger = TrainSenseTagger(Pcfg,CFDist)
    return Pcfg,SenseTagger,TestSet

class WittenBellProbDist(MLEProbDist):
    """
    """
    def __init__(self, freqdist):
        """
        Use the Witten-Bell estimate to create a probability distribution
        for the experiment used to generate C{freqdist}.

        @type freqdist: C{FreqDist}
        @param freqdist: The frequency distribution that the
            probability estimates should be based on.
        """
        assert _chktype(1, freqdist, FreqDist)

        self._freqdist = freqdist
        ### B*N/(B+N) approximates the count of the unseen words in a kinda Witten-Bell like way
        self._freqdist.inc('$UNKNOWN$',self._freqdist.B()*self._freqdist.N()/(self._freqdist.N()+self._freqdist.B()))
        self._N = self._freqdist.N()

    def prob(self, sample):
        prob = self._freqdist.freq(sample)
        ### Appromate the number of unseen (count 0) tokens out there as the number of count 1 tokens plus 1
        if not prob: prob = self._freqdist.freq('$UNKNOWN$')/(1+self._freqdist.Nr(1))
        return prob

    def __repr__(self):
        """
        @rtype: C{string}
        @return: A string representation of this C{ProbDist}.
        """
        return '<Witten-Bell-ProbDist based on %d samples>' % self._freqdist.N()

