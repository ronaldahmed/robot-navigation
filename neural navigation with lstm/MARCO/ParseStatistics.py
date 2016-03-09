import sys,time,os,operator
import nltk.draw.tree
from nltk.tree import Tree
from nltk.token import Token
from Sense import tree2frame

class parseStats(object):
    def __init__(self,accumParses=False):
        self.times = []
        self.average_p = []
        self.num_parses = []
        self.parse_list = []
        self.frame_list = []
        self.all_parses = {}
        self.crossed_brackets = []
        self.accumParses=accumParses
        return self

    def __add__(self,other):
        n = parseStats()
        n.times = self.times + other.times
        n.average_p = self.average_p + other.average_p
        n.num_parses = self.num_parses + other.num_parses
        n.parse_list = self.parse_list #+ other.parse_list
        n.frame_list = self.frame_list #+ other.frame_list
        if self.accumParses:
            n.all_parses.update(self.all_parses)
            for k in other.all_parses:
                n.all_parses[k] = self.all_parses.get(k,0) + other.all_parses[k]
        else:         n.all_parses = {}
        n.crossed_brackets = self.crossed_brackets + other.crossed_brackets
        return n

    def sum(self):
        if len(self.times) <= 1: return self 
        n = parseStats()
        for svar,nvar in zip([self.times, self.average_p, self.num_parses, self.crossed_brackets],
                             [n.times, n.average_p, n.num_parses, n.crossed_brackets]):
            nvar = reduce(operator.add,svar,0)
        for p in self.parse_list:
            if isinstance(p,list): n.parse_list += p
            else: n.parse_list.append(p)
        for f in self.frame_list:
            if isinstance(f,list): n.frame_list += f
            else: n.frame_list.append(f)
        return n

    def __repr__(self):
        s = ''
        s += ' '.join([str(t) for t in self.times]) + '\n'
        s += ' '.join([str(p) for p in self.average_p]) + '\n'
        s += ' '.join([str(n) for n in self.num_parses]) + '\n'
        s += ' '.join([str(c) for c in self.crossed_brackets]) + '\n'
        s += str(len(self.parse_list)) + '\n'
        return s

    def __str__(self):
        s = 'Times:'
        s += ' '.join([str(t) for t in self.times]) + '\n'
        s += 'Avg Parse:'
        s += ' '.join([str(p) for p in self.average_p]) + '\n'
        s += 'Num Parses:'
        s += ' '.join([str(n) for n in self.num_parses]) + '\n'
        s += 'Parses:\n'
        s += '\n'.join([str(p) for p in self.parse_list]) + '\n'
        s += 'Frames:\n'
        s += '\n'.join([str(p) for p in self.frame_list]) + '\n'
        s += 'Crossed Brackets:'
        s += ' '.join([str(c) for c in self.crossed_brackets]) + '\n'
        return s

def parse_stats(parsers, text, stats, trace=0, trueTree=None):
    """Run the parsers on the tokenized sentence text."""
    for parser in parsers:
        print '\ns:%d %s\nparser: %s' % (len(text['WORDS']),text,parser)
        parser.trace(trace)
        t = time.time()
        parses = parser.get_parse_list(text)
        stats.times.append(time.time()-t)
        if parses and parses[0]: p = reduce(lambda a,b:a+b.prob(), parses, 0)/len(parses)
        else: p = 0
        stats.average_p.append(p)
        stats.num_parses.append(len(parses))
        stats.parse_list.append(parses)
        stats.frame_list.append([])
        for p in parses:
            if p:
                print p,type(p)
                stats.all_parses[str(p)] = 1
                stats.frame_list[-1].append(tree2frame(p)[0])
                if trueTree is not []:
                    cross,brackets,percent = crossed_brackets_percentage(p,trueTree)
                    stats.crossed_brackets.append(percent)
    return stats

def print_parse_summary(parsers, stats, interactive, draw_parses, print_parses):
    """Print some summary statistics for a parse"""
    print
    print '       Parser      | Time (secs)   # Parses   Average P(parse)   CB  '
    print '-------------------+-------------------------------------------------'
    for i in range(len(parsers)):
        if stats.crossed_brackets: cb = stats.crossed_brackets[i]
        else: cb = 0
        print '%18s |%11.4f%11d%19.14f%5d ' \
              % (parsers[i].__class__.__name__, stats.times[i], stats.num_parses[i], stats.average_p[i], cb)
    if interactive:
        print
        print 'Draw parses (y/n)? ',
        draw_parses =  sys.stdin.readline().strip().lower().startswith('y')
    if draw_parses:
        print '  please wait...'
        nltk.draw.tree.draw_trees(*parses)
    if interactive:
        print
        print 'Print parses (y/n)? ',
        print_parses = sys.stdin.readline().strip().lower().startswith('y')
    if print_parses:
        for parse in stats.parse_list:
            print 'Parsed:',parse
    return stats.parse_list

def crossed_brackets_percentage(p1,p2):
    """This metric is the percentage of guessed brackets
    which did not cross any correct brackets."""
    bracket_count = 1
    incorrect_count = 0
    if isinstance (p1, Tree) and isinstance (p2, Tree):
        if p1.node != p2.node:
            incorrect_count += 1
        if len(p1.children())!=len(p2.children()): incorrect_count += 1
        for c1,c2 in zip(p1.children(),p2.children()):
            c,b,p = crossed_brackets_percentage(c1,c2)
            bracket_count += b
            incorrect_count += c
    elif isinstance (p1, Token) and isinstance (p2, Token):
        if p1.type() != p2.type():
            incorrect_count += 1
    else: incorrect_count += 1
    return incorrect_count, bracket_count, 100*incorrect_count/bracket_count

def parseTestSet(Parser,Corpus,TestSet,saveParses=0):
    sys.stdout.write('Reading Testing data\n'); sys.stdout.flush()
    for item in TestSet[:]:
        if '-' in item: itemname = item.split('-')[1]
        else: itemname = item
        sys.stdout.write('%s\n'%(item,)); sys.stdout.flush()
        Corpus.parseInstruction(Parser,itemname,saveParses,frames=True,trueTree=Corpus.readDirectionTree(itemname))
        TestSet.remove(item)
