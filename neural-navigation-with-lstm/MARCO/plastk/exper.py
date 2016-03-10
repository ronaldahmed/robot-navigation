"""
Classes and functions for experiment control


$Id: exper.py,v 1.7 2006/03/21 16:04:46 jp Exp $

"""
import base,getopt,sys,os
from base import BaseObject

NYI = "Method must be implemented in subclass."


class Experiment(BaseObject):

    job_id = 'NO_ID'
    gui = False
    verbosity = None
    quiet = False
    pwd = None
    
    USAGE = """
    usage: experiment.py [-ghq]
                         [--verbosity=<print_level>]
                         [--pwd=dir]
                         [--size]
                         [--info]
                         [--id=<job-id>]
                         <condition> [<condition ...]

     -q  Quiet mode
     -g  Allow graphical output, if any.  (for non-batch runs)
     -n  Dry run, initialize, but don't run the condition.
     
     --size outputs the number of conditions in the experiment
     --info outputs a description of the conditions specified,
            then quits.
     --pwd=dir  sets the working directory to dir
     
     
     <print_level> = one of DEBUG, VERBOSE, NORMAL, WARNING,
                     or SILENT (same as -q)
     <job-id> = A unique ID for the job.  Defaults to PIDXXXXX, where
                XXXXX is the process-id of the run.
     <condition> = The number of the experimental condition to run.
                   If more than one condition is specified, they will be
                   run in series. Specifying 'all' runs each condition
                   exactly once.
    """


    def __init__(self,**params):
        super(Experiment,self).__init__(**params)
        self.conditions = cross(self.factors)
        
    def proc(self,*args):
        raise NYI

    def run(self,conds):
        for i in conds:
            level = i%len(self.conditions)
            print "Running Condition:", level
            self.print_condition(level)
            self.proc(i,*self.conditions[level])

    def print_condition(self,i):
        from pprint import pprint
        pprint(self.conditions[i%len(self.conditions)])

    def job_start(self):
        # print some general job info
        print "=============================================="
        print 'Job',self.job_id,'started.'
        print 'Host:',os.getenv('HOSTNAME')
        print 'pwd:', os.getcwd()
        print 'bogomips:',
        sys.stdout.flush()
        os.system('/u/jp/etc/bogomips')
        

    def main(self):
        try:
            opts,args = getopt.getopt(sys.argv[1:],'hgq',
                                            ['id=','size','info','pwd=','verbosity='])
        except getopt.GetoptError,err:
            print err
            sys.exit(self.USAGE)

        opts = dict(opts)

        if 'all' in args:
            conditions = range(len(self.conditions))
        else:
            try:
                conditions = [int(arg) for arg in args]
            except ValueError:
                print "All conditions must be specified as integers."
                print self.USAGE
                return

        if '-h' in opts:
            sys.exit(self.USAGE)

        if '--size' in opts:
            print "Experiment conditions size: %d conditions." % len(self.conditions)

        self.job_id = opts.get('--id','PID'+`os.getpid()`)
        self.verbosity = opts.get('--verbosity')
        self.gui = '-g' in opts
        self.quiet = '-q' in opts
        self.pwd = opts.get('--pwd')
        if self.pwd:
            print 'Changing directory to',self.pwd
            os.chdir(self.pwd)

        if self.quiet: base.BaseObject.print_level = base.SILENT
        if self.verbosity:
            base.min_print_level = getattr(base,self.verbosity)

        if not conditions:
            print "ERROR: no experimental conditions specified."
        elif '--info' in opts:
            for c in conditions:
                print
                print "Condition",c,':'
                self.print_condition(int(c))
        else:
            self.job_start()
            self.run([int(c) for c in conditions])

        print
        print "Job",self.job_id,"finished."

        

class FactorExperiment(Experiment):
    def __init__(self,**params):
        super(Experiment,self).__init__(**params)
        try:
            factors = self.factors
        except AttributeError:
            pass
        else:
            if factors:
                self.conditions = Cross(*self.factors)

    def run(self,conds):
        for i in conds:
            print
            print "Running condition %d:"%i
            level = i%len(self.conditions)
            self.print_condition(i)
            self.proc(i,**self.conditions.levels[level])

    def print_condition(self,i):
        cond = self.conditions[i % len(self.conditions)]
        for k,v in cond.items():
            print k,' = ',v

    def condition_name(self,num):
        def name(item):
            if isinstance(item,type):
                return item.__name__
            else:
                return item
        return '_'.join(['%s=%s'%(k,name(v)) for k,v in self.conditions[num].iteritems()])

class Factor(object):

    def __init__(self,**attribs):
        super(Factor,self).__init__
        if not attribs:
            raise "No attributes specified."
        self.levels = [dict() for v in attribs.values()[0]]
        for attrib,values in attribs.items():
            if len(values) != len(self.levels):
                raise "Not all attributes have the same number of values."
            for i,v in enumerate(values):
                self.levels[i][attrib] = v

    __len__ = lambda self: len(self.levels)
    __getitem__ = lambda self,key: self.levels[key%len(self)]
    __iter__ = lambda self: iter(self.levels)

class Cross(Factor):
    def __init__(self,*factors):
        if len(factors) == 1:
            self.levels = factors[0].levels
        else:
            first = factors[0]
            rest = factors[1:]
            self.levels = []
            for l1 in first.levels:
                crossing = Cross(*rest)
                for l2 in crossing.levels:
                    newlevel = dict(l1)
                    newlevel.update(l2)
                    self.levels.append(newlevel)

class Nest(Factor):
    def __init__(self,nest_factor,nested_factor,**conditions):        
        from operator import __and__

        self.levels = []
        for l1 in nest_factor.levels:
            if reduce(__and__,[l1[k] == v for k,v in conditions.iteritems()]):
                for l2 in nested_factor.levels:
                    new = dict(l1)
                    new.update(l2)
                    self.levels.append(new)
            else:
                self.levels.append(dict(l1))

   
def cross(factors):
    from operator import __add__
    if not factors:
        return [[]]
    else:
        return [[x] + y for y in cross(factors[1:]) for x in factors[0] ]

    

if __name__ == '__main__':
    from pprint import pprint

    class MyExp(Experiment):
        low = (1,2,3)
        med = (4,5,6)
        hi = (7,8)

        factors = (med,low,hi)

        def proc(self,a,b,c):
            print 100*c + 10*b + a 

    MyExp().run()

