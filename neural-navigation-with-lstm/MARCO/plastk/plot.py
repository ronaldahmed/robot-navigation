"""
Plotting Objects

This module implements plotting for PLASTK.  Currently it contains a
subclass of the Gnuplot object from Gnuplot.py, that has been updated
to make some common kinds of plotting more convenient.  [This probably
subsumes most of the important functionality of plastk.rl.data.]

$Id: plot.py,v 1.5 2006/01/18 23:26:00 jp Exp $
"""
import Gnuplot as Gp
from plastk  import utils
from Numeric import array,sqrt

class GPlot(Gp.Gnuplot):
    """
    A subclass of Gnuplot.Gnuplot that makes some common kinds of
    plotting easier.

    Notably, it's easy to plot a single y variable against the
    integers, without haveing explicitly specify the data as a
    sequence of (x,y) pairs.  It also allows the plotting of multiple
    curves in a single call, automatic plotting of averages with error
    bars, and plotting only everyt Nth point in a data set.  See the
    individual method docs for more details.

    Using GPlot, it's not necessary to create separate GPlot.Data
    objects for data.
    """
    
    def __init__(self,*args,**kw):
        Gp.Gnuplot.__init__(self,*args,**kw)
        self.current_style = 0

    def plot(self,x=None, y=None, pts=None, cmd=None,
             replot=False, queue_only=False, clear_queue=False,
             step=1,
             **params):
        """
        General data plotting.  Uses keywords to specify data:

        y -- The Y values to plot
        x -- The X values for the plotted points.  By
             default, x = range(0,len(y)).

        pts -- A sequence of points to plot, similar to the
               standard arg to Gnuplot.Data(), overrides x and y.

        replot -- Keep the old plot contents.  Default = False.
        queue_only -- adds data to PlotItem queue, but doesn't plot it
                      overrides replot=True.  Default=False.
        step -- Only plot every nth data point, default = 1.

        **params -- Keyword arguments that will be passed to
                    Gnuplot.Data(), e.g. with='lines'
        """

        if clear_queue:
            self._clear_queue()
        if queue_only:
            plot_fn = lambda s,d:Gp.Gnuplot._add_to_queue(s,[d])
        elif replot:
            plot_fn = Gp.Gnuplot.replot
        else:
            plot_fn = Gp.Gnuplot.plot

        if pts:
            if step > 1:
                pts = step_select(pts,step)
            data = Gp.Data(pts,inline=1,**params)
        elif y:
            if not x:
                x = range(len(y))
            pts = step_select(zip(x,y),step)
            data = Gp.Data(pts,inline=1,**params)
        elif cmd:
            data = cmd
        else:
            Gp.Gnuplot.replot(self)
            return
        
        plot_fn(self,data)

    def plot_multi(self,x=None,y=None,pts=None,title=None,
                   replot=False,**params):
        """
        Plot several data sets.

        y -- A sequence of sequences of Y coordinates to plot.
        x -- A *single* sequence of X coordinates to match the Ys,
            (can be blank)
        pts -- A sequence of sequences of (x,y) pairs to
        plot. (overrides x and y).
        replot -- Keep the old plot contents. Default False
        title -- A sequence of strings as titles for the respective
                 data sets.
        **params -- keyword args to pass to Gnuplot.Data()
        """
        if pts:
            if not title:
                title = [''] * len(pts)
            self.plot(pts=pts[0],title=title[0],replot=replot,**params)
            for p,t in zip(pts,title):
                self.plot(pts=p,title=t,replot=1,**params)
        elif y:
            if not title:
                title = [''] * len(y)
            self.plot(x=x,y=y[0],title=title[0],replot=replot,**params)
            for p,t in zip(y,title):
                self.plot(x=x,y=p,title=t,replot=1,**params)
        else:
            raise "plot_multi requires either pts, or y as an argument"
        

    def plot_avg(self,x=None,y=None,title=None,replot=False,step=1,
                 errorbars='conf'):

        """
        Plot the average over a set of Y values with error bars
        indicating the 95% confidence interval of the sample mean at
        each point. (i.e. stderr * 1.96)

        y = A sequence of sequences of Y values to average.
            If not all sequences are of equal length, the length
            of the shortest sequence is used for all.
        x = (optional) A single sequence of X values corresponding to
            the Ys.
        title = The title of the average plot.
        replot = Keep the old contents of the plot window.
                 default = False
        step = Plot the average at every Nth point. (default = 1)
        errorbars = What statistic to use for error bars, one of:
                    'conf'   -> 95% confidence interval (stderr * 1.96)
                    'stderr' -> Standard error
                    'stddev  -> Standard deviation
                    'var'    -> Variance
        """
        from Numeric import concatenate as join
        N = min(map(len,y))
        mean,var,stderr = utils.stats(join([array([a[:N]]) for a in y],axis=0))

        if replot:
            self.current_style += 1
        else:
            self.current_style = 1

        self.plot(x=x,y=mean,title=title,
                  with='lines %d'%self.current_style,
                  step=step,replot=replot)
        if not x:
            x = range(len(mean))

        if errorbars == 'conf':
            bars = stderr * 1.96
        elif errorbars == 'stderr':
            bars = stderr
        elif errorbars == 'stddev':
            bars = sqrt(var)
        elif errorbars == 'var':
            bars = var
        else:
            raise 'Unknown error bar type: "%s"' % errorbars
        
        self.plot(pts=zip(x,mean,bars),with='errorbars %d'%self.current_style,
                  step=step,replot=1)

    def plot_stddev(self,x=None,y=None,title=None,replot=False,step=1,with='lines'):
        from Numeric import concatenate as join
        N = min(map(len,y))
        mean,var,stderr = utils.stats(join([array([a[:N]]) for a in y],axis=0))
        self.plot(x=x,y=sqrt(var),title=title,with=with,replot=replot,step=step)
        
    def replot(self,**params):
        self.plot(replot=True,**params)



def step_select(seq,step):
    """
    Given a sequence select return a new sequence containing every
    step'th item from the original.
    """
    if seq:
        return [x for i,x in enumerate(seq)
                if i % step == 0]
    else:
        return seq

