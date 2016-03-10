"""
Functions for manipulating RL data.

$Id: data.py,v 1.5 2005/04/01 22:12:33 jp Exp $
"""

import os,sys
from plastk import pkl
from Scientific.IO.NetCDF import NetCDFFile
from Numeric import concatenate,ones,array



def open_rl_data(filespec):
    if isinstance(filespec,type([])):
        return map(open_rl_data,filespec)    
    ep_data = NetCDFFile(filespec,'r')
    step_file_name = filespec.split('-episodes.cdf')[0]+'-steps.cdf'
    if os.access(step_file_name,os.F_OK):
        step_data = NetCDFFile(step_file_name,'r')
    else:
        step_data = None

    return ep_data,step_data

def get_num_episodes(ep_filename):
    f = NetCDFFile(ep_filename,'r')
    N = len(f.variables['reward'])
    f.close()
    return N


    
##########################
# data manipulation and plotting fns
def get_episode_data(ep_data,step=1):
    from Numeric import array,concatenate,arange
    data = ep_data[0:len(ep_data):step]
    indices = [[float(i)] for i in range(0,len(ep_data),step)]
    
    return concatenate((array(indices)*1.0,data),axis=1)
    
def get_trial_data(trial_data,step=1):
    from Numeric import concatenate
    return [get_episode_data(ep_data,step) for ep_data in trial_data]

def get_trial_avg(trial_data,step=1):
    from Numeric import concatenate,array,transpose
    from plastk.utils import stats
    trial_data = get_trial_data(trial_data,step)
    indices = trial_data[0][:,0]
    data = array([x[:,1] for x in trial_data])

    mean,var,stderr = stats(data)

    return transpose(array((indices, mean,stderr)))
                       

def make_plot(plot=None,title='',xlabel='episodes',ylabel=''):
    import Gnuplot
    if plot:
        return plot
    else:
        g = Gnuplot.Gnuplot()

    g.xlabel('Episode')
    g.ylabel(ylabel)
    g.title(title)

    return g
    

def plot_episodes(ep_data,plot=None,title='',ylabel='reward',with='linespoints',replot=0):
    import Gnuplot
    
    if not plot:
        plot = make_plot(title=title,ylabel=ylabel)

    if replot:
        plot_fn = plot.replot
    else:
        plot_fn = plot.plot

    gpdata = Gnuplot.Data(ep_data,with=with,title=title,inline=1)
    plot_fn(gpdata)
    return plot

def plot_trials(trial_data,plot=None,
                title='',ylabel='Reward',replot=0,step=1,with='linespoints'):

    data = get_trial_data(trial_data,step=step)

    if not plot:
        plot = make_plot(ylabel=ylabel)

    plot_episodes(data[0],plot=plot,replot=replot,ylabel=ylabel,with=with)
    for d in data[1:]:
        plot_episodes(d,plot=plot,replot=True,ylabel=ylabel,with=with)
    return plot


def plot_trial_avg(trial_data,plot=None,
                   title='',ylabel='Reward',replot=0,step=1,errorbars=1,
                   style='',size=''):
    
    import Gnuplot
    from Numeric import average,transpose,concatenate

    data = get_trial_avg(trial_data,step=step)
    if errorbars: line_title=None
    else: line_title=title
    glines_data = Gnuplot.Data(data[:,0:2],
                               with='lines %s %s'%(str(style),str(size)),
                               title=line_title,inline=1)
    gbars_data = Gnuplot.Data(data,
                              with='errorbars %s %s'%(str(style),str(size)),
                              title=title,inline=1)

    if not plot:
        plot = make_plot(ylabel=ylabel)
    if replot:
        plot_fn = plot.replot
    else:
        plot_fn = plot.plot

    plot_fn(glines_data)
    if errorbars:
        plot.replot(gbars_data)
    return plot
