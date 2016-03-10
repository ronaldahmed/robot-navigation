from grid_exper import *
from plastk.plot import GPlot
from Scientific.IO.NetCDF import NetCDFFile as CDF

for c in exp.conditions:
    names = pkl.files_matching('*%s*%s*-episodes.cdf'%(c['agent_name'],c['grid_name']))
    files = [CDF(f) for f in names]
    c['data'] = [f.variables['length'][:,0] for f in files]


plots = dict(grid1 = GPlot(), grid2=GPlot())

for k,p in plots.iteritems():
    p('set logscale y')
    p.xlabel('Episodes')
    p.ylabel('Steps')
    p.title('Episode length for %s'%k.capitalize())

for c in exp.conditions:
    grid_name = c['grid_name']
    agent_name = c['agent_name']
    data = c['data']
    plots[grid_name].plot_avg(y = data,title=agent_name,replot=1,step=10)
    
            

