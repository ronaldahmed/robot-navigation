"""

$Id: loggingrli.py,v 1.41 2006/04/24 14:49:23 jp Exp $
"""
import plastk.rl
from plastk.rl import RLI
from plastk.params import Parameter
from plastk import rand
#from Scientific.IO.NetCDF import NetCDFFile
from scipy.io.netcdf import netcdf_file as NetCDFFile
import time,sys,threading,os

NewColumn = 'new column'

class LoggingRLI(RLI):
    filestem =      Parameter(default='')
    catch_signals = Parameter(default=[])

    ckpt_extension    = Parameter(default = '.ckpt')
    steps_per_ckpt    = Parameter(sys.maxint)
    episodes_per_ckpt = Parameter(sys.maxint)

    rename_old_data = Parameter(default=True)

    gui_button_orient = Parameter(default='horizontal')

    ckpt_attribs = ['ep_count',
                    'step_count',
                    'steps_per_ckpt',
                    'episodes_per_ckpt',
                    'last_ckpt_step',
                    'last_ckpt_episode',
                    'last_sensation',
                    'next_action',
                    'env',
                    'agent']

    
    def __init__(self,**args):
        super(LoggingRLI,self).__init__(**args)
        self.step_vars = {}
        self.ep_vars = {}
        self.caught_signal = None
        if not self.filestem:
            self.filestem = self.name
        self.episode_filename = self.filestem + '-episodes.cdf'
        self.step_filename = self.filestem + '-steps.cdf'

        self.checkpointing = False

        self.gui_root = False
        self.gui_runstate = None

        self.action = ''
        self.last_sensation = ''

            
    def init(self,agent,env,**kw):
        super(LoggingRLI,self).init(agent,env,**kw)

        self.step_count = self.ep_count = 0

        if os.access(self.episode_filename,os.F_OK):
            self.remove_or_rename(self.episode_filename)

        self.episode_data = ed = NetCDFFile(self.episode_filename,'w')
        ed.createDimension('index',None)
        ed.createDimension('value',1)
        ed.createVariable('start','d',('index','value'))
        ed.createVariable('length','d',('index','value'))
        ed.createVariable('reward','f',('index','value'))

        for name,(fn,type,size) in self.ep_vars.items():
            ed.createDimension(name+'_dim',size)
            ed.createVariable(name,type,('index',name+'_dim'))

        if self.step_vars:
            if os.access(self.step_filename,os.F_OK):
                self.remove_or_rename(self.step_filename)

            self.step_data = sd = NetCDFFile(self.step_filename,'a')
            sd.createDimension('index',None)
            for name,(fn,type,size) in self.step_vars.items():
                sd.createDimension(name+'_dim',size)
                sd.createVariable(name,type,('index',name+'_dim'))

        self.last_ckpt_step = 0
        self.last_ckpt_episode = 0


    def remove_or_rename(self,filename):
        # if the  data file already exists either rename it or delete it
        if not self.rename_old_data:
            self.warning("Removing old data file:",filename)
            os.remove(filename)
        else:
            i = 0
            while True:
                stem,ext = filename.split('.cdf')
                new_filename = '%s-%d.cdf'%(stem,i) 
                if os.access(new_filename,os.F_OK):
                    i += 1
                    continue
                self.warning("Renaming old data file to",new_filename)
                os.rename(filename,new_filename)
                break
        
    def steps(self,num_steps,max_episodes=sys.maxint):
        for i in xrange(num_steps):
            if self.ep_count >= max_episodes:
                break
            super(LoggingRLI,self).steps(1)


    def close(self):
        try:
            self.episode_data.close()
            if self.step_vars:
                self.step_data.close()
        except AttributeError:
            self.warning("Error closing data files.")
        
    def add_step_variable(self,name,fn,type,size=1):
        self.step_vars[name] = (fn,type,size)

    def add_episode_variable(self,name,fn,type,size=1):
        self.ep_vars[name] = (fn,type,size)


    def start_episode(self):
        from plastk.rl.data import make_plot,plot_trials


        if (self.checkpointing and self.ep_count - self.last_ckpt_episode >= self.episodes_per_ckpt):
            self.ckpt_save()
        
        if self.gui_runstate == 'Episode':
           self.gui_runstate_control.invoke('Stop')
           self.request_gui_redraw()
           while self.gui_runstate == 'Stop':
               time.sleep(0.1)

        self.message("Starting episode",self.ep_count)
        super(LoggingRLI,self).start_episode()

        if self.step_vars:
            self.step_data.sync()

        epvars = self.episode_data.variables
        epvars['start'][self.ep_count] = self.step_count
        epvars['length'][self.ep_count] = 0            
        epvars['reward'][self.ep_count] = 0
        if self.ep_count > 0:
            for var,(fn,type,size) in self.ep_vars.items():
                epvars[var][self.ep_count-1] = fn(self)
        self.episode_data.sync()
        

        self.ep_count += 1

        

    def collect_data(self,sensation,action,reward,next_sensation):
        from Numeric import array

        self.sensation = sensation
        self.action = action
        self.reward = reward
        self.next_sensation = next_sensation

        if self.caught_signal:
            import sys
            self.close()
            raise "Caught signal %d" % self.caught_signal
        
        epvars = self.episode_data.variables
        epvars['reward'][self.ep_count-1] += array((reward,),'f')
        epvars['length'][self.ep_count-1] += 1

        if self.step_vars:
            stvars = self.step_data.variables
            for var,(fn,type,size) in self.step_vars.items():
                stvars[var][self.step_count] = fn(self)                      
            if self.step_count % 10000 == 0:
                self.step_data.sync()

        if (self.checkpointing and self.step_count - self.last_ckpt_step >= self.steps_per_ckpt):
            self.ckpt_save()

        self.step_count += 1
        


    ###################################################
    # Checkpointing

    def ckpt_steps(self,num_steps,max_episodes=sys.maxint):
        self.checkpointing = True
        self.setup_signals()
        self.steps(num_steps-self.step_count,max_episodes=max_episodes)    
        self.clear_signals()
        self.checkpointing = False
    def ckpt_episodes(self,num_episodes,timeout):
        self.checkpointing = True
        self.setup_signals()
        self.episodes(num_episodes-self.ep_count,timeout)
        self.clear_signals()
        self.checkpointing = False


    def ckpt_filename(self):
        return self.filestem + self.ckpt_extension
    
    def ckpt_save(self):
        from plastk import pkl
        
        self.verbose("Attempting checkpoint, %d episodes, %d steps."%(self.ep_count,self.step_count))
        if self.ckpt_ok():
            self.last_ckpt_step = self.step_count
            self.last_ckpt_episode = self.ep_count

            self.env.sim = self.agent.sim = None
            ckpt = dict(rand_seed = rand.get_seed())

            self.verbose("Checkpointing...")
            for a in self.ckpt_attribs:
                ckpt[a] = getattr(self,a)
                self.verbose(a, ' = ', ckpt[a])

            pkl.dump(ckpt,self.ckpt_filename())
            self.episode_data.sync()
            if self.step_vars:
                self.step_data.sync()
            self.env.sim = self.agent.sim = self
        else:
            self.verbose("No checkpoint, ckpt_ok failed")
            return        

    def ckpt_restore_state(self,filename):
        from plastk import pkl
        ckpt = pkl.load(filename)

        self.verbose("Restoring checkpoint state")
        for a in self.ckpt_attribs:
            self.verbose(a,' = ', ckpt[a])
            setattr(self,a,ckpt[a])
            
        rand.seed(*ckpt['rand_seed'])

        self.env.sim = self.agent.sim = self
        
        self.episode_data = NetCDFFile(self.episode_filename,'a')
        if self.step_vars:
            self.step_data = NetCDFFile(self.step_filename,'a')
        return ckpt

    def ckpt_resume(self):
        import os
        ckpt_filename = self.ckpt_filename()
        if os.access(ckpt_filename,os.F_OK):
            self.message("Found checkpoint file",ckpt_filename)
            self.ckpt_restore_state(ckpt_filename)
            return True
        else:
            return False

    def ckpt_ok(self):
        """
        Override this method to provide a check to make
        sure it's okay to checkpoint.
        (default = True)
        """
        return True

    def setup_signals(self):
        import signal
        for sig in self.catch_signals:
            self.verbose("Setting handler for signal",sig)
            signal.signal(sig,self.signal_handler)
    def clear_signals(self):
        import signal
        for sig in self.catch_signals:
            self.verbose("Clearing handler for signal",sig)
            signal.signal(sig,signal.SIG_DFL)
        
    def signal_handler(self,signal,frame):
        self.caught_signal = signal

    #########################################
    # GUI

    def gui(self,*frame_types):
        """
        Each of frame_types must be either
        (1) the string NewColumn, to start a new column or
        (2) a function that takes (tk_root,rli) and returns a Tkinter widget
            where tk_root is a Tkinter frame and rli is the controlling plastk rli.
            The Tkinter widget must have a redraw method, which takes no arguments.
        """
        import Tkinter as Tk
        from threading import Thread,Event
        self.gui_root = Tk.Tk()
        frame = self.gui_init(self.gui_root,frame_types)
        frame.pack(side='top',expand=1,fill='both')
            
        self.gui_root.title( self.name )
        self.gui_root.bind('<<redraw>>',self.gui_redraw)
        self.gui_root.bind('<<destroy>>', self.gui_destroy)
        self.gui_runloop_thread = Thread(target=self.gui_runloop)
        self.gui_runloop_thread.setDaemon(True)

        self.gui_redraw_event = Event()       

        def destroy():
            self.gui_runstate = 'Quit'
            
        self.gui_root.protocol('WM_DELETE_WINDOW',destroy)
        self.gui_running = True
        self.gui_runloop_thread.start()
        self.gui_root.mainloop()
        print "GUI Finished."
        self.gui_root = False


    def gui_runloop(self):
        while True:
            time.sleep(0.1)
            while  self.gui_runstate != 'Quit' and self.gui_runstate != 'Stop':
                self.steps(1)
                self.request_gui_redraw()
                if self.gui_runstate == 'Step':
                    self.gui_runstate_control.invoke('Stop')
            if self.gui_runstate == 'Quit':
                break
        print "Ending GUI run loop."

          
    def gui_init(self,root,frame_types):
        import Tkinter as Tk
        import Pmw
        gui_frame = Tk.Frame(root)
        control_frame = gui_frame
        #control_frame = Tk.Frame(gui_frame)
        #control_frame.pack(side='left',fill='both',expand=1)
	self.gui_runstate_control = Pmw.RadioSelect(control_frame,
                                                    labelpos = 'w',
                                                    orient = self.gui_button_orient,
                                                    command = self.gui_runstate_callback,
                                                    label_text = '',
                                                    frame_borderwidth = 1,
                                                    frame_relief = 'ridge')
	self.gui_runstate_control.pack(side='top',fill='none')
	# Add some buttons to the RadioSelect.
	for text in ('Quit','Run', 'Stop', 'Step','Episode'):
	    self.gui_runstate_control.add(text)
	self.gui_runstate_control.invoke('Stop')

    
        self.subframes = []
        g_frame = Tk.Frame(control_frame)
        g_frame.pack(side='left',expand=1,fill='both')
        for ft in frame_types:
            if ft == NewColumn:
                g_frame = Tk.Frame(control_frame)
                g_frame.pack(side='left',expand=1,fill='both')
            else:
                f = ft(g_frame,self)
                self.subframes.append(f)
                f.pack(side='top',expand=1,fill='both')
            
        return gui_frame

    def request_gui_redraw(self):
        if self.gui_root:
            self.gui_root.event_generate("<<redraw>>", when='tail')
        self.gui_redraw_event.wait(1.0)
        self.gui_redraw_event.clear()

    def gui_redraw(self,event):
        for f in self.subframes:
            f.redraw()
        self.gui_runstate_control.invoke(self.gui_runstate)
        self.gui_redraw_event.set()

    def gui_runstate_callback(self,tag):
        self.gui_runstate = tag
        if tag == 'Quit':
            self.gui_root.event_generate('<<destroy>>',when='tail')

    def gui_destroy(self,event):
        self.gui_root.quit()
        self.gui_root.destroy()
            


try:
    import Tkinter as Tk
    import Pmw
except ImportError:
    pass
else:
    class VarPlotter(Tk.Frame):

        def __init__(self,root,rli,name,fn,initial_count=500,**args):
            Tk.Frame.__init__(self,root,**args)
            self.fn = fn
            self.rli = rli

            self.trace_len = Tk.StringVar()
            self.trace_len.set('500')
            group = Pmw.Group(self,tag_text=name)
            group.pack(side='top',fill='both',expand=1)
            Pmw.EntryField(group.interior(),
                           label_text='Trace length',
                           labelpos='w',
                           validate='numeric',
                           entry_textvariable=self.trace_len).pack(side='top',fill='x')
#            self.plot = Pmw.Blt.Graph(group.interior())
            self.plot = Pmw.Blt.Graph(group.interior(),height='1i')
            self.plot.pack(side='top',expand=1,fill='both')
            self.plot.line_create('values',label='',symbol='',smooth='step')
            self.plot.grid_on()
            self.last_yvalues = []
            self.last_xvalues = []


        def redraw(self):
            try:
                N = int(self.trace_len.get())
            except ValueError:
                return
            xvalues,yvalues = self.fn(self.rli, N)
            if yvalues and (yvalues != self.last_yvalues or xvalues != self.last_xvalues):
                ydata = tuple(yvalues[:,0])
                xdata = tuple(xvalues)
                self.plot.element_configure('values',
                                            xdata=xdata,
                                            ydata=ydata)
            self.last_yvalues = yvalues
            self.last_xvalues = xvalues
            

    def StepVarPlotter(name,length=500):

        def get_values(rli,N):
            v = rli.step_data.variables[name][:]
            M = len(v[:])
            if M > N:
                return range(M-N,M),v[M-N:M]
            else:
                return range(M),v[:]

        return lambda root,rli: VarPlotter(root,rli,name,get_values,initial_count=length)


    def EpisodeVarPlotter(name,length=500):

        def get_values(rli,N):
            v = rli.episode_data.variables[name][:]
            M = len(v[:])
            if M > N:
                return range(M-N,M-1),v[M-N:M-1]
            else:
                return range(M),v[:M-1]

        return lambda root,rli: VarPlotter(root,rli,name,get_values,initial_count=length)

    class TextList(Tk.Frame):
        def __init__(self, root, rli, **args):
            Tk.Frame.__init__(self, root, **args)
            self.rli = rli
            self.textlist = []
            self.ep_count = rli.ep_count
            
            self.list = Pmw.ComboBox(self, dropdown=0, history=0,
                                        labelpos='nw', label_text=self.name)
            self.list.component('scrolledlist').component('listbox').config(exportselection=0)
            self.list.pack(side='top',fill='both',expand=1)

        def redraw(self):
            self.list.component('listbox').insert(self.rli.step_count, self.get_line())
            listlen = self.list.component('listbox').size()
            if listlen: self.list.selectitem(listlen - 1)
            if self.ep_count != self.rli.ep_count:
                self.list.clear()
                self.ep_count = self.rli.ep_count

    class ActionList(TextList):
        name = 'Actions'
        def get_line(self):
            return self.rli.action

    class SensationList(TextList):
        name = 'Sensations'
        def get_line(self):
            return self.rli.last_sensation
