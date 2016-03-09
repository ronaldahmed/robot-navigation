import Queue, thread, os.path
from PIL.ImageTk import PhotoImage,Image
from Tkinter import Label, Frame 
from Robot.Meanings import str2meaning
from Utility import logger

import plastk.rl
from plastk.rl.facegridworld import MarkovLocPOMDPWorld, FaceGridWorldDisplay
from plastk.rl.loggingrli import LoggingRLI, EpisodeVarPlotter, NewColumn, ActionList, SensationList

class ImageView(Frame):
    def __init__(self, root, rli):
        Frame.__init__(self, root)
        self.rli = rli
        self.envName = ''
        self.envPlatRow = -1
        self.envPlatCol = -1
        self.envPlatDir = -1
        self.label = Label(self)
        self.label.pack()
        self.view = Label(self)
        self.view.pack()

    def setImage(self, imageName):
        size = self.winfo_width()*9/10
        image = Image.open(imageName)
        image = image.resize((size,size))
        self.photo = PhotoImage(image)
        self.label.config(text=imageName)
        self.view.config(image = self.photo)

    def redraw(self):
        envName = self.rli.env.name
        if envName[-1] in '0123456789': envName = envName[:-1]
        envRow,envCol = self.rli.env.curr_pos
        envFace = self.rli.env.curr_face
        envPlatRow,envPlatCol,envPlatDir = self.rli.env.pomdp.coords2plat(envRow,envCol,envFace)
        if ((self.envName,self.envPlatRow, self.envPlatCol, self.envPlatDir)
            != (envName,envPlatRow,envPlatCol,envPlatDir)):
            self.setImage('%s/Direction%s_%d_%d_%d.jpg' %
                          (self.rli.env.pomdp.map_dir,envName,envPlatRow,envPlatCol,envPlatDir))
            self.envName = envName
            self.envPlatRow = envPlatRow
            self.envPlatCol = envPlatCol
            self.envPlatDir = envPlatDir

class AgentProxy(plastk.rl.Agent):
    """
    Act like an rl agent, but forward everything over threading queues.
    """
    def __init__(self,actionQ,observationQ,**args):
        super(AgentProxy,self).__init__(**args)
        self.actionQueue = actionQ
        self.observationQueue = observationQ
    
    def __call__(self,sensation,reward=None):
        self.observationQueue.put((sensation,reward))
        act,time = self.actionQueue.get()
        print 'FollowerAgentProxy',act,time
        return act
