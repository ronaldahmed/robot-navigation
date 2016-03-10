import gzip, logging, os, os.path, sys, time

Sep ={
    'DEBUG' : '~',
    'INFO' : '.',
    'STAGE' : '=',
    'RUN' : '!',
    'WARNING' : '.',
    'ERROR' : '_',
    'CRITICAL' : '#',
    }

def initLogger(loggerName,consoleLevel=logging.INFO,doTrace=True,LogDir='Logs'):
    global timeStamp,logger
    logger = logging.getLogger(loggerName)
    logger.setLevel(consoleLevel)
    timeStamp = time.strftime("%Y-%m-%d-%H-%M")
    logging.addLevelName(24, 'STAGE') # Completion of a stage
    logging.addLevelName(26, 'RUN') # Completion of a run
    Summary = ('Summary', 26, '%(message)s')
    Trace = ('Trace', logging.DEBUG, '%(asctime)s %(levelname)-8s %(message)s')
    if doTrace: Logs = (Summary,Trace)
    else: Logs = tuple()
    for logname,level,fmt in Logs:
        LogFile = '-'.join((loggerName,logname,timeStamp))+'.log'
        logHandler = logging.FileHandler(os.path.join(os.getcwd(), LogDir,LogFile), 'w')
        logHandler.setLevel(level)
        logHandler.setFormatter(logging.Formatter(fmt, '%m-%d %H:%M:%S'))
        logger.addHandler(logHandler)
    
    # Log info and above to console without timestamps
    console = logging.StreamHandler(sys.stdout)
    if doTrace: console.setLevel(logging.INFO)
    else: console.setLevel(24)
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)

def shutdownLogger(): logging.shutdown()
##    for filename in ('Summary','Trace'):
##        gzip('-'.join(('Following',logname,timeStamp))+'.log')

def debug(*msg): logger.debug(*msg)
def info(*msg): logger.info(*msg)
def stageComplete(*msg): logger.log(24,*msg)
def runComplete(*msg): logger.log(26,*msg)
def warning(*msg): logger.warning(*msg)
def error(*msg): logger.error(*msg)
def critical(*msg): logger.critical(*msg)
def flush(): pass
