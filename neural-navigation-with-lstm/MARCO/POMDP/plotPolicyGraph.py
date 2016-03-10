#!/bin/env python
import os,sys

def plotPolicyGraph(file):
    pgfile = open(file,'r')
    basename = '.'.join(file.split('.')[:-1])
    graphfile = open(basename+'.dot','w')
    pomdpfile = open('-'.join(file.split('-')[:-1])+'.pomdp','r')
    POMDP = {}
    actions = []
    observations = []
    while actions == [] or observations == []:
        attribute,values = pomdpfile.readline().split(':')
        if attribute == 'actions':
            actions = values[1:-1].split(' ')
        elif attribute == 'observations':
            observations = values[1:-1].split(' ')
    basename = os.path.basename(basename.replace('-','_'))
    graphfile.write('digraph '+basename+ '{\n')
    for line in pgfile.xreadlines():
        entries = line.split(' ')
        id=entries[0]
        action=entries[1]
        graphfile.write('\t '+id+' [ label = "'+actions[int(action)]+'_'+id+'" ];\n')
        for observation,endstate in zip(observations,entries[3:]): #Skip double space
            if endstate != '-':
                graphfile.write('\t '+id +' -> '+endstate +' [ label = "'+observation+'" ];\n')
    graphfile.write('}\n')

if __name__== '__main__':
    if len(sys.argv)>1 and sys.argv[1]: plotPolicyGraph(sys.argv[1])
