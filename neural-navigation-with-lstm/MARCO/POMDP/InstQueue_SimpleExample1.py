#!/usr/bin/env python
from POMDP.InstQueue import Instruction, POMDP

Views = {
    0 : '.,(?!Y).,.',
    1 : '(?!H.(?!W).H).,.,.',
    2 : 'H,(?!W),H',
    3 : '.,(?!W).,.',
    4 : '.,W,.',
    5 : '.,.,.',
    }

Rewards = [-100,100] #Instructed,NotInstructed
NumInstructions=len(Views)-1
Actions = {}
for i,act in enumerate(['TurnRight','Travel','TurnLeft','Travel','DeclareGoal']):
    Actions[i] = Instruction(i,act,Rewards,NumInstructions)

POMDP(Views,Actions,'SimpleExample1').writefile()
