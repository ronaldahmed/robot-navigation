import numpy as np
import ipdb
import os,sys,re
import xml.etree.ElementTree as ET
from nltk import word_tokenize
sys.path.append('code/MARCO')

from POMDP.MarkovLoc_Grid import getMapGrid
from POMDP.MarkovLoc_Jelly import getMapJelly
from POMDP.MarkovLoc_L import getMapL

#######################################################################################################
maps = {'grid': getMapGrid, 'jelly':getMapJelly, 'l':getMapL}
data_dir = 'data/'

SEED = 42
np.random.seed(SEED)

path_patt = re.compile(r'\d+,\s*\d+,\s*[-]*\d+')

# actions
FW = 0
L = 1
R = 2
STOP = 3
actions_str = [
	"Go",
	"Left",
	"Right",
	"STOP",
]
forward_step = 1 		# 1 cell
rotation_step = 90	# degrees

#######################################################################################################

class MapData:
	def __init__(self,_map_name="",):
		# can be Grid, Jelly or L
		self.name = _map_name.lower()
		# format: instructions: [sent1,sent2] | actions: [act_seq1,act_seq2]
		self.single_sentence = {"instructions":[],"actions":[]}
		# format: instructions: [ [sent11,sent12..],[sent21,sent22,...] ] | actions: [act_seq1.,act_seq2.]
		self.multi_sentence = {"instructions":[],"actions":[]}

	def add_single_sample(self,_instructions,_actions):
		# add new sample (nav_instructions,actions)
		self.single_sentence["instructions"].append(_instructions)
		self.single_sentence["actions"].append(_actions)

	def add_multi_sample(self,_sentences,_actions):
		# add new sample ([sent1,sent2,...],actions)
		self.multi_sentence["instructions"].append(_sentences)
		self.multi_sentence["actions"].append(_actions)


def clean_text(text):
	# Strips spaces and end-lines from both sides of the string
	res = text
	bef = ''
	while res!=bef:
		bef = res
		flag = False
		res = res.strip('\n').strip(' ')
	return res

def preprocess_path(text):
	# process string '[(x,y,o)]' to list of positions
	findings = path_patt.findall(text)
	path = []
	for ins in findings:
		path.append([int(x.strip()) for x in ins.split(',')])
	return path

def move(x,y,th,action):
	if action==FW:
		if th==0:		return x             ,y-forward_step,th
		elif th==90:	return x+forward_step,y             ,th
		elif th==180:	return x             ,y+forward_step,th
		elif th==270:	return x-forward_step,y             ,th
	elif action==R:
		return x,y,(th+rotation_step)%360
	elif action==L:
		return x,y,(th-rotation_step+360)%360

def verbose_actions(actions):
	#print string command for each action
	return [actions_str[act_id] for act_id in actions]

def get_actions_from_path(path):
	"""
	Transform sequence of position states (x,y,orient) to action sequences (FW,L,R,STOP)
	"""
	rand_angle = [0,90,180,270]
	if path[0][-1]==-1:
		#random initial orientation if unknown
		path[0][-1] = rand_angle[np.random.random_integers(0,3,1)[0]]
		if len(path)>1:
			# keep sampling random orientation until it's different from next orientation
			while path[0][-1]==path[1][-1]:
				path[0][-1] = rand_angle[np.random.random_integers(0,3,1)[0]]

	actions = []
	n_path = len(path)
	for k in xrange(1,n_path):
		x,y,theta = path[k]
		x_prev,y_prev,theta_prev = path[k-1]

		# planning movement
		if (x,y)!=(x_prev,y_prev):
			#translation


		if theta==theta_prev:
			x_t,y_t,_ = move(x_prev,y_prev,theta,FW)
			actions.append(FW)
			k=0
			while (x_t,y_t)!=(x,y):
				x_t,y_t,_ = move(x_t,y_t,theta,FW)
				actions.append(FW)
				k+=1
				if k>10:
					print(x_prev,y_prev)
					print(x_t,y_t)
					print(x,y)
					ipdb.set_trace()
		else:
			theta = 360 if theta==0 and abs(theta-theta_prev)>90 else theta
			theta_prev = 360 if theta_prev==0 and abs(theta-theta_prev)>90 else theta_prev
			if theta - theta_prev == 90:
				actions.append(R)
			elif theta - theta_prev == -90:
				actions.append(L)
			elif abs(theta-theta_prev)==180:
				# two LEFT turns in case complete turn
				actions.append(L)
				actions.append(L)
			else:
				print("<--Invalid orientation change-->")
				ipdb.set_trace()
	actions.append(STOP)
	return np.array(actions)


"""
Read single and multiple sentence instructions
"""
def get_data():
	map_data = {
		'grid'  : MapData("grid"),
		'jelly' : MapData("jelly"),
		'l'	  : MapData("l")
	}

	single_sent_file = 'SingleSentence.xml'
	ss_tree = ET.parse(os.path.join(data_dir,single_sent_file)).getroot()
	for example in ss_tree:
		_map = example.attrib['map'].lower()
		nl_ins = clean_text(example.find('instruction').text)
		path_text   = clean_text(example.find('path').text)
		path = preprocess_path(path_text)
		# reformat data
		tokens = word_tokenize(nl_ins)
		actions = get_actions_from_path(path)
		# save into respective map data container
		map_data[_map].add_single_sample(tokens,actions)
	del ss_tree

	multi_sent_file = 'Paragraph.xml'
	ms_tree = ET.parse(os.path.join(data_dir,multi_sent_file)).getroot()
	for example in ms_tree:
		_map = example.attrib['map'].lower()
		nl_ins = clean_text(example.find('instruction').text)
		path_text   = clean_text(example.find('path').text)
		path = preprocess_path(path_text)
		# reformat data_dir
		sentences = [word_tokenize(sent) for sent in nl_ins.split('.') if len(sent)>0]
		actions = get_actions_from_path(path)
		# save into respective map data container
		map_data[_map].add_multi_sample(sentences,actions)
	del ms_tree

	return map_data

data = get_data()
ipdb.set_trace()