import numpy as np
import ipdb
import os,sys,re
from nltk import word_tokenize
from six.moves import cPickle as pickle
from sklearn.cross_validation import train_test_split
from collections import Counter

from MARCO.POMDP.MarkovLoc_Grid import getMapGrid
from MARCO.POMDP.MarkovLoc_Jelly import getMapJelly
from MARCO.POMDP.MarkovLoc_L import getMapL
from MARCO.Robot.Meanings import Wall,End,Empty

#######################################################################################################
data_dir = 'data/'

SEED = 42
np.random.seed(SEED)

path_patt = re.compile(r'\d+,\s*\d+,\s*[-]*\d+')

# actions
FW = 0
L = 1
R = 2
STOP = 3
PAD_decode = 4
actions_str = [
	"FORWARD",
	"LEFT",
	"RIGHT",
	"STOP",
	"<PAD>",
]
num_actions = len(actions_str)
forward_step = 1 		# 1 cell
rotation_step = 90	# degrees

# Special indicator for sequence padding
EOS = '<EOS>'
PAD = '<PAD>'
RARE = '<RARE>'

#######################################################################################################
class Sample(object):
	def __init__(self,_inst=[],_actions=[],path=[],_id='',sP=-1,eP=-1,_map_name=''):
		self._instructions = _inst
		self._actions = _actions
		self._id = _id
		# path: sequence of position states (x,y,th)
		self._path = path
		# start and end position (id_localization) | global (multisentence) start and end
		self._startPos = sP
		self._endPos = eP
		self._map_name = _map_name
	
	def __repr__(self):
		res = ("{ instructions:\n"+
				"   "+str(self._instructions) + '\n'
				" actions:\n"+
				"   "+str(verbose_actions(self._actions)) + '\n'
				" path:\n"+
				"   "+str(self._path)+' }')
		return res
		
		
class MapData:
	def __init__(self,_map_name,_map):
		# can be Grid, Jelly or L
		self.name = _map_name.lower()
		# map object
		self.map = _map
		# format: [Sample(x,y,sample_id)]
		self.samples = []

	def add_sample(self,_instructions,_actions,_path,_id,sP,eP,map_name):
		# add new sample (nav_instructions,actions,sample_id)
		# input: instructions, actions, path, sample_id, start_pos, end_pos, map_name
		self.samples.append( Sample(_instructions,_actions,_path,_id,sP,eP,map_name) )

	def get_multi_sentence_samples(self):
		# return: [[Sample], [Sample]]
		ms_sample_list = []
		prev_id = self.samples[0]._id
		n_sam = len(self.samples)
		ms_sample = []
		for i in xrange(n_sam):
			if self.samples[i]._id != prev_id:
				ms_sample_list.append(ms_sample)
				ms_sample = [self.samples[i]]
			else:
				ms_sample.append(self.samples[i])
			prev_id = self.samples[i]._id
		# add last batch
		ms_sample_list.append(ms_sample)
		return ms_sample_list

def verbose_actions(actions):
	#print string command for each action
	return [actions_str[act_id] for act_id in actions]

def get_actions_and_path(path_text,_map):
	"""
	Extract action and path seq from raw string in data (FW(x,y,th);L(x,y,th)...)
	"""
	list_pre_act = path_text.split(';')
	n_act = len(list_pre_act)
	actions = []
	path = []
	angle_id_dict = _map.plat_orientations
	for i in xrange(n_act):
		x,y,th = -1,-1,-1
		id_act = -1
		if i==n_act-1:
			str_action = list_pre_act[i].strip('(').strip(')').split(',')
			x,y,th = [int(comp.strip()) for comp in str_action]
			id_act = STOP
		else:
			prx = list_pre_act[i].find('(')
			id_act = actions_str.index(list_pre_act[i][:prx])
			x,y,th = [int(comp.strip()) for comp in list_pre_act[i][prx+1:-1].split(',')]
		th_id = angle_id_dict[th]
		xg,yg = _map.gridPos_from_pathPos(x,y)

		if xg < 1 or yg < 1:
			print("Map: ",_map.name)
			print(" xp,yp: ", x,y)
			print(" xg,yg: ", xg,yg)
			print("="*30)
			ipdb.set_trace()

		path.append( (xg,yg,th_id) )
		actions.append(id_act)
	return actions,path



"""
Read single and multiple sentence instructions
return: {map_name : MapData object [with data in 'samples' attribute]}
"""
def get_data():
	map_data = {
		'grid'  : MapData("grid" ,getMapGrid()),
		'jelly' : MapData("jelly",getMapJelly()),
		'l'	  : MapData("l",getMapL())
	}

	for map_name, data_obj in map_data.items():
		filename = map_name + '.settrc'
		sample_id = ''
		flag_toggle = False
		toggle = 0
		actions = path = tokens = []
		start_pos = end_pos = -1
		for line in open( os.path.join(data_dir,filename) ):
			line=line.strip("\n")
			if line=='':
				#ipdb.set_trace()
				# reset variables
				flag_toggle = False
				toggle = 0
				actions = path = tokens = []
				start_pos = end_pos = -1
				sample_id = ''
				continue
			if line.startswith("Cleaned-"):
				prex = "Cleaned-"
				sample_id = line[len(prex):]
			if line.find('map=')!=-1:
				# ignore line: y=...  map=... x=...
				flag_toggle=True
				temp = line.split('\t')
				start_pos = int(temp[0][2:])	# y=...
				end_pos = int(temp[-1][2:])	# x=...
				continue
			if flag_toggle:
				if toggle==0:
					# read instructions
					tokens = word_tokenize(line)
				else:
					# read actions and path
					actions,path = get_actions_and_path(line,data_obj.map)
					# save new single-sentence sample
					data_obj.add_sample(tokens, actions, path, sample_id, start_pos, end_pos, map_name)
					# reset variables
					actions = path = tokens = []
				toggle = (toggle+1)%2
			#END-IF-TOGGLE
		#END-FOR-READ-FILE
	#END-FOR-MAPS

	return map_data

##########################################################################################
##########################################################################################

class Fold:
	def __init__(self,train_set,val_set,test_set,test_multi_set,vocab):
		self.train_data = train_set
		self.valid_data = val_set
		self.test_single_data = test_set
		self.test_multi_data = test_multi_set
		self.vocabulary = vocab
		self.vocabulary_size = len(vocab)

"""
Shuffles and splits data in train.val, and test sets for each fold conf
Each fold is one-map-out conf with (train:0.9,val:0.1)
return: [fold0,fold1,fold2]
"""
def get_folds_vDev(dir='data/',  val=0.1, force=False):
	pickle_file = 'folds_vDev.pickle'
	filename = os.path.join(dir,pickle_file)
	folds = []
	if force or not os.path.exists(filename):
		# Make pickle object
		dataByMap = get_data()
		map_names = dataByMap.keys()
		n_names = len(map_names)
		# Iteration over folds
		for i in range(n_names):
			# reset arrays
			train_set = []
			valid_set = []
			#
			test_single_set = dataByMap[map_names[i]].samples
			test_multi_set  = dataByMap[map_names[i]].get_multi_sentence_samples()
			for j in range(n_names):
				if j != i:
					# shuffle data before splitting
					data = np.array(dataByMap[map_names[j]].samples)	# shuffle in separate array, preserver order for multi_sentence building
					np.random.shuffle(data)
					# split into training and validation sets
					train_samples,valid_samples = train_test_split(	data,
																	test_size=val,
																	random_state = SEED)
					train_set.extend(train_samples)
					valid_set.extend(valid_samples)
			# Reformat to word index
			vocabulary = getVocabulary(train_set)
			train_set 			= reformat_wordid(train_set		,vocabulary)
			valid_set 			= reformat_wordid(valid_set		,vocabulary)
			test_single_set 	= reformat_wordid(test_single_set,vocabulary)
			#   for multi sentences
			temp = []
			for parag in test_multi_set:
				temp.append(reformat_wordid(parag,vocabulary))
			test_multi_set = temp
			# shuffle between maps
			np.random.shuffle(train_set)
			np.random.shuffle(valid_set)
			np.random.shuffle(test_single_set)
			np.random.shuffle(test_multi_set)
			#END-FOR-TRAIN-VAL-SPLIT
			folds.append( Fold(train_set,valid_set,test_single_set,test_multi_set,vocabulary) )
		#END-FOR-FOLDS
		print('Pickling %s.' % filename)
		try:
			with open(filename, 'wb') as f:
				pickle.dump(folds, f, pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save data to', filename, ':', e)
	else:
		with open(filename, 'rb') as f:
			folds = pickle.load(f)
			print('%s read from pickle...' % filename)
	return folds


"""
Shuffles and splits data in train and test sets for each fold conf
Each fold is one-map-out conf with (train:1.0)
Test set is for iteration stopping criteria
return: [fold0,fold1,fold2]
"""
def get_folds_vTest(dir='data/',  val=0.1, force=False):
	pickle_file = 'folds_vTest.pickle'
	filename = os.path.join(dir,pickle_file)
	folds = []
	if force or not os.path.exists(filename):
		# Make pickle object
		dataByMap = get_data()
		map_names = dataByMap.keys()
		n_names = len(map_names)
		# Iteration over folds
		for i in range(n_names):
			# reset arrays
			train_set = []
			valid_set = []
			#
			test_single_set = dataByMap[map_names[i]].samples
			test_multi_set  = dataByMap[map_names[i]].get_multi_sentence_samples()
			for j in range(n_names):
				if j != i:
					train_set.extend(dataByMap[map_names[j]].samples)
			# Reformat to word index
			vocabulary = getVocabulary(train_set)
			train_set 			= reformat_wordid(train_set		,vocabulary)
			valid_set 			= reformat_wordid(valid_set		,vocabulary)
			test_single_set 	= reformat_wordid(test_single_set,vocabulary)
			#   for multi sentences
			temp = []
			for parag in test_multi_set:
				temp.append(reformat_wordid(parag,vocabulary))
			test_multi_set = temp
			# shuffle between maps
			np.random.shuffle(train_set)
			np.random.shuffle(valid_set)
			np.random.shuffle(test_single_set)
			np.random.shuffle(test_multi_set)
			#END-FOR-TRAIN-VAL-SPLIT
			folds.append( Fold(train_set,valid_set,test_single_set,test_multi_set,vocabulary) )
		#END-FOR-FOLDS
		print('Pickling %s.' % filename)
		try:
			with open(filename, 'wb') as f:
				pickle.dump(folds, f, pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save data to', filename, ':', e)
	else:
		with open(filename, 'rb') as f:
			folds = pickle.load(f)
			print('%s read from pickle...' % filename)
	return folds

##########################################################################################
##########################################################################################
def getVocabulary(data):
	vocab = Counter()
	for sample in data:
		vocab.update(sample._instructions)
	frequency_threshold = 1 	# > THR
	vocab = [w for w,f in vocab.items() if f>frequency_threshold]
	vocab.append(EOS)
	vocab.append(PAD)
	vocab.append(RARE)
	vocab_dict = dict( zip(vocab,xrange(len(vocab))) )
	return vocab_dict

def reformat_wordid(data,vocab):
	# data: [Sample]
	wordid_data = []
	for sample in data:
		instructions = sample._instructions
		ref_ins = []
		for w in instructions:
			ref_ins.append( vocab[w] if w in vocab else vocab[RARE] )
		ref_ins.append(vocab[EOS])	# add end of sentence token
		new_sample = Sample(	ref_ins,
									sample._actions,
									sample._path,
									sample._id,
									sample._startPos,sample._endPos,
									sample._map_name)
		wordid_data.append(new_sample)
	return wordid_data

##########################################################################################
##########################################################################################

class BatchGenerator:
	def __init__(self,data,batch_size,vocab):
		self._encoder_unrollings = 49		# experimental
		self._decoder_unrollings = 31	# experimental
		self._data = data 	# format: [Sample(word_id format)]
		self._data_size = len(data)
		self._batch_size = batch_size
		self._vocabulary = vocab
		self._id2word = dict( zip(vocab.values(),vocab.keys()) )
		self._vocabulary_size = len(vocab)
		# batch splitting vars
		segment = self._data_size // batch_size
		self._cursor = [offset * segment for offset in range(batch_size)]	# make <segment> buckets from where extract one batch elmt each time		

	def get_batch(self):
		# Encoder and decoder batches in one-hot format [n_unroll,batch_size,{vocab_size,num_actions}]
		encoder_batch = [np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
								for _ in xrange(self._encoder_unrollings)]
		decoder_batch = [PAD_decode * np.ones(self._batch_size, dtype=np.int32)
								for _ in xrange(self._decoder_unrollings)] # list of action_ids, not one-hot repr

		sample_batch = [0 for _ in xrange(self._batch_size)]
		for b in xrange(self._batch_size):
			encoder_seq = self._data[ self._cursor[b] ]._instructions
			decoder_seq = self._data[ self._cursor[b] ]._actions
			# save original sample in batch
			sample_batch[b] = self._data[ self._cursor[b] ]
			# One-hot formatting for ENCODER
			for roll in xrange(len(encoder_seq)):
				encoder_batch[roll][b,encoder_seq[roll]] = 1.0

			# ZERO PADDING: if outside len of enc, leave empty (0s)
			## Action_id formatting for DECODER
			for i,act in enumerate(decoder_seq):
				decoder_batch[i][b] = act
			# update cursor for current segment
			self._cursor[b] = (self._cursor[b] + 1) % self._data_size
		#END-FOR-BATCH-SEGMENT
		decoder_batch = np.array(decoder_batch)
		return encoder_batch,decoder_batch,sample_batch

	def get_one_sample(self):
		# SGD instead of mini-batch, one sample at a time
		encoder_input = []
		encoder_seq = self._data[ self._cursor[0] ]._instructions
		decoder_seq = self._data[ self._cursor[0] ]._actions
		for i in xrange(len(encoder_seq)):
			roll = np.zeros(shape=(1,self._vocabulary_size),dtype=np.float)
			roll[0,encoder_seq[i]] = 1.0
			encoder_input.append(roll)
		sample = self._data[ self._cursor[0] ]
		self._cursor[0] = (self._cursor[0] + 1) % self._data_size
		return encoder_input,decoder_seq,[sample]


	def batch2string(self,encoder_batch,decoder_batch):
		for b in xrange(self._batch_size):
			print("Batch:",b)
			print("-"*70)
			print(" encoder: ",[self._id2word[encoder_batch[i][b,:].argmax()]
											if sum(encoder_batch[i][b,:])>0 else PAD
												for i in range(self._encoder_unrollings)])
			print(" decoder: ",[  actions_str[decoder_batch[i][b,:].argmax()] 
											if sum(decoder_batch[i][b,:])>0 else PAD
												for i in range(self._decoder_unrollings)])
			print("="*70)
		#END-FOR-BATCH_SIZE


##########################################################################################
def get_landmark_set(_map):
	# Iterates over all locations (intersections and ends) of map, extracting landmarks from map (context festures)
	# return dict {Meaning: id}
	feats = set()
	for loc in xrange(1,_map.NumPlaces):
		for angle in xrange(_map.NumPoses):
			views = _map.getView((loc,angle))
			views = views[0][0]
			[feats.add(feat) for view in views for feat in view if feat!=End and feat!=Empty]	#End and Empty noy in dictionary
	n_feats = len(feats)
	feat_dict = dict( zip(list(feats),xrange(n_feats)) )

	return feat_dict

def get_objects_set(_map):
	mid_objects = set()
	for loc in xrange(1,_map.NumPlaces):
		for angle in xrange(_map.NumPoses):
			views = _map.getView((loc,angle))
			views = views[0][0]
			if views[0][1]!=Empty:
				mid_objects.add(views[0][1]) #mid non-empty
	n_feats = len(mid_objects)
	feat_dict = dict( zip(list(mid_objects),xrange(n_feats)) )
	return feat_dict


def get_world_context_id(_map,place,pose):
	"""
	get bag-of-words repr of wold context from (place,pose)
	return:
	 featsByPose: [set(Meanings inst. in that direction)] x numPoses -> [fw,rg,bw,lf]
	 cell_object: Object in current cell (can be Empty)
	"""
	featsByPose = []
	cell_object = Empty
	for i in range(_map.NumPoses):
		curr_pose = (pose+i)%_map.NumPoses
		views = _map.getView((place,curr_pose))[0][0] # format: [(views,prob)]
		cell_object = views[0][1]	# no problem with overwriting it, bc it's the same in all directions
		curr_view = set()
		for j,view in enumerate(views):
			if j>0:
				curr_view.add(view[1])	#only add object if cell is not current
			curr_view.add(view[3])
			curr_view.add(view[4])
			curr_view.add(view[5])
		if Empty in curr_view:
			curr_view.remove(Empty)
		if End in curr_view:			# if End is found, replace with wall
			curr_view.remove(End)
			curr_view.add(Wall)
		featsByPose.append(curr_view)
	return featsByPose,cell_object


def get_sparse_world_context(_map,place,pose,feature_dict,object_dict):
	num_feats = len(feature_dict)
	num_objects = len(object_dict)
	y_t = np.zeros(shape=(1,num_objects + 4*num_feats),dtype=np.float32)
	featsByPose,cell_object = get_world_context_id(_map,place,pose)
	# add features for every direction
	for i,features in enumerate(featsByPose):
		ids = [feature_dict[feat] + i*num_feats for feat in features]
		y_t[0,ids] = 1.0
	# add object in current cell, if any
	if cell_object != Empty:
		y_t[0,4*num_feats+object_dict[cell_object]] = 1.0
	return y_t

def get_batch_world_context(sample_batch_roll,_t,_maps,feature_dict,object_dict,batch_size):
	"""
	args:
	sample_batch_roll: placeholder of shape [batch_size,1] with Sample obj data for ONE roll
	_t: time step (for path indexing)
	_map: {name: Map object}
	num_feats: number of map features 
	batch_size: <>

	return : world_state vector y [batch_size x 3 * num_feats]
	"""
	num_feats = len(feature_dict)
	num_objects = len(object_dict)
	roll_y = np.zeros(shape=(batch_size,4*num_feats + num_objects),dtype=np.float32)
	for b in xrange(batch_size):
		map_name = sample_batch_roll[b]._map_name
		if _t < len(sample_batch_roll[b]._path):
			# check world state for valid step
			x,y,pose = sample_batch_roll[b]._path[_t]
			place = _maps[map_name].locationByCoord[(x,y)]
			roll_y[b,:] = get_sparse_world_context(_maps[map_name],place,pose,feature_dict,object_dict)
	return roll_y


def move(state,action,_map):
	"""
	state; (xg,yg,pose)
	action: action_id
	_map: map object form getMap.*
	return: (next_loc,next_pos) next position state after applying action
	"""
	if action==STOP or action==PAD_decode:
		return -1

	xg,yg,pose = state
	
	if action==FW:
		nx = ny = -1
		if pose==0:
			nx,ny = xg-2,yg
		elif pose==1:
			nx,ny = xg,yg+2
		elif pose==2:
			nx,ny = xg+2,yg
		else: #pose==3
			nx,ny = xg,yg-2
		if (nx,ny) in _map.locationByCoord:
			return nx,ny,pose
		else:
			return -1
	elif action==L:
		return xg,yg,(pose-1)% _map.NumPoses
	elif action==R:
		return xg,yg,(pose+1)% _map.NumPoses


##########################################################################################
def test_dataset(dataByMap):
	for name,mapdata in dataByMap.items():
		for data in mapdata.samples:
			init_pos = data._path[0]
			end_pos  = data._path[-1]
			# get end pos from actions and init pos
			state = prev_state = init_pos
			end_followed = []
			followed_path = []
			for action in data._actions:
				prev_state = state
				state = move(state,action,mapdata.map)
				if state == -1:
					end_followed = prev_state
					break
				followed_path.append(prev_state)
			if end_followed==[]:
				end_followed = state
			followed_path.append(end_followed)

			if end_followed != end_pos or data._path!=followed_path:
				print("ID: ",data._id)
				print("True seq: %s" % (','.join([actions_str[act] for act in data._actions])))
				ipdb.set_trace()


##########################################################################################

"""
map_data = get_data()
#ms_data = map_data['jelly'].get_multi_sentence_samples()
input_lens = set()
output_lens = set()

for mname,md in map_data.items():
	input_len = set( [len(sample._instructions) for sample in md.samples] )
	output_len = set( [len(sample._actions) for sample in md.samples] )
	input_lens.update(input_len)
	output_lens.update(output_len)

iL = list(input_lens)
oL = list(output_lens)

ipdb.set_trace()
"""
#folds = get_folds_vDev()
#folds_vt = get_folds_vTest()

#batch_gen = BatchGenerator(folds[0].train_data,
#									2,
#									folds[0].vocabulary)

#enc,dec,samples = batch_gen.get_batch()
#batch_gen.batch2string(enc,dec)

#ipdb.set_trace()

"""
mg = getMapGrid()
mj = getMapJelly()
ml = getMapL()
#state = move((1,2),FW,mm)
ff = get_landmark_set(mj)
od = get_objects_set(mj)
y = get_sparse_world_context(mj,15,0,ff,od)
ipdb.set_trace()
"""

#map_data = get_data()
#test_dataset(map_data)