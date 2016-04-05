import tensorflow as tf
import numpy as np
import sys,os
import ipdb
sys.path.append('code/MARCO')

from POMDP.MarkovLoc_Grid import getMapGrid
from POMDP.MarkovLoc_Jelly import getMapJelly
from POMDP.MarkovLoc_L import getMapL

from utils import get_landmark_set, get_objects_set, get_batch_world_context, get_sparse_world_context, move, actions_str
from custom_nn import *

## max instructions length: 48
## max actions length: 31

class Config(object):
	encoder_unrollings = 49
	decoder_unrollings = 31
	num_actions = 5
	max_gradient_norm = 5.0
	def __init__(self,batch_size,
							vocab_size,
							num_nodes=100,
							learning_rate=10.0,
							learning_rate_decay_factor=0.1,
							embedding_world_state_size=30,
							dropout_rate=1.0
							):
		self.vocab_size = vocab_size
		self.num_nodes = num_nodes
		self.learning_rate = learning_rate
		self.learning_rate_decay_factor = learning_rate_decay_factor
		self.embedding_world_state_size = embedding_world_state_size
		self.dropout_rate = dropout_rate


class NavModel(object):
	# define map objects
	_maps = {
			'grid'  : getMapGrid(),
			'jelly' : getMapJelly(),
			'l'	  : getMapL()
		}

	def __init__(self, config, is_training=True):
		# Maps' feature dictionary
		self._map_feature_dict = get_landmark_set(self._maps['grid']) # all featureas are the same for each map
		self._map_objects_dict = get_objects_set(self._maps['grid'])

		self._max_encoder_unrollings 	= config.encoder_unrollings
		self._max_decoder_unrollings 	= config.decoder_unrollings
		self._num_actions				= config.num_actions
		self._vocab_size 				= config.vocab_size
		self._y_size  					= 4*len(self._map_feature_dict) + len(self._map_objects_dict)

		# Model parameters
		self._n_hidden 					 	= config.num_nodes 	# same for encoder and decoder
		self._embedding_world_state_size = config.embedding_world_state_size

		self._init_learning_rate 			= tf.constant(config.learning_rate)		
		self._learning_rate 		 			= self._init_learning_rate
		self._learning_rate_decay_factor = config.learning_rate_decay_factor

		self._max_gradient_norm	= config.max_gradient_norm
		
		# debug parameters
		self._train_dir = "tmp/"			# dir where checkpoint files will be saves
		
		# merging and writing vars
		self._writer = None

		# Dropout rate
		keep_prob = config.dropout_rate

		## TRAINING Placeholders
		self._encoder_inputs = []
		self._encoder_unrollings = tf.placeholder('int64')

		self._decoder_outputs = []
		self._decoder_unrollings = tf.placeholder('int64')
		self._world_state_vectors = [] 	# original sample structure, containing complete path, start_pos, end_pos, and map name
		for i in xrange(self._max_encoder_unrollings):
			self._encoder_inputs.append( tf.placeholder(tf.float32,shape=[1,self._vocab_size], name='x') )
		for i in xrange(self._max_decoder_unrollings):
			self._decoder_outputs.append( tf.placeholder(tf.int32,shape=[1], name='actions') )
			self._world_state_vectors.append( tf.placeholder(tf.float32,shape=[1,self._y_size], name='world_vect') )

		## TESTING / VALIDATION Placeholders
		self._test_st = tf.placeholder(tf.float32, [1, self._n_hidden], name='test_st')
		self._test_ct = tf.placeholder(tf.float32, [1, self._n_hidden], name='test_ct')
		self._test_yt = tf.placeholder(tf.float32, [1, self._y_size], name='test_yt')
		self._test_decoder_output = tf.placeholder(tf.float32,shape=[1,self._num_actions], name='test_action')

		with tf.name_scope('Weights') as scope:
			# Alignment model weights
			W_a = tf.Variable(tf.truncated_normal([self._n_hidden	 , self._n_hidden], -0.1, 0.1), name='W_a')
			U_a = tf.Variable(tf.truncated_normal([self._vocab_size, self._n_hidden], -0.1, 0.1), name='U_a')
			V_a = tf.Variable(tf.truncated_normal([2*self._n_hidden, self._n_hidden], -0.1, 0.1), name='V_a')
			v_a = tf.Variable(tf.truncated_normal([self._n_hidden	 ,1], -0.1, 0.1), name='v_a')
			tanh_bias_a = tf.Variable(tf.truncated_normal([1,1], -0.1, 0.1), name='tanh_bias_a')
			bias_a = tf.Variable(tf.zeros([1, self._n_hidden]), name='linear_bias_a')

			# Embedding weight
			w_emby = tf.Variable(tf.truncated_normal([self._y_size,self._embedding_world_state_size], -0.1, 0.1), name='Ey_w')
			b_emby = tf.Variable(tf.zeros([1, self._embedding_world_state_size]), name='Ey_b')
			# Encoder - decoder transition
			w_trans_s = tf.Variable(tf.truncated_normal([self._n_hidden, self._n_hidden], -0.1, 0.1), name='w_trans_s')
			b_trans_s = tf.Variable(tf.zeros([1,self._n_hidden	]), name='b_trans_s')
			w_trans_c = tf.Variable(tf.truncated_normal([self._n_hidden, self._n_hidden], -0.1, 0.1), name='w_trans_c')
			b_trans_c = tf.Variable(tf.zeros([1,self._n_hidden	]), name='b_trans_c')
			# Action Classifier weights and biases.
			ws = tf.Variable(tf.truncated_normal([self._n_hidden							 , self._embedding_world_state_size	], -0.1, 0.1), name='ws')
			wz = tf.Variable(tf.truncated_normal([2*self._n_hidden + self._vocab_size, self._embedding_world_state_size	], -0.1, 0.1), name='wz')
			wo = tf.Variable(tf.truncated_normal([self._embedding_world_state_size	 , self._num_actions						], -0.1, 0.1), name='wo')
			b_q = tf.Variable(tf.zeros([1,self._embedding_world_state_size	]), name='bq')
			b_o = tf.Variable(tf.zeros([1,self._num_actions						]), name='bo')

		#######################################################################################################################
		## Encoder
		with tf.variable_scope('Encoder') as scope:
			fw_cell = CustomLSTMCell(self._n_hidden, forget_bias=1.0, input_size=self._vocab_size)
			bw_cell = CustomLSTMCell(self._n_hidden, forget_bias=1.0, input_size=self._vocab_size)

			fw_cell_dp = tf.nn.rnn_cell.DropoutWrapper(
									fw_cell, output_keep_prob=keep_prob)
			bw_cell_dp = tf.nn.rnn_cell.DropoutWrapper(
									bw_cell, output_keep_prob=keep_prob)

			h_encoder,c1,h1 = bidirectional_rnn(fw_cell_dp,bw_cell_dp,
										 self._encoder_inputs,
										 dtype=tf.float32,
										 sequence_length = self._encoder_unrollings*tf.ones([1],tf.int64),
										 scope='Encoder'
										 )
		#END-ENCODER-SCOPE

		#######################################################################################################################
		# Alignment model
		with tf.name_scope('Aligner') as scope:
			def context_vector(s_prev,h_encoder,ux_vh,encoder_inputs):
				# alignment model
				beta = []
				for i in xrange(self._max_encoder_unrollings):
					beta.append( tf.cond( tf.less(tf.constant(i,dtype=tf.int64),self._encoder_unrollings),
												 lambda: tf.matmul(tf.tanh(tf.matmul(s_prev,W_a) + ux_vh[i] + bias_a),v_a) + tanh_bias_a,
												 lambda: tf.zeros([1,1])
												)
									)
				beta = tf.concat(1,beta, name='beta')
				# weights of each (xj,hj)
				alpha = tf.nn.softmax(beta)	# shape: batch_size x encoder_unroll
				alpha = tf.split(1,self._max_encoder_unrollings,alpha, name='alpha')	# list of unrolling, each elmt of shape [batch_size x 1]
				z_t = tf.Variable(tf.zeros([1 , 2*self._n_hidden + self._vocab_size]), name='z_t')

				for j in xrange(self._max_encoder_unrollings):
					xh = tf.cond( tf.less(tf.constant(j,dtype=tf.int64),self._encoder_unrollings),
									  lambda: tf.concat(1,[encoder_inputs[j],h_encoder[j]], name='xhj'), # (x_j, h_j)
									  lambda: tf.zeros([1,2*self._n_hidden + self._vocab_size])
									)
					z_t += alpha[j] * xh
				return z_t

			def precalc_Ux_Vh(encoder_inputs,h_enc):
				ux_vh = []
				for i in xrange(self._max_encoder_unrollings):
					ux_vh.append( tf.cond( tf.less(tf.constant(i,dtype=tf.int64),self._encoder_unrollings),
												  lambda: tf.matmul(encoder_inputs[i],U_a) + tf.matmul(h_enc[i],V_a),
												  lambda: tf.zeros([1,self._n_hidden])
												)
						 			)
				return ux_vh

			U_V_precalc = precalc_Ux_Vh(self._encoder_inputs,h_encoder)

		#######################################################################################################################			
		## Decoder loop
		with tf.name_scope('Decoder') as scope:
			dec_cell = CustomLSTMCell(self._n_hidden, forget_bias=1.0, input_size=self._vocab_size)
			dec_cell_dp = tf.nn.rnn_cell.DropoutWrapper(
									dec_cell, output_keep_prob=keep_prob)
			# Initial states
			s_t = tf.tanh( tf.matmul(h1,w_trans_s)+b_trans_s , name='s_0')
			c_t = tf.tanh( tf.matmul(c1,w_trans_c)+b_trans_c , name='c_0')
			state = tf.concat(1,[c_t,s_t])

			logits = [] # logits per rolling
			self._train_predictions = []
			for i in xrange(self._max_decoder_unrollings):
				if i > 0: tf.get_variable_scope().reuse_variables()
				# world state vector at step i
				y_t = tf.cond( tf.less(tf.constant(i,dtype=tf.int64),self._decoder_unrollings),
									lambda: self._world_state_vectors[i],		# batch_size x num_local_feats (feat_id format)
									lambda: tf.zeros([1,self._y_size])
								)
				# embeed world vector | relu nodes
				ey = tf.nn.relu(tf.matmul(y_t,w_emby) + b_emby, name='Ey')
				# context vector
				z_t = context_vector(s_t,h_encoder,U_V_precalc,self._encoder_inputs)
				
				dec_input = tf.concat(1,[ey,z_t])
				s_t,state = dec_cell_dp(dec_input,state)#,scope="CustomLSTMCell")

				# Hidden linear layer before output, proyects z_t,y_t, and s_t to an embeeding-size layer
				hq = ey + tf.matmul(s_t,ws) + tf.matmul(z_t,wz) + b_q
				# Output layer
				logit = tf.matmul(hq,wo) + b_o
				fill_pred = tf.constant([0.,0.,0.,0.,1.])	# one-hot vector for PAD
				prediction = tf.cond( tf.less(tf.constant(i,dtype=tf.int64),self._decoder_unrollings),
									  lambda: tf.nn.softmax(logit,name='prediction'),
									  lambda: fill_pred
								)
				logits.append(logit)
				self._train_predictions.append(prediction)
			#END-FOR-DECODER-UNROLLING
			# Loss definition
			reshaped_dec_outputs = []
			for i in xrange(self._max_decoder_unrollings):
				out = tf.cond( tf.less(tf.constant(i,dtype=tf.int64),self._decoder_unrollings),
									lambda: self._decoder_outputs[i],
									lambda: 4*tf.ones([1],dtype=tf.int32)
					)
				reshaped_dec_outputs.append(out)
			self._loss = tf.nn.seq2seq.sequence_loss(logits,
																 targets=reshaped_dec_outputs,
																 weights=[tf.ones([1],dtype=tf.float32)]*self._max_decoder_unrollings,
																 #np.ones(shape=(self._max_decoder_unrollings,1),dtype=np.float32),
																 name='train_loss')

		###################################################################################################################
		# TESTING
		with tf.variable_scope('Encoder',reuse=True) as scope:
			test_h,c1,h1 = bidirectional_rnn(fw_cell,bw_cell,
										 self._encoder_inputs,
										 dtype=tf.float32,
										 sequence_length = self._encoder_unrollings*tf.ones([1],tf.int64),
										 scope='Encoder'
										 )
			
		self._test_s0 = tf.tanh( tf.matmul(h1,w_trans_s)+b_trans_s, name='test_s0')
		self._test_c0 = tf.tanh( tf.matmul(c1,w_trans_c)+b_trans_c, name='test_c0')

		test_ux_vh = precalc_Ux_Vh(self._encoder_inputs,test_h)

		with tf.variable_scope('Decoder',reuse=True) as scope:
			# embeed world vector | relu nodes
			ey = tf.nn.relu(tf.matmul(self._test_yt,w_emby) + b_emby, name='Ey_test')
			# context vector
			z_t = context_vector(self._test_st,test_h,test_ux_vh,self._encoder_inputs)
			
			state = tf.concat(1,[self._test_ct,self._test_st])
			dec_input = tf.concat(1,[ey,z_t])

			_,temp = dec_cell(dec_input, state)#,scope="CustomLSTMCell")
			self._next_ct,self._next_st = tf.split(1,2,temp)

			# Hidden linear layer before output, proyects z_t,y_t, and s_t to an embeeding-size layer
			hq = ey + tf.matmul(self._next_st,ws) + tf.matmul(z_t,wz) + b_q
			logit = tf.matmul(hq,wo) + b_o
			self._test_prediction = tf.nn.softmax(logit,name='inf_prediction')
			# Loss definition
			self._test_loss = tf.nn.softmax_cross_entropy_with_logits(logit,self._test_decoder_output, name="test_loss")
		#END-DECODER-SCOPE

		
		with tf.variable_scope('Optimization') as scope:
			# Optimizer setup
			self._global_step = tf.Variable(0,trainable=False)
			
			#self._learning_rate = tf.train.exponential_decay(self._init_learning_rate,
			#												 self._global_step, 
			#												 5000,
			#												 self._learning_rate_decay_factor,
			#												 staircase=True)
			
			
			#params = tf.trainable_variables()
			#optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
			optimizer = tf.train.AdamOptimizer(learning_rate=self._init_learning_rate,
														  epsilon=1e-1)
			# Gradient clipping
			#gradients = tf.gradients(self._loss,params)
			gradients,params = zip(*optimizer.compute_gradients(self._loss))
			self._clipped_gradients, self._global_norm = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
			# Apply clipped gradients
			self._optimizer = optimizer.apply_gradients( zip(self._clipped_gradients, params) )#, global_step=self._global_step )

		# Summaries
		clipped_resh = [tf.reshape(tensor,[-1]) for tensor in self._clipped_gradients if tensor]
		clipped_resh = tf.concat(0,clipped_resh)
		_ = tf.scalar_summary("loss",self._loss)
		_ = tf.scalar_summary('global_norm',self._global_norm)
		_ = tf.scalar_summary('learning rate',self._learning_rate)
		_ = tf.histogram_summary('clipped_gradients', clipped_resh)

		# checkpoint saver
		#self.saver = tf.train.Saver(tf.all_variables())
		self._merged = tf.merge_all_summaries()
		
		
	#END-INIT
	##########################################################################################
	def get_end_pos(self, actions, start_pos_grid, map_name):
		"""
		actions: [1 x num_actions]*dec_unrolls | prob distributions of actions
		start_pos_grid: (xg,yg,pose)
		"""
		_map = self._maps[map_name]
		state = start_pos_grid
		prev_state = start_pos_grid
		for action_distro in actions:
			action = action_distro.argmax()
			prev_state = state
			state = move(state,action,_map)
			if state == -1:
				return prev_state
		return state

	def get_endpoint_accuracy(self,samples,predictions):
		"""
		samples: [Sample()]*batch_size
		predictions: [batch_size x num_actions]*num_dec_unroll
		"""
		dec_unrolls = len(predictions)
		batch_size = len(samples)
		positive_samples = 0
		for b in xrange(batch_size):
			init_pos = samples[b]._path[0]
			end_pos  = samples[b]._path[-1]
			actions = [roll[b,:] for roll in predictions]
			try:
				end_pred = self.get_end_pos(actions, init_pos, samples[b]._map_name)
			except:
				ipdb.set_trace()
			#positive_samples += end_pos[:2]==end_pred[:2]	# compare to x,y, not pose
			positive_samples += int(end_pos==end_pred)	# compare to x,y,pose

		return float(positive_samples)/batch_size

	##########################################################################################

	def training_step(self,session,encoder_inputs,decoder_outputs,sample_inputs):
		"""
		Runs training step for one simple sample
		returns:
			loss, summary_string, correct (1:correctly predicted| 0: mispredicted)
		"""
		feed_dict = {}
		n_enc_unrolls = len(encoder_inputs)
		n_dec_unrolls = len(decoder_outputs)
		feed_dict[self._encoder_unrollings] = n_enc_unrolls
		feed_dict[self._decoder_unrollings] = n_dec_unrolls
		sample_input = sample_inputs[0]

		for i in xrange(self._max_encoder_unrollings):
			if i < n_enc_unrolls:
				feed_dict[self._encoder_inputs[i]] = encoder_inputs[i]
			else:
				feed_dict[self._encoder_inputs[i]] = np.zeros(shape=(1,self._vocab_size),dtype=np.float32)
		_map = self._maps[sample_input._map_name]
		for i in xrange(self._max_decoder_unrollings):
			if i < n_dec_unrolls:
				# world_state vector Y
				x,y,pose = sample_input._path[i]
				place = _map.locationByCoord[(x,y)]
				y_roll = get_sparse_world_context(_map, place, pose, self._map_feature_dict, self._map_objects_dict)
				feed_dict[self._decoder_outputs[i]] 	 = [decoder_outputs[i]]
				feed_dict[self._world_state_vectors[i]] = y_roll
			else:
				feed_dict[self._decoder_outputs[i]] 	 = np.zeros(shape=(1),dtype=np.float32)
				feed_dict[self._world_state_vectors[i]] = np.zeros(shape=(1,self._y_size),dtype=np.float32)
		
		output_feed = [
			self._optimizer,
			self._loss,
			self._learning_rate,
			self._merged,
		] + self._train_predictions
		outputs = session.run(output_feed,feed_dict=feed_dict)
		predictions = outputs[4:]

		correct = self.get_endpoint_accuracy(sample_inputs,predictions)
		"""
		temp =[]
		for act in sample_input._actions:
			tt = np.zeros((1,self._num_actions))
			tt[0,act] = 1.0
			temp.append(tt)
		correct = self.get_endpoint_accuracy(sample_inputs,temp)

		if correct==0.0:
			ipdb.set_trace()
			correct = self.get_endpoint_accuracy(sample_inputs,temp)
		"""
		return (outputs[1],outputs[3],correct)	#loss, summary_str, correct


	def inference_step(self,session,encoder_inputs,decoder_output,sample_input):
		"""
		Performs inference with beam search in the decoder.
		session: tensorflow session
		encoder_inputs: [1 x K]*enc_unroll
		decoder_output: true actions [dec_unroll]
		sample_input: Sample instance of current sample
		return : loss, correct_pred (True|False)
		"""
		feed_dict = {}
		n_enc_unrolls = len(encoder_inputs)
		n_dec_unrolls = len(decoder_output)
		feed_dict[self._encoder_unrollings] = n_enc_unrolls
		feed_dict[self._decoder_unrollings] = n_dec_unrolls

		end_state = sample_input._path[-1]
		for i in xrange(self._max_encoder_unrollings):
			if i < n_enc_unrolls:
				feed_dict[self._encoder_inputs[i]] = encoder_inputs[i]
			else:
				feed_dict[self._encoder_inputs[i]] = np.zeros(shape=(1,self._vocab_size),dtype=np.float32)

		# initial values for cell variables
		[st,ct] = session.run([self._test_s0,self._test_c0],feed_dict=feed_dict)
		state = sample_input._path[0]
		prev_state = sample_input._path[0]
		_map = self._maps[sample_input._map_name]
		
		loss = 0.0 	# must be averaged over predicted sequence length
		npreds = 0
		predicted_acts = []

		#ipdb.set_trace()

		while(True):	# keep rolling until stop criterion
			# one hot vector of current true action
			onehot_act = np.zeros((1,self._num_actions),dtype=np.float32)
			if npreds < n_dec_unrolls:
				onehot_act[0,decoder_output[npreds]] = 1.0
			# get world vector for current position
			x,y,pose = state
			place = _map.locationByCoord[(x,y)]
			yt = get_sparse_world_context(_map, place, pose, self._map_feature_dict, self._map_objects_dict)
			# set placeholder for current roll
			feed_dict[self._test_decoder_output] = onehot_act
			feed_dict[self._test_st] = st
			feed_dict[self._test_ct] = ct
			feed_dict[self._test_yt] = yt

			output_feed = [
				self._next_st,
				self._next_ct,
				self._test_prediction,
				self._test_loss,
				]
			st,ct,prediction,step_loss = session.run(output_feed,feed_dict=feed_dict)
			if npreds < n_dec_unrolls:
				loss += step_loss
			npreds += 1
			# greedy prediction
			pred_act = prediction.argmax()
			predicted_acts.append(pred_act)
			# move according to prediction
			prev_state = state
			state = move(state,pred_act,_map)
			if state==-1 or npreds>=self._max_decoder_unrollings:
				break
			
		loss /= min(npreds,n_dec_unrolls)
		"""
		if self.kk>1890:
			## DEBUG
			print("True seq: %s" % (','.join([actions_str[act] for act in decoder_output])))
			print("Pred seq: %s" % (','.join([actions_str[act] for act in predicted_acts])))

			ipdb.set_trace()
		self.kk+=1
		"""
		

		return loss,int(end_state==prev_state)	# for single-sentence