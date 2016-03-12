import tensorflow as tf
from tensorflow.models.rnn.rnn import bidirectional_rnn
import numpy as np
import sys,os
import ipdb
sys.path.append('code/MARCO')

from POMDP.MarkovLoc_Grid import getMapGrid
from POMDP.MarkovLoc_Jelly import getMapJelly
from POMDP.MarkovLoc_L import getMapL

from utils import get_landmark_set, get_batch_world_context, get_sparse_world_context, move


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
							learning_rate=0.1,
							learning_rate_decay_factor=0.1,
							embedding_world_state_size=30,
							dropout_rate=1.0
							):
		self.batch_size = batch_size
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

	def __init__(self, config):
		# Maps' feature dictionary
		self._map_feature_dict = set()
		for _map in self._maps.values():
			self._map_feature_dict.update( get_landmark_set(_map) )
		self._map_feature_dict = dict( zip(list(self._map_feature_dict),range(len(self._map_feature_dict))) )
		self._featmapById = dict( zip(self._map_feature_dict.values(),self._map_feature_dict.keys()) )

		self._batch_size 				= config.batch_size
		self._encoder_unrollings 	= config.encoder_unrollings
		self._decoder_unrollings 	= config.decoder_unrollings
		self._num_actions				= config.num_actions
		self._vocab_size 				= config.vocab_size
		self._y_size  					= len(self._map_feature_dict) * 3

		# Model parameters
		self._n_hidden 					 	= config.num_nodes 	# same for encoder and decoder
		self._embedding_world_state_size = config.embedding_world_state_size

		self._init_learning_rate 			= tf.constant(config.learning_rate)		
		self._learning_rate 		 			= self._init_learning_rate
		self._learning_rate_decay_factor = config.learning_rate_decay_factor

		self._max_gradient_norm	= config.max_gradient_norm
		self._global_norm 		= 0.0
		self._loss 					= 0.0

		# debug parameters
		self._train_dir = "tmp/"			# dir where checkpoint files will be saves
		
		# merging and writing vars
		self._writer = None


		# Dropout rate
		keep_prob = config.dropout_rate

		## TRAINING Placeholders
		self._encoder_inputs = []
		self._decoder_outputs = []
		self._world_state_vectors = [] 	# original sample structure, containing complete path, start_pos, end_pos, and map name
		for i in xrange(self._encoder_unrollings):
			self._encoder_inputs.append( tf.placeholder(tf.float32,shape=[self._batch_size,self._vocab_size], name='x') )
		for i in xrange(self._decoder_unrollings):
			self._decoder_outputs.append( tf.placeholder(tf.int32,shape=[self._batch_size], name='actions') )
			self._world_state_vectors.append( tf.placeholder(tf.float32,shape=[self._batch_size,self._y_size], name='wolrd_vect') )

		## TESTING / VALIDATION Placeholders
		self._test_encoder_inputs = []
		for i in xrange(self._encoder_unrollings):
			self._test_encoder_inputs.append( tf.placeholder(tf.float32,shape=[1,self._vocab_size], name='test_x') )
		self._test_decoder_output = tf.placeholder(tf.float32,shape=[1,self._num_actions], name='test_action_t')
		self._test_st = tf.placeholder(tf.float32, [1, self._n_hidden], name='test_st')
		self._test_ct = tf.placeholder(tf.float32, [1, self._n_hidden], name='test_ct')
		self._test_yt = tf.placeholder(tf.float32, [1, self._n_hidden], name='test_yt')

		with tf.name_scope('Weights') as scope:
			# Alignment model weights
			W_a = tf.Variable(tf.truncated_normal([self._n_hidden	 , self._n_hidden], -0.1, 0.1), name='W_a')
			U_a = tf.Variable(tf.truncated_normal([self._vocab_size, self._n_hidden], -0.1, 0.1), name='U_a')
			V_a = tf.Variable(tf.truncated_normal([2*self._n_hidden, self._n_hidden], -0.1, 0.1), name='V_a')
			v_a = tf.Variable(tf.truncated_normal([self._n_hidden	 ,1], -0.1, 0.1), name='v_a')
			tanh_bias_a = tf.Variable(tf.truncated_normal([1,1], -0.1, 0.1), name='tanh_bias_a')
			bias_a = tf.Variable(tf.zeros([1, self._n_hidden]), name='linear_bias_a')

			## Decoder variables
			# Input gate: input, previous output, context vector, and bias.
			ix = tf.Variable(tf.truncated_normal([self._embedding_world_state_size	 , self._n_hidden], -0.1, 0.1), name='ix')
			im = tf.Variable(tf.truncated_normal([self._n_hidden						 	 , self._n_hidden], -0.1, 0.1), name='ih')
			iz = tf.Variable(tf.truncated_normal([2*self._n_hidden + self._vocab_size, self._n_hidden], -0.1, 0.1), name='iz')
			ib = tf.Variable(tf.zeros([1, self._n_hidden]), name='ib')
			# Forget gate: input, previous output, context vector, and bias.
			fx = tf.Variable(tf.truncated_normal([self._embedding_world_state_size	 , self._n_hidden], -0.1, 0.1), name='fx')
			fm = tf.Variable(tf.truncated_normal([self._n_hidden							 , self._n_hidden], -0.1, 0.1), name='fh')
			fz = tf.Variable(tf.truncated_normal([2*self._n_hidden + self._vocab_size, self._n_hidden], -0.1, 0.1), name='fz')
			fb = tf.Variable(tf.zeros([1, self._n_hidden]), name='fb')
			# Memory cell: input, state, context vector, and bias.                             
			gx = tf.Variable(tf.truncated_normal([self._embedding_world_state_size	 , self._n_hidden], -0.1, 0.1), name='cx')
			gm = tf.Variable(tf.truncated_normal([self._n_hidden							 , self._n_hidden], -0.1, 0.1), name='cc')
			gz = tf.Variable(tf.truncated_normal([2*self._n_hidden + self._vocab_size, self._n_hidden], -0.1, 0.1), name='cz')
			gb = tf.Variable(tf.zeros([1, self._n_hidden]), name='cb')
			# Output gate: input, previous output, context vector, and bias.
			ox = tf.Variable(tf.truncated_normal([self._embedding_world_state_size	 , self._n_hidden], -0.1, 0.1), name='ox')
			om = tf.Variable(tf.truncated_normal([self._n_hidden							 , self._n_hidden], -0.1, 0.1), name='oh')
			oz = tf.Variable(tf.truncated_normal([2*self._n_hidden + self._vocab_size, self._n_hidden], -0.1, 0.1), name='oz')
			ob = tf.Variable(tf.zeros([1, self._n_hidden]), name='ob')
			# Embedding weight
			w_emby = tf.Variable(tf.truncated_normal([self._y_size,self._embedding_world_state_size], -0.1, 0.1), name='Ey_w')
			b_emby = tf.Variable(tf.zeros([1, self._embedding_world_state_size]), name='Ey_b')
			# Action Classifier weights and biases.
			ws = tf.Variable(tf.truncated_normal([self._n_hidden							 , self._embedding_world_state_size	], -0.1, 0.1), name='ws')
			wz = tf.Variable(tf.truncated_normal([2*self._n_hidden + self._vocab_size, self._embedding_world_state_size	], -0.1, 0.1), name='wz')
			wo = tf.Variable(tf.truncated_normal([self._embedding_world_state_size	 , self._num_actions						], -0.1, 0.1), name='wo')
			b_q = tf.Variable(tf.zeros([1,self._embedding_world_state_size	]), name='bq')
			b_o = tf.Variable(tf.zeros([1,self._num_actions						]), name='bo')

		#######################################################################################################################
		## Encoder
		with tf.name_scope('Encoder') as scope:
			lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self._n_hidden, forget_bias=1.0, input_size=self._vocab_size)
			lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self._n_hidden, forget_bias=1.0, input_size=self._vocab_size)

			def encoder(encoder_inputs, batch_size=self._batch_size, is_training=True):
				fw_cell = lstm_fw_cell
				bw_cell = lstm_bw_cell
				if is_training and keep_prob < 1.0:
					fw_cell = tf.nn.rnn_cell.DropoutWrapper(
											fw_cell, output_keep_prob=keep_prob)
					bw_cell = tf.nn.rnn_cell.DropoutWrapper(
											bw_cell, output_keep_prob=keep_prob)

				h = bidirectional_rnn(fw_cell,bw_cell,
											 encoder_inputs,
											 dtype=tf.float32,
											 sequence_length = self._encoder_unrollings * tf.ones([batch_size],tf.int64)
											 )
				return h

		def decoder_cell(i, o, z, c_prev):
			input_gate  = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + tf.matmul(z,iz) + ib)
			forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + tf.matmul(z,fz) + fb)
			output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + tf.matmul(z,oz) + ob)
			# gt
			update = tf.tanh(tf.matmul(i, gx) + tf.matmul(o, gm) + tf.matmul(z,gz) + gb)
			# ct
			c_t = forget_gate * c_prev + input_gate * update
			s_t = output_gate * tf.tanh(c_t)
			return s_t, c_t

		# Alignment model
		with tf.name_scope('Aligner') as scope:
			def context_vector(s_prev):
				# alignment model
				beta = [tf.matmul(tf.tanh(tf.matmul(s_prev,W_a) + u_v + bias_a),v_a) + tanh_bias_a for u_v in U_V_precalc]
				beta = tf.concat(1,beta, name='beta')
				# weights of each (xj,hj)
				alpha = tf.nn.softmax(beta)	# shape: batch_size x encoder_unroll
				alpha = tf.split(1,self._encoder_unrollings,alpha, name='alpha')	# list of unrolling, each elmt of shape [batch_size x 1]
				z_t = tf.Variable(tf.zeros([batch_size , 2*self._n_hidden + self._vocab_size]), name='z_t')
				for j in xrange(self._encoder_unrollings):
					xh = tf.concat(1,[encoder_inputs[j],h_encoder[j]], name='xhj')  # (x_j, h_j)
					z_t += alpha[j] * xh
				return z_t

			def precalc_Ux_Vh(encoder_inputs):
				ux_vh = []
				for i in xrange(self._encoder_unrollings):
					ux_vh.append( tf.matmul(encoder_inputs[i],U_a) + tf.matmul(h_encoder[i],V_a) )
				return ux_vh

		#######################################################################################################################

		def model_encoder_decoder(encoder_inputs, world_state_vectors, batch_size):
			h_encoder = encoder(encoder_inputs)	
			U_V_precalc = precalc_Ux_Vh(encoder_inputs)
			
			## Decoder loop
			with tf.name_scope('Decoder') as scope:
				# Initial states
				s_t = tf.Variable(tf.zeros([batch_size, self._n_hidden]), name='s_0')
				c_t = tf.Variable(tf.zeros([batch_size, self._n_hidden]), name='c_0')
				# Definition of the cell computation.

				logits = [] # logits per rolling
				predictions = []
				current_position = self._initial_position
				for i in xrange(self._decoder_unrollings):
					# world state vector at step i
					y_t = world_state_vectors[i]	# batch_size x num_local_feats (feat_id format)
					# embeed world vector | relu nodes
					ey = tf.nn.relu(tf.matmul(y_t,w_emby) + b_emby, name='Ey')
					# context vector
					z_t = context_vector(s_t)
					# Dropout
					ey = tf.nn.dropout(ey, keep_prob)
					s_t,c_t = decoder_cell(ey,s_t,z_t,c_t)
					s_t = tf.nn.dropout(s_t, keep_prob)
					# Hidden linear layer before output, proyects z_t,y_t, and s_t to an embeeding-size layer
					hq = ey + tf.matmul(s_t,ws) + tf.matmul(z_t,wz) + b_q
					# Output layer
					logit = tf.matmul(hq,wo) + b_o
					prediction = tf.nn.softmax(logit,name='prediction')
					logits.append(logit)
					predictions.append(prediction)
				#END-FOR-DECODER-UNROLLING
			#END-DECODER-SCOPE
			return logits,predictions
		#END-MODEL


		with tf.variable_scope('Train_test_pipeline') as scope:
			logits,self._train_predictions = model_encoder_decoder(self._encoder_inputs,
																					 self._world_state_vectors,
																					 batch_size=self._batch_size)
			scope.reuse_variables()

			self._loss = tf.nn.seq2seq.sequence_loss(logits,
																 targets=self._decoder_outputs,
																 weights=[tf.ones(shape=[self._batch_size],dtype=tf.float32) 
																 				for _ in range(self._decoder_unrollings)],
																 name='train_loss')
			# Optimizer setup
			self._global_step = tf.Variable(0,trainable=False)
			"""
			self._learning_rate = tf.train.exponential_decay(self._init_learning_rate,
															 self._global_step, 
															 5000,
															 self._learning_rate_decay_factor,
															 staircase=True)
			"""
			# debug variables
			params = tf.trainable_variables()

			#optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
			optimizer = tf.train.AdamOptimizer(learning_rate=self._init_learning_rate,
														  epsilon=1e-4)
			# Gradient clipping
			gradients, v = zip(*optimizer.compute_gradients(self._loss,params))
			self._clipped_gradients, self._global_norm = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
			# Apply clipped gradients
			optimizer = optimizer.apply_gradients( zip(self._clipped_gradients, v), global_step=self._global_step )

			##############################################################################################################
			## Testing
			test_h = encoder(self._test_encoder_inputs,1,False)
			test_ux_vh = precalc_Ux_Vh(self._test_encoder_inputs)
			# embeed world vector | relu nodes
			ey = tf.nn.relu(tf.matmul(self._test_yt,w_emby) + b_emby, name='Ey_test')
			# context vector
			z_t = context_vector(self._test_st)
			self._next_st,self._next_ct = decoder_cell(ey, self._test_st, z_t, self._test_ct)
			# Hidden linear layer before output, proyects z_t,y_t, and s_t to an embeeding-size layer
			hq = ey + tf.matmul(self._next_st,ws) + tf.matmul(z_t,wz) + b_q
			logit = tf.matmul(hq,wo) + b_o
			self._test_prediction = tf.nn.softmax(logit,name='inf_prediction')
			self._test_loss = tf.nn.softmax_cross_entropy_with_logits([logit],self._test_decoder_output, name="test_loss")


		# Summaries
		clipped_resh = [tf.reshape(tensor,[-1]) for tensor in self._clipped_gradients]
		clipped_resh = tf.concat(0,clipped_resh)
		_ = tf.scalar_summary("loss",self._loss)
		_ = tf.scalar_summary('global_norm',self._global_norm)
		_ = tf.scalar_summary('learning rate',self._learning_rate)
		_ = tf.histogram_summary('clipped_gradients', clipped_resh)

		# checkpoint saver
		self.saver = tf.train.Saver(tf.all_variables())
		self._merged = tf.merge_all_summaries()
		
	#END-INIT
	##########################################################################################
	def get_end_pos(self, actions, start_pos_grid, map_name):
		"""
		actions: [action_id]*dec_unrolls
		start_pos_grid: (xg,yg,pose)
		"""
		_map = self._maps[map_name]
		state = start_pos_grid
		prev_state = start_pos_grid
		for action in actions:
			prev_state = state
			if state == -1:
				ipdb.set_trace()

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
			actions = [roll[b,:].argmax() for roll in predictions]
			try:
				end_pred = self.get_end_pos(actions, init_pos, samples[b]._map_name)
			except:
				ipdb.set_trace()
			positive_samples += end_pos[:2]==end_pred[:2]	# compare to x,y, not pose

		return float(positive_samples)/batch_size

	##########################################################################################

	def training_step(self,session,encoder_inputs,decoder_outputs,sample_inputs):
		feed_dict = {}
		
		for i in xrange(self._encoder_unrollings):
			feed_dict[self._encoder_inputs[i]] = encoder_inputs[i]

		for i in xrange(self._decoder_unrollings):
			# action sequence
			feed_dict[self._decoder_outputs[i]] = decoder_outputs[i]
			# world_state vector Y
			y_roll = get_batch_world_context(sample_inputs,i,
														self._maps,
														self._map_feature_dict,
														self._batch_size)
			feed_dict[self._world_state_vectors[i]] = y_roll
		
		output_feed = [
			self._merged,
			self._loss,
		] + self._train_predictions
		outputs = session.run(output_feed,feed_dict=feed_dict)

		return outputs

	def inference_step(self,session,encoder_input,decoder_output,sample_input):
		"""
		Performs inference with beam search in the decoder.
		session: tensorflow session
		encoder_input: [1 x K]*enc_unroll
		decoder_output: true actions [dec_unroll]
		sample_input: Sample instance of current sample
		return : loss, correct_pred (True|False)
		"""
		feed_dict = {}
		end_state = sample_input._path[-1]
		for i in xrange(self._encoder_unrollings):
			feed_dict[self._test_encoder_inputs[i]] = encoder_input[i]

		# initial values for cell variables
		st = np.zeros((1,self._n_hidden),dtype=np.float32)
		ct = np.zeros((1,self._n_hidden),dtype=np.float32)
		state = sample_input._path[0]
		prev_state = sample_input._path[0]
		_map = self._maps[sample_input._map_name]
		
		loss = 0.0 	# must be averaged over predicted sequence
		n_preds = -1
		for i in xrange(self._decoder_unrollings):
			# one hot vector of current true action
			onehot_act = np.zeros((1,self._num_actions),dtype=np.float32)
			onehot_act[decoder_output[i]] = 1.0
			# get world vector for current position
			x,y,pose = state
			place = _map.locationByCoord[(x,y)]
			y_t = get_sparse_world_context(_map, place, pose, self._map_feature_dict)
			# set placeholder for current roll
			feed_dict[self._test_decoder_output] = onehot_act
			feed_dict[self._test_st] = st
			feed_dict[self._test_ct] = ct
			feed_dict[self._test_yt] = yt

			output_roll = [
				self._next_st,
				self._next_ct,
				self._test_prediction,
				self._test_loss,
				]
			st,ct,prediction,step_loss = session.run(output_feed,feed_dict=feed_dict)
			loss += step_loss
			# greedy prediction
			pred_act = prediction.argmax()
			# move according to prediction
			prev_state = state
			state = move(state,pred_act,_map)
			if state == -1:
				n_preds = i+1
				break
		if state != -1:
			prev_state = state
		loss /= (n_preds if n_preds!=-1 else self._decoder_unrollings)
		return loss,end_state==prev_state	# for single-sentence