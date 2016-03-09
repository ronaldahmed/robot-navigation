import tensorflow as tf
from tensorflow.models.rnn.rnn import bidirectional_rnn
import numpy as np
import sys,os
import ipdb
sys.path.append('code/MARCO')

from POMDP.MarkovLoc_Grid import getMapGrid
from POMDP.MarkovLoc_Jelly import getMapJelly
from POMDP.MarkovLoc_L import getMapL

from utils import get_landmark_set, get_batch_world_context, get_world_context



## max instructions length: 48
## max actions length: 31

class Config(object):
	encoder_unrollings = 48
	decoder_unrollings = 31
	num_actions = 4
	def __init__(self,batch_size,
							vocab_size,
							num_nodes=100,
							learning_rate=1.0,
							learning_rate_decay_factor=0.95,
							embedding_world_state_size=30,
							dropout_rate=0.5
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

	def __init__(self, config, is_training=True):
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
		self._n_hidden 						= config.num_nodes 	# same for encoder and decoder
		self._learning_rate 					= config.learning_rate
		self._learning_rate_decay_factor = config.learning_rate_decay_factor
		self._embedding_world_state_size = config.embedding_world_state_size
		# Dropout rate
		keep_prob = config.dropout_rate

		## Placeholders
		# Model input
		self._encoder_inputs = []
		self._decoder_outputs = []
		self._world_state_vectors = [] 	# original sample structure, containing complete path, start_pos, end_pos, and map name
		for _ in xrange(self._encoder_unrollings):
			self._encoder_inputs.append( tf.placeholder(tf.float32,shape=[self._batch_size,self._vocab_size]) )
		for _ in xrange(self._decoder_unrollings):
			self._decoder_outputs.append( tf.placeholder(tf.int32,shape=[self._batch_size]) )
			self._world_state_vectors.append( tf.placeholder(tf.int32,shape=[None]) )

		## Encoder variables
		lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self._n_hidden, forget_bias=1.0, input_size=self._vocab_size)
		lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self._n_hidden, forget_bias=1.0, input_size=self._vocab_size)
		if is_training and keep_prob < 1:
			lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
									lstm_fw_cell, output_keep_prob=keep_prob)
			lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
									lstm_bw_cell, output_keep_prob=keep_prob)


		h_encoder = bidirectional_rnn( lstm_fw_cell,lstm_bw_cell,
												 self._encoder_inputs,
												 dtype=tf.float32,
												 sequence_length = self._encoder_unrollings * tf.ones([self._batch_size],tf.int64)
												 )
		# Alignment model weights
		W_a = tf.Variable(tf.truncated_normal([self._n_hidden, self._n_hidden], -0.1, 0.1))
		U_a = tf.Variable(tf.truncated_normal([self._vocab_size,self._n_hidden], -0.1, 0.1))
		V_a = tf.Variable(tf.truncated_normal([2*self._n_hidden, self._n_hidden], -0.1, 0.1))
		v_a = tf.Variable(tf.truncated_normal([self._n_hidden,1], -0.1, 0.1))
		tanh_bias_a = tf.Variable(tf.truncated_normal([1,1], -0.1, 0.1))
		bias_a = tf.Variable(tf.zeros([1, self._n_hidden]))

		## Decoder variables
		# Input gate: input, previous output, context vector, and bias.
		ix = tf.Variable(tf.truncated_normal([self._embedding_world_state_size	 , self._n_hidden], -0.1, 0.1))
		im = tf.Variable(tf.truncated_normal([self._n_hidden						 	 , self._n_hidden], -0.1, 0.1))
		iz = tf.Variable(tf.truncated_normal([2*self._n_hidden + self._vocab_size, self._n_hidden], -0.1, 0.1))
		ib = tf.Variable(tf.zeros([1, self._n_hidden]))
		# Forget gate: input, previous output, context vector, and bias.
		fx = tf.Variable(tf.truncated_normal([self._embedding_world_state_size	 , self._n_hidden], -0.1, 0.1))
		fm = tf.Variable(tf.truncated_normal([self._n_hidden							 , self._n_hidden], -0.1, 0.1))
		fz = tf.Variable(tf.truncated_normal([2*self._n_hidden + self._vocab_size, self._n_hidden], -0.1, 0.1))
		fb = tf.Variable(tf.zeros([1, self._n_hidden]))
		# Memory cell: input, state, context vector, and bias.                             
		gx = tf.Variable(tf.truncated_normal([self._embedding_world_state_size	 , self._n_hidden], -0.1, 0.1))
		gm = tf.Variable(tf.truncated_normal([self._n_hidden							 , self._n_hidden], -0.1, 0.1))
		gz = tf.Variable(tf.truncated_normal([2*self._n_hidden + self._vocab_size, self._n_hidden], -0.1, 0.1))
		gb = tf.Variable(tf.zeros([1, self._n_hidden]))
		# Output gate: input, previous output, context vector, and bias.
		ox = tf.Variable(tf.truncated_normal([self._embedding_world_state_size	 , self._n_hidden], -0.1, 0.1))
		om = tf.Variable(tf.truncated_normal([self._n_hidden							 , self._n_hidden], -0.1, 0.1))
		oz = tf.Variable(tf.truncated_normal([2*self._n_hidden + self._vocab_size, self._n_hidden], -0.1, 0.1))
		ob = tf.Variable(tf.zeros([1, self._n_hidden]))
		# Embedding weight
		emb_y_w = tf.Variable(tf.truncated_normal([self._y_size,self._embedding_world_state_size], -0.1, 0.1))
		# Initial states
		s_t = tf.Variable(tf.zeros([self._batch_size, self._n_hidden]))
		c_t = tf.Variable(tf.zeros([self._batch_size, self._n_hidden]))

		# Action Classifier weights and biases.
		ws = tf.Variable(tf.truncated_normal([self._n_hidden							 , self._embedding_world_state_size	], -0.1, 0.1))
		wz = tf.Variable(tf.truncated_normal([2*self._n_hidden + self._vocab_size, self._embedding_world_state_size	], -0.1, 0.1))
		wo = tf.Variable(tf.truncated_normal([self._embedding_world_state_size	 , self._num_actions						], -0.1, 0.1))
		b_q = tf.Variable(tf.zeros([1,self._embedding_world_state_size	]))
		b_o = tf.Variable(tf.zeros([1,self._num_actions						]))
  
		# Definition of the cell computation.
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
		U_V_precalc = []
		for i in xrange(self._encoder_unrollings):
			U_V_precalc.append( tf.matmul(self._encoder_inputs[i],U_a) + tf.matmul(h_encoder[i],V_a) )
		
		def context_vector(s_prev):
			# alignment model
			beta = [tf.tanh(tf.matmul(s_prev,W_a) + u_v + bias_a) * v_a + tanh_bias_a for u_v in U_V_precalc]
			# weights of each (xj,hj)
			alpha = tf.nn.softmax(beta)	# shape: batch_size x encoder_unroll
			alpha = tf.split(1,self._encoder_unrollings,alpha)	# list of unrolling, each elmt of shape [batch_size x 1]
			z_t = tf.Variable(tf.zeros([self._batch_size , 2*self._n_hidden + self._vocab_size]))
			for j in xrange(self._encoder_unrollings):
				xh = tf.concat(1,[self._encoder_inputs[j],h_encoder[j]])  # (x_j, h_j)
				z_t += alpha[j] * xh
			return z_t


		self._test_ey = tf.nn.embedding_lookup(emb_y_w,self._world_state_vectors[0])
		"""
		# Decoder loop
		logits = [] # logits per rolling
		for i in xrange(self._decoder_unrollings):
			# world state vector at step i
			y_t = self._world_state_vectors[i]	# batch_size x num_local_feats (feat_id format)
			# embeed input
			ey = tf.nn.embedding_lookup(emb_y_w,y_t)
			# context vector
			z_t = context_vector(s_t)
			# Dropout
			if is_training and keep_prob < 1:
				ey = tf.nn.dropout(ey, keep_prob)
				# dropout on s_t discarded, doesn't matter for s_0
			s_t,c_t = decoder_cell(ey,s_t,z_t,c_t)

			if is_training and keep_prob < 1:
				s_t = tf.nn.dropout(s_t, keep_prob)

			# Hidden linear layer before output, proyects z_t,y_t, and s_t to an embeeding-size layer
			hq = ey + tf.matmul(s_t,ws) + tf.matmul(z_t,wz) + b_q
			# Output layer
			logit = tf.matmul(hq,wo) + b_o
			logits.append(logit)
			#predictions = tf.softmax(logits)
		self._loss = tf.nn.seq2seq.sequence_loss( logits,
																self._decoder_outputs,
																tf.ones([self._decoder_unrollings])	)
		"""
		
	#END-INIT
	##########################

	def step(self,session,encoder_inputs,decoder_outputs,sample_inputs):
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
			ipdb.set_trace()
			feed_dict[self._world_state_vectors[i]] = y_roll
		
		#output_feed = [self._loss]
		output_feed = [self._test_ey]
		outputs = session.run(output_feed,feed_dict=feed_dict)

		ipdb.set_trace()

		return outputs