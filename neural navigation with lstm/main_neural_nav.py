"""
author: Ronald A. Cardenas
"""

import numpy as np
import tensorflow as tf
from utils import *
from nav_model import *
import ipdb
np.random.seed(SEED)


if __name__=="__main__":
	# Get folds for experiments
	folds_vDev = get_folds_vDev()
	#folds_vTest = get_folds_vTest()
	train_data = folds_vDev[0].train_data
	vocab = folds_vDev[0].vocabulary
	V_size = folds_vDev[0].vocabulary_size
	batch_size = 5

	batch_gen = BatchGenerator(train_data,batch_size,vocab)


	encoder_batch,decoder_batch,sample_batch = batch_gen.get_batch()


	model_config = Config(batch_size=batch_size,vocab_size=V_size)
	model = NavModel(model_config,True)
	with tf.Session() as sess:
		outputs = model.step(sess,
							 encoder_batch,
							 decoder_batch,
							 sample_batch)

		ipdb.set_trace()
	