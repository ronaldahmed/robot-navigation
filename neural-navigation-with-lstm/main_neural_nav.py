"""
author: Ronald A. Cardenas
"""

import numpy as np
import tensorflow as tf
from utils import *
from nav_model import *
import time
import math
import ipdb

np.random.seed(SEED)


def create_model(session,config_object,is_training=True,force_new=False):
	model = NavModel(config_object,is_training)
	ckpt = tf.train.get_checkpoint_state(model._train_dir)
	if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and not force_new:
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.initialize_all_variables())
		# Merge summaries and write them
		model._writer = tf.train.SummaryWriter(model._train_dir+"nw_logs", session.graph_def)

	return model

if __name__=="__main__":
	# Get folds for experiments
	folds_vDev = get_folds_vDev()
	#folds_vTest = get_folds_vTest()
	train_data = folds_vDev[0].train_data
	vocab = folds_vDev[0].vocabulary
	V_size = folds_vDev[0].vocabulary_size
	batch_size = 5

	batch_gen = BatchGenerator(train_data,batch_size,vocab)
	model_config = Config(batch_size=batch_size,vocab_size=V_size)

	num_steps = 7
	steps_per_checkpoint = 2	# How many training steps to do per checkpoint

	with tf.Session() as sess:
		model = create_model(sess,model_config,True,True)
		loss,step_time = 0.0,0.0
		for step in xrange(num_steps):
			# Get a batch and make a step.
			start_time = time.time()
			encoder_batch,decoder_batch,sample_batch = batch_gen.get_batch()
			outputs = model.step(sess,
								 encoder_batch,
								 decoder_batch,
								 sample_batch)
			summary_str = outputs[0]
			step_loss = outputs[1]
			step_time += (time.time() - start_time) / steps_per_checkpoint
			loss += step_loss / steps_per_checkpoint

			if step % steps_per_checkpoint == 0:
				perplexity = math.exp(loss) if loss < 300 else float('inf')
				print ("step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (step, model._learning_rate.eval(),
                         step_time, perplexity))
				# Save checkpoint
				checkpoint_path = os.path.join(model._train_dir, "neural_walker.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model._global_step)
				# Save summaries
				model._writer.add_summary(summary_str,step)
				# zero timer and loss.
				step_time, loss = 0.0, 0.0
		#END-FOR-STEPS
		model._writer.flush()
	#END-SESSION-SCOPE
	