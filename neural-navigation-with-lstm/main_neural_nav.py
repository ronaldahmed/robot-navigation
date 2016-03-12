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

def create_model(session,config_object,force_new=False):
	"""
	session: tensorflow session
	config_object: Config() instance with model parameters
	force_new: True: creates model with fresh parameters, False: load parameters from checkpoint file
	"""
	model = NavModel(config_object)
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


def train(fold, params, batch_size, num_steps, steps_per_checkpoint):
	"""
	fold: Fold instance
	params: {dropout:[float], num_hidden: [int]}
	steps_per_checkpoint: How many training steps to do per checkpoint
	"""
	train_data = fold.train_data
	valid_data = fold.valid_data[:10]
	test_single_data = fold.test_single_data[:10]

	valid_size = len(valid_data)
	test_size = len(test_single_data)

	vocab = fold.vocabulary
	batch_size = batch_size

	print("Organize in mini-batches...")
	train_batch_gen = BatchGenerator(train_data		 , batch_size,vocab)
	valid_gen  		 = BatchGenerator(valid_data 		 , 			1,vocab)	# Batch of 1 sample, since 
	test_single_gen = BatchGenerator(test_single_data, 			1,vocab)	# test size changes across folds

	# best params according to ACC
	best_dropout = 0.0
	best_nh = 0
	max_acc = -np.inf
	test_acc = 0.0
	test_loss = 0.0
	test_perpl = 0.0

	# Cross validation iteration for FOLD
	for dropout_rate in params["dropout"]:
		for nh in params['num_hidden']:
			model_config = Config(batch_size=batch_size,
										 vocab_size=fold.vocabulary_size,
										 num_nodes=nh,
										 dropout_rate=dropout_rate)
			valid_task_metrics = []	# for early stopping criteria
			with tf.Session() as sess:
				print ("Creating model...")
				model = create_model(sess,model_config,True)	# create fresh model
				loss,step_time = 0.0,0.0
				print ("Training...")
				for step in xrange(num_steps):
					# Get a batch and make a step.
					start_time = time.time()
					encoder_batch,decoder_batch,sample_batch = train_batch_gen.get_batch()
					# run predictions
					outputs = model.training_step(sess,
										 					encoder_batch,
										 					decoder_batch,
										 					sample_batch)
					summary_str = outputs[0]
					step_loss = outputs[1]
					step_time += (time.time() - start_time) / steps_per_checkpoint
					loss += step_loss / steps_per_checkpoint

					if step % steps_per_checkpoint == 0:
						perplexity = math.exp(loss) if loss < 300 else float('inf')
						predictions = outputs[2:]
						accuracy = model.get_endpoint_accuracy(sample_batch,predictions)

						print ("step %d learning rate %.4f step-time %.2f" % (step, model._learning_rate.eval(),step_time) )
						print ("   Training set  : perplexity: %.2f, accuracy: %.2f" % (perplexity, 100.0*accuracy) )

						# Save checkpoint
						#checkpoint_path = os.path.join(model._train_dir, "neural_walker.ckpt")
						#model.saver.save(sess, checkpoint_path, global_step=model._global_step)
						# Save summaries
						#model._writer.add_summary(summary_str,step)
						# zero timer and loss.
						step_time, loss = 0.0, 0.0
						
						# Validation eval
						valid_loss = 0.0
						accuracy = 0
						for _ in xrange(valid_size):
							encoder_batch,decoder_batch,sample_batch = valid_gen.get_batch()
							vloss,corr = model.inference_step(sess,
														   			 encoder_batch,
														   			 decoder_batch,
														   			 sample_batch[0])	# since it's only one sample
							valid_loss += vloss
							accuracy += corr
						accuracy = float(accuracy)/valid_size
						perplexity = math.exp(valid_loss) if valid_loss < 300 else float('inf')
						print ("   Validation set: perplexity: %.2f, accuracy: %.2f" % (perplexity, 100.0*accuracy) )
						print ("-"*80)

						# update best parameters
						if accuracy > max_acc:
							max_acc = accuracy
							best_dropout = dropout_rate
							best_nh = nh
							# calc test single data metrics
							test_loss = 0.0
							test_acc = 0.0
							for _ in xrange(test_size):
								encoder_batch,decoder_batch,sample_batch = valid_gen.get_batch()
								vloss,corr = model.inference_step(sess,
															   			 encoder_batch,
															   			 decoder_batch,
															   			 sample_batch[0])	# since it's only one sample
								test_loss += vloss
								test_acc += corr
							test_acc = float(test_acc)/test_size
							test_perpl = math.exp(test_loss) if test_loss < 300 else float('inf')

						# early stoping criteria
						if len(valid_task_metrics)>3:
							if np.std(valid_task_metrics[-3:]) < 1e-6:
								break
						valid_task_metrics.append(accuracy)

					#END-IF-CHECKPOINT
				#END-FOR-STEPS
				#model._writer.flush()
			#END-SESSION-SCOPE
		#END-FOR-NUM_HIDDEN_UNITS
	#END-FOR-DROPOUT_RATE
	print ("="*80)
	print ("Best parameters: dropout:%.2f, num_hidden:%i" % (best_dropout,best_nh))
	print ("Testing set: loss:%.2f, perplexity: %.2f, accuracy: %.2f" %(test_loss, test_perpl, test_acc))




if __name__=="__main__":
	# Get folds for experiments
	print("Reading data...")
	folds_vDev = get_folds_vDev()
	#folds_vTest = get_folds_vTest()
	batch_size = 5
	num_steps = 3
	steps_per_checkpoint = 1	# How many training steps to do per checkpoint
	params = {
		'dropout': [1.0],
		'num_hidden': [10]
	}
	fold = folds_vDev[0]
	train(fold, params, batch_size, num_steps, steps_per_checkpoint)

	