"""
author: Ronald A. Cardenas
"""

import numpy as np
import tensorflow as tf
from utils import *
from nav_model import *
from test_baseline import Baseline
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
	#model = NavModel(config_object)
	model = Baseline(config_object)
	ckpt = tf.train.get_checkpoint_state(model._train_dir)
	if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and not force_new:
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		#model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.initialize_all_variables())
		# Merge summaries and write them
		model._writer = tf.train.SummaryWriter(model._train_dir+"nw_logs", session.graph_def)
	return model


def crossvalidate(folds,params,batch_size, num_steps, steps_per_checkpoint, verbose=True):
	"""
	args:
		folds: [Fold() instance]
		params: {dropout:[float], num_hidden: [int]}
		batch_size: To define batch generators
		num_steps: Total number of iterations for training
		steps_per_checkpoint: How many training steps to do per checkpoint
		verbose: print training log
	returns:
		best_params: {dropout: best_dp, num_hidden: best_nh}
		accuracies: [train_acc,valid_acc,test_acc], correspoding to the best parameters' configuration
	"""
	# best params according to ACC
	best_dropout = 0.0
	best_nh = 0
	max_acc = -np.inf
	final_train_acc = 0.0
	final_test_acc = 0.0
	n_folds = len(folds)
	# grid search through all parameter's combinations
	for dropout_rate in params["dropout"]:
		for nh in params['num_hidden']:
			# avg accuracies across folds
			avg_train_acc = 0.0
			avg_val_acc = 0.0
			avg_test_acc = 0.0

			for fold in folds:
				train_data = fold.train_data
				valid_data = fold.valid_data
				test_single_data = fold.test_single_data

				vocab = fold.vocabulary
				batch_size = batch_size
				if verbose:
					print("Initialize batch generators...")
				train_batch_gen = BatchGenerator(train_data		 , batch_size,vocab)
				valid_gen  		 = BatchGenerator(valid_data 		 , 			1,vocab)	# Batch of 1 sample, since 
				test_single_gen = BatchGenerator(test_single_data, 			1,vocab)	# test size changes across folds
				batch_gens = [train_batch_gen,valid_gen,test_single_gen]
				model_config = Config(batch_size=batch_size,
											 vocab_size=fold.vocabulary_size,
											 num_nodes=nh,
											 dropout_rate=dropout_rate)
				train_acc,val_acc,test_acc = train(model_config, batch_gens, num_steps, steps_per_checkpoint,verbose)
				avg_train_acc += train_acc / n_folds
				avg_val_acc   += val_acc   / n_folds
				avg_test_acc  += test_acc  / n_folds
			#END-FOR-FOLDS
			if avg_val_acc > max_acc:
				max_acc = avg_val_acc
				final_train_acc = avg_train_acc
				final_test_acc  = avg_test_acc
				best_dropout = dropout_rate
				best_nh = nh
			#END-UPDATE-PARAMS
		#END-FOR-N_HIDDEN
	#END-FOR-DROPOUT
	best_params = {
		'dropout': best_dropout,
		'num_hidden': best_nh
	}
	accs = [final_train_acc,max_acc,final_test_acc]
	return best_params,accs



def train(config,batch_gens,num_steps,steps_per_checkpoint,verbose=True,force_new_model=True):
	"""
	config: Config() instance, to initialize model
	batch_gens: batch generators
	num_steps: Total number of iterations for training
	steps_per_checkpoint: How many training steps to do per checkpoint
	verbose: print training log
	force_new_model: create fresh model with new parameters
	"""
	train_batch_gen, valid_gen, test_gen = batch_gens
	valid_size = valid_gen._data_size
	test_size = test_gen._data_size
	
	valid_task_metrics = []	# for early stopping criteria
	train_accuracy = 0.0
	valid_accuracy = 0.0
	test_accuracy  = 0.0
	with tf.Session() as sess:
		if verbose:
			print ("Creating model...")
		model = create_model(sess,config,force_new_model)	# create fresh model
		loss,step_time = 0.0,0.0
		train_acc = 0.0
		if verbose:
			print ("Training...")
		for step in xrange(num_steps):
			# Get a batch and make a step.
			start_time = time.time()
			if train_batch_gen._batch_size==1:
				encoder_batch,decoder_batch,sample_batch = train_batch_gen.get_one_sample()
			else:
				encoder_batch,decoder_batch,sample_batch = train_batch_gen.get_batch()
			# run predictions
			step_loss,summary_str,step_corr = model.training_step(sess,
																 					encoder_batch,
																 					decoder_batch,
																 					sample_batch)
			step_time += (time.time() - start_time) / steps_per_checkpoint
			loss += step_loss / steps_per_checkpoint
			train_acc += step_corr

			if step % steps_per_checkpoint == 0:
				# calculate last accuracy in training set
				train_accuracy = float(train_acc)/(step+1.0)
				if verbose:
					perplexity = math.exp(loss) if loss < 100 else float('inf')
					print ("step %d | learning rate %.4f | step-time %.2f" % (step,model._learning_rate.eval(),step_time) )
					#print ("step %d step-time %.2f" % (step,step_time) )
					print ("   Training set  : loss: %.2f, perplexity: %.2f, accuracy: %.2f" % (loss, perplexity, 100.0*train_accuracy) )
				# Save checkpoint
				#checkpoint_path = os.path.join(model._train_dir, "neural_walker.ckpt")
				#model.saver.save(sess, checkpoint_path, global_step=model._global_step)
				
				# Save summaries
				model._writer.add_summary(summary_str,step)
				# zero timer and loss.
				step_time, loss = 0.0, 0.0
				
				# Validation eval
				valid_loss = 0.0
				valid_accuracy = 0
				for _ in xrange(valid_size):
					encoder_batch,decoder_batch,sample_batch = valid_gen.get_one_sample()
					vloss,corr = model.inference_step(sess,
												   			 encoder_batch,
												   			 decoder_batch,
												   			 sample_batch[0])	# since it's only one sample
					valid_loss += vloss
					valid_accuracy += corr
				valid_accuracy = float(valid_accuracy)/valid_size
				if verbose:
					valid_loss /= float(valid_size)
					perplexity = math.exp(valid_loss) if valid_loss < 100 else float('inf')
					print ("   Validation set: loss: %.3f, perplexity: %.3f, accuracy: %.4f" % (valid_loss, perplexity, 100.0*valid_accuracy) )
					print ("-"*80)
				# early stoping criteria
				"""
				if len(valid_task_metrics)>3:
					if np.std(valid_task_metrics[-3:]) < 1e-6:
						break

				valid_task_metrics.append(accuracy)
				"""
			#END-IF-CHECKPOINT
		#END-FOR-STEPS
		model._writer.flush()
		# calc test single data metrics
		test_loss = 0.0
		for _ in xrange(test_size):
			encoder_batch,decoder_batch,sample_batch = test_gen.get_one_sample()
			vloss,corr = model.inference_step(sess,
										   			 encoder_batch,
										   			 decoder_batch,
										   			 sample_batch[0])	# since it's only one sample
			test_loss += vloss
			test_accuracy += corr
		test_accuracy = float(test_accuracy)/test_size
		if verbose:
			test_loss /= float(test_size)
			test_perpl = math.exp(test_loss) if test_loss < 100 else float('inf')
			print ("   Test set: loss: %.3f, perplexity: %.3f, accuracy: %.4f" % (test_loss, test_perpl, 100.0*test_accuracy) )
			print ("-"*80)
	#END-SESSION-SCOPE
	return train_accuracy,valid_accuracy,test_accuracy
	


if __name__=="__main__":
	# Get folds for experiments
	print("Reading data...")
	folds_vDev = get_folds_vDev(force=True)
	folds_vTest = get_folds_vTest(force=True)
	batch_size = 1
	num_steps = 60001
	steps_per_checkpoint = 2000	# How many training steps to do per checkpoint
	params = {
		'dropout': [0.9],
		'num_hidden': [500]
	}

	best_params,accs = crossvalidate(folds_vDev[:1],params,batch_size,num_steps,steps_per_checkpoint)
	train_acc,valid_acc,test_acc = accs

	print("Avg accuracies: train: %.2f | val: %.2f | test : %.2f" % (train_acc, valid_acc, test_acc))
	print("Best parameters: dropout: %.2f | number of hidden units: %i" % (best_params['dropout'],best_params['num_hidden']))