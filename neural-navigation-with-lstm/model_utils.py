import numpy as np
import tensorflow as tf
import time
import math
import ipdb

from utils import *
np.random.seed(SEED)
from nav_model import *
from test_baseline import *


def create_model(session,config_object,is_training=True, force_new=False):
	"""
	session: tensorflow session
	config_object: Config() instance with model parameters
	force_new: True: creates model with fresh parameters, False: load parameters from checkpoint file
	"""
	#model = NavModel(config_object,is_training=is_training)
	model = Baseline(config_object,is_training=is_training)
	ckpt = tf.train.get_checkpoint_state(model._train_dir)
	if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and not force_new:
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
		session.run(tf.initialize_variables(model.vars_to_init))
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
	num_ensembles = 3
	
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
				train_accuracy = float(train_acc)/steps_per_checkpoint
				train_acc = 0
				if verbose:
					perplexity = math.exp(loss) if loss < 100 else float('inf')
					print ("step %d | learning rate %.4f | step-time %.2f" % (step,model._learning_rate.eval(),step_time) )
					#print ("step %d step-time %.2f" % (step,step_time) )
					print ("   Training set  : loss: %.2f, perplexity: %.2f, accuracy: %.2f" % (loss, perplexity, 100.0*train_accuracy) )
				# Save checkpoint
				checkpoint_path = os.path.join(model._train_dir, "neural_walker.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model._global_step)

				# zero timer and loss.
				step_time, loss = 0.0, 0.0
				
				# Validation eval
				valid_accuracy,valid_loss = inference(model,sess,valid_gen)
				#valid_accuracy,valid_loss = inference_ensemble(config,valid_gen)  # Ensembling and beamsearch
				if verbose:
					perplexity = math.exp(valid_loss) if valid_loss < 100 else float('inf')
					print ("   Validation set: loss: %.3f, perplexity: %.3f, accuracy: %.4f" % (valid_loss, perplexity, 100.0*valid_accuracy) )
					print ("-"*80)
				# Save summaries
				feed_sum = {model._train_acc: 100.0*train_accuracy, model._val_acc: 100.0*valid_accuracy}
				[tacc,vacc] = sess.run([model._train_acc_sum,model._val_acc_sum],feed_dict=feed_sum)

				model._writer.add_summary(summary_str,step)
				model._writer.add_summary(tacc,step)
				model._writer.add_summary(vacc,step)
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
		test_accuracy,test_loss = inference(model,sess,test_gen)
		#test_accuracy,test_loss = inference_ensemble(config,test_gen)
		if verbose:
			test_perpl = math.exp(test_loss) if test_loss < 100 else float('inf')
			print ("   Test set: loss: %.3f, perplexity: %.3f, accuracy: %.4f" % (test_loss, test_perpl, 100.0*test_accuracy) )
			print ("-"*80)
	#END-SESSION-SCOPE
	return train_accuracy,valid_accuracy,test_accuracy
	
def inference(model,session,dataset_gen,beam_size=10):
	"""
	Performs inference over all dataset with beamsearch but not ensemble model averaging
	"""
	data_size = dataset_gen._data_size
	loss = 0.0
	accuracy = 0
	for _ in xrange(data_size):
		encoder_sample,decoder_sample,sample_obj = dataset_gen.get_one_sample()
		step_loss,corr = model.step_inference(
									session,
									encoder_sample,
									decoder_sample,
									sample_obj[0], # since it's only one sample
									beam_size=beam_size)
		loss += step_loss
		accuracy += corr
	accuracy = float(accuracy)/data_size
	loss /= data_size
	return accuracy,loss

def inference_ensemble(conf_object,dataset_gen,beam_size=10,num_ensembles=5):
	"""
	Performs inference over all dataset using ensemble model averaging and beam search
	"""
	sessions   = [tf.Session()]*num_ensembles
	ens_models = [create_model(sessions[i],conf_object,is_training=False) for i in xrange(num_ensembles)]
	data_size = dataset_gen._data_size
	loss = 0.0
	accuracy = 0
	step_time = time.time()
	for k in xrange(data_size):
		encoder_sample,decoder_sample,sample_obj = dataset_gen.get_one_sample()
		step_loss,corr = step_inference(
									ens_models,
									sessions,
									encoder_sample,
									decoder_sample,
									sample_obj[0], # since it's only one sample
									beam_size=beam_size)
		loss += step_loss
		accuracy += corr

		if k%10==0:
			new_t = time.time()
			print("-->%i: %.4f" % (k,new_t-step_time))
			step_time = new_t
	accuracy = float(accuracy)/data_size
	loss /= data_size
	return accuracy,loss


def step_inference(ens_models,sessions,encoder_inputs,decoder_output,sample_input,beam_size=10):
	"""
	Performs inference with beam search and ensemble models in the decoder.
	encoder_inputs: [1 x K]*enc_unroll
	decoder_output: true actions [dec_unroll]
	sample_input: Sample instance of current sample
	beam_size: beam size to use in beam search
	num_ensembles: number of ensemble models
	return : loss, correct_pred (True|False)
	"""
	# reinitialize models for new inference
	[session.run(tf.initialize_variables(model.vars_to_init)) for session,model in zip(sessions,ens_models)]
	num_ensembles = len(ens_models)

	feed_dict = [{}]*num_ensembles
	sts = list(xrange(num_ensembles))
	cts = list(xrange(num_ensembles))

	n_enc_unrolls = len(encoder_inputs)
	n_dec_unrolls = len(decoder_output)
	num_actions = ens_models[0]._num_actions

	for i,model in enumerate(ens_models):
		feed_dict[i][model._encoder_unrollings] = n_enc_unrolls
		feed_dict[i][model._decoder_unrollings] = n_dec_unrolls

		for roll in xrange(model._max_encoder_unrollings):
			if roll < n_enc_unrolls:
				feed_dict[i][model._encoder_inputs[roll]] = encoder_inputs[roll]
			else:
				feed_dict[i][model._encoder_inputs[roll]] = np.zeros(shape=(1,model._vocab_size),dtype=np.float32)
	
		# initial values for cell variables
		sts[i],cts[i] = sessions[i].run([model._test_s0,model._test_c0],feed_dict=feed_dict[i])

	end_state = sample_input._path[-1]
	pos_state = sample_input._path[0]
	prev_state = sample_input._path[0]
	_map = ens_models[0]._maps[sample_input._map_name]
	
	### DECODER
	def run_decoder_step(model,session,_feed_dict,_st,_ct,_state,len_seq):
		# one hot vector of current true action
		onehot_act = np.zeros((1,num_actions),dtype=np.float32)
		if len_seq < n_dec_unrolls:
			onehot_act[0,decoder_output[len_seq]] = 1.0
		# get world vector for current position
		x,y,pose = _state
		place = _map.locationByCoord[(x,y)]
		yt = get_sparse_world_context(_map, place, pose, model._map_feature_dict, model._map_objects_dict)
		# set placeholder for current roll
		_feed_dict[model._test_decoder_output] = onehot_act
		_feed_dict[model._test_st] = _st
		_feed_dict[model._test_ct] = _ct
		_feed_dict[model._test_yt] = yt

		output_feed = [
			model._next_st,
			model._next_ct,
			model._test_prediction,
			model._test_loss,
			]
		st,ct,prediction,step_loss = session.run(output_feed,feed_dict=_feed_dict)
		return [st,ct,prediction,step_loss]

	## BEAM SEARCH vars
	max_sequence=40
	nodes = {} 			# nodes[v] = parent(v)
	act_id = {} 		# act_id[node] = action id
	dist = {}
	node_loss={}		# node_loss[node] = loss of sequence so far
	terminal_nodes=[]	# [(final_prob,node)]
	Q = []				# [(log_prob,node)]
	n_nodes = 0

	#ipdb.set_trace()

	# first move
	avg_pred = avg_loss = 0.0
	for i,model in enumerate(ens_models):
		sts[i],cts[i],prediction,loss = run_decoder_step(
															ens_models[i],
															sessions[i],
															feed_dict[i],
															sts[i],cts[i],
															pos_state,0)
		avg_pred += prediction
		avg_loss += loss
	avg_pred /= num_ensembles
	avg_loss /= num_ensembles
	for i in range(num_actions):
		logprob = np.log(avg_pred[0,i]+1e-12)
		Q.append((logprob,n_nodes))
		new_node = BeamS_Node(_id=n_nodes,
									logprob=logprob,
									loss=avg_loss,
									parent=-1,
									pos_state=pos_state,
									dec_st=sts,dec_ct=cts,
									dist=0,
									act_id=i
									)
		nodes[n_nodes]=new_node
		n_nodes+=1

	while len(Q)!=0:	# keep rolling until stop criterion
		new_Q = []
		# use all current elmts in Q
		for prob,curr_node_id in Q:
			curr_node = nodes[curr_node_id]
			# discard long sequences and PAD-ended sequences
			if any([curr_node._dist>max_sequence,
					curr_node._act_id==PAD_decode,
					curr_node._pos_state==-1,
				]):
				continue
			# check if it's terminal
			if curr_node._act_id==STOP: # if it's STOP:
				terminal_nodes.append((prob,curr_node_id))
				continue
			# get next prob dist
			pos_state = move(curr_node._pos_state,curr_node._act_id,_map)
			if pos_state==-1:	# invalid move in current map
				continue

			avg_pred = avg_loss = 0.0
			new_st = new_ct = [0.0]*num_ensembles
			for i,model in enumerate(ens_models):
				new_st[i],new_ct[i],new_prediction,step_loss = run_decoder_step(
																					model,
																					sessions[i],
																					feed_dict[i],
																					curr_node._dec_st[i],
																					curr_node._dec_ct[i],
																					pos_state,
																					curr_node._dist)
				avg_pred += new_prediction
				avg_loss += step_loss	
			avg_pred /= num_ensembles
			avg_loss /= num_ensembles

			avg_pred = np.log(avg_pred+1.e-12)

			for i in range(num_actions):
				logprob = prob+avg_pred[0,i]
				new_Q.append((logprob,n_nodes))
				new_node = BeamS_Node(_id=n_nodes,
									logprob=logprob,
									loss=avg_loss,
									parent=curr_node_id,
									pos_state=pos_state,
									dec_st=new_st,dec_ct=new_ct,
									dist=curr_node._dist+1,
									act_id=i
									)
				nodes[n_nodes]=new_node
				n_nodes+=1
		new_Q.sort(reverse=True)
		Q = new_Q[:beam_size]
	#END-WHILE-BEAM_SEARCH
	terminal_nodes.sort(reverse=True)
	pred_actions = []
	node=nodes[terminal_nodes[0][1]]
	pred_end_state=node._pos_state
	loss = node._loss
	idx = node._id
	while idx!=-1:
		node = nodes[idx]
		pred_actions.append(node._act_id)
		idx = node._parent
	pred_actions.reverse()
		
	loss /= len(pred_actions)

	return loss,int(end_state==pred_end_state)	# for single-sentence


	