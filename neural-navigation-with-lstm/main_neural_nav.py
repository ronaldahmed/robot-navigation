"""
author: Ronald A. Cardenas
"""

import numpy as np
import tensorflow as tf
from model_utils import *
import time
import math
import ipdb



if __name__=="__main__":
	# Get folds for experiments
	print("Reading data...")
	folds_vDev = get_folds_vDev()
	#folds_vTest = get_folds_vTest()
	batch_size = 1
	num_steps = 200001
	steps_per_checkpoint = 2000	# How many training steps to do per checkpoint
	params = {
		'dropout': [0.9],
		'num_hidden': [500]
	}

	best_params,accs = crossvalidate(folds_vDev[:1],params,batch_size,num_steps,steps_per_checkpoint)
	train_acc,valid_acc,test_acc = accs

	print("Avg accuracies: train: %.2f | val: %.2f | test : %.2f" % (train_acc, valid_acc, test_acc))
	print("Best parameters: dropout: %.2f | number of hidden units: %i" % (best_params['dropout'],best_params['num_hidden']))