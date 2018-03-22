import os.path
import sys
import json
import tensorflow as tf
import numpy as np
import extract_training_data as extractor
from models.conv_lstm_model import Model
#from models.deep_conv_model import Model
from preprocessing.talk import Talk
from preprocessing.audio_tools import Sound
from preprocessing.subtitle import Subtitle
from nltk import word_tokenize
from timing_demo import TimingDemo
import scipy.stats
from scipy.optimize import fmin_cobyla
from scipy.optimize import fmin_slsqp
import argparse
import re

def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	current_dir = os.path.dirname(__file__)
	return os.path.abspath(os.path.join(current_dir, relpath))

if not os.path.isfile(extractor.frequent_words_path) or not os.path.isfile(extractor.word_timings_path):
	print("Execute extract_training_data first.")
	exit()
frequent_words = json.load(open(extractor.frequent_words_path))
word_timings = json.load(open(extractor.word_timings_path))

prior_probabilities_path = _path("data/training/word_priors.json")
if not os.path.isfile(prior_probabilities_path):
	print("Execute prior_probabilities.py first.")
	exit()
prior_probabilities_dict = json.load(open(prior_probabilities_path))
prior_probabilities = np.array([p for _, p in prior_probabilities_dict.items()])

def argmax_n(arr, n=1):
	"""
	Computes the n maximum entries and returns the respective indices
	"""
	max_ind = []
	while len(max_ind) < n:
		ind = np.argmax(arr)
		arr[ind] = -np.inf
		max_ind.append(ind)
	return max_ind

def cost_function(x, probs, interval_count, word_indices):
	out = 0.0
	for word_ind, t in enumerate(x):
		interval_midpoints = np.linspace(0.5*extractor.INTERVAL_SIZE, interval_count*extractor.INTERVAL_SIZE, num=interval_count)
		interval_diffs = interval_midpoints-t
		sd_scalar = 6
		interval_scalars = np.exp(-interval_diffs**2 / (2*(sd_scalar*extractor.DATA_SD)**2)) / (np.sqrt(2*np.pi*(sd_scalar*extractor.DATA_SD)**2))
		# interval_scalars = scipy.stats.norm(t, extractor.DATA_SD).pdf(interval_midpoints)
		out += interval_scalars.dot(probs[:,word_indices[word_ind]])
	return -out

def cost_function_gradient(x, probs, interval_count, word_indices):
	print("starting gradient eval")
	out = np.zeros((len(x),))
	for word_ind, t in enumerate(x):
		interval_midpoints = np.linspace(0.5*extractor.INTERVAL_SIZE, interval_count*extractor.INTERVAL_SIZE, num=interval_count)
		interval_diffs = interval_midpoints-t
		interval_scalars = np.exp(-interval_diffs**2 / (2*extractor.DATA_SD**2)) / (np.sqrt(2*np.pi*extractor.DATA_SD**2))
		interval_scalars = np.multiply(interval_scalars, probs[:,word_indices[word_ind]])
		for j, tj in enumerate(x):
			gradient_inner = (interval_midpoints-tj)*tj / (2*extractor.DATA_SD**2)
			out[j] += interval_scalars.dot(gradient_inner)
	print("done")
	return -out

def constraint_function(x, probs=[], interval_count=0, word_indices=[]):
	"""
	Enforces correct word order
	"""
	return np.array(x[1:])-np.array(x[:-1])

def constrain_function_jacobian(x, probs=[], interval_count=0, word_indices=[]):
	print("starting cost function jac eval")
	out = np.zeros((len(x)-1, len(x)))
	for i in range(len(x)-1):
		out[i,i] = -1
		out[i,i+1] = 1
	print("done")
	return out

if __name__ == '__main__':
	arguments = argparse.ArgumentParser()
	arguments.add_argument("-baseline", action="store_true")
	arguments.add_argument("id", help="TED talk id.")
	options = arguments.parse_args()
	talk_id = int(options.id)

	# start a new tensorflow session
	sess = tf.InteractiveSession()

	# load model
	model_load_checkpoint = _path("training_data/run_2018-03-21-23_9fbe4594184ad6a9224f865a2bdfd407/train/model.ckpt-18")
	input_3d = tf.placeholder(tf.float32, [None, 80, 13], name="input_3d")
	model = Model()
	prediction = model.test_model(input_3d)
	model.load_variables_from_checkpoint(sess, model_load_checkpoint)

	# load input
	talk = Talk(talk_id)

	mfcc_per_interval = int(extractor.INTERVAL_SIZE / 0.005)
	mfcc_features = np.load(talk.features_path())
	interval_count = int(mfcc_features.shape[0] // mfcc_per_interval)
	mfcc_features = mfcc_features[:interval_count*mfcc_per_interval]
	mfcc_features = mfcc_features.reshape((interval_count,mfcc_per_interval,13))

	# perform prediction
	prediction_vals = np.zeros((0,1500))
	if not options.baseline:
		batch_size = 50
		while mfcc_features.shape[0] > 0:
			if batch_size > mfcc_features.shape[0]:
				batch_size = mfcc_features.shape[0]
			chunk = mfcc_features[:batch_size]
			mfcc_features = mfcc_features[batch_size:]
			val_prediction = sess.run(
				[prediction],
				feed_dict={
					input_3d: chunk
				}
			)
			prediction_vals = np.concatenate((prediction_vals, np.array(val_prediction).reshape((batch_size, 1500))), axis=0)

	# release gpu resources
	sess.close()

	print("Prediction for {} intervals was successful.".format(prediction_vals.shape[0]))

	# # compute deviations from the prior probabilities
	# # scale deviations nonlinearly to pronounce their differences
	# if not options.baseline:
	# 	for i in range(interval_count):
	# 		#prediction_vals[i,:] -= prior_probabilities
	# 		prediction_vals[i,:] = np.maximum(prediction_vals[i,:], np.zeros((1500,)))
	# 		max_ind = argmax_n(prediction_vals[i,:], n=5)
	# 		prediction_vals[i,max_ind] *= 80

	if not options.baseline:
		for i in range(interval_count):
			threshold = 0.15
			prediction_vals[i,:] = 1/(1+np.exp(-(prediction_vals[i,:]-threshold)))


	presave_path = _path("optimization_demos/optimized_predictions_{}.npy".format(talk_id))
	baseline_path = _path("optimization_demos/optimized_predictions_baseline_{}.npy".format(talk_id))

	# compute initial guess
	sound = Sound(talk.audio_path())
	#p_dotspace = re.compile(r'(?:\W|^)\w+?(\.)\w+?(?:\W|$)')
	#transcript = p_dotspace.sub(" ", talk.transcript)
	#words = word_tokenize(transcript)
	filter_words = [".", ",", ";", "-", "!", "?", "--", "(Laughter)", "Laughter", "(", ")", "\""]
	clean_words = [w for (w,_) in word_timings[str(talk_id)] if not w in filter_words]
	#clean_words = [w for w in words if not w in filter_words]
	start_off = 12.0
	if options.baseline:
		if not os.path.isfile(baseline_path):
			print("No baseline data found. Please run full procedure first.")
			exit()
		word_offsets = np.load(baseline_path)
	else:
		word_offsets = np.array(sound.interpolate_without_silence(start_off, -10.0, len(clean_words)))
	frequent_words_with_timing = [(start_off+t,w) for t,w in zip(word_offsets, clean_words) if extractor.ps.stem(w) in frequent_words]
	initial_guess = [t for (t,_) in frequent_words_with_timing]
	word_indices = [frequent_words.index(extractor.ps.stem(w)) for (_,w) in frequent_words_with_timing]
	optimization_words = [w for (_,w) in frequent_words_with_timing]

	if not os.path.isfile(baseline_path):
		np.save(baseline_path, initial_guess)
	if os.path.isfile(presave_path):
		word_offsets = np.load(presave_path)
	else:
		print("Starting optimization....\n (This could take minutes.)")
		# optimization
		word_offsets = fmin_cobyla(
							cost_function, 
							initial_guess, 
							constraint_function, 
							args=[prediction_vals, interval_count, word_indices], 
							consargs=[],
							maxfun=800
						)
		# bounds = [(10, talk.duration) for _ in range(len(initial_guess))]
		# word_offsets = fmin_slsqp(
		# 				cost_function,
		# 				initial_guess,
		# 				f_ieqcons = constraint_function,
		# 				fprime = cost_function_gradient,
		# 				fprime_ieqcons = constrain_function_jacobian,
		# 				args = (prediction_vals, interval_count, word_indices),
		# 				bounds=bounds,
		# 				iter = 10
		# 			)

		np.save(presave_path, word_offsets)

	# output sum of squared errors for computed alignment
	print("SSE prediction to initial guess: {}".format(np.sum((word_offsets-initial_guess)**2)))
	data_offsets = np.array([t for (w,t) in word_timings[str(talk_id)] if not w in filter_words])
	print("SSE prediction to (true) data guess: {}".format(np.sum((word_offsets-data_offsets)**2)))
	print("SSE initial guess to (true) data guess: {}".format(np.sum((data_offsets-initial_guess)**2)))

	# demonstrate computed alignment
	frequent_words_with_timing = [(t,w) for t,w in zip(word_offsets, optimization_words)]
	demo = TimingDemo(talk.audio_path(), Subtitle(None, None, words_with_timing=frequent_words_with_timing))
	demo.play()

