import os.path
import json
import time
import tensorflow as tf
import numpy as np
import extract_training_data as extractor
from preprocessing.talk import Talk
from preprocessing.audio_tools import Sound
from preprocessing.subtitle import Subtitle
from training_routines import get_model_from_run
from training_routines import compute_full_vector_labels
from timing_demo import TimingDemo
from scipy.optimize import fmin_cobyla
from scipy.optimize import fmin_slsqp
import argparse
import datetime
import hashlib
import csv

# abspath utitility function
def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	current_dir = os.path.dirname(__file__)
	return os.path.abspath(os.path.join(current_dir, relpath))

def cost_function(x, probs, interval_count, word_indices):
	out = 0.0
	for word_ind, t in enumerate(x):
		interval_midpoints = np.linspace(0.5*extractor.INTERVAL_SIZE, interval_count*extractor.INTERVAL_SIZE, num=interval_count)
		interval_diffs = interval_midpoints-t
		sd_scalar = 1
		interval_scalars = np.exp(-interval_diffs**2 / (2*(sd_scalar*extractor.DATA_SD)**2)) / (np.sqrt(2*np.pi*(sd_scalar*extractor.DATA_SD)**2))
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

def get_optimal_predictions(talk):
	mfcc_features = np.load(talk.features_path())
	mfcc_per_interval = int(extractor.INTERVAL_SIZE / 0.005)
	interval_count = int(mfcc_features.shape[0] // mfcc_per_interval)
	return compute_full_vector_labels(talk, interval_count)

if __name__ == '__main__':
	arguments = argparse.ArgumentParser()
	arguments.add_argument("id", help="TED talk id.", type=int)
	arguments.add_argument("optimizer", help="The method of optimization to use. (cobyla or slsqp)", type=str, default="cobyla")
	arguments.add_argument("run", help="The training run from which to load the model (Path relative to ml_subtitle_align/training_data). This path needs to contain a training_config.json and a train/ directory with one or more checkpoints.", type=str)
	arguments.add_argument("model_loss", help="Final loss of the model to be run.", type=float, default=-1.0)
	arguments.add_argument("-baseline", action="store_true", help="Examine baseline rather than optimizing.")
	arguments.add_argument("-save", action="store_true", help="Save the results of this run.")
	arguments.add_argument("-demo", action="store_true", help="Play demo after optimization.")
	arguments.add_argument("-scale_predictions", action="store_true", help="Scale model predictions to reduce uniformity.")
	arguments.add_argument("-fake_optimal", action="store_true", help="Use optimal labels from the dataset instead of predictions.")
	options = arguments.parse_args()
	talk_id = options.id

	# load data preprocessing results
	if not os.path.isfile(extractor.frequent_words_path) or not os.path.isfile(extractor.word_timings_path):
		print("Execute extract_training_data first.")
		exit()
	frequent_words = json.load(open(extractor.frequent_words_path))
	word_timings = json.load(open(extractor.word_timings_path))

	# start a new tensorflow session
	sess = tf.InteractiveSession()
	input_3d = tf.placeholder(tf.float32, [None, 80, 13], name="input_3d")

	# load model
	model_load_checkpoint, training_config = get_model_from_run(options.run)
	if training_config["model"] == "simple_conv":
		from models.conv_model import Model
	elif training_config["model"] == "dense_conv":
		from models.conv_model import Model
	elif training_config["model"] == "conv_lstm":
		from models.conv_lstm_model import Model
	elif training_config["model"] == "deep_conv":
		from models.deep_conv_model import Model
	if training_config["model"] == "dense_conv":
		model = Model(hyperparams=["dense"])
	else:
		model = Model()
	prediction = model.test_model(input_3d)
	print("Loading model from checkpoint {}".format(model_load_checkpoint))
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

	if options.fake_optimal:
		prediction_vals = get_optimal_predictions(talk)

	print("Prediction for {} intervals was successful.".format(prediction_vals.shape[0]))

	# scale predicted probabilities to reduce uniformity
	if not options.baseline and options.scale_predictions:
		for i in range(interval_count):
			threshold = 0.15
			slope = 3
			prediction_vals[i,:] = 1/(1+np.exp(-slope*(prediction_vals[i,:]-threshold)))


	presave_path = _path("optimization_demos/optimized_predictions_{}.npy".format(talk_id))
	baseline_path = _path("optimization_demos/optimized_predictions_baseline_{}.npy".format(talk_id))

	# compute initial guess
	sound = Sound(talk.audio_path())
	filter_words = [".", ",", ";", "-", "!", "?", "--", "(Laughter)", "Laughter", "(", ")", "\""]
	clean_words = [w for (w,_) in word_timings[str(talk_id)] if not w in filter_words]
	start_off = 10.0
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
	opt_time = 0

	cobyla_limit = 800
	slsqp_limit = 10
	if not os.path.isfile(baseline_path) and options.save:
		np.save(baseline_path, initial_guess)
	if os.path.isfile(presave_path):
		word_offsets = np.load(presave_path)
	else:
		print("Starting optimization....\n (This could take minutes.)")
		start_time = time.time()
		
		# perform optimization
		if options.optimizer == "cobyla":
			word_offsets = fmin_cobyla(
								cost_function, 
								initial_guess, 
								constraint_function, 
								args=[prediction_vals, interval_count, word_indices], 
								consargs=[],
								maxfun=cobyla_limit
							)
		else:
			bounds = [(start_off, talk.duration) for _ in range(len(initial_guess))]
			word_offsets = fmin_slsqp(
							cost_function,
							initial_guess,
							f_ieqcons = constraint_function,
							fprime = cost_function_gradient,
							fprime_ieqcons = constrain_function_jacobian,
							args = (prediction_vals, interval_count, word_indices),
							bounds=bounds,
							iter = slsqp_limit
						)

		opt_time = "{}".format(time.time()-start_time)
		print("Optimization took {} seconds".format(opt_time))

		if options.save:
			np.save(presave_path, word_offsets)

	# output sum of squared errors for computed alignment
	initial_sse = np.sum((word_offsets-initial_guess)**2)
	print("SSE prediction to initial guess: {}".format(initial_sse))
	data_offsets = np.array([t for (w,t) in word_timings[str(talk_id)] if not w in filter_words])
	prediction_sse = np.sum((word_offsets-data_offsets)**2)
	print("SSE prediction to (true) data guess: {}".format(prediction_sse))
	moved_sse = np.sum((data_offsets-initial_guess)**2)
	print("SSE initial guess to (true) data guess: {}".format(moved_sse))

	if not options.baseline:
		# save prediction summary
		summary = [
					[
						"model", 
						"loss", 
						"loss_func", 
						"loss_hyperparam", 
						"scale_predictions"
						"initial_sse", 
						"prediction_sse", 
						"moved_sse", 
						"talk_id", 
						"optimizer", 
						"steps", 
						"duration"
					],
					[
						training_config["model"], 
						options.model_loss, 
						training_config["loss_function"], 
						(str(training_config["loss_hyperparam"]) if training_config["loss_function"] == "reg_hit_top" else "0"), 
						str(options.scale_predictions),
						str(initial_sse), 
						str(prediction_sse), 
						str(moved_sse), 
						str(talk_id), 
						options.optimizer, 
						(str(cobyla_limit) if options.optimizer == "cobyla" else str(slsqp_limit)), 
						str(opt_time)
					]
				  ]
		summary_hash = hashlib.md5("{}".format(summary).encode('utf-8')).hexdigest()
		summary_path = _path("prediction_summaries/sum_{}-{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), summary_hash))
		with open(summary_path, 'w') as f:
			writer = csv.writer(f)
			for row in summary:
				writer.writerow(row)
		print("Prediction summary was written to {}".format(summary_path))

	if options.demo:
		# demonstrate computed alignment
		frequent_words_with_timing = [(t,w) for t,w in zip(word_offsets, optimization_words)]
		demo = TimingDemo(talk.audio_path(), Subtitle(None, None, words_with_timing=frequent_words_with_timing))
		demo.play()

