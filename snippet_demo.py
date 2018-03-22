import os.path
import json
import tensorflow as tf
import extract_training_data as extractor
import numpy as np
import argparse
from training_routines import get_model_from_run

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

prior_probabilities_path = _path("data/training/word_priors.json")
if not os.path.isfile(prior_probabilities_path):
	print("Execute prior_probabilities.py first.")
	exit()
prior_probabilities_dict = json.load(open(prior_probabilities_path))
prior_probabilities = np.array([p for _, p in prior_probabilities_dict.items()])

if __name__ == '__main__':
	arguments = argparse.ArgumentParser()
	arguments.add_argument("run", help="The training run from which to load the model (Path relative to ml_subtitle_align/training_data). This path needs to contain a training_config.json and a train/ directory with one or more checkpoints.", type=str)
	arguments.add_argument("top_guesses", help="How many guesses to output for the snippet.", type=int, default=10)
	arguments.add_argument("-scale_predictions", action="store_true", help="Scale model predictions to reduce uniformity.")
	options = arguments.parse_args()

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
	model.load_variables_from_checkpoint(sess, model_load_checkpoint)

	# load input
	mfcc_features = np.load(_path("snippet_demo/snippet.npy"))
	mfcc_features = mfcc_features.reshape((1,80,13))

	val_prediction = sess.run(
		[prediction],
		feed_dict={
			input_3d: mfcc_features
		}
	)
	val_prediction = np.array(val_prediction).reshape((1500,))

	# if the model outputs logits, we need to transform them to probabilities first
	if training_config["loss_function"] in ["softmax", "sigmoid_cross_entropy"]:
		odds = np.exp(val_prediction)
		val_prediction = odds / (1 + odds)

	# scale predicted probabilities to reduce uniformity
	if options.scale_predictions:
		threshold = 0.15
		slope = 3
		val_prediction = 1/(1+np.exp(-slope*(val_prediction-threshold)))

	# sort and output top predictions
	idx = (-val_prediction).argsort()[:options.top_guesses].astype(int)
	print(" --- The computed {} most likely words in the snippet are --- ".format(options.top_guesses))
	for i in idx:
		print("{} - {}".format(val_prediction[i], frequent_words[i]))

	# output predictions with top deviation from prior probability
	val_prediction -= prior_probabilities
	idx = (-val_prediction).argsort()[:options.top_guesses].astype(int)
	print(" --- The computed {} most interesting words (most deviation from prior) in the snippet are --- ".format(options.top_guesses))
	for i in idx:
		print("{} - {}".format(val_prediction[i], frequent_words[i]))


