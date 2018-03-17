import os.path
import sys
import json
import tensorflow as tf
import numpy as np
import extract_training_data as extractor
from model.model import Model
from preprocessing.talk import Talk
from preprocessing.audio_tools import Sound
from preprocessing.subtitle import Subtitle
from nltk import word_tokenize
from timing_demo import TimingDemo

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

if __name__ == '__main__':

	if len(sys.argv) == 1:
		print("Please enter talk ID.")
		exit()
	talk_id = int(sys.argv[1])

	# start a new tensorflow session
	sess = tf.InteractiveSession()

	# load model
	model_load_checkpoint = _path("pretrained_models/0316/model.ckpt-15")
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

	print("Prediction for {} intervals was successful.\n Starting optimization...".format(prediction_vals.shape[0]))

	# compute initial guess
	sound = Sound(talk.audio_path())
	words = word_tokenize(talk.transcript)
	filter_words = [".", ",", ";", "-", "!", "?", "--", "(Laughter)", "Laughter"]
	clean_words = [w for w in words if not w in filter_words]
	start_off = 12.0
	word_offsets = sound.interpolate_without_silence(start_off, -10.0, len(clean_words))

	# TODO: optimization

	# demonstrate computed alignment
	words_with_timing = [(start_off+t,w) for t,w in zip(word_offsets, clean_words)]
	demo = TimingDemo(talk.audio_path(), Subtitle(None, None, words_with_timing=words_with_timing))
	demo.play()

