from model.model import Model
import os.path
import json
import tensorflow as tf
import extract_training_data as extractor
import numpy as np

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

	# start a new tensorflow session
	sess = tf.InteractiveSession()

	# load model
	model_load_checkpoint = _path("pretrained_models/0319_dense/model.ckpt-50")
	input_3d = tf.placeholder(tf.float32, [None, 80, 13], name="input_3d")
	model = Model()
	prediction = model.test_model(input_3d, dense=True)
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
	n = 10
	idx = (-val_prediction).argsort()[:n].astype(int)
	print(" --- The computed {} most likely words in the snippet are --- ".format(n))
	for i in idx:
		print("{} - {}".format(val_prediction[i], frequent_words[i]))


