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
	model_load_checkpoint = _path("pretrained_models/0316/model.ckpt-10")
	input_3d = tf.placeholder(tf.float32, [None, 80, 13], name="input_3d")
	model = Model()
	prediction = model.test_model(input_3d)
	model.load_variables_from_checkpoint(sess, model_load_checkpoint)

	# load input
	mfcc_per_interval = int(extractor.INTERVAL_SIZE / 0.005)
	mfcc_features = np.load(_path("data/audio_features/1011.npy"))
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

	# TODO: optimization

