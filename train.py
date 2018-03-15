from model.model import Model
import numpy as np
import os
import os.path
import json
import tensorflow as tf
import extract_training_data as extractor
from preprocessing.talk import AllTalks
import nltk
import scipy.stats
from sklearn.model_selection import train_test_split


def _get_full_path(*rel_path):
	"""Make absolute path to a file or directory in the project folder ml_subtitle_align.

	Arguments:
        *rel_path: List of path elements.

    Returns:
        `str`: Absolute path to requested file or directory.
	"""
	path = os.path.abspath(__file__) # `.../ml_subtitle_align/train.py`
	path = os.path.dirname(path) # `.../ml_subtitle_align/`
	return os.path.join(path, *rel_path)

if not os.path.isfile(extractor.frequent_words_path) or not os.path.isfile(extractor.word_timings_path):
	print("Execute extract_training_data first.")
	exit()
frequent_words = json.load(open(extractor.frequent_words_path))
word_timings = json.load(open(extractor.word_timings_path))
mfcc_per_interval = int(extractor.INTERVAL_SIZE / 0.005)

def compute_full_vector_labels(talk, interval_count):
	"""
	For each interval, return a vector of shape (,1500) containing every
	probability for each single word.
	"""
	talk_labels = np.zeros((1500, interval_count))
	for (w, t) in word_timings[str(talk.ID)]:
		distrib = scipy.stats.norm(t, extractor.DATA_SD)
		word_ind = None
		try:
			word_ind = frequent_words.index(extractor.ps.stem(w))
		except ValueError:
			print("Could not find word {}".format(w))
			continue
		interval_ind = int(t // extractor.INTERVAL_SIZE)
		index_range = [interval_ind+i for i in range(-2,2) if interval_ind+i >= 0 and interval_ind+i+1 < interval_count]
		talk_labels[word_ind][index_range] += np.array([extractor.compute_word_probability(i, distrib) for i in index_range])
	return talk_labels.T

def compute_categorical_labels(talk, interval_count):
	"""
	For each interval, return an int64 (categorical) label.
	"""
	talk_labels = []
	for i in range(interval_count):
		midpoint = (i+0.5)*extractor.INTERVAL_SIZE
		closest_word = None
		diff = float(talk.duration)
		for (w, t) in word_timings[str(talk.ID)]:
			if diff > abs(midpoint-t):
				diff = abs(midpoint-t)
				closest_word = w
		word_ind = frequent_words.index(extractor.ps.stem(w))
		talk_labels.append(word_ind)
	return np.array(talk_labels, dtype=np.int64)

def xbatches(batch_size, training=True):
	"""Batch MFCC data, labels together.
	"""
	talk_limit = None
	all_ids = [talk.ID for talk in AllTalks(limit=talk_limit)]
	train_ids, test_ids = train_test_split(all_ids, test_size=0.1, shuffle=True)
	features = np.zeros((0,mfcc_per_interval,13))
	labels = np.array([])
	talks = AllTalks(limit=talk_limit)

	try:
		while True:
			if labels.shape[0] < batch_size:
				talk = next(talks)
				if training and not talk.ID in train_ids:
					continue
				if not training and talk.ID in train_ids:
					continue
				if not str(talk.ID) in word_timings:
					continue
				mfcc_features = np.load(talk.features_path())
				interval_count = int(mfcc_features.shape[0] // mfcc_per_interval)
				mfcc_features = mfcc_features[:interval_count*mfcc_per_interval]
				mfcc_features = mfcc_features.reshape((interval_count,mfcc_per_interval,13))
				features = np.concatenate((features, mfcc_features), axis=0)

				#talk_labels = compute_full_vector_labels(talk, interval_count)
				talk_labels = compute_categorical_labels(talk, interval_count)
				labels = np.concatenate((labels, talk_labels))
			else:
				yield (
					features[:batch_size],
					labels[:batch_size]
				)
				features = features[batch_size:]
				labels = labels[batch_size:]
	except StopIteration:
		return

def main():
	tf.logging.set_verbosity(tf.logging.INFO)

    # change directory to ml_subtitle_align/ folder
	# start Tensorboard with command:
	#   tensorboard --logdir model/retrain_logs/
	# open Tensorboard in browser:
	#   http://127.0.0.1:6006

	# pull all data
	# `all_features` and `all_labels` are 3d numpy arrays with
	#   shape[0] <- number of samples
	#   shape[1], shape[2] <- shape of mfcc data
	# all_features = np.load(_get_full_path("data", "tmp", "x_all.npy")) # tbc
	# all_labels = np.load(_get_full_path("data", "tmp", "y_all.npy")) # tbc
	num_labels = 1500
	batch_size = 180

	# split features and labels into training and validation set or load presplitted data
	# train_features = np.load(_get_full_path("data", "tmp", "x_train.npy")) # tbc
	# train_labels = np.load(_get_full_path("data", "tmp", "y_train.npy")) # tbc
	# val_features = np.load(_get_full_path("data", "tmp", "x_val.npy")) # tbc
	# val_labels = np.load(_get_full_path("data", "tmp", "y_val.npy")) # tbc

	# start a new tensorflow session
	sess = tf.InteractiveSession()

	input_3d = tf.placeholder(tf.float32, [None, 80, 13], name="input_3d")

	model = Model()
	model.set_config(label_count=num_labels)

	logits, dropout_prob = model.train_model(input_3d)

	# define loss and optimizer
	# ground_truth_input = tf.placeholder(tf.int64, [batch_size, 1500], name="ground_truth_input")
	ground_truth_input = tf.placeholder(tf.int64, [None], name="ground_truth_input")

	# create back propagation and training evaluation machinery in the graph
	with tf.name_scope("cross_entropy"):
		cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
			labels=ground_truth_input,
			logits=logits
		)
	tf.summary.scalar("cross_entropy", cross_entropy_mean)

	with tf.name_scope("train"), tf.control_dependencies([tf.add_check_numerics_ops()]):
		learning_rate_input = tf.placeholder(tf.float32, [], name="learning_rate_input")
		train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(cross_entropy_mean)

	predicted_indices = tf.argmax(logits, 1)
	correct_prediction = tf.equal(predicted_indices, ground_truth_input)
	confusion_matrix = tf.confusion_matrix(ground_truth_input, predicted_indices, num_classes=num_labels)

	evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar("accuracy", evaluation_step)

	global_step = tf.train.get_or_create_global_step()
	increment_global_step = tf.assign(global_step, global_step + 1)

	saver = tf.train.Saver(tf.global_variables())

	# Merge all the summaries and write them out to /model/retrain_logs
	merged_summaries = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(_get_full_path("model", "retrain_logs", "train"), sess.graph)
	validation_writer = tf.summary.FileWriter(_get_full_path("model", "retrain_logs", "validation"))

	tf.global_variables_initializer().run()

	start_step = 1
	if False:
		model.load_variables_from_checkpoint(session, _get_full_path("model", "train", "model.ckpt"))
		start_step = global_step.eval(session=sess)

	tf.logging.info("Training from step: %d", start_step)

	# save graph.pbtxt
	tf.train.write_graph(sess.graph_def, _get_full_path("model", "train"), "graph.pbtxt")

	# training loop
	validation_step_interval = 3 # tbc - interval how often to evaluate model
	save_step_interval = 100 # tbc - interval how often the model should be saved
	training_steps_list = [1, 1]
	#training_steps_list = [1000, 300] # tbc - number of training steps to do with associated learning rate
	learning_rates_list = [0.0003, 0.0001] # tbc - learning rates, associated with `training_steps_list`
	training_steps_max = np.sum(training_steps_list)
	for training_step in range(start_step, training_steps_max + 1):
		# get current learning rate
		training_steps_sum = 0
		for i in range(len(training_steps_list)):
			training_steps_sum += training_steps_list[i]
			if training_step <= training_steps_sum:
				learning_rate_value = learning_rates_list[i]
				break

		# run the graph with batches of data
		for batch_ii, (train_input, train_ground_truth) in enumerate(xbatches(batch_size, training=True)):
			train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
				[merged_summaries, evaluation_step, cross_entropy_mean, train_step, increment_global_step],
				feed_dict={
					input_3d: train_input,
					ground_truth_input: train_ground_truth,
					learning_rate_input: learning_rate_value,
					dropout_prob: 0.8
				}
			)
		train_writer.add_summary(train_summary, training_step)
		tf.logging.info("Step #%d: rate %f, accuracy %.1f%%, cross entropy %f" %
			(training_step, learning_rate_value, train_accuracy * 100, cross_entropy_value)
		)

		# evaluate
		is_last_step = (training_step == training_steps_max)
		if training_step % validation_step_interval == 0 or is_last_step:
			total_accuracy = 0
			total_cf_matrix = None
			for batch_ii, (val_input, val_ground_truth) in enumerate(xbatches(batch_size, training=False)):
				val_summary, val_accuracy, cf_matrix = sess.run(
					[merged_summaries, evaluation_step, confusion_matrix],
					feed_dict={
						input_3d: val_input,
						ground_truth_input: val_ground_truth,
						dropout_prob: 1.0
					}
				)
				validation_writer.add_summary(val_summary, training_step)
				total_accuracy += val_accuracy
				if total_cf_matrix is None:
					total_cf_matrix = cf_matrix
				else:
					total_cf_matrix += cf_matrix
				tf.logging.info("Confusion matrix:\n %s" % (total_cf_matrix))
				tf.logging.info("Step %d: Validation accuracy = %.1f%%" % (training_step, total_accuracy * 100))

		# save model checkpoint
		if training_step % save_step_interval == 0 or is_last_step:
			checkpoint_path = _get_full_path("model", "train", "model.ckpt")
			tf.logging.info("Saving to `%s-%d`", checkpoint_path, training_step)
			saver.save(sess, checkpoint_path, global_step=training_step)


if __name__ == "__main__":
	main()
