from model.model import Model
import numpy as np
import os
import os.path
import tensorflow as tf


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


def xbatches(mfcc, labels, batch_size):
	"""Batch MFCC data, labels together.
	"""
	for ii in range(0, len(mfcc) // batch_size):
		offset = ii * batch_size
		yield (
			mfcc[offset:(offset + batch_size)],
			labels[offset:(offset + batch_size)]
		)


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
	all_features = np.load(_get_full_path("data", "tmp", "x_all.npy")) # tbc
	all_labels = np.load(_get_full_path("data", "tmp", "y_all.npy")) # tbc
	num_labels = 3 # tbc

	# split features and labels into training and validation set or load presplitted data
	train_features = np.load(_get_full_path("data", "tmp", "x_train.npy")) # tbc
	train_labels = np.load(_get_full_path("data", "tmp", "y_train.npy")) # tbc
	val_features = np.load(_get_full_path("data", "tmp", "x_val.npy")) # tbc
	val_labels = np.load(_get_full_path("data", "tmp", "y_val.npy")) # tbc

	# start a new tensorflow session
	sess = tf.InteractiveSession()

	input_3d = tf.placeholder(tf.float32, [None, all_features.shape[1], all_features.shape[2]], name="input_3d")

	model = Model()
	model.set_config(label_count=num_labels)

	logits, dropout_prob = model.train_model(input_3d)

	# define loss and optimizer
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
	batch_size = 100
	validation_step_interval = 200 # tbc - interval how often to evaluate model
	save_step_interval = 100 # tbc - interval how often the model should be saved
	training_steps_list = [1000, 300] # tbc - number of training steps to do with associated learning rate
	learning_rates_list = [0.001, 0.0001] # tbc - learning rates, associated with `training_steps_list`
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
		for batch_ii, (train_input, train_ground_truth) in enumerate(xbatches(train_features, train_labels, batch_size)):
			train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
				[merged_summaries, evaluation_step, cross_entropy_mean, train_step, increment_global_step],
				feed_dict={
					input_3d: train_input,
					ground_truth_input: train_ground_truth,
					learning_rate_input: learning_rate_value,
					dropout_prob: 0.5
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
			for batch_ii, (val_input, val_ground_truth) in enumerate(xbatches(val_features, val_labels, batch_size)):
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
