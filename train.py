from model.experimental_model import Model
import numpy as np
import tensorflow as tf
import extract_training_data as extractor
from training_routines import compute_full_vector_labels
from training_routines import xbatches
from training_routines import _get_full_path


if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)

    # change directory to ml_subtitle_align/ folder
	# start Tensorboard with command:
	#   tensorboard --logdir model/retrain_logs/
	# open Tensorboard in browser:
	#   http://127.0.0.1:6006

	batch_size = 50
	keep_probability = 0.8
	model_load_checkpoint = None # _get_full_path("model", "train", "model.ckpt-1")

	# start a new tensorflow session
	sess = tf.InteractiveSession()

	input_3d = tf.placeholder(tf.float32, [None, 80, 13], name="input_3d")

	model = Model()
	predictions, keep_prob = model.train_model(input_3d)

	# define loss and optimizer
	ground_truth_input = tf.placeholder(tf.float32, [batch_size, 1500], name="ground_truth_input")

	# create back propagation and training evaluation machinery in the graph
	with tf.name_scope("loss"):
		loss = tf.losses.mean_squared_error(
			labels=ground_truth_input,
			predictions=predictions
		)
	tf.summary.scalar("loss", loss)

	with tf.name_scope("train"), tf.control_dependencies([tf.add_check_numerics_ops()]):
		learning_rate_input = tf.placeholder(tf.float32, [], name="learning_rate_input")
		train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(loss)

	global_step = tf.train.get_or_create_global_step()
	increment_global_step = tf.assign(global_step, global_step + 1)

	saver = tf.train.Saver(tf.global_variables())

	# Merge all the summaries and write them out to /model/retrain_logs
	merged_summaries = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(_get_full_path("model", "retrain_logs", "train"), sess.graph)
	validation_writer = tf.summary.FileWriter(_get_full_path("model", "retrain_logs", "validation"))

	tf.global_variables_initializer().run()

	start_step = 1
	if model_load_checkpoint != None:
		model.load_variables_from_checkpoint(sess, model_load_checkpoint)
		start_step = global_step.eval(session=sess)

	tf.logging.info("Training from step: %d", start_step)

	# save graph.pbtxt
	tf.train.write_graph(sess.graph_def, _get_full_path("model", "train"), "graph.pbtxt")

	# training loop
	save_step_interval = 100 # tbc - interval how often the model should be saved
	passes_list = [2, 2]
	batch_log_interval = 1000
	save_interval = 3000
	learning_rates_list = [0.001, 0.0001] # tbc - learning rates, associated with `passes_list`
	passes_max = np.sum(passes_list)
	global_batch_step = 0
	for data_pass in range(start_step, passes_max + 1):
		# get current learning rate
		passes_sum = 0
		for i in range(len(passes_list)):
			passes_sum += passes_list[i]
			if data_pass <= passes_sum:
				learning_rate_value = learning_rates_list[i]
				break

		# run the graph with batches of data
		for batch_ii, (train_input, train_ground_truth) in enumerate(xbatches(batch_size, training=True)):
			train_summary, loss_value, _, _ = sess.run(
				[merged_summaries, loss, train_step, increment_global_step],
				feed_dict={
					input_3d: train_input,
					ground_truth_input: train_ground_truth,
					learning_rate_input: learning_rate_value,
					keep_prob: keep_probability
				}
			)
			if batch_ii % batch_log_interval == 0:
				train_writer.add_summary(train_summary, data_pass)
				tf.logging.info("Pass {} - batch {}: rate {}, mean squared error {}".format(data_pass, batch_ii, learning_rate_value, loss_value))
			if batch_ii % save_interval == 0:
				global_batch_step += 1
				# save model checkpoint
				checkpoint_path = _get_full_path("model", "train", "model.ckpt")
				tf.logging.info("Saving to `%s-%d`", checkpoint_path, data_pass)
				saver.save(sess, checkpoint_path, global_step=global_batch_step)

		# evaluate
		total_loss = 0
		validation_batches = 0
		total_cf_matrix = None
		for batch_ii, (val_input, val_ground_truth) in enumerate(xbatches(batch_size, training=False)):
			val_summary, val_loss = sess.run(
				[merged_summaries, loss],
				feed_dict={
					input_3d: val_input,
					ground_truth_input: val_ground_truth,
					keep_prob: 1.0
				}
			)
			validation_batches += 1
			total_loss += val_loss
		validation_writer.add_summary(val_summary, data_pass)
		tf.logging.info("Pass {}: Mean squared error {}".format(data_pass, (total_loss/validation_batches) * 100))
			
		# save model checkpoint
		checkpoint_path = _get_full_path("model", "train", "model.ckpt")
		tf.logging.info("Saving to `%s-%d`", checkpoint_path, data_pass)
		saver.save(sess, checkpoint_path, global_step=data_pass)
