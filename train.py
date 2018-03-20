import models.conv_model
import models.experimental_model
import numpy as np
import json
import os.path
from shutil import copyfile
import tensorflow as tf
import extract_training_data as extractor
from training_routines import compute_full_vector_labels
from training_routines import xbatches
from training_routines import _get_full_path

if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)
	model_load_checkpoint = None # _get_full_path("models", "train", "model.ckpt-1")

    # change directory to ml_subtitle_align/ folder
	# start Tensorboard with command:
	#   tensorboard --logdir model/retrain_logs/
	# or:
	# 	python -m tensorboard.main --logdir=model/retrain_logs
	# open Tensorboard in browser:
	#   http://127.0.0.1:6006

	# load config
	config_path = _get_full_path("training_config.json")
	if not os.path.isfile(config_path):
		copyfile(config_path+".default", config_path)
	config = json.load(open(config_path))

	batch_size = config["batch_size"]
	keep_probability = config["keep_probability"]

	# start a new tensorflow session
	sess = tf.InteractiveSession()

	# setup input shape
	input_3d = tf.placeholder(tf.float32, [None, 80, 13], name="input_3d")

	# instantiate model
	if config["model"] == "simple_conv":
		model = models.conv_model.Model()
	elif config["model"] == "dense_conv":
		model = models.conv_model.Model(hyperparams=["dense"])
	else: # if config["model"] == "experimental"
		model = models.model.Model()
	predictions, keep_prob = model.train_model(input_3d)

	#
	# 	--- define loss ---
	#
	ground_truth_input = tf.placeholder(tf.float32, [batch_size, 1500], name="ground_truth_input")
	with tf.name_scope("loss"):
		if config["loss_function"] == "logsumexp":
			# this is a smooth approximation to the maximum function
			loss = tf.reduce_logsumexp(tf.abs(tf.subtract(ground_truth_input, predictions)))
		else: # if config["loss_function"] == "mean_squared_error":
			loss = tf.losses.mean_squared_error(
				labels=ground_truth_input,
				predictions=predictions
			)
	tf.summary.scalar("loss", loss)

	#
	# 	--- define optimizer ---
	#
	with tf.name_scope("train"), tf.control_dependencies([tf.add_check_numerics_ops()]):
		learning_rate_input = tf.placeholder(tf.float32, [], name="learning_rate_input")
		train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(loss)

	global_step = tf.train.get_or_create_global_step()
	increment_global_step = tf.assign(global_step, global_step + 1)

	#
	# 	--- setup training logs ---
	#
	saver = tf.train.Saver(tf.global_variables())

	# Merge all the summaries and write them out to /model/retrain_logs
	merged_summaries = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(_get_full_path("models", "retrain_logs", "train"), sess.graph)
	validation_writer = tf.summary.FileWriter(_get_full_path("models", "retrain_logs", "validation"))

	#
	# 	--- initialize training variables ---
	#
	tf.global_variables_initializer().run()
	start_step = 1
	if model_load_checkpoint != None:
		model.load_variables_from_checkpoint(sess, model_load_checkpoint)
		start_step = global_step.eval(session=sess)

	tf.logging.info("Training from step: %d", start_step)

	# save graph.pbtxt
	tf.train.write_graph(sess.graph_def, _get_full_path("models", "train"), "graph.pbtxt")
	checkpoint_path = _get_full_path("models", "train", "model.ckpt")

	# training loop
	save_step_interval = config["save_step_interval"] # interval how often the model should be saved
	passes_list = config["passes_list"] # how many passes to use for training
	learning_rates_list = config["learning_rates_list"] # learning rates, associated with `passes_list`
	batch_log_interval = config["batch_log_interval"]
	passes_max = np.sum(passes_list)
	global_batch_step = 0
	validation_loss = 0
	for data_pass in range(start_step, passes_max + 1):
		# get current learning rate
		passes_sum = 0
		for i in range(len(passes_list)):
			passes_sum += passes_list[i]
			if data_pass <= passes_sum:
				learning_rate_value = learning_rates_list[i]
				break

		#
		# 	--- training inner loop ---
		#
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
			if batch_ii % save_step_interval == 0:
				global_batch_step += 1
				# save model checkpoint
				tf.logging.info("Saving to `%s-%d`", checkpoint_path, data_pass)
				saver.save(sess, checkpoint_path, global_step=global_batch_step)

		#
		# 	--- evaluation inner loop ---
		#
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
		validation_loss = total_loss/validation_batches
		tf.logging.info("Pass {}: Mean squared error {}".format(data_pass, validation_loss))
			
		# save model checkpoint
		tf.logging.info("Saving to `%s-%d`", checkpoint_path, data_pass)
		saver.save(sess, checkpoint_path, global_step=data_pass)
	sess.close()

	# print training run summary
	print(" -----------------------------------------\n \
Training has finished with a final validation loss of\n {}\nParameters were:".format(validation_loss))
	for key, val in config.items():
		print("{}: {}".format(key, val))