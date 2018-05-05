import numpy as np
import json
import os.path
from shutil import copyfile
import tensorflow as tf
import extract_training_data as extractor
import training_routines
import training_data_provider

if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)
	model_load_checkpoint = None #training_routines._get_full_path("training_data", "run_2018-03-21-14_a3d133d1fb017e1980ae91f7c2345a2f", "train", "model.ckpt-11")

	# change directory to ml_subtitle_align/ folder
	# start Tensorboard with command:
	#   tensorboard --logdir model/retrain_logs/
	# or:
	# 	python -m tensorboard.main --logdir=model/retrain_logs
	# open Tensorboard in browser:
	#   http://127.0.0.1:6006

	# load config
	config_path = training_routines._get_full_path("training_config.json")
	if not os.path.isfile(config_path):
		copyfile(config_path+".default", config_path)
	config = json.load(open(config_path))
	hardware_config_path = training_routines._get_full_path("hardware_config.json")
	if not os.path.isfile(hardware_config_path):
		copyfile(hardware_config_path+".default", hardware_config_path)

	batch_size = config["batch_size"]
	keep_probability = config["keep_probability"]

	# start a new tensorflow session
	sess = tf.InteractiveSession()

	# setup data provider for training
	data_provider = training_data_provider.DataProvider(config)

	# setup input shape
	input_3d = tf.placeholder(tf.float32, [None, data_provider.get_feature_count(), 13], name="input_3d")

	# instantiate model
	model = training_routines.get_model_obj_from_config(config)
	predictions, keep_prob = model.train_model(input_3d)

	#
	# 	--- define loss ---
	#
	if config["data"] == "interval_sequence":
		ground_truth_input = tf.placeholder(tf.float32, [batch_size, 2*training_data_provider.word_embedding.EMBEDDING_DIMENSION], name="ground_truth_input")
	else:
		ground_truth_input = tf.placeholder(tf.float32, [batch_size, 1500], name="ground_truth_input")
	with tf.name_scope("loss"):
		if config["loss"]["function"] == "logsumexp":
			# this is a smooth approximation to the maximum function
			loss = tf.reduce_logsumexp(tf.abs(tf.subtract(ground_truth_input, predictions)))
		elif config["loss"]["function"] == "reg_logsumexp":
			loss = tf.add(
						tf.reduce_logsumexp(tf.abs(tf.subtract(ground_truth_input, predictions))), 
						tf.reduce_mean(tf.multiply(tf.abs(predictions), 4e3))
					)
		elif config["loss"]["function"] == "reg_max":
			loss = tf.add(
						tf.reduce_max(tf.abs(tf.subtract(ground_truth_input, predictions))), 
						tf.reduce_mean(tf.multiply(tf.abs(predictions), 4e2))
					)
		elif config["loss"]["function"] == "power_max":
			loss = tf.reduce_mean(tf.reduce_max(tf.abs(
									tf.subtract(tf.pow(ground_truth_input, 4), tf.pow(predictions, 4))
								), axis=1)
							)
		elif config["loss"]["function"] == "power_mean":
			loss = tf.reduce_mean(tf.pow(tf.subtract(ground_truth_input, predictions), 4))
		elif config["loss"]["function"] == "softmax":
			loss = tf.losses.softmax_cross_entropy(
							onehot_labels=tf.one_hot(tf.argmax(ground_truth_input, axis=1), depth=1500),
							logits=predictions
						)
		elif config["loss"]["function"] == "reg_hit_top":
			top_truth_mask = tf.one_hot(tf.argmax(ground_truth_input, axis=1), on_value=True, off_value=False, dtype=bool, depth=1500)
			loss = tf.reduce_mean(tf.add(
					tf.squared_difference(
							tf.boolean_mask(predictions, top_truth_mask),
							tf.reduce_max(ground_truth_input, axis=1)
						),
					tf.multiply(tf.reduce_mean(predictions, axis=1), config["loss"]["hyperparam"])
				))
		elif config["loss"]["function"] == "reg_span_mse":
			# this turns out to be a very bad idea in practice
			# the model just picks a random word, sets it to 100% probability and chooses 0% probability for every other word
			loss = tf.subtract(
						tf.reduce_mean(tf.multiply(tf.squared_difference(predictions, ground_truth_input), config["loss"]["hyperparam"])),
						tf.reduce_mean(tf.squared_difference(
								tf.reduce_max(predictions, axis=1),
								tf.reduce_mean(predictions, axis=1)
							))
					)
		elif config["loss"]["function"] == "sigmoid_cross_entropy":
			loss = tf.losses.sigmoid_cross_entropy(
						ground_truth_input, 
						predictions
					)
			#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ground_truth_input, logits = predictions))
		elif config["loss"]["function"] == "reg_hit_top_soft":
			# 
			loss = tf.subtract(
					tf.reduce_mean(tf.multiply(tf.reduce_mean(predictions, axis=1), config["loss"]["hyperparam"])),
					tf.divide(tf.trace(tf.matmul(
						predictions,
						ground_truth_input,
						transpose_b = True,
						b_is_sparse = True
					)), 1500)
				)
		elif config["loss"]["function"] == "interval_sequence":
			# the model will predict a sequence of 8 words for the current interval
			# but we only have 2 words as a label
			# choose the minimum squared distance between the label and all predicted 2-shingles
			# the loss is the mean of these minimums for the current batch
			word_embedding = training_data_provider.word_embedding
			sq_diffs_0 = tf.reduce_mean(tf.squared_difference(
							predictions[:,(0*word_embedding.EMBEDDING_DIMENSION):(2*word_embedding.EMBEDDING_DIMENSION)],
							ground_truth_input
						), axis=1)
			sq_diffs_1 = tf.reduce_mean(tf.squared_difference(
							predictions[:,(1*word_embedding.EMBEDDING_DIMENSION):(3*word_embedding.EMBEDDING_DIMENSION)],
							ground_truth_input
						), axis=1)
			sq_diffs_2 = tf.reduce_mean(tf.squared_difference(
							predictions[:,(2*word_embedding.EMBEDDING_DIMENSION):(4*word_embedding.EMBEDDING_DIMENSION)],
							ground_truth_input
						), axis=1)
			sq_diffs_3 = tf.reduce_mean(tf.squared_difference(
							predictions[:,(3*word_embedding.EMBEDDING_DIMENSION):(5*word_embedding.EMBEDDING_DIMENSION)],
							ground_truth_input
						), axis=1)
			sq_diffs_4 = tf.reduce_mean(tf.squared_difference(
							predictions[:,(4*word_embedding.EMBEDDING_DIMENSION):(6*word_embedding.EMBEDDING_DIMENSION)],
							ground_truth_input
						), axis=1)
			sq_diffs_5 = tf.reduce_mean(tf.squared_difference(
							predictions[:,(5*word_embedding.EMBEDDING_DIMENSION):(7*word_embedding.EMBEDDING_DIMENSION)],
							ground_truth_input
						), axis=1)
			sq_diffs_6 = tf.reduce_mean(tf.squared_difference(
							predictions[:,(6*word_embedding.EMBEDDING_DIMENSION):(8*word_embedding.EMBEDDING_DIMENSION)],
							ground_truth_input
						), axis=1)
			sq_diffs = tf.stack([sq_diffs_0, sq_diffs_1, sq_diffs_2, sq_diffs_3, sq_diffs_4, sq_diffs_5, sq_diffs_6], axis=1)
			loss = tf.reduce_mean(tf.reduce_min(sq_diffs, axis=1))
		else: # if config["loss"]["function"] == "mean_squared_error":
			loss = tf.losses.mean_squared_error(
				labels=ground_truth_input,
				predictions=predictions
			)
	tf.summary.scalar("loss", loss)

	#
	# 	--- define optimizer ---
	#
	with tf.name_scope("train"):
		learning_rate_input = tf.placeholder(tf.float32, [], name="learning_rate_input")
		if config["optimizer"] == "adam":
			train_step = tf.train.AdamOptimizer(learning_rate_input).minimize(loss)
		elif config["optimizer"] == "adadelta":
			train_step = tf.train.AdadeltaOptimizer(learning_rate_input).minimize(loss)
		else:
			train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(loss)

	global_step = tf.train.get_or_create_global_step()
	increment_global_step = tf.assign(global_step, global_step + 1)

	#
	# 	--- setup training logs ---
	#
	saver = tf.train.Saver(tf.global_variables())

	# Merge all the summaries and write them out to /training_data/run_[...]_[...]/retrain_logs
	merged_summaries = tf.summary.merge_all()
	training_data_path, retrain_logs_path, train_path = training_routines.get_training_save_paths(config)
	train_writer = tf.summary.FileWriter(os.path.join(retrain_logs_path, "train"), sess.graph)
	validation_writer = tf.summary.FileWriter(os.path.join(retrain_logs_path, "validation"))

	# copy training and hardware config to training data output path
	copyfile(config_path, os.path.join(training_data_path, "training_config.json"))
	copyfile(hardware_config_path, os.path.join(training_data_path, "hardware_config.json"))

	#
	# 	--- initialize training variables ---
	#
	tf.global_variables_initializer().run()
	start_step = 1
	if model_load_checkpoint != None:
		model.load_variables_from_checkpoint(sess, model_load_checkpoint)
		#start_step = global_step.eval(session=sess)

	tf.logging.info("Training from step: %d", start_step)

	print("\n\nYou can monitor the training progress using a Tensorboard:\n $ python3 -m tensorboard.main --logdir={}\n\n".format(retrain_logs_path))

	# save graph.pbtxt
	tf.train.write_graph(sess.graph_def, train_path, "graph.pbtxt")
	checkpoint_path = os.path.join(train_path, "model.ckpt")

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
		#for batch_ii, (train_input, train_ground_truth) in enumerate(data_provider.xbatches(training=True)):
		for batch_ii, (train_input, train_ground_truth) in enumerate(data_provider.load_batches(training=True)):
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
				tf.logging.info("Pass {} - batch {}: rate {}, loss {}".format(data_pass, batch_ii, learning_rate_value, loss_value))
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
		#for batch_ii, (val_input, val_ground_truth) in enumerate(data_provider.xbatches(training=False)):
		for batch_ii, (val_input, val_ground_truth) in enumerate(data_provider.load_batches(training=False)):
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
		tf.logging.info("Pass {}: loss {}".format(data_pass, validation_loss))
			
		# save model checkpoint
		tf.logging.info("Saving to `%s-%d`", checkpoint_path, data_pass)
		saver.save(sess, checkpoint_path, global_step=data_pass)
	sess.close()

	# print training run summary
	print(" -----------------------------------------\n \
Training has finished with a final validation loss of\n {}\nParameters were:".format(validation_loss))
	for key, val in config.items():
		print("{}: {}".format(key, val))
