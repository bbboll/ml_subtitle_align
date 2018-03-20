import numpy as np
import tensorflow as tf
from .base_model import BaseModel
from .base_model import OUTPUT_DIM

class Model(BaseModel):

	def _create_model(self, input_3d, is_training=True):
		"""
		Model structure:
		Input -> Conv2d -> relu activation -> pooling
		      -> Conv2d -> relu activation (-> dropout) -> pooling
		      -> Conv2d -> relu activation -> pooling
		      -> Conv2d -> relu activation -> pooling
		      -> Conv2d -> relu activation -> pooling
		      -> Conv2d -> relu activation -> pooling
		      -> Dense  -> Dense -> Output
		Six convolutional layers are used to perform translation-invariant classification.
		The embedding of covolution output into the label space is performed by two dense layers.
		"""

		conv_params = [
			{"height": 10, "width": 26, "count": 16},
			{"height": 5, "width": 13, "count": 16},
			{"height": 5, "width": 5, "count": 32},
			{"height": 3, "width": 3, "count": 32},
			{"height": 3, "width": 3, "count": 32},
			{"height": 3, "width": 3, "count": 32}
		]

		if is_training:
			# load hyperparameter for dropout layers
			# lower keep probability can be used to control overfitting
			keep_prob = tf.placeholder(tf.float32, name="keep_prob")

		# prepare input for convolution
		input_4d = tf.expand_dims(input_3d, -1)

		# convolution 1 parameters
		first_weights = tf.Variable(
			tf.truncated_normal([
				conv_params[0]["height"],
				conv_params[0]["width"],
				1,
				conv_params[0]["count"]
			], stddev=0.01)
		)

		# convolution 1
		first_bias = tf.Variable(tf.zeros([conv_params[0]["count"]]))
		first_conv = tf.nn.conv2d(input_4d, first_weights, [1, 1, 1, 1], "SAME") + first_bias

		# activation + pooling for convolution 1
		first_relu = tf.nn.relu(first_conv)
		first_max_pool = tf.nn.max_pool(first_relu, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

		# convolution 2 parameters
		second_weights = tf.Variable(
			tf.truncated_normal([
				conv_params[1]["height"],
				conv_params[1]["width"],
				conv_params[0]["count"],
				conv_params[1]["count"]
			], stddev=0.01)
		)
		
		# convolution 2
		second_bias = tf.Variable(tf.zeros([conv_params[1]["count"]]))
		second_conv = tf.nn.conv2d(first_max_pool, second_weights, [1, 1, 1, 1], "SAME") + second_bias

		# activation (+ dropout) for convolution 2
		second_relu = tf.nn.relu(second_conv)
		if is_training:
			second_dropout = tf.nn.dropout(second_relu, keep_prob)
		else:
			second_dropout = second_relu
		second_max_pool = tf.nn.max_pool(second_dropout, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

		# convolution 3 parameters
		third_weights = tf.Variable(
			tf.truncated_normal([
				conv_params[2]["height"],
				conv_params[2]["width"],
				conv_params[1]["count"],
				conv_params[2]["count"]
			], stddev=0.01)
		)

		# convolution 3
		third_bias = tf.Variable(tf.zeros([conv_params[2]["count"]]))
		third_conv = tf.nn.conv2d(second_max_pool, third_weights, [1, 1, 1, 1], "SAME") + third_bias

		# activation for convolution 3
		third_relu = tf.nn.relu(third_conv)

		# convolution 4 parameters
		fourth_weights = tf.Variable(
			tf.truncated_normal([
				conv_params[3]["height"],
				conv_params[3]["width"],
				conv_params[2]["count"],
				conv_params[3]["count"]
			], stddev=0.01)
		)

		# convolution 4
		fourth_bias = tf.Variable(tf.zeros([conv_params[3]["count"]]))
		fourth_conv = tf.nn.conv2d(third_relu, fourth_weights, [1, 1, 1, 1], "SAME") + fourth_bias

		# activation for convolution 4
		fourth_relu = tf.nn.relu(fourth_conv)

		# convolution 5 parameters
		fifth_weights = tf.Variable(
			tf.truncated_normal([
				conv_params[4]["height"],
				conv_params[4]["width"],
				conv_params[3]["count"],
				conv_params[4]["count"]
			], stddev=0.01)
		)

		# convolution 5
		fifth_bias = tf.Variable(tf.zeros([conv_params[4]["count"]]))
		fifth_conv = tf.nn.conv2d(fourth_relu, fifth_weights, [1, 1, 1, 1], "SAME") + fifth_bias

		# activation for convolution 5
		fifth_relu = tf.nn.relu(fifth_conv)

		# convolution 6 parameters
		final_weights = tf.Variable(
			tf.truncated_normal([
				conv_params[5]["height"],
				conv_params[5]["width"],
				conv_params[4]["count"],
				conv_params[5]["count"]
			], stddev=0.01)
		)

		# convolution 6
		final_bias = tf.Variable(tf.zeros([conv_params[5]["count"]]))
		final_conv = tf.nn.conv2d(fifth_relu, final_weights, [1, 1, 1, 1], "SAME") + final_bias

		# activation for convolution 6
		final_relu = tf.nn.relu(final_conv)

		# flatten convolutional output
		final_conv_shape = final_relu.get_shape()
		final_conv_output_width = final_conv_shape[2]
		final_conv_output_height = final_conv_shape[1]
		final_conv_element_count = int(final_conv_output_width * final_conv_output_height * conv_params[5]["count"])
		flattened_final_conv = tf.reshape(final_relu, [-1, final_conv_element_count])

		# encode convolution output into label space
		dense_layer = tf.layers.dense(flattened_final_conv, 1000, activation=tf.nn.sigmoid)
		final_fc = tf.layers.dense(dense_layer, OUTPUT_DIM, activation=tf.nn.sigmoid)

		if is_training:
			return final_fc, keep_prob
		return final_fc