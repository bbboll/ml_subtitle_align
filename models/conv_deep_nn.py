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
		      -> Dense  -> Dense -> Output
		Two convolutional layers are used to produce translation invariant features.
		Classification is consequently performed by eight dense layers.
		"""

		conv_params = [
			{"height": 10, "width": 26, "count": 16},
			{"height": 5, "width": 13, "count": 16}
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
		final_weights = tf.Variable(
			tf.truncated_normal([
				conv_params[1]["height"],
				conv_params[1]["width"],
				conv_params[0]["count"],
				conv_params[1]["count"]
			], stddev=0.01)
		)

		# convolution 2
		final_bias = tf.Variable(tf.zeros([conv_params[1]["count"]]))
		final_conv = tf.nn.conv2d(first_max_pool, final_weights, [1, 1, 1, 1], "SAME") + final_bias

		# activation for convolution 2
		final_relu = tf.nn.relu(final_conv)

		# flatten convolutional output
		final_conv_shape = final_relu.get_shape()
		final_conv_output_width = final_conv_shape[2]
		final_conv_output_height = final_conv_shape[1]
		final_conv_element_count = int(final_conv_output_width * final_conv_output_height * conv_params[1]["count"])
		flattened_final_conv = tf.reshape(final_relu, [-1, final_conv_element_count])

		# encode convolution output into label space
		sizes = [2500, 1500, 800, 400, 200, 100, 500]
		dense_layers = [tf.layers.dense(flattened_final_conv, sizes[0], activation=tf.nn.sigmoid)]
		for i in range(6):
			dense_layers.append(tf.layers.dense(dense_layers[i], sizes[i+1], activation=tf.nn.sigmoid))

		final_fc = tf.layers.dense(dense_layers[-1], OUTPUT_DIM, activation=tf.nn.sigmoid)

		if is_training:
			return final_fc, keep_prob
		return final_fc
