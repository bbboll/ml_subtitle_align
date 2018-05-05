import numpy as np
import tensorflow as tf
from .base_model import BaseModel
from .base_model import OUTPUT_DIM

class Model(BaseModel):

	def _create_model(self, input_3d, is_training=True):
		"""
		"""

		units = [
			1000, 500, 300, 300, 200, 200, 200, 180, 150, 140, OUTPUT_DIM
		]

		# prepare input
		input_4d = tf.expand_dims(input_3d, -1)

		if is_training:
			# load hyperparameter for dropout layers
			keep_prob = tf.placeholder(tf.float32, name="keep_prob")

		# setup network layers
		layers = []
		layers.append(tf.layers.dense(tf.reshape(input_3d, [-1, 13*399]), units[0], activation=tf.nn.sigmoid))		

		for u in units[1:]:
			layers.append(tf.layers.dense(layers[-1], u, activation=tf.nn.sigmoid))

		if is_training:
			return layers[-1], keep_prob
		return layers[-1]
