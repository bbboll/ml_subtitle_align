import numpy as np
import os.path
import tensorflow as tf


def _get_full_path(*rel_path):
	"""Make absolute path to a file or directory in the project folder ml_subtitle_align.

	Arguments:
        *rel_path: List of path elements.

    Returns:
        `str`: Absolute path to requested file or directory.
	"""
	path = os.path.abspath(__file__) # `.../ml_subtitle_align/model/model.py`
	path = os.path.dirname(path) # `.../ml_subtitle_align/model/`
	path = os.path.dirname(path) # `.../ml_subtitle_align/`
	return os.path.join(path, *rel_path)

EMBEDDING_DIM = 15
OUTPUT_DIM = 1500 # 8*EMBEDDING_DIM 

class BaseModel(object):
	"""

	Attributes:
	"""

	def __init__(self, hyperparams=[]):
		"""Initialize a new empty object.
		"""
		self.hyperparams = hyperparams

	def train_model(self, input_3d):
		"""
		"""
		return self._create_model(input_3d, is_training=True)

	def test_model(self, input_3d):
		"""
		"""
		return self._create_model(input_3d, is_training=False)

	def load_variables_from_checkpoint(self, session, checkpoint):
		"""
		"""
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(session, checkpoint)

	def _create_model(self, input_3d, is_training=True):
		print("Please extend base model class. No default implementation available.")
		exit()