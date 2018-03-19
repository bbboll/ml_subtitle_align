##
## /preprocessing/audio_features.py
##
## Created by Bastian Boll <bastibboll@googlemail.com>
##        and Paul Warkentin <p.warkentin@stud.uni-heidelberg.de>.
##

import numpy as np
import os
import os.path
import scipy.io.wavfile as wav
from pydub import AudioSegment
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank


def _path(*rel_path):
	"""Make absolute path to a file or directory in the project folder `ml_subtitle_align`.

	Arguments:
        *rel_path: List of path elements.

    Returns:
        `str`: Absolute path to requested file or directory.
	"""
	path = os.path.abspath(__file__) # `.../ml_subtitle_align/preprocessing/audio_features.py`
	path = os.path.dirname(path) # `.../ml_subtitle_align/preprocessing/`
	path = os.path.dirname(path) # `.../ml_subtitle_align/`
	return os.path.join(path, *rel_path)


class AudioFeatures(object):
	"""Objects of this class handle extracting and loading audio features from files.

	Attributes:
		features (obj:`numpy.ndarray`): NumPy array holding audio features. Default is None.
	"""

	def __init__(self):
		"""Initialize a new empty object.
		"""
		self.features = None

	def load_from_wav(self, path):
		"""Load audio features from existing wav file.

		Arguments:
			path (str): Path to an existing wav file.
		"""
		if not os.path.isfile(path):
			print("No such file or directory: `{}`.".format(path))
			exit()

		# load wav audio data
		(rate, sig) = wav.read(path)
		self.features = mfcc(sig, samplerate=rate, nfft=2048)

	def load_from_mp3(self, path):
		"""Load audio features from existing mp3 file.

		Arguments:
			path (str): Path to an existing mp3 file.
		"""
		if not os.path.isfile(path):
			print("No such file or directory: `{}`.".format(path))
			exit()

		# convert mp3 to wav
		try:
			sound = AudioSegment.from_mp3(path)
			sound.export(path + ".wav", format="wav")
		except:
			print("Conversion to WAV format failed!")
			return

		# load wav audio data
		(rate, sig) = wav.read(path + ".wav")
		self.features = mfcc(sig, samplerate=rate, nfft=2048)

		# remove wav file
		os.remove(path + ".wav")

	def load_from_numpy(self, path):
		"""Load extracted audio features from existing numpy file.

		Arguments:
			path (str): Path to an existing numpy file.
		"""
		if not os.path.isfile(path):
			print("No such file or directory: `{}`.".format(path))
			exit()

		# load audio features
		self.features = np.load(path)

	def save_to_numpy(self, path):
		"""Save extracted features to a numpy file.

		Arguments:
			path (str): Path to save features to.
		"""
		if self.features is None:
			print("Please extract features first. Did it fail?")
			exit()

		# save features
		np.save(path, self.features)
