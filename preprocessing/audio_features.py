import numpy as np
import os
import os.path
import scipy.io.wavfile as wav
from pydub import AudioSegment
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank


class AudioFeatures(object):
	"""Objects of this handle extracting and loading audio features from files.

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
			print("Please download audio first.")
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
			print("Please download audio first.")
			exit()

		# convert mp3 to wav
		try:
			sound = AudioSegment.from_mp3(path)
			sound.export(path + ".wav", format="wav")
		except:
			print("Conversion to MP3 format failed!")
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
			print("Please download audio first.")
			exit()

		# load audio features
		self.features = np.load(path)

	def save_to_numpy(self, path):
		"""Save extracted features to a numpy file.

		Arguments:
			path (str): Path to save features to.
		"""
		if self.features is None:
			print("Please extract features first.")
			exit()

		# save features
		np.save(path, self.features)


if __name__ == "__main__":
	"""Testing the audio features extraction.
	"""

	audio = AudioFeatures()
	audio.load_from_wav("audio_features_example/english.wav") # https://raw.githubusercontent.com/jameslyons/python_speech_features/master/english.wav
	audio.save_to_numpy("audio_features_example/english.npy")

	print(audio.features.shape)
	print(audio.features[:3, :])
