import scipy.io.wavfile as wav
import numpy as np
from pydub import AudioSegment
import math
import os

SILENCE_THRESHOLD = 800
PARTITION_LENGHT = 0.02

def convert_to_wav(in_filename):
	sound = AudioSegment.from_mp3(in_filename)
	sound.export(in_filename[:-3]+"wav", format="wav")

# def cut_off(in_filename, threshold):
# 	(rate, sig) = wav.read(in_filename)
# 	duration = sig.shape[0] / rate
# 	partitions = math.floor(duration / 0.02)
# 	remainder = sig.shape[0] % partitions
# 	if remainder != 0:
# 		sig = sig[:-remainder]
# 	out_filename = in_filename[:-4]+"_"+str(threshold)+".wav"
# 	data = np.array([x for x in np.split(sig, partitions) if np.max(np.linalg.norm(x, axis=1)) > threshold], dtype=sig.dtype)
# 	wav.write(out_filename, rate, np.reshape(data, (len(data)*len(data[0]),2)))

class Sound(object):
	"""
	Objects of this class provide an interface for seamless use of
	mp3 encoded audio files.
	Instantiation of an object results in a temporary wav file being
	generated.
	"""
	def __init__(self, mp3_filepath):
		convert_to_wav(mp3_filepath)
		self.mp3_filepath = mp3_filepath
		self.wav_filepath = mp3_filepath[:-3]+"wav"
		(self.rate, self.sig) = wav.read(self.wav_filepath)
		duration = self.sig.shape[0] / self.rate
		self.partitions = math.floor(duration / PARTITION_LENGHT)
		remainder = self.sig.shape[0] % self.partitions
		if remainder != 0:
			self.sig = self.sig[:-remainder]
		self.duration = self.sig.shape[0] / self.rate

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		os.remove(self.wav_filepath)

	def interpolate_without_silence(self, start, end, num_words):
		"""
		Interpolates [start, end] linearly with silence removed
		"""
		sig_interval = self.sig[math.floor(start*self.rate):math.floor(end*self.rate)]
		interval_partitions = math.floor((end-start) / PARTITION_LENGHT)
		remainder = sig_interval.shape[0] % interval_partitions
		if remainder != 0:
			sig_interval = sig_interval[:-remainder]
		bitmap = np.array([(np.max(np.linalg.norm(x, axis=1)) > SILENCE_THRESHOLD) for x in np.split(sig_interval, interval_partitions)], dtype=bool)
		reduced_length = np.sum(bitmap)*PARTITION_LENGHT
		offsets = []
		cursor = 0
		word_ind = 0
		reduced_offsets = np.linspace(reduced_length/num_words/2, reduced_length, num=num_words, endpoint=False)
		for full_ind, audible in enumerate(bitmap):
			if audible:
				cursor += PARTITION_LENGHT
			if word_ind >= len(reduced_offsets):
				break
			if cursor > reduced_offsets[word_ind]:
				offsets.append(full_ind*PARTITION_LENGHT+cursor-reduced_offsets[word_ind])
				word_ind += 1
		return offsets

	# def silence_in_interval(self, start, end):
	# 	"""
	# 	Compute the amount of silence (in ms) in the interval [start, end]
	# 	"""
	# 	silent_parts = 0
	# 	for x in np.split(self.sig, self.partitions):
	# 		silent_parts += 1 if np.max(np.linalg.norm(x, axis=1)) > SILENCE_THRESHOLD
	# 	return silent_parts*PARTITION_LENGHT