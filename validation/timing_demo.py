import sys
import _thread
import json
import os
import time
from math import floor
from subtitle import Subtitle
from talk import Talk

def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	parent = os.path.join(os.path.dirname(__file__), "..")
	return os.path.abspath(os.path.join(parent, relpath))

class TimingDemo(object):
	"""
	Provides functionality to play an audio file while 
	concurrently displaying associated subtitles.
	"""
	def __init__(self, audio_filename, subtitle):
		self.audio_filename = audio_filename
		self.subtitle = subtitle

	def play(self):
		"""
		Plays the audio and concurrently displays subtitles
		"""

		# play audio on seperate thread
		if os.path.isfile(self.audio_filename):
			_thread.start_new_thread(os.system, ("cvlc \"{}\"".format(self.audio_filename),))
		else:
			print("Audio file {} not found.".format(self.audio_filename))
			exit()
		
		# display subtitles concurrently on the main thread
		start_time = time.time()
		print("Starting subtitles for talk {}".format(self.audio_filename))
		while True:
			word = self.subtitle.get_word_for_timestamp(time.time()-start_time)
			if word != None:
				print(word)
			time.sleep(0.005)

if __name__ == '__main__':
	if len(sys.argv) == 1:
		print("Please pass a talk ID")
		exit()
	talk_id = int(sys.argv[1])
	talk = Talk(talk_id)
	
	# perform demo
	demo = TimingDemo(_path("data/audio/{}.mp3".format(talk_id), talk.subtitle))
	demo.play()
