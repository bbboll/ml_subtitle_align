from subtitle import Subtitle
from math import floor
import os.path
import json

class Talk(object):
	"""
	A talk object binds metadata for a single TED talk in the dataset.
	"""
	def __init__(self, talk_id):
		self.ID = talk_id

		# load talk subtitle from file
		metamap = json.load(open("../data/talks/metamap.json"))
		filename = "../data/talks/ted_talks_{}.json".format(metamap[str(talk_id)])
		if not os.path.isfile(filename):
			raise Exception("talk metadata file {} is missing (trying to load talk {})".format(filename, talk_id))
		file = json.load(open(filename))
		talk = None
		for t in file:
			if t["id"] == talk_id:
				talk = t
		if talk == None:
			raise Exception("Talk for ID {} not found in the dataset.".format(talk_id))

		# check audio file and extracted data availability
		self.audio = "../data/audio/{}.mp3".format(talk_id)
		if not os.path.isfile(self.audio):
			raise Exception("Audio file for talk {} missing.".format(talk_id))
		if not os.path.isfile("../data/audio_features/{}.npy".format(talk_id)):
			raise Exception("Audio features for talk {} missing.".format(talk_id))

		self.title = talk["title"]
		self.subtitle = Subtitle(talk["subtitle"])
		self.transcript = talk["transcript"]
		self.duration = talk["duration"]
		self.url = talk["url"]
