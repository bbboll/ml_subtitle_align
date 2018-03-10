from subtitle import Subtitle
from math import floor
import os.path
import json

def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	parent = os.path.join(os.path.dirname(__file__), "..")
	return os.path.abspath(os.path.join(parent, relpath))

class AllTalks(object):
	"""
	Iterator for all talks in the dataset.
	"""
	def __init__(self, limit=None):
		self.limit = limit
		self.metamap = json.load(open(_path("data/talks/metamap.json")))
		# TODO


class Talk(object):
	"""
	A talk object binds metadata for a single TED talk in the dataset.
	"""
	def __init__(self, talk_id):
		self.ID = talk_id

		# load talk subtitle from file
		metamap = json.load(open(_path("data/talks/metamap.json")))
		filename = _path("data/talks/ted_talks_{}.json".format(metamap[str(talk_id)]))
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
		self.audio = _path("data/audio/{}.mp3".format(talk_id))
		if not os.path.isfile(self.audio):
			raise Exception("Audio file for talk {} missing.".format(talk_id))
		if not os.path.isfile(_path("data/audio_features/{}.npy".format(talk_id))):
			raise Exception("Audio features for talk {} missing.".format(talk_id))

		self.subtitle = Subtitle(talk["subtitle"], self.ID)
		self.transcript = talk["transcript"]
		self.duration = talk["duration"]
		self.url = talk["url"]

if __name__ == '__main__':
	t = Talk(1)
	it = AllTalks()