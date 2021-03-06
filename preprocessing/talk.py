from .subtitle import Subtitle
from math import floor
import os.path
import json


def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	parent = os.path.join(os.path.dirname(__file__), "..")
	return os.path.abspath(os.path.join(parent, relpath))

def _get_full_path(*rel_path):
	"""Make absolute path to a file or directory in the project folder ml_subtitle_align.

	Arguments:
        *rel_path: List of path elements.

    Returns:
        `str`: Absolute path to requested file or directory.
	"""
	path = os.path.abspath(__file__) # `.../ml_subtitle_align/preprocessing/talk.py`
	path = os.path.dirname(path) # `.../ml_subtitle_align/preprocessing/`
	path = os.path.dirname(path) # `.../ml_subtitle_align/`
	return os.path.join(path, *rel_path)

class AllTalks(object):
	"""
	Iterator for all talks in the dataset.
	"""
	def __init__(self, limit=None):
		self.limit = limit
		self.count = 0
		self.chunk_start = 0
		self.chunk = None
		self.chunk_iter = None
		self.seen_ids = []

	def __iter__(self):
		return self

	def __next__(self):
		if self.limit != None and self.count >= self.limit:
			raise StopIteration

		if self.chunk == None:
			chunk_filepath = _path("data/talks/ted_talks_{}.json".format(self.chunk_start))
			if not os.path.isfile(chunk_filepath):
				raise StopIteration
			self.chunk = json.load(open(chunk_filepath))
			self.chunk_iter = iter(self.chunk)

		talk = None
		talk_obj = None
		try:
			talk = next(self.chunk_iter)
			if talk["id"] in self.seen_ids:
				return self.__next__()
			self.seen_ids.append(talk["id"])
		except StopIteration:
			self.chunk_start += 20
			self.chunk = None
			return self.__next__()
		self.count += 1
		try:
			return Talk(talk["id"], talk=talk)
		except:
			return self.__next__()

class Talk(object):
	"""
	A talk object binds metadata for a single TED talk in the dataset.
	"""
	def __init__(self, talk_id, talk=None):
		self.ID = talk_id

		if talk == None:
			metamap = json.load(open(_path("data/talks/metamap.json")))

			# load talk subtitle from file
			filename = _path("data/talks/ted_talks_{}.json".format(metamap[str(talk_id)]))
			if not os.path.isfile(filename):
				raise Exception("talk metadata file {} is missing (trying to load talk {})".format(filename, talk_id))
			file = json.load(open(filename))
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

		self.subtitle = None
		self.raw_subtitle = talk["subtitle"]
		self.transcript = talk["transcript"]
		self.duration = int(talk["duration"])
		self.url = talk["url"]
		self.title = talk["title"]

	def load_subtitle(self):
		self.subtitle = Subtitle(self.raw_subtitle, self.ID)

	def features_path(self):
		return _path("data/audio_features/{}.npy".format(self.ID))

	def audio_path(self):
		return _get_full_path("data", "audio", "{}.mp3".format(self.ID))