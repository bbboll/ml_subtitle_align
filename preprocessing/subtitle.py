import os.path
import json
from webvtt import WebVTT

class Subtitle(object):
	"""
	Objects of this class can be used to abstract around 
	subtitle file access.
	"""
	def __init__(self, raw_string, transcript):
		"""
		Take a raw string for instantiation
		"""
		self.raw_string = raw_string
		self.words_with_timing = parse_raw(raw_string, transcript)

	def parse_raw(raw_string, transcript):
		"""
		A raw subtitle string consists of groups

		00:00:12.958 --> 00:00:14.958
		Sergey Brin: I want to discuss a question\n\n 
		
		where the text group is optional and additional metadata
		such as speaker names or audience reaction descriptions may
		not be included in the transcript.

		The purpose of this function is to extract the above information,
		interpolate a point in time for each single word and return a list
		of shape
		[(word0, time0), (word1, time1), ...]
		"""

if __name__ == '__main__':
	"""
	Testing the subtitle data extraction
	"""

	# load metadata from json file
	talks_json_path = "../data/talks/ted_talks_100.json"
	if not os.path.isfile(talks_json_path):
		print("Please perform subtitle mining first.")
		exit()
	talks_json = json.load(open(talks_json_path))

	print(talks_json[1]["subtitle"])
	print(talks_json[1]["transcript"])