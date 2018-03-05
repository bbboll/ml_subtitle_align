import os.path
import json
import re
import numpy as np

class Subtitle(object):
	"""
	Objects of this class can be used to abstract around 
	subtitle file access.
	"""
	def __init__(self, raw_string):
		"""
		Take a raw string for instantiation
		"""
		self.raw_string = raw_string
		self.words_with_timing = self.parse_raw(raw_string)
		self.current_id = 0

	def parse_raw(self, raw_string):
		"""
		A raw subtitle string consists of groups

		00:00:12.958 --> 00:00:14.958
		Sergey Brin: I want to discuss a question\n\n 
		
		where the text group is optional and additional metadata
		such as speaker names or audience reaction descriptions may
		be included.

		The purpose of this function is to extract the above information,
		interpolate a point in time for each single word and return a list
		of shape
		[(time0, word0), (time1, word1), ...]
		"""
		out = []

		# find all groups
		m = re.findall(r'\n{2}(.+)\n{1}(.+\n?)+(?:\n{2}|\Z)', raw_string[7:])
		for group in m:
			(time_range, text) = group
			out.extend(self.extract_single_words(time_range, text))
		return out

	def extract_single_words(self, time_range, text):
		"""
		We extract a mapping
			time -> (single) word
		from the given mapping
			time range -> sequence of words
		This is done by linear interpolation.
		In real spoken audio, each word has an actual time interval 
		associated with it. We try to give a time offset which is the
		center of this interval.


		TODO: consider nonlinear interpolation
		"""
		# find word tokens. These may include punctuation
		tokens = re.findall(r'\s*(\S*)\s*', text)

		# TODO: filter speaker annotations

		# parse time
		start = self.parse_to_timestamp(time_range[0:12])
		end = self.parse_to_timestamp(time_range[17:])
		duration = end-start
		offsets = np.linspace(0.0, duration, num=len(tokens))
		const_off = (duration/len(tokens))/2
		return zip([start+const_off+off for off in offsets], tokens)

	def parse_to_timestamp(self, s):
		"""
		Take a string of shape
		00:15:04.958
		and parse it into a timestamp
		"""
		hours = int(s[0:2])
		minutes = int(s[3:5])
		seconds = int(s[6:8])
		return float(seconds + 60*minutes + 60*60*hours)

	def get_word_for_timestamp(self, t):
		if len(self.words_with_timing) <= self.current_id:
			return None
		if self.words_with_timing[self.current_id][0] < t:
			self.current_id += 1
			return self.words_with_timing[self.current_id][1]
		return None



if __name__ == '__main__':
	"""
	Testing the subtitle data extraction
	"""

	# load metadata from json file
	talks_json_path = "../data/talks/ted_talks_2000.json"
	if not os.path.isfile(talks_json_path):
		print("Please perform subtitle mining first.")
		exit()
	talks_json = json.load(open(talks_json_path))

	ind = 5
	for i, t in enumerate(talks_json):
		if t["id"] == 2011:
			ind = i

	sub = talks_json[ind]["subtitle"]
	print(sub)

	s = Subtitle(sub)



