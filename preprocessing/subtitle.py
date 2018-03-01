
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
		"""