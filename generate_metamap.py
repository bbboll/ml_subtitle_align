import os.path
import json

def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	current_dir = os.path.dirname(__file__)
	return os.path.abspath(os.path.join(current_dir, relpath))

talk_files = {}

chunk_size = 20
for chunk_start in range(0, 2560, chunk_size):
	filename = _path("data/talks/ted_talks_{}.json".format(chunk_start))
	if not os.path.isfile(filename):
		break

	data = json.load(open(filename))
	for talk in data:
		talk_files[talk["id"]] = chunk_start

with open(_path("data/talks/metamap.json"), "w", encoding="utf-8") as f:
	json.dump(talk_files, f)