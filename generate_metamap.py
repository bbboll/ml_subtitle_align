import os.path
import json

talk_files = {}

chunk_size = 20
for chunk_start in range(0, 2560, chunk_size):
	filename = "data/talks/ted_talks_{}.json".format(chunk_start)
	if not os.path.isfile(filename):
		break

	data = json.load(open(filename))
	for talk in data:
		talk_files[talk["id"]] = chunk_start

with open("data/talks/metamap.json", "w", encoding="utf-8") as f:
	json.dump(talk_files, f)