import os.path
import json

ids = []
total_duration = 0

chunk_size = 20
for chunk_start in range(0, 2560, chunk_size):
	filename = "data/talks/ted_talks_{}.json".format(chunk_start)
	if not os.path.isfile(filename):
		break

	data = json.load(open(filename))
	for talk in data:
		if not talk["id"] in ids:
			ids.append(talk["id"])
			total_duration += int(talk["duration"]) / 60

print("The total duration of all talks is {0:.2f}h".format(total_duration/60))
# 561.54h