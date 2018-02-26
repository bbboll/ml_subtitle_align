import httplib2
from bs4 import BeautifulSoup
import csv
import json
import time
import os.path

talks = []
transcripts = {}
http = httplib2.Http()

def fetch_id_for_url(url):
	time.sleep(2.5)
	# fetch url
	try:
		headers, body = http.request(url)
		if headers["status"] != "200":
			# TODO: http error handling
			print("Unexpected HTTP status {}".format(headers["status"]))
			return 0, False

		# fish talk ID out of markup soup
		soup = BeautifulSoup(body, 'html.parser')
		meta = soup.find("meta",  property="al:ios:url")
		# we are looking for <meta content="ted://talks/633?source=facebook" property="al:ios:url"/>
		if meta:
			content = meta["content"]
			id_str = ""
			content = content[12:]
			for char in content:
				if char in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
					id_str += char
				else:
					break
			return int(id_str), True
		else:
			# TODO: error handling
			print("Markup did not contain standard meta tag")
			return 0, False
	except httplib2.ServerNotFoundError:
		# TODO: handle connection problems
		print("Unable to hit url {}".format(url))
		return 0, False

def fetch_subtitle_for_id(talk_id):
	time.sleep(2.5)
	url = "https://hls.ted.com/talks/{}/subtitles/en/full.vtt".format(talk_id)
	try:
		headers, body = http.request(url)
		if headers["status"] != "200":
			# TODO: http error handling
			print("Unexpected HTTP status {}".format(headers["status"]))
			return "", False
		return body.decode("utf-8"), True
	except httplib2.ServerNotFoundError:
		# TODO: handle connection problems
		print("Unable to hit url {}".format(url))
		return "", False

# load talk list from kaggle dataset
with open('data/kaggle/ted_main.csv', 'r') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=',')
	for row in csvreader:
		# skip csv header
		if row[7] == 'name':
			continue
		url = row[15]

		# clean url field
		if url.endswith('\n'):
			url = url[0:len(url)-1]

		# save talk
		talks.append({
			"title": row[14],
			"duration": row[2],
			"url": url})

# load transcripts from kaggle dataset
with open('data/kaggle/transcripts.csv', 'r') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=',')
	for row in csvreader:
		# skip csv header
		if row[1] == 'url':
			continue
		url = row[1]

		# clean url field
		if url.endswith('\n'):
			url = url[0:len(url)-1]

		# save transcript
		transcripts[url] = row[0]

# fetch ID and subtitle for each talk
output = []
chunk_size = 20
for chunk_start in range(0, len(talks), chunk_size):
	out_filename = "data/ted_talks_{}.json".format(chunk_start)
	if os.path.isfile(out_filename):
		continue
	for talk in talks[chunk_start:(chunk_start+chunk_size)]:
		trans_ok = False
		try:
			transcript = transcripts[talk["url"]]
			talk["transcript"] = transcript
			trans_ok = True
		except KeyError:
			print("Transcript missing for \"{}\"".format(talk["title"]))
			continue

		print("Fetching ID+subtitles for talk: {}".format(talk["title"]))
		talk_id, id_ok = fetch_id_for_url(talk["url"])
		if id_ok:
			talk["id"] = talk_id
		subtitle, sub_ok = fetch_subtitle_for_id(talk_id)
		if sub_ok:
			talk["subtitle"] = subtitle
		
		if id_ok and sub_ok and trans_ok:
			output.append(talk)
		else:
			print("loading subtitle and/or transcript for \"{}\" failed.".format(talk["title"]))

	with open(out_filename, "w", encoding="utf-8") as f:
		json.dump(output, f)
	time.sleep(40)









