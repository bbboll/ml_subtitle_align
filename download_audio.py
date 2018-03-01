import httplib2
import json
from bs4 import BeautifulSoup
import time
import os.path
import urllib.request
import urllib.error

http = httplib2.Http()
url_mapping = {}

def convert_video_download_link(link):
	if link.endswith('?apikey=TEDDOWNLOAD'):
		link = link[:-19]
	if link.endswith('-light.mp4'):
		link = link[:-10]+'.mp3'
	return link

def fetch_meta_site(num):
	url = "https://www.ted.com/talks/quick-list?page={}".format(num)
	try:
		headers, body = http.request(url)

		if headers["status"] != "200":
			print("Unexpected status {}".format(headers["status"]))
			# 429 may happen if we fetch too quickly
			return headers["status"], []

		# TODO: parse url -> download url mapping from markup
		out = []
		soup = BeautifulSoup(body, 'html.parser')
		lists = soup.find_all('div', class_='quick-list__row')
		for l in lists:
			link_containers = [c for i, c in enumerate(l.children) if i in [3, 9]]
			try:
				url = "https://www.ted.com{}".format(link_containers[0].find("a")["href"])
				download_url = link_containers[1].find_all("a")
				if len(download_url) > 0:
					download_url = download_url[0]["href"]
				else:
					continue
				out.append([url, convert_video_download_link(download_url)])
			except KeyError:
				print("Unable to parse markup for num {}".format(num))
				return "500", []
		return "200", out
	except httplib2.ServerNotFoundError:
		# TODO: handle connection problems
		print("Unable to hit url {}".format(url))
		return "404", []

# fetch (url -> download link) mapping
mapping_filename = "data/download_link_mapping.json"
if not os.path.isfile(mapping_filename):
	out = {}
	for i in range(1, 76):
		print("fetching site {}".format(i))
		status, mapping = fetch_meta_site(i)
		time.sleep(2.5)
		if status == '429':
			time.sleep(120)
			status, mapping = fetch_meta_site(i)
		if status == '429' or status == '404' or status == '500':
			print("Fetching site {} failed permanently with status {}".format(i, status))
			continue
		for m in mapping:
			out[m[0]] = m[1]
	with open(mapping_filename, "w", encoding="utf-8") as f:
		json.dump(out, f)

download_urls = json.load(open(mapping_filename))

# download audio for all talks with available metadata
chunk_size = 20
for chunk_start in range(0, 10000, chunk_size):
	filename = "data/ted_talks_{}.json".format(chunk_start)
	if not os.path.isfile(filename):
		break
	
	data = json.load(open(filename))
	for talk in data:
		audio_filename = "data/audio/{}.mp3".format(talk["id"])
		if os.path.isfile(audio_filename):
			continue
		if talk["url"] in download_urls.keys():
			print("Fetching {}".format(talk["url"]))
			try:
				fetched_data = urllib.request.urlopen(download_urls[talk["url"]]).read()
				with open(audio_filename, "wb") as audio_file:
					audio_file.write(fetched_data)
			except urllib.error.HTTPError as e:
				print("Fetching audio failed: {}".format(e.reason))