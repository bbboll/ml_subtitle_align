import httplib2
import json
from bs4 import BeautifulSoup
import time
import os
import os.path
import subprocess
import urllib.request
import urllib.error
import ssl

def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	current_dir = os.path.dirname(__file__)
	return os.path.abspath(os.path.join(current_dir, relpath))

http = httplib2.Http()
url_mapping = {}

# def convert_video_download_link(link):
# 	if link.endswith('?apikey=TEDDOWNLOAD'):
# 		link = link[:-19]
# 	return link

# def fetch_meta_site(num):
# 	url = "https://www.ted.com/talks/quick-list?page={}".format(num)
# 	try:
# 		headers, body = http.request(url)

# 		if headers["status"] != "200":
# 			print("Unexpected status {}".format(headers["status"]))
# 			# 429 may happen if we fetch too quickly
# 			return headers["status"], []

# 		# TODO: parse url -> download url mapping from markup
# 		out = []
# 		soup = BeautifulSoup(body, 'html.parser')
# 		lists = soup.find_all('div', class_='quick-list__row')
# 		for l in lists:
# 			link_containers = [c for i, c in enumerate(l.children) if i in [3, 9]]
# 			try:
# 				url = "https://www.ted.com{}".format(link_containers[0].find("a")["href"])
# 				download_url = link_containers[1].find_all("a")
# 				if len(download_url) > 0:
# 					download_url = download_url[0]["href"]
# 				else:
# 					continue
# 				out.append([url, convert_video_download_link(download_url)])
# 			except KeyError:
# 				print("Unable to parse markup for num {}".format(num))
# 				return "500", []
# 		return "200", out
# 	except httplib2.ServerNotFoundError:
# 		# TODO: handle connection problems
# 		print("Unable to hit url {}".format(url))
# 		return "404", []

def fetch_audio_from_url(audio_filename, url):
	url = url[:-4]+"-light.mp4"
	context = ssl._create_unverified_context()
	try:
		fetched_data = urllib.request.urlopen(url, context=context).read()
		with open(_path("data/audio/tmp.mp4"), "wb") as video_file:
			video_file.write(fetched_data)
		command = "ffmpeg -i data/audio/tmp.mp4 -ab 160k -ac 2 -ar 44100 -vn {}".format(audio_filename)
		subprocess.call(command, shell=True)
		os.remove(_path("data/audio/tmp.mp4"))
	except (urllib.error.HTTPError, urllib.error.URLError) as e:
		print("Fetching audio failed: {}".format(e.reason))

# fetch (url -> download link) mapping
mapping_filename = _path("data/audio/download_link_mapping.json")
# if not os.path.isfile(mapping_filename):
# 	out = {}
# 	for i in range(1, 76):
# 		print("fetching site {}".format(i))
# 		status, mapping = fetch_meta_site(i)
# 		time.sleep(2.5)
# 		if status == '429':
# 			time.sleep(120)
# 			status, mapping = fetch_meta_site(i)
# 		if status == '429' or status == '404' or status == '500':
# 			print("Fetching site {} failed permanently with status {}".format(i, status))
# 			continue
# 		for m in mapping:
# 			out[m[0]] = m[1]
# 	with open(mapping_filename, "w", encoding="utf-8") as f:
# 		json.dump(out, f)

download_urls = json.load(open(mapping_filename))

# download audio for all talks with available metadata
# 380
chunk_size = 20
for chunk_start in range(0, 2560, chunk_size):
	print("chunk start: {}".format(chunk_start))
	filename = _path("data/talks/ted_talks_{}.json".format(chunk_start))
	if not os.path.isfile(filename):
		break

	data = json.load(open(filename))
	for talk in data:
		audio_filename = _path("data/audio/{}.mp3".format(talk["id"]))
		if os.path.isfile(audio_filename):
			continue
		if talk["url"] in download_urls.keys():
			print("Fetching {}".format(talk["url"]))
			fetch_audio_from_url(audio_filename, download_urls[talk["url"]])