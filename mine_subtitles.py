import httplib2
from bs4 import BeautifulSoup
import csv

talks = []
http = httplib2.Http()

def fetch_id_for_url(url):
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

print(talks[312])
url = talks[312]["url"]
talk_id, ok = fetch_id_for_url(url)










