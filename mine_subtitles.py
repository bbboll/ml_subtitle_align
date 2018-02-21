import csv

talks = []

with open('data/kaggle/ted_main.csv', 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=',')
	for row in csvreader:
		if row[7] == 'name':
			continue
		url = row[15]
		if url.endswith('\n'):
			url = url[0:len(url)-1]
		talks.append({
			"title": row[14],
			"duration": row[2],
			"url": url})
print(talks[135])