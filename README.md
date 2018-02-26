# Subtitle Alignment

## Prerequisites
The mining script requires Python 3.* as well as `httplib2` and `beautifulsoup4`

## Data aquisition
We use TED-talk transcripts and subtitles as well as their audio recordings. The audio in this dataset is very high-quality and consists largely of clean spoken words.

This [kaggle dataset](https://www.kaggle.com/rounakbanik/ted-talks) contains structural metadata about TED talks such as title, url and even transcripts of each talk. The subtitles can be requested from TEDs servers at URLs with shape 

`https://hls.ted.com/talks/66/subtitles/en/full.vtt`

Hence, we need to find the unique ID associated with each talk in order to fetch the respective subtitle. Fortunately, these IDs can be found in the markup of the site reached by the talk url we already have sitting in the above dataset.

After downloading the kaggle dataset and unpacking it to `ml_subtitle_align/data/kaggle/` you can execute `mine_subtitles.py` which will fetch the subtitles and save all gathered data into a newly created file `ml_subtitle_align/data/all_data.json`.

Once all previous steps have finished successfully, you can execute `download_audio.py` which will fetch all required MP3 audio files from TED and save them into `ml_subtitle_align/data/audio`.