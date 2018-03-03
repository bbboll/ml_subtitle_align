# Subtitle Alignment

## Prerequisites
The code in this repo is expected to be run with Python `3.*`. Additional library requirements are listed in `requirements.txt`. You will also need to install `ffmpeg` for extracting audio from fetched video files.

## Data aquisition
We use TED-talk transcripts and subtitles as well as their audio recordings. The audio in this dataset is very high-quality and consists largely of clean spoken words.

This [kaggle dataset](https://www.kaggle.com/rounakbanik/ted-talks) contains structural metadata about TED talks such as title, url and even transcripts of each talk. The subtitles can be requested from TEDs servers at URLs with shape 

`https://hls.ted.com/talks/66/subtitles/en/full.vtt`

Hence, we need to find the unique ID associated with each talk in order to fetch the respective subtitle. Fortunately, these IDs can be found in the markup of the site reached by the talk url we already have sitting in the above dataset.

After downloading the kaggle dataset and unpacking it to `ml_subtitle_align/data/kaggle/` you can execute `mine_subtitles.py` which will fetch the subtitles incrementally.

Once all previous steps have (at least partially) finished successfully, you can execute `download_audio.py` which will incrementally fetch all required MP3 audio files from TED and save them into `ml_subtitle_align/data/audio`. Consequently, `convert_audio.py` will convert all MP3 files from `ml_subtitle_align/data/audio` to WAV for easier handling and save the resulting files in `ml_subtitle_align/data/audio_wav`.