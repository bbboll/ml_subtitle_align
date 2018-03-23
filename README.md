# Subtitle Alignment

The aim of this project is to compute time alignment information between (English) spoken word audio and a provided transcript of the same audio. This is done by using an acoustic model trained on a large number of [TED talks](https://ted.com) and performing a global optimization procedure with the model predictions as input. A much more in-depth discussion of this process can found in `ml_subtitle_align/doc`.

This project is a mini research project created in fulfillment of requirements for the course "Fundamentals of Machine Learning" at Heidelberg University in the winter semester 2017/18.

## Getting started
This repository contains all necessary code for both training and prediction. To perform training, you will need to follow the procedure outlined under "Data aquisition" to download the necessary audio material (about 40GB) and extract MFCC features from it (another 40GB of disk space required). 

For prediction, you can use one of the pretrained models located in `ml_subtitle_align/training_data/run_*/train/`. Simple prediction for an audio snippet can be performed by executing `snippet_demo.py`. To compute alignment information for a whole TED-talk, choose one of the example talks in `ml_subtitle_align/data/audio` and execute `predict.py` with the required arguments.

## Prerequisites
The code in this repo is expected to be run with Python `3.*`. Additional library requirements are listed in `requirements.txt`. You will also need to install `ffmpeg` for extracting audio from fetched video files.
If you wish to demo subtitle synchronization, you will also need to have [VLC](https://de.wikipedia.org/wiki/VLC_media_player) installed.

## Data aquisition
We use TED-talk transcripts and subtitles as well as their audio recordings. The audio in this dataset is very high-quality and consists largely of clean spoken words.

This [kaggle dataset](https://www.kaggle.com/rounakbanik/ted-talks) contains structural metadata about TED talks such as title, url and even transcripts of each talk. The subtitles can be requested from TEDs servers at URLs with shape

`https://hls.ted.com/talks/66/subtitles/en/full.vtt`

Hence, we need to find the unique ID associated with each talk in order to fetch the respective subtitle. Fortunately, these IDs can be found in the markup of the site reached by the talk url we already have sitting in the above dataset.

After downloading the kaggle dataset and unpacking it to `ml_subtitle_align/data/kaggle/` you can execute `mine_subtitles.py` which will fetch the subtitles incrementally.

Once all previous steps have (at least partially) finished successfully, you can execute `download_audio.py` which will incrementally fetch all required MP3 audio files from TED and save them into `ml_subtitle_align/data/audio`. Consequently, `extract_features.py` will extract the audio features from MP3 and WAV files in `ml_subtitle_align/data/audio` and save the resulting files in `ml_subtitle_align/data/audio_features`.

More specifically, [python_speech_features](https://github.com/jameslyons/python_speech_features) is used to extract [MFCC](https://de.wikipedia.org/wiki/Mel_Frequency_Cepstral_Coefficients) features with a 25ms analysis window and 10ms step between successive windows.

## License
All python code in this repository is available under an MIT license. All files in `ml_subtitle_align/data/` are copies or derivatives of content owned by [TED](https://www.ted.com) and are hence [subject to additional restrictions](https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy).