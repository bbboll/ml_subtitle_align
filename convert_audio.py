from pydub import AudioSegment
import os
from shutil import copyfile

filenames = [f for f in os.listdir('data/audio') if f != ".gitkeep"]
for file in filenames:
	if file.endswith(".mp3"):
		if os.path.isfile("data/audio_wav/{}".format(file[:-3]+"wav")):
			continue
		print("converting file: {}".format(file))
		try:
			sound = AudioSegment.from_mp3("data/audio/{}".format(file))
			sound.export("data/audio_wav/{}".format(file[:-3]+"wav"), format="wav")
		except:
			print("conversion failed!")
	if file.endswith(".wav"):
		if os.path.isfile("data/audio_wav/{}".format(file)):
			continue
		print("copying file: {}".format(file))
		try:
			copyfile("data/audio/"+file, "data/audio_wav/"+file)
		except:
			print("copying failed!")
