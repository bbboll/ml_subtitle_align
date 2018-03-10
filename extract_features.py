import os
from preprocessing.audio_features import AudioFeatures
from shutil import copyfile

def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	current_dir = os.path.dirname(__file__)
	return os.path.abspath(os.path.join(current_dir, relpath))

# extract features from audio
filenames = [ff for ff in os.listdir(_path("data/audio")) if ff != ".gitkeep"]
for ff in filenames:
	features = AudioFeatures()

	# skip if already extracted
	output_path = _path("data/audio_features/{}".format(ff[:-3] + "npy"))
	if os.path.isfile(output_path):
		continue

	print("Extracting features from 'data/audio/{}' ...".format(ff))

	# load mp3 / wav
	path = _path("data/audio/{}".format(ff))
	if ff.endswith(".mp3"):
		features.load_from_mp3(path)
	elif ff.endswith(".wav"):
		features.load_from_wav(path)
	else:
		print("Unexpected filename: {}. Skipping.".format(ff))
		continue

	# save features
	if not features.features is None:
		features.save_to_numpy(output_path)
