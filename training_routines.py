import numpy as np
import os
import datetime
import hashlib
import json
import extract_training_data as extractor
import re
import models.conv_model
import models.conv_lstm_model
import models.deep_conv_model
import models.big_deep_conv_model
import models.conv_deep_nn

def _get_full_path(*rel_path):
	"""Make absolute path to a file or directory in the project folder ml_subtitle_align.

	Arguments:
        *rel_path: List of path elements.

    Returns:
        `str`: Absolute path to requested file or directory.
	"""
	path = os.path.abspath(__file__) # `.../ml_subtitle_align/train.py`
	path = os.path.dirname(path) # `.../ml_subtitle_align/`
	return os.path.join(path, *rel_path)

def get_training_save_paths(model_config):
	config_hash = hashlib.md5("-".join(["{}:{}".format(k,v) for k,v in model_config.items()]).encode('utf-8')).hexdigest()
	dirpath = _get_full_path("training_data", "run_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H"), config_hash))
	if not os.path.isdir(dirpath):
		os.mkdir(dirpath)
	return dirpath, os.path.join(dirpath, "retrain_logs"), os.path.join(dirpath, "train")

def get_model_from_run(run):
	"""
	Given a relative path of shape
		run_2018-03-21-23_9fbe4594184ad6a9224f865a2bdfd407/
	extract the (absolute) model checkpoint and training_config.json paths
	"""
	checkpoint_meta_path = _get_full_path("training_data", run, "train", "checkpoint")
	checkpoint_path = ""
	with open(checkpoint_meta_path, "r") as f:
		line = f.readline()
		abs_checkpoint_path = re.search(r'\"(.+)\"', line).group(0)
		checkpoint_num = re.search(r'model.ckpt-([0-9]+)', abs_checkpoint_path).group(0)
		checkpoint_path = _get_full_path("training_data", run, "train", checkpoint_num)
	train_config = json.load(open(_get_full_path("training_data", run, "training_config.json")))
	return checkpoint_path, train_config

def get_model_obj_from_config(training_config):
	if training_config["model"] == "simple_conv":
		return models.conv_model.Model()
	elif training_config["model"] == "dense_conv":
		return models.conv_model.Model(hyperparams=["dense"])
	elif training_config["model"] == "conv_lstm":
		return models.conv_lstm_model.Model()
	elif training_config["model"] == "deep_conv":
		return models.deep_conv_model.Model()
	elif training_config["model"] == "big_deep_conv":
		return models.big_deep_conv_model.Model()
	elif training_config["model"] == "conv_deep_nn":
		return models.conv_deep_nn.Model()
	else: # if training_config["model"] == "experimental"
		return models.model.Model()
