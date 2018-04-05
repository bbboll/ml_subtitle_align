import os.path

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
