import os

import matplotlib.pyplot as plt
from PIL import Image

import random


def get_file_names():	
	test_dir = [x[0] for x in os.walk('archive/test/HGG_LGG')]
	test_files = []

	for patients in test_dir[1:]:
		for (dirpath, dirnames, filenames) in os.walk(patients):
			test_files += [patients + "/" + x for x in filenames]
	

	train_files = []
	train_dir_HGG = [x[0] for x in os.walk('archive/train/HGG')]
	for patients in train_dir_HGG[1:]:
		for (dirpath, dirnames, filenames) in os.walk(patients):
			train_files += [patients + "/" + x for x in filenames]
	
	train_dir_LGG = [x[0] for x in os.walk('archive/train/LGG')]
	for patients in train_dir_LGG[1:]:
		for (dirpath, dirnames, filenames) in os.walk(patients):
			train_files += [patients + "/" + x for x in filenames]
	
	for files in test_files:
		if not("T1" in files or "Flair" in files or "T2" in files or "T1c" in files):
			test_files.remove(files)
		if "0T" in files:
			test_files.remove(files)
	
	for files in train_files:
		if not("T1" in files or "Flair" in files or "T2" in files or "T1c" in files):
			train_files.remove(files)
		if "0T" in files:
			train_files.remove(files)
	return test_files, train_files
