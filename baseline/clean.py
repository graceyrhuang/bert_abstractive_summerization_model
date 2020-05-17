import argparse
import json

def add_arguments():
	parser = argparse \
	.ArgumentParser(description='Process some integers.')
	
	parser.add_argument(
		'filename',
		help='input the file name'
		)
	args = parser.parse_args()

	return args

def transform(file_name):

	with open("file_name", "r") as f:
		data = json.load(f)

	for index, sample in enumerate(data):
		with open("")
		text = ''.join(sample["src_txt"])
