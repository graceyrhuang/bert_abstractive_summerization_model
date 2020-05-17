'''
python compare_json.py --target_file_name attn_vis_data_0007.json --path_to_compared /scratch/kll482/pretrain/pretrained_model/decode
'''
import json
import glob
import argparse
from tqdm import tqdm

def get_abstract_str(file_name):
	with open(file_name) as json_file:
	    data = json.load(json_file)
	    abstract_str = data["abstract_str"]
	json_file.close()
	return abstract_str

def find_repeated_output_json(target_file_name, path_to_compared):
	target_abstract_str = get_abstract_str(path_to_compared+"/"+target_file_name)
	for file_name in tqdm(glob.glob(path_to_compared+"/*.json")):
		if target_abstract_str == get_abstract_str(file_name):
			if target_file_name != file_name:
				print(file_name)
	return

def checked_unique_json_number(path_to_compared):
	file_set = set([])
	for file_name in tqdm(glob.glob(path_to_compared+"/*.json")):
		file_set.add(get_abstract_str(file_name))
	print("Total unrepeated json: ", len(file_set))
	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--target_file_name", help="location of target json file name needed to be compared")
	parser.add_argument("--path_to_compared", help="location of comparing json")
	args = parser.parse_args()
	# find_repeated_output_json(args.target_file_name, args.path_to_compared)
	checked_unique_json_number(args.path_to_compared)
