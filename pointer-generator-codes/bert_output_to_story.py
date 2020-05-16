'''
python bert_output_to_story.py --folder_path=./bert_dir/ --gold_name=bert_linear_step33000.gold --candidate_name=bert_linear_step33000.candidate
'''
from tqdm import tqdm
import argparse

# def read_gold(path):
# 	with open(path, "r") as file:
# 		data = file.read()
# 	gold = data.split("\n")[:-1]

# 	return gold

def read_text(path):
	with open(path, "r") as file:
		data = file.read()
	data = data.replace("<q>", ". ")
	text = data.split("\n")[:-1]

	return text

def merge_gold_candidate(gold, candidate):
	story_string = ""
	if len(gold) != len(candidate):
		print("The number of documents in gold and candidate are different.")
	else:
		for i in tqdm(range(len(gold))):
			story_string += candidate[i]+"\n"+gold[i]+"\n"

	return story_string

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--folder_path", help="the folder where the gold and candidate locate.")
	parser.add_argument("--gold_name", help="the name of the gold file.")
	parser.add_argument("--candidate_name", help="the name of the candidate file.")
	args = parser.parse_args()

	gold = read_text(args.folder_path+args.gold_name)
	candidate = read_text(args.folder_path+args.candidate_name)

	story_string = merge_gold_candidate(gold, candidate)

	# save story_string to a .story file
	with open(args.folder_path+args.gold_name[:-4]+"story", "w+") as file:
		file.write(story_string)
