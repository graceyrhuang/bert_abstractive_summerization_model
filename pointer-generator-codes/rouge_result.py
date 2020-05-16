'''
python rouge_result.py --path_of_decoded_folder ../pretrain_10_nocov/pretrained_model/decode/ --path_for_saving ../pretrain_10_nocov/rouge.txt --path_unique_name ../pretrain_10_nocov/unique_files.txt

python rouge_result.py --path_of_decoded_folder --path_for_saving --path_unique_name
'''

import json
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os
from rouge_score import rouge_scorer
import argparse
from tqdm import tqdm
from random import sample, seed
import rouge
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
#nltk.download('wordnet')

def get_article_and_summary(path_of_decoded_folder):
	
	file_list = os.listdir(path_of_decoded_folder)
	file_list = [f for f in file_list if f.endswith(".json")]

	dict_of_data = {"body": [], "summary": []}
	print("Reading and pre-processing all of the files.")
	for file in tqdm(file_list):
		with open(path_of_decoded_folder+file, "r") as f:
			data = json.load(f)
			dict_of_data["body"].append(data["abstract_str"])
			dict_of_data["summary"].append(detokenized(data["decoded_lst"]))
	
	return dict_of_data

def get_unique_json_file(path_of_decoded_folder):
	# check unique data by abstract_str
	# initialize stemmer and lemmatizer
	porter_stemmer = PorterStemmer()
	wordnet_lemmatizer = WordNetLemmatizer()

	unique_data_hash = {} # for keeping the unique json data
	dict_of_data = {"reference": [], "summary": [], "decoded_lst": []} # save the body and summary as above
	unique_file_name = set() # for saving the name of the file

	file_list = os.listdir(path_of_decoded_folder)
	file_list = [f for f in file_list if f.endswith(".json")]

	print("Reading and extracting the unique json data.")
	for file in tqdm(file_list):
		try:
			with open(path_of_decoded_folder+file, "r") as f:
				data = json.load(f)
				if unique_data_hash.get(data["abstract_str"]) is None:
					unique_data_hash[data["abstract_str"]] = 1
					unique_file_name.add(file)

					dict_of_data["reference"].append(data["abstract_str"])

					#detokenized_data = detokenized(["<q>" if token=="." else token for token in data["decoded_lst"]])
					detokenized_data = detokenized(data["decoded_lst"])
					dict_of_data["summary"].append(detokenized_data)

					preprocessed_decoded_lst = [wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token)) for token in data["decoded_lst"]]
					dict_of_data["decoded_lst"].append(preprocessed_decoded_lst)
		except ValueError:
				pass
		except Exception:
			pass
	return dict_of_data, unique_file_name

def detokenized(word_list):
	# transform a list of words to a string by detokenizing.
	TWD = TreebankWordDetokenizer() 
	# to transform a word list to a string
	return TWD.detokenize(word_list)

def sample_summary(dict_of_data, k=5000, set_seed=123):
	data_length = len(dict_of_data["summary"])

	seed(set_seed)
	sample_index = sample(range(data_length), k)

	dict_of_data["reference"] = [dict_of_data["reference"][index] for index in sample_index]
	dict_of_data["summary"] = [dict_of_data["summary"][index] for index in sample_index]
	dict_of_data["decoded_lst"] = [dict_of_data["decoded_lst"][index] for index in sample_index]
	
	return dict_of_data

def repetition_precentage(dict_of_data):
	decoded_2darray = dict_of_data["decoded_lst"]
	total_percentage = 0
	for file in decoded_2darray:
		total_percentage += 1-(len(set(file))/len(file))

	return total_percentage/len(decoded_2darray)

def prepare_results(metric, p, r, f):
	return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def compute_and_save_rouge(dict_of_data, path_for_saving):

	print("Computing the rouge score.")

	with open(path_for_saving, "w+") as write_file:
		for aggregator in ['Avg', 'Best']:
			write_file.write('Evaluation with {}\n'.format(aggregator))
			apply_avg = aggregator == 'Avg'
			apply_best = aggregator == 'Best'
			evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
									max_n=4,
									limit_length=True,
									length_limit=100,
									length_limit_type='words',
									apply_avg=apply_avg,
									apply_best=apply_best,
									alpha=0.5, # Default F1_score
									weight_factor=1.2,
									stemming=True)
			all_hypothesis = dict_of_data["reference"]
			all_references = dict_of_data["summary"]

			scores = evaluator.get_scores(all_hypothesis, all_references)
	
			for metric, results in sorted(scores.items(), key=lambda x: x[0]):
				if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
					for hypothesis_id, results_per_ref in enumerate(results):
						nb_references = len(results_per_ref['p'])
						for reference_id in range(nb_references):
							write_file.write("\tHypothesis #{} & Reference #{}: \n".format(hypothesis_id, reference_id))
							write_file.write("\t" + prepare_results(metric, results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id])+"\n")
					write_file.write("\n")
				else:
					write_file.write(prepare_results(metric, results['p'], results['r'], results['f'])+"\n")

		write_file.write("\n")
		# compute repetition
		avg_repetition = repetition_precentage(dict_of_data)
		write_file.write("repetition rate: " + str(avg_repetition))
		write_file.write("\n")

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--path_of_decoded_folder", help="the path of the decoded folder.")
	parser.add_argument("--path_for_saving", help="the path where we save the rouge score.")
	parser.add_argument("--path_unique_name", help="the path for saving the unique files' name.")
	args = parser.parse_args()

	# read and process the json files
	# dict_of_data = get_article_and_summary(args.path_of_decoded_folder)
	dict_of_data, file_names = get_unique_json_file(args.path_of_decoded_folder)
	
	# select 5000 samples
	dict_of_data_sample = sample_summary(dict_of_data, 3, 123)

	# compute and save the rouge score
	compute_and_save_rouge(dict_of_data_sample, args.path_for_saving)

	# write the unique file names
	with open(args.path_unique_name, "w+") as f:
		f.write("\n".join(file_names))



