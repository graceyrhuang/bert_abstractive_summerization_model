"""
USAGE EXAMPLE:
python rouge_result.py \
    --temp_dir /Users/gracehuang/Desktop/NLU/project/results \
    --candidate new_test_step4000.candidate \
    --reference new_test_step4000.gold

"""

import os
import json
from rouge_score import rouge_scorer
import time
import argparse
import logging



def test_rouge(temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding='utf-8')]
    references = [line.strip() for line in open(ref, encoding='utf-8')]
    print("test number", len(candidates))
    # print(len(references))
    assert len(candidates) == len(references)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    results = { 'rouge1' : {'precision':0, 'recall':0, 'fmeasure':0},
    			'rouge2' : {'precision':0, 'recall':0, 'fmeasure':0},
    			'rougeL' : {'precision':0, 'recall':0, 'fmeasure':0}
    			}

    cnt = len(candidates)
    for i in range(cnt):
    	# print(candidates[i])
    	# print(references[i])
    	scores = scorer.score(candidates[i], references[i])
    	results['rouge1']['precision'] += scores['rouge1'].precision
    	results['rouge1']['recall'] += scores['rouge1'].recall
    	results['rouge1']['fmeasure'] += scores['rouge1'].fmeasure

    	results['rouge2']['precision'] += scores['rouge2'].precision
    	results['rouge2']['recall'] += scores['rouge2'].recall
    	results['rouge2']['fmeasure'] += scores['rouge2'].fmeasure
    	
    	results['rougeL']['precision'] += scores['rougeL'].precision
    	results['rougeL']['recall'] += scores['rougeL'].recall
    	results['rougeL']['fmeasure'] += scores['rougeL'].fmeasure
    
    for k,v in results.items():
    	for key, value in v.items():
    		results[k][key] = value/cnt

    return results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--temp_dir", default=".", type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--candidate", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--reference", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.") 
    parser.add_argument("--log_file", default='./log/rouge.log', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")  

    args = parser.parse_args()
    # log file
    logging.basicConfig(filename=args.log_file,level=logging.DEBUG)
    candidate = os.path.join(args.temp_dir, args.candidate)
    reference = os.path.join(args.temp_dir, args.reference)
    rouge_result = test_rouge(args.temp_dir, candidate, reference)
    # print("candidate:", args.candidate)
    # print("reference:", args.reference)
    # print(rouge_result)
    logging.info("candidate:" + args.candidate)
    logging.info("reference:" + args.reference)
    logging.info(rouge_result)



if __name__ == '__main__':
    main()



