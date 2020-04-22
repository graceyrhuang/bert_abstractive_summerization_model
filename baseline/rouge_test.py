import os
import numpy as np
import json
from rouge_score import rouge_scorer

filename = 'xsum' + '.txt'

with open(filename) as json_file:
    data = json.load(json_file)

rouge1_p = 0
rouge1_r = 0
rouge1_f = 0
rougelsum_p = 0
rougelsum_r = 0
rougelsum_f = 0
rougeL_p = 0
rougeL_r = 0
rougeL_f = 0

count = 0
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
for i in range(100):
        body = data[str(i)]['body']
        summary = data[str(i)]['summary']
        print(i)
        scores = scorer.score(body, summary)
        print(scores)

        rouge1_p += scores['rouge1'].precision
        rouge1_r += scores['rouge1'].recall
        rouge1_f += scores['rouge1'].fmeasure

        rougelsum_p += scores['rougeLsum'].precision
        rougelsum_r += scores['rougeLsum'].recall
        rougelsum_f += scores['rougeLsum'].fmeasure

        rougeL_p += scores['rougeL'].precision
        rougeL_r += scores['rougeL'].recall
        rougeL_f += scores['rougeLsum'].fmeasure
        count += 1

N = count
print('rouge1 precision:', rouge1_p/N)
print('rouge1 recall:', rouge1_r/N)
print('rouge1 fmeasure:', rouge1_f/N)
print('rougelsum precision:', rougelsum_p/N)
print('rougelsum recall:', rougelsum_r/N)
print('rougelsum fmeasure:', rougelsum_f/N)
print('rougeL precision:', rougeL_p/N)
print('rougeL recall:', rougeL_r/N)
print('rougeL fmeasure:', rougeL_f/N)


