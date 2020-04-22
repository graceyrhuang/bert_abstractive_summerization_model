import torch
import os
from summarizer import Summarizer
import numpy as np
import json

# from rouge_score import rouge_scorer

def sumerize(body, model):
	result = model(body, min_length=30)
	sum_text = ''.join(result)
	return sum_text

story_path = '/Users/gracehuang/Desktop/NLU/project/bert_data_xsum_new'
story_file = []
for folder, subfolders, files in os.walk(story_path):
    for file in files:
        filePath = os.path.join(os.path.abspath(folder), file)
        story_file.append(filePath)
        # print(filePath)
        
model = Summarizer()
# scorer = rouge_scorer.RougeScorer(['rouge1', '"rougeLsum"', 'rougeL'], use_stemmer=True)
count = 0
json_data = {}
for i, passage in enumerate(story_file[:1]):
    data = torch.load(passage)
    for iter_ in range(len(data)):
        body = ''.join(data[iter_]['src_txt'])
        sum_text = sumerize(body, model)
        if (len(body) != len(sum_text)) and (len(sum_text) != 0):
            # output them into json file
            summerdict = {'body': body, 
                          'summary': sum_text}
            # print(summerdict)
            json_data[count] = summerdict
            count += 1


# print(json_data)
filename = 'xsum' + i + '.txt'
with open(filename, 'w') as json_file:
    json.dump(json_data, json_file)
#         # print('body length', len(body))
#         # print('sum length', len(sum_text))
#         print('{}:{}'.format(i, sum_text))
#         try:
#             score = scorer.score(body, sum_text)[0]
#             print('score = ', score)
#             print('passage {} with score {}'.format(i, score))
#             # print(type(score))