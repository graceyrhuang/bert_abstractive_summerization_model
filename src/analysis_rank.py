import torch
import numpy as np
import matplotlib.pyplot as plt
import os

path = '/Users/gracehuang/github/PreSumm/bert_data'
ul_file = '/Users/gracehuang/github/PreSumm/logs/new_test_step4000.sentence'
select = []

with open(ul_file, 'r') as f:
    ind = f.read()


for filename in os.listdir(path):
    if filename.endswith('.pt'):
        filename = os.path.join(path, filename)
        if filename.split('.')[1] == 'test':
            test_dataset = torch.load(filename)
            print(filename)
            for i in range(len(test_dataset)):
                class_list = test_dataset[i]['src_sent_labels']
                select_id = [j for j, e in enumerate(class_list) if e == 1]
                for item in select_id:
                    select.append(item)

oracle_result = []
ul_result = []
# max_len = max(select)
for i in range(15):
    print('origin: sentence {}, select times {}'.format(i+1, select.count(i)))
    print('ul: sentence {}, select times {}'.format(i+1, ind.count(str(i))))
    oracle_result.append(select.count(i)/len(select))
    ul_result.append(ind.count(str(i))/(11489*3))

print(oracle_result)
print(ul_result)

print('total senstence:', len(select))

x = np.arange(len(oracle_result))
total_width, n = 0.8, 4
width = total_width / n
x = x - (total_width - width) / 2

# plt.bar(range(1, len(oracle_result)+1), oracle_result, label='oracle')
# plt.bar(range(1, len(oracle_result)+1), ul_result, label='Bert')
plt.bar(x, oracle_result, width=width,label='oracle')
plt.bar(x + width, bert_sum, width=width, label='bert_sum')
plt.bar(x + 2 * width, ul_result, width=width,label='bert_sum_tr')
plt.bar(x + 3 * width, baseline_result, width=width, label='baseline')
plt.ylabel('Proportion of selected sentences')
plt.xlabel('Sentence position (in source document)')
plt.legend()
plt.show()