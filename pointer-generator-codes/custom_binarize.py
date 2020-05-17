'''
# refer to https://blog.csdn.net/hfutdog/article/details/78447860
python custom_binarize.py --story_path=./json_dir/story_dir/alldata.story --finished_path=./json_dir/finished --bin_data_name=alldata.bin
python custom_binarize.py --story_path=./bert_dir/bert_linear_step33000.story --finished_path=./bert_dir/finished --bin_data_name=bert_linear_step33000.bin
'''

import os
import struct
import collections
from tensorflow.core.example import example_pb2
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--story_path", help="file path to save the .story file.")
parser.add_argument("--finished_path", help="file path to save the .story file.")
parser.add_argument("--bin_data_name", help="name of the bin data")
args = parser.parse_args() 

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
 
# train_file = './train/train.txt'
# val_file = './val/val.txt'

try:
    os.mkdir(args.finished_path)
except:
    pass

data_file = args.story_path
finished_files_dir = args.finished_path
chunks_dir = os.path.join(finished_files_dir, "chunked")
 
VOCAB_SIZE = 200000
CHUNK_SIZE = 500  # number of sample for a partition
 
 
def chunk_file(set_name):
    in_file = os.path.join(finished_files_dir, '%s.bin' % set_name)
    print(in_file)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk))  # 新的分块
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1
 
 
def chunk_all():
    # create chunked directory
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # partition the data
    for set_name in [args.bin_data_name[:-4]]:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunks_dir)
 
 
def read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines
 
 
def write_to_bin(input_file, out_file, makevocab=False):
    if makevocab:
        vocab_counter = collections.Counter()
 
    with open(out_file, 'wb') as writer:
        lines = read_text_file(input_file)
        print("Writing the story file to a .bin file.")
        for i, new_line in tqdm(enumerate(lines)):
            if i % 2 == 0: # articles locate at the even number of lines.
                article = lines[i]
            if i % 2 != 0:
                abstract = "%s %s %s" % (SENTENCE_START, lines[i], SENTENCE_END)

                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([bytes(article, encoding='utf-8')])
                tf_example.features.feature['abstract'].bytes_list.value.extend([bytes(abstract, encoding='utf-8')])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))
 
                # if makevocab==true, write vocabulary to a document
                if makevocab:
                    art_tokens = article.split(' ')
                    abs_tokens = abstract.split(' ')
                    abs_tokens = [t for t in abs_tokens if
                                  t not in [SENTENCE_START, SENTENCE_END]]   
                    tokens = art_tokens + abs_tokens
                    tokens = [t.strip() for t in tokens] 
                    tokens = [t for t in tokens if t != ""] 
                    vocab_counter.update(tokens)
 
    print("Finished writing file %s\n" % out_file)
 
    # if makevocab==true, write vocabulary to a document
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w', encoding='utf-8') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")
 
 
if __name__ == '__main__':
 
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)
 
    # 
    write_to_bin(data_file, os.path.join(finished_files_dir, args.bin_data_name))
    #write_to_bin(val_file, os.path.join(finished_files_dir, "val.bin"))
    #write_to_bin(train_file, os.path.join(finished_files_dir, "train.bin"), makevocab=True)
    chunk_all()
