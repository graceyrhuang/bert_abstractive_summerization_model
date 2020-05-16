"""
USAGE:
python merge_files.py \
    --train_data_dir /Users/gracehuang/Desktop/NLU/project/data/raw_data/ \
    --output_dir /Users/gracehuang/Desktop/NLU/project/data/gpt/
"""
import io
import os
import argparse

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_dir", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    file_start_token = '<|startoftext|>'
    file_end_token = '<|endoftext|>'

    output_file = os.path.join(args.output_dir, 'gpt_2.txt')
    f_out = open(output_file, "w")


    for filename in os.listdir(args.train_data_dir):
        input_ = ""
        f_in = io.open(os.path.join(args.train_data_dir, filename), mode="r", encoding="utf-8")
        input_ = f_in.read()
        input.split('hightlight')

        f_out.write(file_start_token)
        f_out.write(input_)
        f_out.write(file_end_token)





if __name__ == '__main__':
    main()
