## Pointer-Generator Requirements and Code

### 1. Download the Pretrained Model

- Download the preprocessed CNN/Daily Mail data set from [here](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail).
- Find the pretrained model from [here](https://github.com/abisee/pointer-generator) or directly download by [pretrained-model.zip](https://drive.google.com/file/d/0B7pQmm-OfDv7ZUhHZm9ZWEZidDg/view).

### 2. Git Clone the Repository

- Git clone the [seungwonpark/pointer-generator](https://github.com/seungwonpark/pointer-generator) directly on PRINCE for the Python 3 version.

- Use the run_summarization.py to to fine-tune.

### 3. Preprocess the Output From the First Stage

- Step 1: Transform the output from Bert to a .story text file.

```
python bert_output_to_story.py --folder_path=/path/for/saving/file/ --gold_name=${OUTPUT.gold} --candidate_name=${OUTPUT.candidate}
```

- Step 2: Transform the .story file to the required binary format. 

```
python custom_binarize.py --story_path=${OUTPUT.story} --finished_path=/folder/path/for/saving/binary/files/ --bin_data_name=$NEWDATA.bin$
```

Note: the code for custom_binarize.py is referred to [here](refer to https://blog.csdn.net/hfutdog/article/details/78447860).

### 4. Fine-tune on the Decoder

For example,

```
python run_summarization.py --mode=decode --data_path=${DATAPATH} --vocab_path=${VOCABPATH} --log_root=${LOGROOT} --exp_name=pretrained_model --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --cov_loss_wt=1
```

### 5. Compute the ROUGE Score and Repetition Rate

```
python rouge_result.py --path_of_decoded_folder /path/to/pretrained_model/decode/ --path_for_saving /path/for/saving/the/result.txt
```

### Note: Settings for TensorFlow

- More instruction for installing tensorflow on the Prince system can be found [here](https://github.com/ppriyank/Prince-Set-UP#Installing-Tensorflow-on-Prince).

```
module purge
module load cudnn/9.0v7.0.5  
module load cuda/9.0.176
pip install tensorflow-gpu==1.7.0
```