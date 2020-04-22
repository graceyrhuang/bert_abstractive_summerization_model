# Bert Summarization

### Baseline

First install the reqirements,  the rouge-score package is a package implementation from google research, but sometimes it may not complie with pytorch

```bash
pip install -r reqirement.txt
```

To test model with dataset

```
python extractive_summery.py
python rouge_test.py
```



### Rouge-Result

|        model        | rouge-1 | rouge-2 | rouge-l |
| :-----------------: | :-----: | :-----: | :-----: |
| Extractive baseline |  39.4   |  18.02  |  36.07  |

