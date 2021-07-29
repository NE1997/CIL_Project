# Tweet Sentiment Classification
Project for Computational Intelligence Lab 2021 Spring @ ETH Zurich [[Report Link]()] <br/>

## Authors
Ying Jiao, Yuyan Zhao, , 

## Project Description



## 1. Data shuffle and split
- Requirements: csv, random
- Put train_pos_full.txt and train_neg_full.txt under ```./twitter-datasets```
- Run
```
python dataset.py
```

## 2. Baseline ML Models

## 3. LSTM Models

### 3.1 stacked BLSTM
- Requirements: Python 3.7.4, TensorFlow 2.2.0, matplotlib, pandas, numpy, sklearn
- Option 1: run python file
```
python3 stacked_blstm.py
```
- Option 2: run on jupyter notebook
```
jupyter notebook stacked_blstm.ipynb
```

### 3.2 LSTM-AT & LSTM-SAT
- Requirements: Python 3.7.4, TensorFlow 2.0.0, pickle, gensim, numpy, pandas, re, time, nltk
- Module load on Leonhard
```
module load gcc/6.3.0 python_gpu/3.7.4
```
- Download tweet opinion words from [Link](https://www.kaggle.com/nltkdata/opinion-lexicon), put positive-words.txt and negative-words.txt under ```./opinion_lexicon```
- Generate sentiment lexicon
```
python LSTM-SAT/sentiment_score.py
```
- Data preprocessing, embedding; sentiment vector generation; data and sentiment vector batching
```
bsub -n 4 -W 24:00 -R "rusage[mem=40960, ngpus_excl_p=1]" python LSTM-SAT/utils.py
```
- Train LSTM-AT
```
bsub -n 4 -W 24:00 -R "rusage[mem=81920, ngpus_excl_p=1]" python LSTM-SAT/train_lstm_at.py
```
- Train LSTM-SAT
```
bsub -n 4 -W 24:00 -R "rusage[mem=81920, ngpus_excl_p=1]" python LSTM-SAT/train_lstm_sat.py
```
- Generate submission file
```
python LSTM-SAT/pred_to_sub.py
```

## 4. BERT Models

### 4.1 BERT-NN
- Requirements: torch, transformers, sklearn, tqdm
- Module load on Leonhard
```
module load python_gpu/3.7.1
```
- Fine-tune BERT-NN
```
bsub -n 4 -W 24:00 -R "rusage[mem=81920, ngpus_excl_p=1]" python BERT/bert_nn.py
```

### 4.2 BertForSequenceClassification
- Requirements: torch, transformers, sklearn, tqdm
- Module load on Leonhard
```
module load python_gpu/3.7.4
```
- Fine-tune BertForSequenceClassification
```
bsub -n 4 -W 24:00 -R "rusage[mem=81920, ngpus_excl_p=1]" python BertForSequenceClassification/BertForSequenceClassification.py
```

## Acknowledgement
We would like to thank the following Github repos and tutorials: <br/>
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [Huggingface](https://huggingface.co/)
- [bert fine-tuning 1](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
- [bert fine-tuning 2](https://skimai.com/fine-tuning-bert-for-sentiment-analysis/)
- [Text classification with an RNN] (https://www.tensorflow.org/text/tutorials/text_classification_rnn)
