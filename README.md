# Tweet Sentiment Classification
Project for Computational Intelligence Lab 2021 Spring @ ETH Zurich [[Report Link]](./document/CIL2021-SNYY.pdf) <br/>

## Authors
Ying Jiao, Yuyan Zhao, Shuting Li, Nils Ebeling
## Project Description
Neural network ensemble for sentiment classification (positive/1 or negative/-1) of tweets. Individual models are based on BERT and bidirectional RNNs using LSTM cells, augmented with sentiment information.


## 1. Data shuffle and split
- Requirements: csv, random
- Put train_pos.txt, train_neg.txt, train_pos_full.txt and train_neg_full.txt under ```./twitter-datasets```
- Run
```
python dataset.py
```

## 2. Baseline ML Models
- Requirements: Python 3.8.5, numpy, gensim, sklearn

### 2.1 File structure
```
.
├── GloVe_random_forest.py
├── GloVe_svm.py
├── twitter-datasets
|   ├── test_data.txt
|   ├── train_neg.txt
|   ├── train_neg_full.txt
|   ├── train_pos.txt
|   ├── train_pos_full.txt
|   └── glove
|       ├── glove.twitter.27B.25d.txt
|       ├── glove.twitter.27B.50d.txt
|       ├── glove.twitter.27B.100d.txt
|       └── glove.twitter.27B.200d.txt
```

### 2.2 Download GloVe pre-trained vectors for twitter
- Please download 25d, 50d, 100d, 200d vectors using the following link: https://nlp.stanford.edu/data/glove.twitter.27B.zip; source website: https://nlp.stanford.edu/projects/glove/.
- After downloading, unzip the file and put all .txt files in the a folder named "glove". Place the "glove" folder according to the file structure.
- If it is not the first time you have run the code for the corresponding dimension (i.e., you have obtained "glove.twitter.27B.--d.word2vec" where -- is the dimensionality), you can comment line 42 in both python files to speed up the process.

### 2.3 How to run
- Choose the GloVe dimension in line 47 (acceptable values: 25, 50, 100, 200; default: 200)
- Choose the data size in line 68 (acceptable values: "partial", "full"; default: "partial")
- For svm only: choose the kernel function by commenting line 107 and 108 to utilize the linear kernel, or commenting line 110 and 111 to utilize the rbf kernel
- Run python file according to the classifier you want to choose:
```
python GloVe_random_forest.py
```
or
```
python GloVe_svm.py
```
- Note: These codes include their own data shuffling and splitting, depending on whether partial data or full data is used for training. Make sure to organize files in an identical manner as shown in the file structure in order for programs to run properly!

## 3. LSTM Models

### 3.1 stacked BLSTM
- Requirements: Python 3.7.4, TensorFlow 2.2.0, matplotlib, pandas, numpy, sklearn
- Option 1: run python file
```
python stacked_BLSTM/stacked_blstm.py
```
- Option 2: run on jupyter notebook
```
jupyter notebook stacked_BLSTM/stacked_blstm.ipynb
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
python LSTM-SAT/setiment_score.py
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

### 4.2 BERT-Seq
- Requirements: torch, transformers, sklearn, tqdm
- Module load on Leonhard
```
module load python_gpu/3.7.4
```
- Fine-tune BERT-Seq
```
bsub -n 4 -W 24:00 -R "rusage[mem=81920, ngpus_excl_p=1]" python BERT/bert_seq.py
```

## Acknowledgement
We would like to thank the following Github repos and tutorials: <br/>
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [Huggingface](https://huggingface.co/)
- [bert fine-tuning 1](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
- [bert fine-tuning 2](https://skimai.com/fine-tuning-bert-for-sentiment-analysis/)
- [Text classification with an RNN](https://www.tensorflow.org/text/tutorials/text_classification_rnn)
- [Getting started with NLP: Word Embeddings, GloVe and Text classification](https://edumunozsala.github.io/BlogEms/jupyter/nlp/classification/embeddings/python/2020/08/15/Intro_NLP_WordEmbeddings_Classification.html)
