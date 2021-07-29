# -*- coding:utf-8 -*-



import pickle

from gensim.corpora.dictionary import Dictionary
from gensim.models.word2vec import Word2Vec


import numpy as np
import pandas as pd
import re


np.random.seed(1337)

TRAIN_PATH = 'twitter-datasets/train_full_all_shuffled.csv'
TEST_PATH = 'twitter-datasets/test_data.csv'
SENTIMENT_WORDS = './sentiment_words.txt'
WORD2VEC_DIM = 200
WORD2VEC_MIN_FRE = 5
WORD2VEC_WINDOW = 5
W_IN_S = 128
LABEL_NUM = 2
TRAIN_VAL = 0.1
BATCH_SIZE = 128



def text_to_index_array(dic, sen):
    new_words = []
    for word in sen:
        try:
            new_word = dic[word]
        except:
            new_word = 0
        new_words.append(new_word)
    return new_words


def normalize_X(dict, docs):
    normalize_result = list()
    for doc in docs:
        temp = list()
        sen_array = text_to_index_array(dict, doc)
        if len(sen_array) == 0:
            continue
        if len(sen_array) >= W_IN_S:
            temp.append(sen_array[:W_IN_S])
        else:
            temp.append(sen_array + [0] * (W_IN_S - len(sen_array)))
        normalize_result.append(temp)
    return np.array(normalize_result)


def normalize_y(labels):
    result = []
    for label in labels:
        result_temp = np.zeros(LABEL_NUM)
        result_temp[int(label)] = 1
        result.append(result_temp)
    return np.array(result)


def getdocs_labels(path, havelabel):
    data = pd.DataFrame()        
    temp_df = pd.read_csv(path, encoding='utf-8').reset_index(drop=True)
    data = pd.concat([data, temp_df])

    if havelabel:

        X = list(data.tweet.values)
        y = list(data.label.values)
        #print(y)

        return X, y

    else:
        X = list(data.tweet.values)
        y = [1]*len(X)
        #print(y)

        return X, y


def text_preprocessing(text):

    text = re.sub(r'http(\S)+', r'', text)
    text = re.sub(r'http ...', r'', text)
    text = re.sub(r'http', r'', text)

    text = re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+',r'', text)
    text = re.sub(r'@[\S]+',r'', text)

    text = ''.join([i if ord(i) < 128 else '' for i in text])
    text = re.sub(r'_[\S]?',r'', text)

    text = re.sub(r'(@.*?)[\s]', ' ', text)

    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;',r'<', text)
    text = re.sub(r'&gt;',r'>', text)

    text = re.sub(r'[ ]{2, }',r' ', text)
    text = re.sub(r'([\w\d]+)([^\w\d ]+)', r'\1 \2', text)
    text = re.sub(r'([^\w\d ]+)([\w\d]+)', r'\1 \2', text)
    text = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def text_to_sen_list(sents):

    result_list = list()

    for sent in sents:
        
        all_words = sent.split()
        result_list.append(all_words)

    return result_list



def create_dictionaries(model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}
    w2vec = {word: model[word] for word in w2indx.keys()}
    return w2indx, w2vec


def create_sentiment_dict():
    file = open(SENTIMENT_WORDS, 'r', encoding='utf8')
    file_content = file.readlines()
    sentiment_dict = dict()
    for line in file_content:
        line_split = line.strip('\n').split('\t')
        label = float(line_split[1])
        word = line_split[0]
        sentiment_dict[word]=label
    file.close()
    return sentiment_dict

def get_sentiment_mask(dict, sens):
    senti_mask = list()
    for sen in sens:
        sen_senti = list()
        temp = list()
        for word in sen:
            try:
                senti = dict[word]
            except:
                senti = 0
            sen_senti.append(senti)
        if len(sen_senti) >= W_IN_S:
            temp.append(sen_senti[:W_IN_S])
        else:
            temp.append(sen_senti + [0]*(W_IN_S - len(sen_senti)))
        senti_mask.append(temp)
    return np.array(senti_mask)


def batch_iter(input_x, input_y, senti, batch_size):
    data_size = len(input_x)
    batch_no = int(data_size/batch_size) + 1
    input_x_pad = np.append(input_x, input_x[:(batch_no * batch_size - data_size)], axis=0)
    input_y_pad = np.append(input_y, input_y[:(batch_no * batch_size - data_size)], axis=0)
    senti_pad = np.append(senti, senti[:(batch_no * batch_size - data_size)], axis=0)
    for batch_index in range(batch_no):
        start_index = batch_index * batch_size
        end_index = (batch_index+1) * batch_size
        return_x = input_x_pad[start_index:end_index]
        return_y = input_y_pad[start_index:end_index]
        return_senti = senti_pad[start_index:end_index]

        yield(return_x, return_y, return_senti)



X, y = getdocs_labels(TRAIN_PATH, True)
X_test, y_test = getdocs_labels(TEST_PATH, False)

test_no = len(X_test)


X_all = X + X_test
#print(len(X))
#print(X_test)

for i in range(len(X_all)):
    X_all[i] = text_preprocessing(X_all[i])

sentences_list = text_to_sen_list(X_all)

model = Word2Vec(sentences_list,
                 size=WORD2VEC_DIM,
                 min_count=WORD2VEC_MIN_FRE,
                 window=WORD2VEC_WINDOW)


model.save('word2vec' + u'.model')
print("W2C done!")

index_dict, word_vectors = create_dictionaries(model)

new_dic = index_dict

n_symbols = len(index_dict) + 1
embedding_weights = np.zeros((n_symbols, WORD2VEC_DIM))

for w, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[w]

sentiment_dict = create_sentiment_dict()

X = X_all[:len(X)]
X_test = X_all[len(X):]


X_train = X[:int(len(X)*(1-TRAIN_VAL))]
y_train = y[:int(len(X)*(1-TRAIN_VAL))]


X_val = X[int(len(X)*(1-TRAIN_VAL)):]
y_val = y[int(len(X)*(1-TRAIN_VAL)):]


sentiment_mask_train = get_sentiment_mask(sentiment_dict, X_train)
sentiment_mask_val = get_sentiment_mask(sentiment_dict, X_val)
sentiment_mask_test = get_sentiment_mask(sentiment_dict, X_test)

X_train = normalize_X(new_dic, X_train)
X_val = normalize_X(new_dic, X_val)
X_test = normalize_X(new_dic, X_test)

y_train = normalize_y(y_train)
y_val = normalize_y(y_val)
y_test = normalize_y(y_test)

X_train = X_train.reshape(int(len(X)*(1-TRAIN_VAL)), W_IN_S)
X_val = X_val.reshape(int(len(X)*TRAIN_VAL), W_IN_S)
X_test = X_test.reshape(test_no, W_IN_S)
sentiment_mask_train = sentiment_mask_train.reshape(int(len(X)*(1-TRAIN_VAL)), W_IN_S)
sentiment_mask_val = sentiment_mask_val.reshape(int(len(X)*TRAIN_VAL), W_IN_S)
sentiment_mask_test = sentiment_mask_test.reshape(test_no, W_IN_S)

X_train_batch = []
X_val_batch = []
X_test_batch = []
y_train_batch = []
y_val_batch = []
y_test_batch = []
senti_train_batch = []
senti_val_batch = []
senti_test_batch = []


for i, (x, y, s) in enumerate(batch_iter(X_train, y_train, sentiment_mask_train, batch_size=BATCH_SIZE)):
    X_train_batch.append(x)
    y_train_batch.append(y)
    senti_train_batch.append(s)


for i, (x, y, s) in enumerate(batch_iter(X_val, y_val, sentiment_mask_val, batch_size=BATCH_SIZE)):
    X_val_batch.append(x)
    y_val_batch.append(y)
    senti_val_batch.append(s)


for i, (x, y, s) in enumerate(batch_iter(X_test, y_test, sentiment_mask_test, batch_size=BATCH_SIZE)):
    X_test_batch.append(x)
    y_test_batch.append(y)
    senti_test_batch.append(s)



data_file = open('./data_senti_batched_all.pkl', 'wb')
pickle.dump([X_train_batch, y_train_batch, X_val_batch, y_val_batch, X_test_batch, y_test_batch, senti_train_batch, senti_val_batch, senti_test_batch, n_symbols, embedding_weights], data_file)

if __name__ == "__main__":
    pass
