import time
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

"""
Function "vectorizer()" adapted from:
*    Title: Getting started with NLP: Word Embeddings, GloVe and Text classification
*    Author: Eduardo Muñoz Sala
*    Section: Applying the word embedding to a text classification task
*    Date: Aug 15, 2020
*    Availability: https://edumunozsala.github.io/BlogEms/jupyter/nlp/classification/embeddings/python/2020/08/15/Intro_NLP_WordEmbeddings_Classification.html
*
"""
# Transforming tweets to vectors
def vectorizer(word2vec_model, tweets_data):
    #print("Transforming tweets to vectors...")

    colon = word2vec_model.get_vector(':')
    dimension = colon.shape[0]

    X = np.zeros((len(tweets_data), dimension))
    cnt = 0
    for tweet in tweets_data:
        #print("Round {0}".format(cnt+1))
        words = tweet.split()
        vectors = []
        for word in words:
            try:
                vector = word2vec_model.get_vector(word)
                vectors.append(vector)
            except KeyError:
                pass

        # Using average of all words in a tweet to represent it
        if len(vectors) != 0:
            vectors = np.array(vectors)
            X[cnt] = vectors.mean(axis=0)
        else:
            X[cnt] = np.zeros(dimension)

        cnt += 1

    return X

# Transfering a GloVe model to word2vec
glove_dimension = 200
glove_filename = "glove.twitter.27B.{0}d".format(glove_dimension)
glove_path = "twitter-datasets/glove/{0}.txt".format(glove_filename)
word2vec_path = "twitter-datasets/glove/{0}.word2vec".format(glove_filename)
glove2word2vec(glove_path, word2vec_path)   # Comment this line if you already have the .word2vec file for corresponding dimension
#print("Converting GloVe to word2vec: done.")

word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
#print("Loading word2vec model: done.")

# Loading data
tweets = []
labels = []

def load_tweets(filename, label):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tweets.append(line.rstrip())
            labels.append(label)

# Choose the data size
data_size = "partial"
if data_size == "partial":
    load_tweets("twitter-datasets/train_neg.txt", -1)
    load_tweets("twitter-datasets/train_pos.txt", 1)
elif data_size == "full":
    load_tweets("twitter-datasets/train_neg_full.txt", -1)
    load_tweets("twitter-datasets/train_pos_full.txt", 1)

# Converting to NumPy array to facilitate indexing
tweets = np.array(tweets)
labels = np.array(labels)
#print("Loading training data: done ({0} tweets loaded).".format(len(tweets)))

# Spliting data into training set and validation set
np.random.seed(243) # Reproducibility

shuffled_indices = np.random.permutation(len(tweets))
split_idx = int(0.9 * len(tweets))
train_indices = shuffled_indices[:split_idx]
val_indices = shuffled_indices[split_idx:]

X_train = vectorizer(word2vec_model, tweets[train_indices])
#np.savetxt("twitter-datasets/X_train_glove_{0}d_full_90.csv".format(glove_dimension), X_train, delimiter=",")
X_val = vectorizer(word2vec_model, tweets[val_indices])
#np.savetxt("twitter-datasets/X_val_glove_{0}d_full_10.csv".format(glove_dimension), X_val, delimiter=",")

Y_train = labels[train_indices]
#np.savetxt("twitter-datasets/Y_train_glove_{0}d_full_90.csv".format(glove_dimension), Y_train, delimiter=",")
Y_val = labels[val_indices]
#np.savetxt("twitter-datasets/Y_val_glove_{0}d_full_10.csv".format(glove_dimension), Y_val, delimiter=",")

######################################
###### Random forest classifier ######
######################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

n_estimator = 100
clf = RandomForestClassifier(n_estimators=n_estimator)
start = time.time()

#X_train = np.loadtxt("twitter-datasets/X_train_glove_{0}d_full_90.csv".format(glove_dimension), delimiter=",")
#Y_train = np.loadtxt("twitter-datasets/Y_train_glove_{0}d_full_90.csv".format(glove_dimension), delimiter=",")
#print("Training random forest classifier...")
clf.fit(X_train, Y_train)

#X_val = np.loadtxt("twitter-datasets/X_val_glove_{0}d_full_10.csv".format(glove_dimension), delimiter=",")
#Y_val = np.loadtxt("twitter-datasets/Y_val_glove_{0}d_full_10.csv".format(glove_dimension), delimiter=",")
print("train score:", cross_val_score(clf, X_train, Y_train, cv=5))
print("test score:", cross_val_score(clf, X_val, Y_val, cv=5))
end = time.time()
#print("Training runs in: {0}s.".format(end-start))

test = []
def load_test(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tweet = line.rstrip()
            char = ','
            pos = tweet.find(char)
            test.append(tweet[pos+1:])

load_test('twitter-datasets/test_data.txt')
test = np.array(test)
#print("Loading test data: done ({0} tweets loaded).".format(len(test)))
X_test = vectorizer(word2vec_model, test)
#np.savetxt("twitter-datasets/X_test_glove_{0}d.csv".format(glove_dimension), X_test, delimiter=",")
Y_test = clf.predict(X_test)

output_filename = "Y_test_{0}_glove_{1}d_90-10_nEst_{2}.csv".format(data_size, glove_dimension, n_estimator)
output_path = "twitter-datasets/" + output_filename
with open(output_path, 'w') as file:
    file.write("Id,Prediction\n")
    for i in range(len(Y_test)):
        file.write("{0},{1}\n".format(i+1, Y_test[i]))
#print("Dumping prediction to {0} file: done.".format(output_filename))
