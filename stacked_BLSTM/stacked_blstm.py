import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import pathlib
import itertools

from sklearn.model_selection import train_test_split
from nltk.tokenize.api import StringTokenizer

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
from plot_keras_history import plot_history

from tensorflow.math import confusion_matrix

tf.get_logger().setLevel('ERROR')

AUTOTUNE = tf.data.AUTOTUNE


#data preparation
neg = []
pos = []
neg_label = []
pos_label = []

data = []
label = []

with open("./train_neg_full.txt",'r') as file:
    neg = file.read().splitlines()
    neg_label = [0 for i in range(len(neg))]
    file.close()
            
with open("./train_pos_full.txt",'r') as file:
    pos = file.read().splitlines()
    pos_label = [1 for i in range(len(pos))]
    file.close()
    
data.append(neg)
data.append(pos)
data = np.ravel(data)
label.append(neg_label)
label.append(pos_label)
label = np.ravel(label)

with open("./vocab_cut.txt",'r') as file:
    cut_vocab = file.read().splitlines()

print("pos: ", np.shape(pos), np.shape(pos_label))
print("neg: ", np.shape(neg), np.shape(neg_label))
print("total: ", np.shape(data), np.shape(label))
print("cut_vocab words: ", np.shape(cut_vocab))


#train and validation data split
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42, shuffle=True)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

VOCAB_SIZE = len(cut_vocab)


#encoder
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary())

#define model
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(5e-4),
              metrics=['binary_accuracy'])

#fit model
history = model.fit(train_dataset, epochs=2,
                    validation_data=test_dataset,
                    validation_steps=30)

#evaluate model
test_loss, test_err = model.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Mean Squared Error:', test_err)

#plot loss and accuracy
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'binary_accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)

#predict for test data
y_pred = model.predict(test_dataset)

with open("./test_data_cleaned.txt",'r') as file:
    test_data = file.read().splitlines()
    file.close()

y_final = model.predict(test_data)

y_final[y_final<0.5]=-1
y_final[y_final>=0.5]=1
y_final = np.ravel(y_final.astype('int64'))

#save solution as csv
y_solution = zip(range(1, len(y_final)+1), y_final)
sol = pd.DataFrame(y_solution, columns= ['Id', 'Prediction'])
sol.to_csv('stacked_blstm_submission.csv', index=False)