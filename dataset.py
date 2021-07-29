import csv 
import random

train_data = []

with open("twitter-datasets/train_pos_full.txt", "r") as pos_f:
    pos_lines = pos_f.readlines()

for line in pos_lines:
    train_data.append((line.strip("\n"), "1"))

with open("twitter-datasets/train_neg_full.txt", "r") as neg_f:
    neg_lines = neg_f.readlines()
    
for line in neg_lines:
    train_data.append((line.strip("\n"), "0"))

random.Random(1337).shuffle(train_data)


header = ['id','tweet', 'label']

with open("twitter-datasets/train_full_all_shuffled.csv", "w", encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    for i in range(len(train_data)):
      data = [i, train_data[i][0], train_data[i][1]]
      writer.writerow(data)

f.close()