train_data_pos = []
train_data_neg = []

with open("twitter-datasets/train_pos_full.txt", "r") as pos_f:
    pos_lines = pos_f.readlines()

for line in pos_lines:
    train_data_pos.append(line.strip("\n"))

with open("twitter-datasets/train_neg_full.txt", "r") as neg_f:
    neg_lines = neg_f.readlines()
    
for line in neg_lines:
    train_data_neg.append(line.strip("\n"))


pos_words = []
neg_words = []

with open("opinion_lexicon/positive-words.txt", "r") as pos_f:
    pos_lines = pos_f.readlines()


for line in pos_lines:
  if line[0] != ";" and line != "NULL":
    line = line.strip("\n")
    if len(line) !=0:
      pos_words.append(line.strip("\n"))


with open("opinion_lexicon/negative-words.txt", "r") as neg_f:
    neg_lines = neg_f.readlines()


for line in neg_lines:
  if line[0] != ";" and line != "NULL":
    line = line.strip("\n")
    if len(line) !=0:
      neg_words.append(line.strip("\n"))


pos_sentiment = {}

for pos_sen in train_data_pos:
  for pos_word in pos_words:
    if pos_word in pos_sen:
      if pos_word in pos_sentiment.keys():
        pos_sentiment[pos_word] += 1
      else:
        pos_sentiment[pos_word] = 1

max_fre = max(pos_sentiment.values())
pos_sentiment = {k: v / max_fre for k, v in pos_sentiment.items()}


neg_sentiment = {}

for neg_sen in train_data_neg:
  for neg_word in neg_words:
    if neg_word in neg_sen:
      if neg_word in neg_sentiment.keys():
        neg_sentiment[neg_word] += 1
      else:
        neg_sentiment[neg_word] = 1

max_fre = max(neg_sentiment.values())
neg_sentiment = {k: -v / max_fre for k, v in neg_sentiment.items()}


with open('sentiment_words.txt', 'w') as of:
  for k, v in pos_sentiment.items():
    line = k+"\t"+str(v)+"\n"
    of.write(line)

  for k, v in neg_sentiment.items():
    line = k+"\t"+str(v)+"\n"
    of.write(line)

of.close()


