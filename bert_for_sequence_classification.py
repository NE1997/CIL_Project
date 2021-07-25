import os
import re
#from tqdm import tqdm
import numpy as np
import pandas as pd
import csv 
from sklearn.model_selection import train_test_split
import torch
#import transformers
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import multiprocessing as mp

import requests as r

import random
import time

import torch.nn.functional as F

#from contextlib import contextmanager;
#@contextmanager
#def timethis(s=None):
#    start = time.time(); s = f'{s}\t' or ''
#    try: yield
#    finally: print(f'{s}{time.time() - start} s')



def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """

    text = re.sub(r'http(\S)+', r'', text)
    text = re.sub(r'http ...', r'', text)
    text = re.sub(r'http', r'', text)

    text = re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+',r'', text)
    text = re.sub(r'@[\S]+',r'', text)

    text = ''.join([i if ord(i) < 128 else '' for i in text])
    text = re.sub(r'_[\S]?',r'', text)

    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;',r'<', text)
    text = re.sub(r'&gt;',r'>', text)

    text = re.sub(r'[ ]{2, }',r' ', text)
    text = re.sub(r'([\w\d]+)([^\w\d ]+)', r'\1 \2', text)
    text = re.sub(r'([^\w\d ]+)([\w\d]+)', r'\1 \2', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    #for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
     #   encoded_sent = tokenizer.encode_plus(
     #       text=text_preprocessing(sent),  # Preprocess sentence
     #       add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
     #       max_length=MAX_LEN,                  # Max length to truncate/pad
     #       pad_to_max_length=True,         # Pad sentence to max length
     #       #return_tensors='pt',           # Return PyTorch tensor
     #       return_attention_mask=True,     # Return attention mask
     #       truncation=True
     #   )

    encoded_sents = [tokenizer.encode_plus(text=text_preprocessing(sent),add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True,return_attention_mask=True,truncation=True) for sent in data]
        
        # Add the outputs to the lists
    #   input_ids.append(encoded_sent.get('input_ids'))
    #    attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = [encoded_sent.get('input_ids') for encoded_sent in encoded_sents]
    attention_masks = [encoded_sent.get('attention_mask') for encoded_sent in encoded_sents]

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


# Create the BertClassifier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        #model = BertForSequenceClassification.from_pretrained(
        #    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        #    num_labels = 2, # The number of output labels--2 for binary classification.
        #                    # You can increase this for multi-class tasks.   
        #    output_attentions = False, # Whether the model returns attentions weights.
        #    output_hidden_states = True, # Whether the model returns all hidden-states.
        #)
        self.bert = BertForSequenceClassification.from_pretrained('./bert_pretrained_model')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        #print("outputs[0]", np.shape(outputs[0]))
        #last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = outputs.logits#self.classifier(last_hidden_state_cls)

        return logits


def initialize_model(epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    #bert_classifier = BertClassifier(freeze_bert=False)
    #bert_classifier = BertForSequenceClassification.from_pretrained(
    #    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    #    num_labels = 2, # The number of output labels--2 for binary classification.
    #                    # You can increase this for multi-class tasks.   
    #    output_attentions = False, # Whether the model returns attentions weights.
    #    output_hidden_states = False, # Whether the model returns all hidden-states.
    #)
    bert_classifier = BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler



def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        #print('Epoch', 'Batch', 'Train Loss', 'Val Loss', 'Val Acc', 'Elapsed')
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            #print(np.shape(b_input_ids), np.shape(b_attn_mask))
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                #print(epoch_i + 1, step, batch_loss / batch_counts, "-", "-", time_elapsed)

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            #print(epoch_i + 1, "-", avg_train_loss, val_loss, val_accuracy, time_elapsed)
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs




MAX_LEN = 128


train_data = []
with open("train_pos_full.txt", "r") as pos_f:
    pos_lines = pos_f.readlines()
#pos_lines = pos_lines[:50]
for line in pos_lines:
    train_data.append(line.strip("\n"))
with open("train_neg_full.txt", "r") as neg_f:
    neg_lines = neg_f.readlines()
#neg_lines = neg_lines[:50]
for line in neg_lines:
    train_data.append(line.strip("\n"))

labels = ["1"] * len(pos_lines) + ["0"] * len(neg_lines)
header = ['id','tweet', 'label']
with open("train_full_all.csv", "w", encoding='UTF8') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    # write the data
    for i in range(len(train_data)):
      data = [i, train_data[i], labels[i]]
      writer.writerow(data)
f.close()
test_data = []
with open("test_data.txt", "r") as test_f:
    test_lines = test_f.readlines()
for line in test_lines:
    test_data.append(line.strip("\n"))
test_header = ['id','tweet']
with open("test_data.csv", "w", encoding='UTF8') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(test_header)
    # write the data
    for i in range(len(test_data)):
      data = [i, test_data[i]]
      writer.writerow(data)
f.close()


data = pd.DataFrame()        
temp_df = pd.read_csv("train_full_all.csv", encoding='utf-8', memory_map=True).reset_index(drop=True)
data = pd.concat([data, temp_df])


X = data.tweet.values
y = data.label.values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2020)


test_data = pd.read_csv("test_data.csv")


if torch.cuda.is_available():       
    device = torch.device("cuda")
    print('There are ', torch.cuda.device_count(), ' GPU(s) available.')
    print('Device name: ', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# Load the BERT 
#r.get('https://huggingface.co/bert-base-uncased/resolve/main/config.json', verify=False)
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenizer = BertTokenizer.from_pretrained('bert_pretrained')


# Run function `preprocessing_for_bert` on the train set and the validation set
print('######## Tokenizing data... ########')

#train_inputs, train_masks = preprocessing_for_bert(X_train)
#val_inputs, val_masks = preprocessing_for_bert(X_val)

#pool = mp.Pool(mp.cpu_count())
#train_inputs, train_masks = pool.map(preprocessing_for_bert, X_train)
train_inputs, train_masks = preprocessing_for_bert(X_train)

#pool = mp.Pool(mp.cpu_count())
#val_inputs, val_masks = pool.map(preprocessing_for_bert, X_val)
val_inputs, val_masks = preprocessing_for_bert(X_val)

# Convert other data types to torch.Tensor
train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
batch_size = 64

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


# Specify loss function
loss_fn = nn.CrossEntropyLoss()

set_seed(42)    # Set seed for reproducibility


bert_classifier, optimizer, scheduler = initialize_model(epochs=3)
print('######## Training train... ########')

#pool = mp.Pool(mp.cpu_count())
#pool.map(train, bert_classifier, train_dataloader, 5)
train(bert_classifier, train_dataloader, epochs=3)


# Concatenate the train set and the validation set
full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])
full_train_sampler = RandomSampler(full_train_data)
full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size=32)

# Train the Bert Classifier on the entire training data
set_seed(42)
bert_classifier, optimizer, scheduler = initialize_model(epochs=3)
print('######## Training full... ########')

#pool = mp.Pool(mp.cpu_count())
#pool.map(train, bert_classifier, full_train_dataloader, 5)
train(bert_classifier, full_train_dataloader, epochs=3)

# Run `preprocessing_for_bert` on the test set
print('######## Tokenizing data... ########')

#pool = mp.Pool(mp.cpu_count())
#test_inputs, test_masks = pool.map(preprocessing_for_bert, test_data.tweet)
test_inputs, test_masks = preprocessing_for_bert(test_data.tweet)

# Create the DataLoader for our test set
test_dataset = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

# Compute predicted probabilities on the test set
print('######## Predicting... ########')

#pool = mp.Pool(mp.cpu_count())
#probs = pool.map(bert_predict, bert_classifier, test_dataloader)
probs = bert_predict(bert_classifier, test_dataloader)

# Get predictions from the probabilities
threshold = 0.5
preds = np.where(probs[:, 1] > threshold, 1, -1)

header = ['Id','Prediction']

with open("submission.csv", "w", encoding='UTF8') as of:
    writer = csv.writer(of)

    # write the header
    writer.writerow(header)

    # write the data
    for i in range(len(preds)):
      data = [i+1, preds[i]]
      writer.writerow(data)

of.close()