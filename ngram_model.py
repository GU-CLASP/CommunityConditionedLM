import os
import torch
import random
import data
from torchtext.data.utils import ngrams_iterator
from collections import Counter, defaultdict

data_dir = "data/reddit2015"
model_dir = "model/reddit2015"
max_seq_len = 64
file_limit = 50000

dataset, fields = data.load_data_and_fields(data_dir, model_dir,
        max_seq_len, file_limit)

vocab_size = len(fields['text'].vocab.itos)
comm_vocab_size = len(fields['community'].vocab.itos)
comm_unk_idx = fields['community'].vocab.stoi['<unk>']
text_pad_idx = fields['text'].vocab.stoi['<pad>']

random.seed(42)
random_state = random.getstate()
train_data, val_data, test_data = dataset.split(split_ratio=[0.8,0.1,0.1], stratified=True, strata_field='community', random_state=random_state)

def bigrams_iter(tokens):
    return zip(*[tokens[i:] for i in range(2)])

counts = defaultdict(lambda: Counter())
for i, example in enumerate(train_data):
    if (i % 1000) == 0:
        print(f"{i}/{len(train_data)}", end='\r')
    tokens = ['<pad>'] + example.text + ['<eos>'] # use bad for BoS
    tokens = [fields['text'].vocab.stoi[w] for w in tokens]
    for w1,w2 in bigrams_iter(tokens):
        counts[w1][w2] += 1

model = {}
for i, w1 in enumerate(counts):
    print(f"{i}/{vocab_size}", end='\r')
    denom = sum(counts[w1].values()) + vocab_size
    model[w1] = defaultdict(lambda x: 1/denom)
    for w2 in counts:
        model[w2] = (counts[w1][w2] + 1) / denom 

import pickle
with open('model/reddit2015/bigram_model.pickle', 'wb') as f:
    pickle.dump(model, f)
