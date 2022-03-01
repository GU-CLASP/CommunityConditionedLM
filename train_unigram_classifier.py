import data
from collections import Counter, defaultdict
import numpy as np
import csv
import pickle
import os
from pathlib import Path
import torch
import torchtext as tt
from classifier_model import NaiveBayesUnigram 
from IPython import embed

data_dir = Path('data/reddit_splits')
model_family_dir = Path('model/reddit')
save_dir = model_family_dir/'unigram-cond'
max_seq_len = 64
batch_size = 1024
gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')


unigram_counts_file = save_dir/'unigram_counts.pickle'

fields = data.load_fields(model_family_dir, lower_case=False, use_eosbos=False)
fields['text'].include_lengths = True

comms = fields['community'].vocab.itos
vocab = fields['text'].vocab.itos
vocab_size = len(vocab)

if not os.path.exists(unigram_counts_file):
    print("Counting unigrams per community...")
    unigram_counts = defaultdict(Counter)
    comm_N = Counter()
    for i, c in enumerate(comms):
        with open(f'data/unigram_counts/{c}.count.txt') as f: # generate from test_data
            print(f"{i+1}/{len(comms)}", end='\r')
            for line in f.readlines():
                line = line.lstrip().split()
                if len(line) == 1: # handle a few weird whitespace characters
                    w = '<unk>'
                    count = line[0]
                else:
                    count, w = line
                count = int(count)
                comm_N[c] += count
                if w in vocab:
                    unigram_counts[c][w] = count
                else:
                    unigram_counts[c]['<unk>'] += count
    with open(unigram_counts_file, 'wb') as f:
        pickle.dump(unigram_counts, f)
else:
    print("Loading unigram counts...")
    unigram_counts = defaultdict(Counter)
    with open(unigram_counts_file, 'rb') as f:
        unigram_counts = pickle.load(f)
        comm_N = {comm: sum(unigram_counts[comm].values()) for comm in comms }

indices = ([],[])
values = []
for comm_i, comm in enumerate(comms):
    for w, count in unigram_counts[comm].items():
        w_i = fields['text'].vocab.stoi[w]
        indices[0].append(comm_i)
        indices[1].append(w_i)
        values.append(count)

unigram_counts_tensor = torch.sparse_coo_tensor(indices, values, size=[len(comms), vocab_size]).to_dense().to(device)
comm_N_tensor = unigram_counts_tensor.sum(axis=1)
unigram_freq_tensor = (unigram_counts_tensor.float().T / comm_N_tensor.float()).T

model = NaiveBayesUnigram(vocab_size, len(comms))
model.load_state_dict({
    'unigram_freq': unigram_freq_tensor,
    'comm_N': comm_N_tensor,
    'alpha': torch.tensor(0.01)
    })
model.to(device)


def search_param(f, alpha_min, alpha_max, alpha_mid, epsilon=0.01, from_left=True):
    best_loss = f(alpha_mid)
    assert best_loss < f(alpha_min) and best_loss < f(alpha_max)
    while alpha_max - alpha_min > epsilon:
        from_left = alpha_mid - alpha_min > alpha_max - alpha_mid
        print(f"testing {alpha_max:0.3f} to {alpha_min:0.3f} from_left: {from_left}")
        if from_left:
            alpha_candidate = (alpha_min + alpha_mid) / 2
            f_candidate = f(alpha_candidate)
            if f_candidate < best_loss:
                alpha_max = alpha_mid
                alpha_mid = alpha_candidate
                best_loss = f_candidate
            else:
                alpha_min = alpha_candidate
        else: # from right
            alpha_candidate = (alpha_max + alpha_mid) / 2
            f_candidate = f(alpha_candidate)
            if f_candidate < best_loss:
                alpha_min = alpha_mid
                alpha_mid = alpha_candidate
                best_loss = f_candidate
            else:
                alpha_max = alpha_candidate
        print(f"alpha={alpha_candidate:0.3f} f(alpha) = {f_candidate}")
    return alpha_mid, best_loss

print("Loading dev data...")

dev_data = data.load_data(data_dir, fields, 'dev', max_seq_len, file_limit=None, lower_case=False)
dev_iterator = tt.data.BucketIterator(
    dev_data,
    device=device,
    batch_size=batch_size,
    sort_key=lambda x: len(x.text),
    train=False)

f = lambda x: model.test_cross_entropy_laplace(dev_iterator, alpha=x)

alpha, mean_entropy = search_param(f, .001,2,0.1, epsilon=0.01)
print(alpha, mean_entropy)

model.update_alpha(alpha)
torch.save(model.state_dict(), save_dir/'model.bin')
