import data
from collections import Counter, defaultdict
import numpy as np
import csv
import pickle
import os

data_dir = 'data/reddit_splits'
model_family_dir = 'model/reddit'
max_seq_len = 64

unigram_counts_file = 'model/reddit/unigram-cond/unigram_counts.pickle'

fields = data.load_fields(model_family_dir)

comms = fields['community'].vocab.itos
vocab = fields['text'].vocab.itos

test_data = data.load_data(data_dir, fields, 'test', max_seq_len, None)


if os.path.exists(unigram_counts_file):
    print("Loding unigram counts...")
    unigram_counts = pickle.load(open(unigram_counts_file, 'rb'))
else:
    print("Counting unigrams per community...")
    train_data = data.load_data(data_dir, fields, 'train', max_seq_len, None)
    train_messages = [ex.text for ex in test_data.examples + train_data.examples]
    train_comms = [ex.community for ex in test_data.examples + train_data.examples]
    n_train_messages = len(train_messages)
    unigram_counts = defaultdict(Counter)
    for i, (m, c) in enumerate(zip(train_messages, train_comms)):
        if (i % 1000) == 0:
            print(f'{i}/{n_train_messages}', end='\r')
        m = [w if w in vocab else '<unk>' for w in m]
        for w in m:
            unigram_counts[c][w] += 1
            unigram_counts['<comm_totals>'][w] += 1
    with open(unigram_counts_file, 'wb') as f:
        pickle.dump(unigram_counts, f)


community_totals = {comm: sum(unigram_counts[comm].values()) for comm in comms + ['<comm_totals>']}

unigram_freqs = {comm: {word: unigram_counts[comm][word] / community_totals[comm]
    for word in unigram_counts[comm]} for comm in comms + ['<comm_totals>']}

def cross_entropy(p, q):
    sum_ = 0
    for word in p:
        if not word in q:
            return -np.inf
        else:
            sum_ += p[word] * np.log(q[word])
    return -sum_

def word_freq_in_message(m):
    message_counts = Counter(m)
    len_m = len(m)
    message_freq = {w: message_counts[w]/len_m for w in m}
    return message_freq

cond_file = 'model/reddit/unigram-cond/nll.csv'
ucond_file = 'model/reddit/unigram/nll.csv'

print("Computing conditional and unconditional message likelihoods...")
with open(cond_file, 'w') as f_cond, open(ucond_file, 'w') as f_ucond:
    meta_fields = ['community', 'example_id', 'length']
    writer_cond = csv.DictWriter(f_cond, fieldnames=meta_fields+comms)
    writer_cond.writeheader()
    writer_ucond = csv.DictWriter(f_ucond, fieldnames=meta_fields+['nll'])
    writer_ucond.writeheader()
    for i, ex in enumerate(test_data):
        if (i % 1000) == 0:
            print(f'{i}/{len(test_data)}', end='\r')
        m = [w if w in vocab else '<unk>' for w in ex.text]
        message_len = len(m)
        m_word_freq = word_freq_in_message(m)
        row = {'community': ex.community, 'example_id': ex.example_id, 'length': message_len}
        row['nll'] = cross_entropy(m_word_freq, unigram_freqs['<comm_totals>']) 
        out = writer_ucond.writerow(row)
        del row['nll']
        for c in comms:
            row[c] = cross_entropy(m_word_freq, unigram_freqs[c]) 
        out = writer_cond.writerow(row)

