import data
from collections import Counter, defaultdict
import numpy as np
import csv
import pickle
import os

def cross_entropy(p, q):
    sum_ = 0
    for word in p:
        if not word in q:
            return np.inf
        else:
            sum_ += p[word] * np.log(q[word])
    return -sum_

def cross_entropy_smoothed(p, q, vocab_size, alpha):
    sum_ = 0
    for word in p:
        if not word in q:
            sum_ += np.log(alpha) - np.log(vocab_size)
        else:
            sum_ += p[word] * np.log(alpha/vocab_size + (1-alpha) * q[word])
    return -sum_

def cross_entropy_message(m, q):
    sum_ = 0
    for word in m:
        if not word in q:
            return np.inf
        else:
            sum_ += np.log(q[word])
    return -sum_

def word_freq_in_message(m):
    message_counts = Counter(m)
    len_m = len(m)
    message_freq = {w: message_counts[w]/len_m for w in m}
    return message_freq

def cross_entropy_message_smoothed(m, q, vocab_size, alpha):
    sum_ = 0
    for word in m:
        if not word in q:
            sum_ += np.log(alpha) - np.log(vocab_size)
        else:
            sum_ += np.log(alpha/vocab_size + (1-alpha) * q[word])
    return -sum_

def test_cross_entropy_smoothed(word_freq, test_data, vocab_size, alpha):
    entropy = 0
    for i, (c, m) in enumerate(test_data):
        entropy += cross_entropy_message_smoothed(m, word_freq[c], vocab_size, alpha)
    return entropy

def cross_entropy_message_laplace(m, q, vocab_size, alpha, N):
    sum_ = 0
    for word in m:
        if not word in q:
           sum_ += np.log(alpha / (N + vocab_size * alpha))
        else:
           sum_ += np.log(q[word] * N / (N + vocab_size * alpha))
    return -sum_

def test_cross_entropy_laplace(word_freq, test_data, vocab, alpha, N):
    entropy = 0
    vocab_size = len(vocab)
    for i, (c, m) in enumerate(test_data):
        entropy += cross_entropy_message_laplace(m, word_freq[c], vocab_size, alpha, N[c])
    return entropy / len(test_data)


data_dir = 'data/reddit_splits'
model_family_dir = 'model/reddit'
max_seq_len = 64

unigram_counts_file = 'model/reddit/unigram-cond/unigram_counts.pickle.cheating'

fields = data.load_fields(model_family_dir)

comms = fields['community'].vocab.itos
vocab = fields['text'].vocab.itos
vocab_size = len(vocab)


dev_data = data.load_data(data_dir, fields, 'dev', max_seq_len, None)
dev_data = [(ex.community, [w if w in vocab else '<unk>' for w in ex.text]) for ex in dev_data]

print("Counting unigrams per community...")
unigram_counts = defaultdict(Counter)
comm_N = Counter()
for i, c in enumerate(comms):
    with open(f'data/unigram_counts/{c}.count.txt') as f:
        print(f"{i+1}/{len(comms)}", end='\r')
        for line in f.readlines():
            line = line.lstrip().split()
            if len(line) == 1:
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

with open('model/reddit/unigram-cond/unigram_counts.pickle', 'wb') as f:
    pickle.dump(unigram_counts, f)

# community_totals = {comm: sum(unigram_counts[comm].values()) for comm in comms + ['<comm_totals>']}
# unigram_freqs = {comm: {word: unigram_counts[comm][word] / community_totals[comm]
    # for word in unigram_counts[comm]} for comm in comms + ['<comm_totals>']}

community_totals = {comm: sum(unigram_counts[comm].values()) for comm in comms }
unigram_freqs = {comm: {word: unigram_counts[comm][word] / community_totals[comm]
    for word in unigram_counts[comm]} for comm in comms}


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

f = lambda x: test_cross_entropy_laplace(unigram_freqs, dev_data, vocab, x, comm_N)
alpha, mean_entropy = search_param(f, 0,3,1.5, epsilon=0.01)
# (0.0908203125, 182.08458470675262)

alpha, mean_entropy

cond_file = 'model/reddit/unigram-cond/nll.csv'
ucond_file = 'model/reddit/unigram/nll.csv'

test_data = data.load_data(data_dir, fields, 'test', max_seq_len, None)
test_data = [(ex.example_id, ex.community, [w if w in vocab else '<unk>' for w in ex.text]) for ex in test_data]

print("Computing message likelihoods...")
with open(cond_file, 'w') as f_cond: #, open(ucond_file, 'w') as f_ucond:
    meta_fields = ['community', 'example_id', 'length']
    writer_cond = csv.DictWriter(f_cond, fieldnames=meta_fields+comms)
    writer_cond.writeheader()
    # writer_ucond = csv.DictWriter(f_ucond, fieldnames=meta_fields+['nll'])
    # writer_ucond.writeheader()
    for i, c, m in test_data:
        if (i % 1000) == 0:
            print(f'{i}/{len(test_data)}', end='\r')
        message_len = len(m)
        # m_word_freq = word_freq_in_message(m)
        row = {'community': c, 'example_id': i, 'length': message_len}
        # row['nll'] = cross_entropy_message_smoothed(m, unigram_freqs['<comm_totals>'], vocab_size, alpha)
        # out = writer_ucond.writerow(row)
        # del row['nll']
        for c in comms:
            row[c] = cross_entropy_message_laplace(m, unigram_freqs[c], vocab_size, alpha, comm_N[c])
        out = writer_cond.writerow(row)

