from pathlib import Path
import util
import data
import random
import itertools
import numpy as np

def iter_ngrams(text, n=2):
    ts = itertools.tee(text, n)
    for i, t in enumerate(ts[1:]):
        for _ in range(i+1):
            next(t)
    return zip(*ts)

# @click.argument('model_dir', type=click.Path(exists=True))
# @click.argument('model_name', type=str)
# @click.argument('data_dir', type=click.Path(exists=True))
# @click.option('--rebuild-vocab/--no-rebuild-vocab', default=False)
# @click.option('--vocab-size', default=40000)
# @click.option('--max-seq-len', default=64)
# @click.option('--file-limit', type=int, default=None,
        # help="Number of examples per file (community).")

model_dir = 'model/synth'
model_name = 'bigram'
data_dir = 'data/synth_data'
rebuild_vocab = False
vocab_size = 40000
max_seq_len = 64
file_limit = None

model_dir = Path(model_dir)
data_dir = Path(data_dir)
save_dir = model_dir/model_name

util.mkdir(model_dir)
util.mkdir(save_dir)

log = util.create_logger('ngram', save_dir/'ngram.log', True)

dataset, fields = data.load_data_and_fields(data_dir, model_dir,
        max_seq_len, file_limit, vocab_size, rebuild_vocab)

comm_unk_idx = fields['community'].vocab.stoi['<unk>']
text_pad_idx = fields['text'].vocab.stoi['<pad>']
log.info(f"Loaded {len(dataset)} examples.")

random.seed(42)
random_state = random.getstate()
train_data, val_data, test_data = dataset.split(split_ratio=[0.8,0.1,0.1], stratified=True, strata_field='community', random_state=random_state)
log.info(f"Splits: train: {len(train_data)} val: {len(val_data)} test: {len(test_data)} ")

def iter_examples(n, data):
    fields = data.fields
    bos_token = fields['text'].init_token
    bos_int = fields['text'].vocab[bos_token]
    for example in data:
        text = fields['text'].process([example.text]).numpy().squeeze()
        comm = fields['community'].vocab.stoi[example.community]
        if n > 2:
            ngram_padding = np.array((bos_int,) * (n - 2))
            text = np.concatenate((ngram_padding, text)) 
        yield comm, text
    
def count_ngrams(n, data):
    n_comms = len(data.fields['community'].vocab.itos)
    vocab_size = len(data.fields['text'].vocab.itos)
    C = np.zeros((n_comms, *((vocab_size,) * n)),int)
    for comm, text in iter_examples(n,data):
        for ngram in iter_ngrams(text, n):
            C[comm][ngram] += 1 
    return C

def build_lm(C, alpha):
    n_comms = C.shape[0]
    vocab_size = C.shape[1]
    n = len(C.shape) - 1
    divisor_shape = (n_comms, *((vocab_size,) * (n - 1)), 1)
    P = C / C.sum(axis=n).reshape(*divisor_shape)
    P = np.nan_to_num(P, 0) # zeros where the ngram prefix has 0 prob
    P_smooth_unnorm = P + alpha 
    P_smooth = P_smooth_unnorm / P_smooth_unnorm.sum(axis=n).reshape(*divisor_shape)
    return P_smooth

def example_nll(n, P, comm, text):
    p = []
    for y_hat, ngram in zip(text[n-1:], iter_ngrams(text, n-1)):
        p.append(P[comm][ngram][y_hat])
    p = np.array(p)
    return -np.log(p)

def test_lm(P, data):

n = 2
C = count_ngrams(n, train_data)

alpha = 100/(vocab_size)
P = build_lm(C, alpha)

results = []
for comm, text in iter_examples(n,test_data):
    nll = example_nll(n, P, comm,text)
    mean_nll = nll.sum() / len(nll)
    ppl = np.exp(mean_nll)
    results.append({'comm': comm, 'ppl': ppl})

import pandas as pd
df = pd.DataFrame(results)
df.groupby('comm').mean('ppl').mean()
