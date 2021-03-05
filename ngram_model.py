from pathlib import Path
import util
import data
import random
import itertools
import numpy as np

def iter_ngrams(text, n=3):
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

vocab_size = len(fields['text'].vocab.itos)
n_comms = len(fields['community'].vocab.itos)
comm_unk_idx = fields['community'].vocab.stoi['<unk>']
text_pad_idx = fields['text'].vocab.stoi['<pad>']
log.info(f"Loaded {len(dataset)} examples.")

random.seed(42)
random_state = random.getstate()
train_data, val_data, test_data = dataset.split(split_ratio=[0.8,0.1,0.1], stratified=True, strata_field='community', random_state=random_state)
log.info(f"Splits: train: {len(train_data)} val: {len(val_data)} test: {len(test_data)} ")

# bigram counts
C = np.zeros((n_comms,vocab_size,vocab_size),int)

for example in train_data:
    text = fields['text'].process([example.text]).numpy().squeeze()
    comm = fields['community'].vocab.stoi[example.community]
    for bigram in iter_ngrams(text, 2):
        C[comm][bigram[0]][bigram[1]] += 1 

# P(c,x,y) = C(c,x,y) / Sum_j(C(c,x,j))
# Laplace smoothing: Add alpha to every 
# P'(c,x,y) = (P(c,x,y) + alpha) / (1 + |V| * alpha)
# Heurisitc: alpha should be smaller than smallest P(c,x,y) (certanily very much smaller than 1/|V|, possibly 1/|V|**2)
# Hyperparameter search for alpha using the test set

alpha = 1/vocab_size
P = C / C.sum(axis=2).reshape(n_comms, vocab_size, 1)
P = np.nan_to_num(P, 0) # zeros where the ngram prefix has 0 prob
P_smooth_unnorm = P + alpha 
P_smooth = P_smooth_unnorm / P_smooth_unnorm.sum(axis=2).reshape(n_comms, vocab_size,1)


