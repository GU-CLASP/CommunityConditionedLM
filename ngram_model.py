from pathlib import Path
import util
import data
import itertools
import numpy as np
import pandas as pd
import click
from collections import Counter, defaultdict
import pickle

def iter_ngrams(text, n=2):
    ts = itertools.tee(text, n)
    for i, t in enumerate(ts[1:]):
        for _ in range(i+1):
            next(t)
    return zip(*ts)

def iter_examples(n, data):
    fields = data.fields
    bos_token = fields['text'].init_token
    bos_int = fields['text'].vocab[bos_token]
    for example in data:
        text = fields['text'].process([example.text]).numpy().squeeze()
        cond = fields['community'].vocab.stoi[example.community]
        if n > 2:
            ngram_padding = np.array((bos_int,) * (n - 2))
            text = np.concatenate((ngram_padding, text)) 
        yield cond, text

class CondNGramLM():

    def __init__(self, N, n_conds, vocab_size, alpha, C_w=None, C=None):
        self.N = N
        self.n_conds = n_conds
        self.vocab_size = vocab_size
        self.C_w = C_w if C_w else {cond: defaultdict(Counter) for cond in range(n_conds)} 
        self.C = C if C else {cond: Counter() for cond in range(n_conds)}
        self.alpha = alpha

    def prob(self, cond, ngram_prefix, w, alpha=None):
        alpha = alpha if alpha else self.alpha
        return ((self.C_w[cond][ngram_prefix][w] + alpha) / 
               (self.C[cond][ngram_prefix] + self.vocab_size * alpha))

    def fit(self, data):
        for cond, text in iter_examples(self.N, data):
            for ngram in iter_ngrams(text, self.N):
                prefix, w = ngram[:-1], ngram[-1]
                self.C_w[cond][prefix][w] += 1
                self.C[cond][prefix] += 1
        return self

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open('filename', 'rb') as f:
            return pickle.load(f)

def example_nll(lm, cond, text, alpha=None):
    n = lm.N
    p = []
    for y_hat, ngram_prefix in zip(text[lm.N-1:], iter_ngrams(text, lm.N-1)):
        p.append(lm.prob(cond, ngram_prefix, y_hat, alpha=alpha))
    p = np.array(p)
    return -np.log(p)

def test_lm(lm, data, alpha=None):
    results = []
    N = lm.N 
    for cond, text in iter_examples(N, data):
        nll = example_nll(lm, cond, text, alpha)
        mean_nll = nll.sum() / len(nll)
        ppl = np.exp(mean_nll)
        results.append({'cond': cond, 'ppl': ppl})
    df = pd.DataFrame(results)
    return df

def grid_search(func, n_points, min_value, max_value, n_iters, log):
    """ Grid search of one parameter """
    def grid_search_iter(min_value, max_value):
        log.info(f"Testing {n_points} points between {min_value:0.4E} and {max_value:0.4E}")
        step = (max_value - min_value) / n_points
        params = np.arange(start=min_value, stop=max_value, step=step)
        results = np.array([func(param) for param in params])
        best = np.argmin(results)
        best_value = params[best]
        log.info(f"Lowest loss {results[best]:0.6f} with parameter {best_value:0.4E} (point {best+1}/{n_points})")
        return params[max(best-1,0)], params[min(best+1, len(params)-1)], best_value
    for i in range(n_iters):
        min_value, max_value, best_value = grid_search_iter(min_value, max_value)
    return best_value

def test_alpha(lm, alpha, test_data):
    results = test_lm(lm, test_data, alpha=alpha) 
    return float(results.groupby('cond').mean('ppl').mean()) # macro average ppl

@click.command()
@click.argument('model_family_dir', type=click.Path(exists=False))
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('ngram_size', type=int) # N-gram size
@click.option('--max-seq-len', default=64)
@click.option('--vocab-size', default=40000)
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gridsearch-iters', type=int, default=10)
@click.pass_context
def cli(ctx, model_family_dir, data_dir, ngram_size, max_seq_len, vocab_size, file_limit, gridsearch_iters):
    ctx.ensure_object(dict)

    model_name = f"{ngram_size}-gram"
    model_family_dir = Path(model_family_dir)
    data_dir = Path(data_dir)
    model_dir = model_family_dir/model_name
    results_file = model_dir/'test_ppl.pickle'

    util.mkdir(model_dir)

    log = util.create_logger('ngram', model_dir/'ngram.log', True)

    log.info(f"Loading data from {data_dir}.")
    fields = data.load_fields(model_family_dir, data_dir, vocab_size)
    train_data = data.load_data(data_dir, fields, 'train', max_seq_len, file_limit)
    vocab_size = len(fields['text'].vocab.itos)
    cond_vocab_size = len(fields['community'].vocab.itos)
    text_pad_idx = fields['text'].vocab.stoi['<pad>']
    dev_data = data.load_data(data_dir, fields, 'dev', max_seq_len, None)
    log.info(f"Loaded {len(train_data)} train and {len(dev_data)} dev examples.")

    text_pad_idx = fields['text'].vocab.stoi['<pad>']
    vocab_size = len(fields['text'].vocab.stoi)

    lm = CondNGramLM(ngram_size, cond_vocab_size, vocab_size, None)
    log.info("Fitting n-gram model to the training data")
    lm.fit(train_data)
    log.info("Finding optimal smoothing parameter")

    min_alpha = 1 / (vocab_size**2)
    max_alpha = 1
    grid_points = 10
    alpha = grid_search(lambda x: test_alpha(lm, x, dev_data), grid_points, min_alpha, max_alpha, gridsearch_iters, log)
    lm.alpha = alpha
    lm.save(model_dir/'model')

    test_data = data.load_data(data_dir, fields, 'test', max_seq_len, None)
    df = test_lm(lm, test_data)
    df.to_pickle(results_file)

if __name__ == '__main__':
    cli()
