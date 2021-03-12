from data import load_data_and_fields
from pathlib import Path
import util
import data
import random
import itertools
import numpy as np
import pandas as pd
import click
import os

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
    if n == 1:
        for y_hat in text[n-1:]:
            p.append(P[comm][y_hat])
    else:
        for y_hat, ngram in zip(text[n-1:], iter_ngrams(text, n-1)):
            p.append(P[comm][ngram][y_hat])
    p = np.array(p)
    return -np.log(p)

def test_lm(P, data):
    results = []
    n = len(P.shape) - 1
    for comm, text in iter_examples(n,data):
        nll = example_nll(n, P, comm,text)
        mean_nll = nll.sum() / len(nll)
        ppl = np.exp(mean_nll)
        results.append({'comm': comm, 'ppl': ppl})
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

def test_alpha(C, alpha, test_data):
    P = build_lm(C, alpha)
    results = test_lm(P, test_data) 
    return float(results.groupby('comm').mean('ppl').mean()) # macro average ppl

@click.command()
@click.argument('model_family_dir', type=click.Path(exists=False))
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('ngram_size', type=int) # N-gram size
@click.option('--max-seq-len', default=64)
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gridsearch-iters', type=int, default=10)
@click.pass_context
def cli(ctx, model_family_dir, data_dir, ngram_size, max_seq_len, file_limit, gridsearch_iters):
    ctx.ensure_object(dict)

    model_name = f"{ngram_size}-gram"
    model_family_dir = Path(model_family_dir)
    data_dir = Path(data_dir)
    model_dir = model_family_dir/model_name
    results_file = model_dir/'test_ppl.pickle'

    util.mkdir(model_dir)

    log = util.create_logger('ngram', model_dir/'ngram.log', True)

    dataset, fields = load_data_and_fields(data_dir, model_family_dir,
            max_seq_len, file_limit)
    log.info(f"Loaded {len(dataset)} examples.")

    text_pad_idx = fields['text'].vocab.stoi['<pad>']
    vocab_size = len(fields['text'].vocab.stoi)


    random.seed(42)
    random_state = random.getstate()
    train_data, val_data, test_data = dataset.split(split_ratio=[0.8,0.1,0.1], stratified=True, strata_field='community', random_state=random_state)
    log.info(f"Splits: train: {len(train_data)} val: {len(val_data)} test: {len(test_data)} ")

    log.info("Counting n-grams in the training data")
    C = count_ngrams(ngram_size, train_data)
    log.info("Finding optimal smoothing parameter")

    min_alpha = 1 / (vocab_size**2)
    max_alpha = 1
    grid_points = 10
    alpha = grid_search(lambda x: test_alpha(C, x, val_data), grid_points, min_alpha, max_alpha, gridsearch_iters, log)

    P = build_lm(C, alpha)
    np.save(model_dir/'model', P)
    with open(os.path.join(model_dir, 'smoothing-alpha.txt'), 'w') as f:
        f.write(f'{alpha:0.8f}')

    df = test_lm(P, test_data)
    df.to_pickle(results_file)

if __name__ == '__main__':
    cli()
