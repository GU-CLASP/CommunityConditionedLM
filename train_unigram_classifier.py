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
import click
from IPython import embed
import util

def compile_unigram_counts(comms, vocab_stoi, counts_dir):
    indices = ([],[])
    values = []
    unks = {c: 0 for c in comms}
    unk_index = vocab_stoi['<unk>']
    for c_i, c in enumerate(comms):
        with open(counts_dir/f'{c}.count.txt') as f: # generate from test_data
            print(f"{c_i+1}/{len(comms)}", end='\r')
            for line in f.readlines():
                line = line.lstrip().split()
                if len(line) == 1: # handle a few weird whitespace characters
                    w = '<unk>'
                    count = line[0]
                else:
                    count, w = line
                count = int(count)
                if w in vocab_stoi:
                    indices[0].append(c_i)
                    indices[1].append(vocab_stoi[w])
                    values.append(count)
                else:
                    unks[c] += count
    for c_i, c in enumerate(comms):
        indices[0].append(c_i)
        indices[1].append(unk_index)
        values.append(unks[c])
    return indices, values 

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

@click.command()
@click.argument('model_dir', type=click.Path(exists=True))
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--vocab-size', default=40000)
@click.option('--lower-case/--no-lower-case', default=False)
@click.option('--batch-size', default=128)
@click.option('--max-seq-len', default=64)
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(model_dir, data_dir, vocab_size, lower_case, batch_size,
        max_seq_len, file_limit, gpu_id):

    model_dir = Path(model_dir)
    model_name = f"unigram_classifier"

    save_dir = model_dir/model_name
    util.mkdir(save_dir)
    unigram_counts_file = save_dir/'unigram_counts.pickle'

    log = util.create_logger('find-alpha', save_dir/'training.log', True)
    log.info(f"Model will be saved to {save_dir}.")

    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')

    log.info(f"Loading data from {data_dir}.")
    fields = data.load_fields(model_dir, data_dir, vocab_size, 
            lower_case=lower_case, use_eosbos=False)
    fields['text'].include_lengths = True
    lower_case = all(w.lower() == w for w in fields['text'].vocab.itos) # really don't want to fuck this up again...
    log.info(f"Using lower-case vocab: {lower_case!s}.")

    vocab_size = len(fields['text'].vocab.itos)
    comms = fields['community'].vocab.itos
    comm_vocab_size = len(comms)
    text_pad_idx = fields['text'].vocab.stoi['<pad>']
    dev_data = data.load_data(data_dir, fields, 'dev',
            max_seq_len, file_limit, lower_case)
    log.info(f"Loaded {len(dev_data)} dev examples.")

    log.debug("Compiling unigrams per community...")
    indices, values = compile_unigram_counts(comms, fields['text'].vocab.stoi, Path('data/unigram_counts/'))
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

    dev_iterator = tt.data.BucketIterator(
        dev_data,
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        train=False)

    f = lambda x: model.test_cross_entropy_laplace(dev_iterator, alpha=x)
    alpha, mean_entropy = search_param(f, .001,2,0.1, epsilon=0.01)
    log.info(f"Final alpha = {alpha} w/ mean entropy = {mean_entropy}.")

    model.update_alpha(alpha)
    torch.save(model.state_dict(), save_dir/'model.bin')


if __name__ == '__main__':
    cli()

