""" Use the trained models on test data, testing both CCLM and LMCC.
    Produce results tabels that will be read by analysis.py to create plots, tables, etc.
"""

import util
import data
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext as tt
import numpy as np
import random
import json
import csv
import os
import click
from pathlib import Path

def exp_normalize(x, axis):
    """ https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/ """
    b = x.max(axis=axis)
    y = np.exp(x - b)
    return y / y.sum(axis)

def batch_nll(lm, batch, pad_idx, comm=None):
    """
    Compute the negative log likelihood of the batch for the given community
    If comm is None, then the actual communities for the batch will be used.
    """
    if comm is None:
        comm = batch.community
    text, lengths = batch.text
    x_text = text[:-1]
    y = text[1:]
    y_hat = lm(x_text, comm)
    vocab_size = y_hat.shape[-1]
    nll_seq = F.nll_loss(y_hat.view(-1,vocab_size), y.view(-1), 
            reduction='none', ignore_index=pad_idx).view(y.shape).sum(axis=0)
    return nll_seq

@click.command()
@click.argument('model_family_dir', type=click.Path(exists=False))
@click.argument('model_name', type=str)
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--batch-size', default=512)
@click.option('--max-seq-len', default=64)
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(model_family_dir, model_name, data_dir, batch_size, max_seq_len, file_limit, gpu_id):

    model_family_dir = Path(model_family_dir)
    model_dir = model_family_dir/model_name
    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
    log = util.create_logger('test', os.path.join(model_dir, 'testing.log'), True)

    log.info(f"Loading data from {data_dir}.")
    fields = data.load_fields(model_family_dir)
    fields['text'].include_lengths = True
    test_data = data.load_data(data_dir, fields, 'test', max_seq_len, file_limit)

    vocab_size = len(fields['text'].vocab.itos)
    comm_vocab_size = len(fields['community'].vocab.itos)
    comms = fields['community'].vocab.itos
    pad_idx = fields['text'].vocab.stoi['<pad>']
    log.info(f"Loaded {len(test_data)} test examples.")

    model_args = json.load(open(model_dir/'model_args.json'))
    lm = model.CommunityConditionedLM.build_model(**model_args).to(device)
    lm.load_state_dict(torch.load(model_dir/'model.bin'))
    lm.to(device)
    lm.eval()
    log.debug(str(lm))

    test_iterator = tt.data.BucketIterator(
        test_data,
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: -len(x.text),
        shuffle=True,
        train=False)

    def batchify_comm(comm, batch_size):
        comm_idx = fields['community'].vocab.stoi[comm]
        return torch.tensor(comm_idx).repeat(batch_size).to(device)

    with torch.no_grad(), open(model_dir/'nll.csv', 'w') as f:
        meta_fields = ['community', 'example_id', 'length']
        writer = csv.DictWriter(f, fieldnames=meta_fields+comms)
        writer.writeheader()
        for i, batch in enumerate(test_iterator):
            nlls_batch = [
                dict(zip(meta_fields, meta_values)) for meta_values in zip(
                    batch.community.tolist(),
                    batch.example_id.tolist(),
                    batch.text[1].tolist()
                )
            ]
            for comm in comms:
                batch_comm = batchify_comm(comm, batch.batch_size)
                nlls_comm = batch_nll(lm, batch, pad_idx, comm=batch_comm)
                for j, nll in enumerate(nlls_comm):
                    nlls_batch[j][comm] = nll.item()
            writer.writerows(nlls_batch)
            log.info(f"Completed {i+1}/{len(test_iterator)}")


    # comm_probs_batch = exp_normalize(-nlls, axis=0)
    # comm_probs.append(comm_probs_batch)
    # actual_comms += batch.community.tolist()
    # comm_probs = np.concatenate(comm_probs, axis=1).t


    # def test_ppl(lm, test_batches):
        # ppls = []
        # with torch.no_grad():
            # for i, batch in enumerate(test_batches):
                # text, lengths = batch.text
                # print(f'{i+1}/{len(test_batches)}', end='\r')
                # nll_seq = batch_nll(lm, batch)
                # nll_mean = nll_seq / lengths.float()
                # ppl = nll_mean.exp()
                # ppls.append(ppl.cpu().numpy())
        # ppls = np.concatenate(ppls)
        # return ppls

if __name__ == '__main__':
    cli(obj={})
