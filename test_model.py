""" Use the trained models on test data, testing both CCLM and LMCC.
    Produce results tabels that will be read by analysis.py to create plots, tables, etc.
"""

from model import CommunityConditionedLM
from data import load_data_and_fields
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext as tt
import pandas as pd
import numpy as np
import random
import json
import os
import click
from pathlib import Path

def exp_normalize(x, axis):
    """ https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/ """
    b = x.max(axis=axis)
    y = np.exp(x - b)
    return y / y.sum(axis)

def mask_seqs(max_seq_len, batch_size, lengths):
    idxs = torch.arange(max_seq_len).expand(batch_size, max_seq_len).to(lengths.device)
    mask = idxs < lengths.unsqueeze(1)
    return mask.T.float()

def batch_nll(model, batch, comm=None):
    """
    Compute the negative log likelihood of the batch for the given community
    If comm is None, then the actual communities for the batch will be used.
    """
    if comm is None:
        comm = batch.community
    text, lengths = batch.text
    max_seq_len = text.shape[0] - 1
    x_text = text[:-1]
    y = text[1:]
    y_hat = model(x_text, comm)
    vocab_size = y_hat.shape[-1]
    nll_word = F.nll_loss(y_hat.view(-1,vocab_size), y.view(-1), reduction='none').view(y.shape)
    mask = mask_seqs(max_seq_len, batch.batch_size, lengths-1)
    nll_word = nll_word * mask
    nll_seq = nll_word.sum(axis=0) 
    return nll_seq

@click.group()
@click.argument('model_family_dir', type=click.Path(exists=False))
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--batch-size', default=512)
@click.option('--max-seq-len', default=64)
@click.option('--file-limit', type=int, default=50000,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
@click.pass_context
def cli(ctx, model_family_dir, data_dir, batch_size, max_seq_len, file_limit, gpu_id):
    ctx.ensure_object(dict)

    ctx.obj['device'] = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')

    # load the dataset
    dataset, fields = load_data_and_fields(data_dir, model_family_dir,
            max_seq_len, file_limit)
    fields['text'].include_lengths = True
    ctx.obj['fields'] = fields

    # split into train/val/test
    random.seed(42)
    random_state = random.getstate()
    _, _, test_data = dataset.split(split_ratio=[0.8,0.1,0.1],
            stratified=True, strata_field='community', random_state=random_state)
    ctx.obj['test_data'] = test_data

    ctx.obj['test_iterator'] = tt.data.BucketIterator(test_data, device=ctx.obj['device'],
        batch_size=batch_size, sort=False, shuffle=False, train=False)

    ctx.obj['model_family_dir'] = Path(model_family_dir)
    ctx.obj['conditional_architectures'] = [x.stem for x in ctx.obj['model_family_dir'].glob('*-*-*')]
    ctx.obj['architectures'] = [x.stem for x in ctx.obj['model_family_dir'].glob('*-*')]

@cli.command()
@click.pass_context
def extract_comm_embeds(ctx):
    for arch in ctx.obj['conditional_architectures']:
        model_dir = ctx.obj['model_family_dir']/arch 
        model_weights = torch.load(model_dir/'model.bin', map_location='cpu')
        comm_embed = model_weights['comm_embed.weight'].numpy()
        np.save(model_dir/'comm_embed', comm_embed)

@cli.command()
@click.pass_context
def test_cclm(ctx):
    """ Find the test perpelixity for each model on each test example."""

    results_file = ctx.obj['model_family_dir']/'test_ppl.pickle'

    def test_ppl(lm, test_batches):
        ppls = []
        with torch.no_grad():
            for i, batch in enumerate(test_batches):
                text, lengths = batch.text
                print(f'{i+1}/{len(test_batches)}', end='\r')
                nll_seq = batch_nll(lm, batch)
                nll_mean = nll_seq / lengths.float()
                ppl = nll_mean.exp()
                ppls.append(ppl.cpu().numpy())
        ppls = np.concatenate(ppls)
        return ppls

    df = pd.DataFrame([(example.community, ' '.join(example.text))
        for example in ctx.obj['test_data']], columns=['community', 'comment'])
    for model_name in ctx.obj['architectures']:
        print(f"Testing {model_name}")
        model_dir = ctx.obj['model_family_dir']/model_name
        model_args = json.load(open(model_dir/'model_args.json'))
        model = CommunityConditionedLM.build_model(**model_args).to(ctx.obj['device'])
        model.load_state_dict(torch.load(model_dir/'model.bin'))
        model.to(ctx.obj['device'])
        model.eval()
        ppls = test_ppl(model, ctx.obj['test_iterator'])
        df[model_name] = ppls
    df.to_pickle(results_file)

@cli.command()
@click.argument('model_name', type=str)
@click.pass_context
def test_lmcc(ctx, model_name):
    """ Language model-based community classification. """

    model_dir = os.path.join(ctx.obj['model_family_dir'], model_name)

    fields = ctx.obj['fields']
    n_communities = len(fields['community'].vocab)
    comms = fields['community'].vocab.itos

    model_args = json.load(open(os.path.join(model_dir, 'model_args.json')))
    model = CommunityConditionedLM.build_model(**model_args).to(ctx.obj['device'])
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    model.to(ctx.obj['device'])
    model.eval()

    def batchify_comm(comm, batch_size):
        comm_idx = fields['community'].vocab.stoi[comm]
        return torch.tensor(comm_idx).repeat(batch_size).to(ctx.obj['device'])

    comm_probs = []
    actual_comms = []
    with torch.no_grad():
        for i, batch in enumerate(ctx.obj['test_iterator']):
            print(f"{i}/{len(ctx.obj['test_iterator'])}",end='\r')
            with torch.no_grad():
                nlls = torch.stack([
                    batch_nll(
                        model, batch,
                        comm=batchify_comm(comm, batch.batch_size)
                    ) for comm in comms[1:]
                    ], dim=0).cpu().numpy()
            comm_probs_batch = exp_normalize(-nlls, axis=0)
            comm_probs.append(comm_probs_batch)
            actual_comms += batch.community.tolist()
    comm_probs = np.concatenate(comm_probs, axis=1).T

    df = pd.DataFrame(comm_probs, columns=comms[1:])
    df['actual_comm'] = [comms[c] for c in actual_comms]
    df.to_pickle(os.path.join(model_dir, 'comm_probs.pickle'))

if __name__ == '__main__':
    cli(obj={})


