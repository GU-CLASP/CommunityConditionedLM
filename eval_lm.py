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

def test_perplexity():
    pass

@click.command()
@click.argument('model_family_dir', type=click.Path(exists=False))
@click.argument('model_name', type=str)
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--batch-size', default=512)
@click.option('--max-seq-len', default=64)
@click.option('--file-limit', type=int, default=50000,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(model_family_dir, model_name, data_dir, batch_size, max_seq_len, 
        file_limit, gpu_id):

    model_dir = os.path.join(model_family_dir, model_name)
    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')

    random.seed(42)
    random_state = random.getstate()
    dataset, fields = load_data_and_fields(data_dir, model_family_dir, max_seq_len, file_limit)
    fields['text'].include_lengths = True
    _, _, test_data = dataset.split(split_ratio=[0.8,0.1,0.1], stratified=True, strata_field='community', random_state=random_state)

    test_iterator = tt.data.BucketIterator(
        test_data,
        device=device,
        batch_size=batch_size,
        sort=False,
        shuffle=False,
        train=False)

    vocab_size = len(fields['text'].vocab)
    n_communities = len(fields['community'].vocab)
    text_pad_idx = fields['text'].vocab.stoi['<pad>']
    comm_unk_idx = fields['community'].vocab.stoi['<unk>']
    comms = fields['community'].vocab.itos

    model_args = json.load(open(os.path.join(model_dir, 'model_args.json')))
    model = CommunityConditionedLM.build_model(**model_args).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    model.to(device)

    def batch_liklihood(model, x_text, comm):
        comm_idx = fields['community'].vocab.stoi[comm]
        x_comm = torch.tensor(comm_idx).repeat(batch.batch_size).to(device)
        y_hat = model(x_text, x_comm)
        nll = F.nll_loss(y_hat.view(-1,vocab_size), y.view(-1), reduction='none', ignore_index=text_pad_idx).view(y.shape)
        nll_per_word = nll.sum(axis=0) / lengths.float()
        ppl_per_word = nll_per_word.exp()
        likelihood = 1/ppl_per_word
        return likelihood

    probs = []
    actual_comms = []
    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            print(f'{i}/{len(test_iterator)}',end='\r')
            x_comm = batch.community
            text, lengths = batch.text
            x_text = text[:-1]
            y = text[1:]
            likelihoods = torch.stack([batch_liklihood(model, x_text, comm)
                for comm in comms[1:]], dim=0).cpu().numpy()
            probs.append((likelihoods / likelihoods.sum(axis=0)).T)
            actual_comms += x_comm.tolist()
    probs = np.concatenate(probs)

    df = pd.DataFrame(probs, columns=comms[1:])
    df['actual_comm'] = [comms[c] for c in actual_comms]

    df.to_pickle(os.path.join(model_dir, 'comm_probs.pickle'))

if __name__ == '__main__':
    cli()
