import click
import data
from classifier_model import LSTMClassifier
import torch
import torch.nn as nn
import torchtext as tt
import torch.nn.functional as F
import math
import random
import util
import os
from pathlib import Path
import csv
from IPython import embed

@click.command()
@click.argument('model_dir', type=click.Path(exists=True))
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--max-seq-len', default=64)
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(model_dir, data_dir, max_seq_len, file_limit, gpu_id):

    # model_dir = "model/reddit/lstm_classifier"
    # data_dir = "data/reddit_splits"
    # gpu_id = 0
    # max_seq_len = 64
    # file_limit = None

    model_dir = Path(model_dir)
    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
    log = util.create_logger('test', os.path.join(model_dir, 'testing.log'), True)

    log.info(f"Loading data from {data_dir}.")
    fields = data.load_fields('model/reddit')
    fields['text'].include_lengths = True
    test_data = data.load_data(data_dir, fields, 'test', max_seq_len, file_limit)

    vocab_size = len(fields['text'].vocab.itos)
    comm_vocab_size = len(fields['community'].vocab.itos)
    comms = fields['community'].vocab.itos
    pad_idx = fields['text'].vocab.stoi['<pad>']
    log.info(f"Loaded {len(test_data)} test examples.")

    model = LSTMClassifier(vocab_size, comm_vocab_size, 512)
    model.load_state_dict(torch.load(model_dir/'model.bin'))
    model.to(device)
    model.eval()
    log.debug(str(model))

    test_iterator = tt.data.BucketIterator(
        test_data,
        device=device,
        batch_size=1024,
        sort_key=lambda x: -len(x.text),
        shuffle=True,
        train=False)

    analytics = ['entropy', 'cross_entropy', 'pred']
    layers =  ['embed', 'lstm1', 'lstm2', 'all']
    meta_fields = ['community', 'example_id', 'length']
    data_fields = list(range(1, max_seq_len+2)) # take data point at each token in sequence excluding <start> but including <end>

    def make_writer(f):
        writer = csv.DictWriter(f, fieldnames=meta_fields + data_fields)
        writer.writeheader()
        return writer

    results_files = {l: {a: open(model_dir/f"{a}_{l}.csv", 'w') for a in analytics} for l in layers}
    writers = {l: {a: make_writer(results_files[l][a]) for a in analytics} for l in layers}

    with torch.no_grad():
        data_fields = comms
        for batch_no, batch in enumerate(test_iterator):
            batch_max_len = batch.text[1].max().item()
            activations = model.depth_stratified_activations(batch.text[0])
            batch_results = {l: {a: [
                dict(zip(meta_fields, item)) for item in zip(
                batch.community.tolist(),
                batch.example_id.tolist(),
                batch.text[1].tolist()
            )] for a in analytics} for l in layers}
            for i in range(1, batch_max_len+1): # skip evaluating only the start token
                for y, l in zip(activations, layers):
                    prob_dist = F.softmax(y[0:i+1].max(dim=0)[0], dim=-1)
                    entropy = (-prob_dist * prob_dist.log()).sum(dim=-1)
                    pred_probs, preds = prob_dist.max(dim=-1)
                    for j in range(batch.batch_size):
                        if i < batch.text[1][j]:
                            cross_entropy = -(prob_dist[j][batch.community[j]].log())
                            batch_results[l]['entropy'][j][i] = f'{entropy[j].item():0.4f}'
                            batch_results[l]['cross_entropy'][j][i] = f'{cross_entropy.item():0.4f}'
                            batch_results[l]['pred'][j][i] = preds[j].item()
            for l in layers:
                for a in analytics:
                    writers[l][a].writerows(batch_results[l][a])
            log.info(f"Completed {batch_no+1}/{len(test_iterator)}")

    for l in layers:
        for a in analytics:
            results_files[l][a].close()

if __name__ == '__main__':
    cli(obj={})

