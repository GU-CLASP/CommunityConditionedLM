import click
import data
from classifier_model import LSTMClassifier, NaiveBayesUnigram
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

def entropy(t, dim):
    nats = t * t.log()
    nats[torch.isnan(nats)] = 0 # inf * 0 treated as 0 
    return -(nats).sum(dim=dim)

@click.command()
@click.argument('model_architecture', type=click.Choice(['unigram', 'lstm']))
@click.argument('model_dir', type=click.Path(exists=True))
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--batch-size', default=1024)
@click.option('--max-seq-len', default=64)
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(model_architecture, model_dir, data_dir, batch_size, max_seq_len, file_limit, gpu_id):

    # model_architecture = 'unigram-cond'
    # model_dir = "model/reddit/"
    # data_dir = "data/reddit_splits"
    # gpu_id = None
    # max_seq_len = 64
    # file_limit = None

    model_dir = Path(model_dir)
    save_dir = model_dir/model_architecture
    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
    log = util.create_logger('test', os.path.join(save_dir, 'testing.log'), True)

    log.info(f"Loading data from {data_dir}.")
    fields = data.load_fields(model_dir, use_eosbos=False)
    fields['text'].include_lengths = True
    lower_case = all(w.lower() == w for w in fields['text'].vocab.itos) # really don't want to fuck this up again...
    log.info(f"Using lower-case vocab: {lower_case!s}.")

    test_data = data.load_data(data_dir, fields, 'test', max_seq_len, file_limit)

    vocab_size = len(fields['text'].vocab.itos)
    comm_vocab_size = len(fields['community'].vocab.itos)
    comms = fields['community'].vocab.itos
    pad_idx = fields['text'].vocab.stoi['<pad>']
    log.info(f"Loaded {len(test_data)} test examples.")

    state_dict = torch.load(save_dir/'model.bin')

    if model_architecture == 'lstm':
        embedding_size = state_dict['token_embed.weight'].size(1)
        hidden_size = state_dict['lstm.weight_hh_l0'].size(1)
        n_layers = len([k for k in state_dict.keys() if k.startswith('lstm.weight_hh_l')])
        model = LSTMClassifier(vocab_size, comm_vocab_size, embedding_size, hidden_size, n_layers)
        eval_func = lambda x, x_lens: model(x, x_lens, agg_seq=None)
    elif model_architecture == 'unigram':
        model = NaiveBayesUnigram(vocab_size, comm_vocab_size)
        eval_func = lambda x, x_lens: model.infer_comm_incremental(x, x_lens)
    else:
        raise ValueError(f"Unknown architecture: {model_architecture}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    log.debug(str(model))
    embed()
    raise

    test_iterator = tt.data.BucketIterator(
        test_data,
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: -len(x.text),
        shuffle=True,
        train=False)

    analytics = ['entropy', 'cross_entropy', 'pred']
    meta_fields = ['community', 'example_id', 'length']
    data_fields = list(range(max_seq_len))

    def make_writer(f):
        writer = csv.DictWriter(f, fieldnames=meta_fields + data_fields)
        writer.writeheader()
        return writer

    results_files = {a: open(save_dir/f"{a}.csv", 'w') for a in analytics}
    writers = {a: make_writer(results_files[a]) for a in analytics}

    with torch.no_grad():
        for batch_no, batch in enumerate(test_iterator):
            if batch_no % 1 == 0 and batch_no > 0:
                log.info(f"{batch_no:5d}/{len(test_iterator):5d} batches")
            batch_text, batch_text_lens = batch.text

            batch_max_len = batch_text_lens.max().item()
            batch_comms = batch.community.unsqueeze(0).expand(batch_text.size(0), -1)

            prob_dist = eval_func(batch_text, batch_text_lens)

            entr = entropy(prob_dist, dim=-1)
            cross_entr = F.cross_entropy(
                    prob_dist.flatten(0,1).log(), batch_comms.flatten(), 
                    reduction='none').reshape(batch_max_len, batch.batch_size)
            pred_probs, preds = prob_dist.max(dim=-1)

            batch_results = {a: [
                dict(zip(meta_fields, item)) for item in zip(
                batch.community.tolist(),
                batch.example_id.tolist(),
                batch_text_lens.tolist()
            )] for a in analytics}

            for i in range(batch_max_len): 
                for j in range(batch.batch_size):
                    if i < batch.text[1][j]:
                        batch_results['cross_entropy'][j][i] = f'{cross_entr[i][j].item():0.4f}'
                        batch_results['entropy'][j][i] = f'{entr[i][j].item():0.4f}'
                        batch_results['pred'][j][i] = preds[i][j].item()
            for a in analytics:
                writers[a].writerows(batch_results[a])

    for a in analytics:
        results_files[a].close()

if __name__ == '__main__':
    cli(obj={})

