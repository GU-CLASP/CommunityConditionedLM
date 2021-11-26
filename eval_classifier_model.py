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

def make_seq_mask(seq_lens):
    max_len = seq_lens.max().item()
    return (torch.arange(max_len).to(seq_lens.device).expand(len(seq_lens), max_len) < seq_lens.unsqueeze(1)).float()


@click.command()
@click.argument('model_dir', type=click.Path(exists=True))
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--max-seq-len', default=64)
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(model_dir, data_dir, max_seq_len, file_limit, gpu_id):

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

    model = LSTMClassifier(vocab_size, comm_vocab_size, 512, 2, 0.1)
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

    results_files = [model_dir/f"cc_probs{suffix}.csv" for suffix in ('', '_e', '_1', '_2')]
    fs = [open(filename, 'w') for filename in results_files]

    y_e_means =  torch.zeros(510).to(device)
    y_1_means =  torch.zeros(510).to(device)
    y_2_means =  torch.zeros(510).to(device)

    with torch.no_grad():
        meta_fields = ['community', 'example_id']
        data_fields = comms
        writers = (csv.DictWriter(f, fieldnames=meta_fields+data_fields) for f in fs)
        for writer in writers:
            writer.writeheader()
        for i, batch in enumerate(test_iterator):
            y_e, y_1, y_2 = model.depth_stratified_preds(batch.text[0])

            mask = make_seq_mask(batch.text[1]).T.unsqueeze(-1).expand(-1, -1, 510).to(device)

            y_e_means += (y_e * mask.float()).pow(2).mean(0).mean(0)
            y_1_means += (y_1 * mask.float()).pow(2).mean(0).mean(0)
            y_2_means += (y_2 * mask.float()).pow(2).mean(0).mean(0)

            # batch_results = [
                # dict(zip(meta_fields, meta_values)) for meta_values in zip(
                    # [comms[i] for i in batch.community.tolist()],
                    # batch.example_id.tolist()
                # )
            # ]
            # preds  = model.depth_stratified_preds(batch.text[0])
            # for y, writer in zip(preds, writers):
                # for j, probs_item in enumerate(y.exp()):
                    # batch_results[j].update(dict(zip(comms, probs_item.tolist())))
                # writer.writerows(batch_results)
            log.info(f"Completed {i+1}/{len(test_iterator)}")

    for f in fs:
        f.close()

if __name__ == '__main__':
    cli(obj={})

