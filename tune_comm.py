import os
import click
import data
from train import evaluate
import model
import torch
import torch.nn as nn
import torchtext as tt
import torch.nn.functional as F
import math
import random
import util
import json

def tune(lm, batches, vocab_size, criterion, optimizer, log):
    lm.train()
    batches.init_epoch()
    train_loss = 0
    for batch_no, batch in enumerate(batches):
        optimizer.zero_grad()
        batch_size_ = len(batch)
        x_comm = batch.community
        text = batch.text
        x_text = text[:-1]
        y = text[1:]
        y_hat = lm(x_text, x_comm)
        loss = criterion(y_hat.view(-1, vocab_size), y.view(-1)).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_no % 1000 == 0 and batch_no > 0:
            cur_loss = train_loss / 1000
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            sim = F.cosine_similarity(torch.matmul(lm.comm_inference.weight, lm.comm_embed.weight), lm.comm_embed.weight)[1:] # ignore <unk>
            log.info(f"{batch_no:5d}/{len(batches):5d} batches | loss {cur_loss:5.4f} | ppl {math.exp(cur_loss):0.2f} | comm embed sim {sim.mean():0.4f} |lr {lr}")
            train_loss = 0
    return lm

@click.command()
@click.argument('model_family_dir', type=click.Path(exists=False))
@click.argument('model_name', type=str)
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--dropout', default=0.1)
@click.option('--batch-size', default=32)
@click.option('--max-seq-len', default=64)
@click.option('--lr', default=0.001)
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(model_family_dir, model_name, data_dir, dropout,
        batch_size, max_seq_len, lr, file_limit, gpu_id):

    model_dir = os.path.join(model_family_dir, model_name)
    log = util.create_logger('tune', os.path.join(model_dir, 'training.log'), True)

    log.info(f"Loading dataset from {data_dir} files.")
    dataset, fields = data.load_data_and_fields(data_dir, model_family_dir,
            max_seq_len, file_limit)
    vocab_size = len(fields['text'].vocab.itos)
    comm_vocab_size = len(fields['community'].vocab.itos)
    comm_unk_idx = fields['community'].vocab.stoi['<unk>']
    text_pad_idx = fields['text'].vocab.stoi['<pad>']
    log.info(f"Loaded {len(dataset)} examples.")

    random.seed(42)
    random_state = random.getstate()
    train_data, val_data, test_data = dataset.split(split_ratio=[0.8,0.1,0.1], stratified=True, strata_field='community', random_state=random_state)
    log.info(f"Splits: train: {len(train_data)} val: {len(val_data)} test: {len(test_data)} ")

    log.info(f"Loading model from {model_dir}.")
    model_args = json.load(open(os.path.join(model_dir, 'model_args.json')))
    lm = model.CommunityConditionedLM.build_model(**model_args)
    lm.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))

    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
    lm.to(device)
    lm.tune_comm()

    params = list(lm.named_parameters())
    log.debug(f"Model parameters ({len(params)} total):")
    for n, p in params:
        log.debug("{:<25} | {:<10} | {}".format(
            str(p.size()),
            'training' if p.requires_grad else 'frozen',
            n if n else '<unnamed>'))


    tune_iterator = tt.data.BucketIterator(
        train_data,
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        train=True)
    val_iterator = tt.data.BucketIterator(
        val_data,
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        train=False)

    criterion = nn.NLLLoss(ignore_index=text_pad_idx, reduction='none')
    optimizer = torch.optim.AdamW(lm.parameters(), lr=lr)

    val_losses = []
    epoch = 0
    while True:
        epoch += 1
        log.info(f'Starting epoch {epoch}')
        lm = tune(lm, tune_iterator, vocab_size, criterion, optimizer, log)
        val_loss = evaluate(lm, val_iterator, vocab_size, comm_unk_idx, criterion)
        val_loss = sum(val_loss) / len(val_loss)
        if epoch == 1 or val_loss < min(val_losses):
            torch.save(lm.state_dict(), os.path.join(model_dir, 'model_tuned.bin'))
            with open(os.path.join(model_dir, 'saved-epoch_tuned.txt'), 'w') as f:
                f.write(f'{epoch:03d}')
        val_losses.append(val_loss)
        log.info(f"Epoch {epoch:3d} | val loss {val_loss:5.6f} | ppl {math.exp(val_loss):0.2f}")
        if val_losses[-1] > min(val_losses) and val_losses[-2] > min(val_losses):
            log.info(f'Stopping early after epoch {epoch}.')
            break

if __name__ == '__main__':
    cli()
