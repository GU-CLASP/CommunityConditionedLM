from pathlib import Path
import os
import click
import data
import model
import torch
import torch.nn as nn
import torchtext as tt
import torch.nn.functional as F
import math
import random

def train(lm, batches, vocab_size, condition_community, comm_unk_idx, criterion, optimizer):
    lm.train()
    batches.init_epoch()
    train_loss = 0
    for batch_no, batch in enumerate(batches):
        optimizer.zero_grad()
        batch_size_ = len(batch)
        x_sub = batch.sub if condition_community else None 
        x_text = batch.text[:-1]
        y = batch.text[1:]
        y_hat = lm(x_text, x_sub)
        loss = criterion(y_hat.view(-1, vocab_size), y.view(-1)).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_no % 1000 == 0 and batch_no > 0:
            cur_loss = train_loss / 1000
            click.echo(f"{batch_no:5d}/{len(batches):5d} batches | loss {cur_loss:5.2f} | ppl {math.exp(cur_loss)}")
            train_loss = 0
    return lm

def evaluate(lm, batches, vocab_size, condition_community, comm_unk_idx, criterion):
    lm.eval()
    batches.init_epoch()
    eval_losses = []
    for batch in batches:
        with torch.no_grad():
            batch_size_ = len(batch)
            x_sub = batch.sub if condition_community else None 
            x_text = batch.text[:-1]
            y = batch.text[1:]
            y_hat = lm(x_text, x_sub)
            loss = criterion(y_hat.view(-1, vocab_size), y.view(-1))
            eval_losses += list(loss)
    return eval_losses


@click.command()
@click.argument('architecture', type=click.Choice(['Transformer', 'LSTM'], case_sensitive=False))
@click.argument('model_filename', type=click.Path(exists=False))
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--rebuild-vocab/--no-rebuild-vocab', default=False)
@click.option('--vocab-size', default=40000)
@click.option('--emsize', default=128)
@click.option('--nhead', default=8)
@click.option('--nhid', default=256)
@click.option('--condition-community/--no-condition-community', default=True)
@click.option('--community-emsize', default=64)
@click.option('--layers-before', default=2)
@click.option('--layers-after', default=2)
@click.option('--dropout', default=0.1)
@click.option('--epochs', default=5)
@click.option('--batch-size', default=20)
@click.option('--max-seq-len', default=64)
@click.option('--chosen-subs-file', type=click.Path(exists=True), 
        default='chosen_subs.txt',
        help="File to save the list of chosen subs to.")
@click.option('--file-limit', type=int, default=10000,
        help="Number of examples per subreddit per month")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(architecture, model_filename, data_dir, rebuild_vocab, vocab_size, emsize, nhead, nhid,
        condition_community, community_emsize, layers_before, layers_after, dropout,
        epochs, batch_size, max_seq_len, chosen_subs_file, file_limit, gpu_id):
    data_dir = Path(data_dir)
    vocab_size = 40000
    subs = data.get_subs(chosen_subs_file)

    click.echo(f"Training {architecture}, will be saved with prefix {model_filename}.")
    click.echo(f"Loading dataset from {data_dir}.")
    dataset, fields = data.load_data_and_fields(data_dir, subs, max_seq_len, vocab_size, rebuild_vocab, file_limit)
    vocab_size = len(fields['text'].vocab.itos)
    comm_vocab_size = len(fields['sub'].vocab.itos)
    comm_unk_idx = fields['sub'].vocab.stoi['<unk>']
    text_pad_idx = fields['text'].vocab.stoi['<pad>']
    click.echo(f"Loaded {len(dataset)} examples.")

    if architecture == 'Transformer':
        lm = model.TransformerLM(vocab_size, comm_vocab_size, nhead, nhid, layers_before, layers_after, community_emsize, dropout)
    elif architecture == 'LSTM':
        lm = model.LSTMLM(vocab_size, comm_vocab_size, nhid, layers_before, layers_after, community_emsize, dropout)

    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
    lm.to(device)

    random.seed(42)
    random_state = random.getstate()
    train_data, val_data, test_data = dataset.split(split_ratio=[0.8,0.1,0.1], stratified=True, strata_field='sub', random_state=random_state)
    click.echo(f"Split {len(train_data)} to train and {len(val_data)} to validation and {len(test_data)} to test.")

    train_iterator = tt.data.BucketIterator(
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
    optimizer = torch.optim.SGD(lm.parameters(), lr=0.01, momentum=0.9)

    val_losses = []
    for epoch in range(epochs):
        click.echo(f'Starting epoch {epoch}')
        lm = train(lm, train_iterator, vocab_size, condition_community, comm_unk_idx, criterion, optimizer)
        torch.save(lm.state_dict(), f'model/{model_filename}-E{epoch:02d}.bin')
        val_loss = evaluate(lm, val_iterator, vocab_size, condition_community, comm_unk_idx, criterion)
        val_loss = sum(val_loss) / len(val_loss)
        val_losses.append(val_loss)
        click.echo(f"Epoch {epoch:3d} | val loss {val_loss:5.2f} | ppl {math.exp(val_loss)}")
        if val_losses[-1] > min(val_losses) and val_losses[-2] > min(val_losses):
            with open(f'model/{model_filename}-val_losses.txt', 'w') as f:
                f.write('\n'.join(map(str, val_losses)))
            click.echo(f'Stopping early after epoch {epoch}.')
            break


if __name__ == '__main__':
    cli()
