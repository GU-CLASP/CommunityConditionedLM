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
import util

def init_model(architecture, encoder_layers, condition_community, community_layer_no,
        vocab_size, comm_vocab_size, hidden_size, community_emsize, dropout, log):

    if community_layer_no > encoder_layers:
        raise ValueError(f"Community layer position cannot be greater than the number of encoder layers.")
    layers_before = community_layer_no
    layers_after = encoder_layers - community_layer_no

    log.info(f"Building {architecture} LM {'with' if condition_community else 'without'} community conditioning.")

    if condition_community:
        log.info(f"Encoder layers before community: {layers_before}")
        log.info(f"Encoder layers after community:  {layers_after}")
    else:
        log.info(f"Encoder layers: {encoder_layers}.")
    log.info(f"Vocab size: {vocab_size}")
    log.info(f"Hidden size: {hidden_size}")
    if architecture == 'Transformer':
        log.info(f"Attention heads: {heads}")

    if architecture == 'Transformer':
        encoder_model = model.TransformerLM
        encoder_args = (vocab_size, heads, hidden_size)
    elif architecture == 'LSTM':
        encoder_model = model.LSTMLM
        encoder_args = (vocab_size, hidden_size)
    encoder_before = encoder_model(*encoder_args, layers_before, dropout) if layers_before > 0 else None
    encoder_after  = encoder_model(*encoder_args, layers_after,  dropout) if layers_after  > 0 else None
    lm = model.CommunityConditionedLM(vocab_size, comm_vocab_size, hidden_size, community_emsize,
            encoder_before, encoder_after, condition_community, dropout)
    total_params = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    log.info(f"Built model with {total_params} parameters.")
    log.debug(str(lm))
    return lm


def train(lm, batches, vocab_size, condition_community, comm_unk_idx, criterion, optimizer, log):
    lm.train()
    batches.init_epoch()
    train_loss = 0
    for batch_no, batch in enumerate(batches):
        optimizer.zero_grad()
        batch_size_ = len(batch)
        x_comm = batch.community if condition_community else None
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
            log.info(f"{batch_no:5d}/{len(batches):5d} batches | loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):0.2f}")
            train_loss = 0
    return lm

def evaluate(lm, batches, vocab_size, condition_community, comm_unk_idx, criterion):
    lm.eval()
    batches.init_epoch()
    eval_losses = []
    for batch in batches:
        with torch.no_grad():
            batch_size_ = len(batch)
            text = batch.text
            x_comm = batch.community if condition_community else None
            x_text = text[:-1]
            y = text[1:]
            y_hat = lm(x_text, x_comm)
            loss = criterion(y_hat.view(-1, vocab_size), y.view(-1))
            eval_losses += list(loss)
    return eval_losses


@click.command()
@click.argument('architecture', type=click.Choice(['Transformer', 'LSTM'], case_sensitive=False))
@click.argument('model_family_dir', type=click.Path(exists=True))
@click.argument('model_name', type=str)
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--rebuild-vocab/--no-rebuild-vocab', default=False)
@click.option('--vocab-size', default=40000)
@click.option('--encoder-layers', default=1)
@click.option('--heads', default=8)
@click.option('--hidden-size', default=256)
@click.option('--condition-community/--no-condition-community', default=True)
@click.option('--community-emsize', default=16)
@click.option('--community-layer-no', default=0)
@click.option('--dropout', default=0.1)
@click.option('--batch-size', default=32)
@click.option('--max-seq-len', default=64)
@click.option('--lr', default=0.01)
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(architecture, model_family_dir, model_name, data_dir, rebuild_vocab,
        vocab_size, encoder_layers, heads, hidden_size,
        condition_community, community_emsize, community_layer_no, dropout,
        batch_size, max_seq_len, lr, file_limit, gpu_id):

    model_dir = os.path.join(model_family_dir, model_name)
    util.mkdir(model_dir)
    log = util.create_logger('train', os.path.join(model_dir, 'training.log'), True)
    log.info(f"Model will be saved to {model_dir}.")

    log.info(f"Loading dataset from {data_dir} files.")
    dataset, fields = data.load_data_and_fields(data_dir, model_family_dir,
            max_seq_len, vocab_size, rebuild_vocab, file_limit)
    vocab_size = len(fields['text'].vocab.itos)
    comm_vocab_size = len(fields['community'].vocab.itos)
    comm_unk_idx = fields['community'].vocab.stoi['<unk>']
    text_pad_idx = fields['text'].vocab.stoi['<pad>']
    log.info(f"Loaded {len(dataset)} examples.")

    lm = init_model(architecture, encoder_layers, condition_community, community_layer_no,
        vocab_size, comm_vocab_size, hidden_size, community_emsize, dropout, log)

    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
    lm.to(device)

    random.seed(42)
    random_state = random.getstate()
    train_data, val_data, test_data = dataset.split(split_ratio=[0.8,0.1,0.1], stratified=True, strata_field='community', random_state=random_state)
    log.info(f"Splits: train: {len(train_data)} val: {len(val_data)} test: {len(test_data)} ")

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
    optimizer = torch.optim.Adam(lm.parameters(), lr=lr)

    val_losses = []
    epoch = 0
    while True:
        epoch += 1
        log.info(f'Starting epoch {epoch}')
        lm = train(lm, train_iterator, vocab_size, condition_community, comm_unk_idx, criterion, optimizer, log)
        val_loss = evaluate(lm, val_iterator, vocab_size, condition_community, comm_unk_idx, criterion)
        val_loss = sum(val_loss) / len(val_loss)
        if epoch == 1 or val_loss < min(val_losses):
            torch.save(lm.state_dict(), os.path.join(model_dir, 'model.bin'))
            with open(os.path.join(model_dir, 'saved-epoch.txt'), 'w') as f:
                f.write(f'{epoch:03d}')
        val_losses.append(val_loss)
        log.info(f"Epoch {epoch:3d} | val loss {val_loss:5.2f} | ppl {math.exp(val_loss):0.2f}")
        if val_losses[-1] > min(val_losses) and val_losses[-2] > min(val_losses):
            log.info(f'Stopping early after epoch {epoch}.')
            break


if __name__ == '__main__':
    cli()
