import os
import click
import data
from train import init_model, evaluate
import model
import torch
import torch.nn as nn
import torchtext as tt
import torch.nn.functional as F
import math
import random
import util

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
            print(lm.comm_inference.weight)
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            log.info(f"{batch_no:5d}/{len(batches):5d} batches | loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):0.2f} | lr {lr}")
            train_loss = 0
    return lm

@click.command()
@click.argument('architecture', type=click.Choice(['Transformer', 'LSTM'], case_sensitive=False))
@click.argument('model_filename', type=click.Path(exists=False))
@click.argument('data_dir', type=click.Path(exists=True))
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
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(architecture, model_filename, data_dir, vocab_size, encoder_layers, heads, hidden_size,
        condition_community, community_emsize, community_layer_no, dropout,
        batch_size, max_seq_len, file_limit, gpu_id):

    log = util.create_logger('tune', f"model/{model_filename}_tuning.log", True)

    log.info(f"Loading dataset from {data_dir} files.")
    dataset, fields = data.load_data_and_fields(data_dir, max_seq_len, vocab_size, rebuild_vocab, file_limit)
    vocab_size = len(fields['text'].vocab.itos)
    comm_vocab_size = len(fields['community'].vocab.itos)
    comm_unk_idx = fields['community'].vocab.stoi['<unk>']
    text_pad_idx = fields['text'].vocab.stoi['<pad>']
    log.info(f"Loaded {len(dataset)} examples.")

    random.seed(42)
    random_state = random.getstate()
    _, _, tune_data = dataset.split(split_ratio=[0.8,0.1,0.1], stratified=True, strata_field='community', random_state=random_state)
    tune_train_data, tune_val_data, tune_test_data = tune_data.split(split_ratio=[0.8,0.1,0.1], stratified=True, strata_field='community', random_state=random_state)

    log.info(f"Split {len(tune_train_data)} to train and {len(tune_val_data)} to validation and {len(tune_test_data)} to test.")

    log.info(f"Model loading model {model_filename}.")
    lm = init_model(architecture, encoder_layers, condition_community, community_layer_no,
        vocab_size, comm_vocab_size, hidden_size, community_emsize, dropout, log)

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
        tune_train_data,
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        train=True)
    val_iterator = tt.data.BucketIterator(
        tune_val_data,
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        train=False)

    criterion = nn.NLLLoss(ignore_index=text_pad_idx, reduction='none')
    optimizer = torch.optim.Adam(lm.parameters(), lr=0.01)

    val_losses = []
    epoch = 0
    while True:
        epoch += 1
        log.info(f'Starting epoch {epoch}')
        lm = tune(lm, tune_iterator, vocab_size, criterion, optimizer, log)
        val_loss = evaluate(lm, val_iterator, vocab_size, condition_community, comm_unk_idx, criterion)
        val_loss = sum(val_loss) / len(val_loss)
        if epoch == 1 or val_loss < min(val_losses):
            torch.save(lm.state_dict(), f'model/{model_filename}_tuned.bin')
            with open(f'model/{model_filename}_saved-epoch_tuned.txt', 'w') as f:
                f.write(f'{epoch:03d}')
        val_losses.append(val_loss)
        log.info(f"Epoch {epoch:3d} | val loss {val_loss:5.2f} | ppl {math.exp(val_loss):0.2f}")
        if val_losses[-1] > min(val_losses) and val_losses[-2] > min(val_losses):
            log.info(f'Stopping early after epoch {epoch}.')
            break

if __name__ == '__main__':
    cli()
