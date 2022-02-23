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
from IPython import embed


def train(model, batches, vocab_size, criterion, optimizer, log):
    model.train()
    batches.init_epoch()
    train_loss = 0
    for batch_no, batch in enumerate(batches):
        optimizer.zero_grad()
        batch_size_ = len(batch)
        y = batch.community 
        x = batch.text
        y_hat = model(x)
        loss = criterion(y_hat, y).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_no % 1000 == 0 and batch_no > 0:
            cur_loss = train_loss / 1000
            log.info(f"{batch_no:5d}/{len(batches):5d} batches | loss {cur_loss:5.2f}")
            train_loss = 0
    return model

def evaluate(model, batches, vocab_size, criterion):
    model.eval()
    batches.init_epoch()
    eval_losses = []
    num_correct = 0
    for i, batch in enumerate(batches):
        with torch.no_grad():
            batch_size_ = len(batch)
            y = batch.community 
            x = batch.text
            y_hat = model(x)
            loss = criterion(y_hat, y).mean()
            eval_losses += [loss.item()]
            num_correct += (torch.max(y_hat, dim=-1)[1] == y).sum().item()
    eval_acc = num_correct / (i+1)
    return eval_losses, eval_acc


@click.command()
@click.argument('model_dir', type=click.Path(exists=True))
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--resume-training/--no-resume-training', default=False)
@click.option('--rebuild-vocab/--no-rebuild-vocab', default=False)
@click.option('--vocab-size', default=40000)
@click.option('--lower-case/--no-lower-case', default=False)
@click.option('--hidden-size', default=512)
@click.option('--dropout', default=0.1)
@click.option('--batch-size', default=128)
@click.option('--max-seq-len', default=64)
@click.option('--lr', default=0.001)
@click.option('--max-epochs', type=int, default=None)
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(model_dir, data_dir, resume_training, rebuild_vocab,
        vocab_size, lower_case, hidden_size, dropout,
        batch_size, max_seq_len, lr, max_epochs, file_limit, gpu_id):

    model_dir = Path(model_dir)
    model_name = f"lstm_classifier"

    save_dir = model_dir/model_name
    util.mkdir(save_dir)
    log = util.create_logger('train', save_dir/'training.log', True)
    log.info(f"Model will be saved to {save_dir}.")

    log.info(f"Loading data from {data_dir}.")
    fields = data.load_fields(model_dir, data_dir, vocab_size)
    train_data = data.load_data(data_dir, fields, 'train',
            max_seq_len, file_limit, lower_case)
    vocab_size = len(fields['text'].vocab.itos)
    comm_vocab_size = len(fields['community'].vocab.itos)
    text_pad_idx = fields['text'].vocab.stoi['<pad>']
    dev_data = data.load_data(data_dir, fields, 'dev',
            max_seq_len, file_limit, lower_case)
    log.info(f"Loaded {len(train_data)} train and {len(dev_data)} dev examples.")

    log.info(f"Vocab size: {vocab_size}")
    log.info(f"Hidden size: {hidden_size}")

    model = LSTMClassifier(vocab_size, comm_vocab_size, hidden_size, dropout)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Built model with {total_params} parameters.")
    log.debug(str(model))

    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
    model.to(device)

    train_iterator = tt.data.BucketIterator(
        train_data,
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        shuffle=True,
        train=True)
    val_iterator = tt.data.BucketIterator(
        dev_data,
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        shuffle=False,
        train=False)

    criterion = nn.CrossEntropyLoss(ignore_index=text_pad_idx, reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if resume_training and os.path.exists(save_dir/'saved-epoch.txt'):
        with open(save_dir/'saved-epoch.txt', 'r') as f:
            epoch = int(f.read().strip())
        model.load_state_dict(torch.load(save_dir/'model.bin'))
        log.info(f"Resuming trainng after epoch {epoch}.")
        val_ppls = util.read_logged_val_ppls(save_dir)
        saved_val_ppl = val_ppls[epoch-1]
        if val_ppls[-1] > saved_val_ppl and val_ppls[-2] > saved_val_ppl:
            log.info("Training is already finished.")
            exit()
        val_ppls = val_ppls[:epoch]
    else:
        epoch = 0
        val_ppls = []

    while True:
        epoch += 1
        log.debug(f'Starting epoch {epoch} training.')
        model = train(model, train_iterator, vocab_size, criterion, optimizer, log)
        log.debug(f'Starting epoch {epoch} validation.')
        val_loss, val_acc = evaluate(model, val_iterator, vocab_size, criterion)
        val_loss = sum(val_loss) / len(val_loss)
        val_ppl = math.exp(val_loss)
        if val_ppls == [] or val_ppl < min(val_ppls):
            log.debug(f'Saving epoch {epoch} model.')
            torch.save(model.state_dict(), save_dir/'model.bin')
            with open(save_dir/'saved-epoch.txt', 'w') as f:
                f.write(f'{epoch:03d}')
        val_ppls.append(val_ppl)
        log.info(f"Epoch {epoch:3d} | val loss {val_loss:5.2f} | ppl {val_ppl:7.2f} | acc {val_acc:5.2f}")
        if (val_ppls[-1] > min(val_ppls) and val_ppls[-2] > min(val_ppls)) or epoch == max_epochs:
            log.info(f'Stopping early after epoch {epoch}.')
            break


if __name__ == '__main__':
    cli()

