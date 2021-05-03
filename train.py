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

def train(lm, batches, vocab_size, criterion, optimizer, log):
    lm.train()
    batches.init_epoch()
    train_loss = 0
    for batch_no, batch in enumerate(batches):
        optimizer.zero_grad()
        batch_size_ = len(batch)
        x_comm = batch.community if lm.use_community else None
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

def evaluate(lm, batches, vocab_size, criterion):
    lm.eval()
    batches.init_epoch()
    eval_losses = []
    for batch in batches:
        with torch.no_grad():
            batch_size_ = len(batch)
            text = batch.text
            x_comm = batch.community if lm.use_community else None
            x_text = text[:-1]
            y = text[1:]
            y_hat = lm(x_text, x_comm)
            loss = criterion(y_hat.view(-1, vocab_size), y.view(-1))
            eval_losses += list(loss)
    return eval_losses


@click.command()
@click.argument('architecture', type=click.Choice(['Transformer', 'LSTM'], case_sensitive=False))
@click.argument('model_dir', type=click.Path(exists=True))
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--resume-training/--no-resume-training', default=False)
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
@click.option('--lr', default=0.001)
@click.option('--max-epochs', type=int, default=None)
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(architecture, model_dir, data_dir, resume_training, rebuild_vocab,
        vocab_size, encoder_layers, heads, hidden_size,
        condition_community, community_emsize, community_layer_no, dropout,
        batch_size, max_seq_len, lr, max_epochs, file_limit, gpu_id):

    model_name = f"{architecture.lower()}-{encoder_layers}" + (f"-{community_layer_no}" if condition_community else "")
    save_dir = os.path.join(model_dir, model_name)
    util.mkdir(save_dir)
    log = util.create_logger('train', os.path.join(save_dir, 'training.log'), True)
    log.info(f"Model will be saved to {save_dir}.")

    log.info(f"Loading data from {data_dir}.")
    fields = data.load_fields(model_dir, data_dir, vocab_size)
    train_data = data.load_data(data_dir, fields, 'train', max_seq_len, file_limit)
    vocab_size = len(fields['text'].vocab.itos)
    comm_vocab_size = len(fields['community'].vocab.itos)
    text_pad_idx = fields['text'].vocab.stoi['<pad>']
    dev_data = data.load_data(data_dir, fields, 'dev', max_seq_len, None)
    log.info(f"Loaded {len(train_data)} train and {len(dev_data)} dev examples.")

    if not condition_community:
        community_layer_no = 0
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

    lm = model.CommunityConditionedLM.build_model(
            architecture, heads, hidden_size, vocab_size, 
            condition_community, community_emsize, 
            layers_before, layers_after, comm_vocab_size,
            dropout, save_args_file=os.path.join(save_dir, 'model_args.json'))

    total_params = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    log.info(f"Built model with {total_params} parameters.")
    log.debug(str(lm))

    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
    lm.to(device)

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

    criterion = nn.NLLLoss(ignore_index=text_pad_idx, reduction='none')
    optimizer = torch.optim.AdamW(lm.parameters(), lr=lr)

    if resume_training:
        with open(os.path.join(save_dir, 'saved-epoch.txt'), 'r') as f:
            epoch = int(f.read().strip())
        lm.load_state_dict(torch.load(os.path.join(save_dir, 'model.bin')))
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
        log.info(f'Starting epoch {epoch}')
        lm = train(lm, train_iterator, vocab_size, criterion, optimizer, log)
        val_loss = evaluate(lm, val_iterator, vocab_size, criterion)
        val_loss = sum(val_loss) / len(val_loss)
        val_ppl = math.exp(val_loss)
        if val_ppls == [] or val_ppl < min(val_ppls):
            torch.save(lm.state_dict(), os.path.join(save_dir, 'model.bin'))
            with open(os.path.join(save_dir, 'saved-epoch.txt'), 'w') as f:
                f.write(f'{epoch:03d}')
        val_ppls.append(val_ppl)
        log.info(f"Epoch {epoch:3d} | val loss {val_loss:5.2f} | ppl {val_ppl}")
        if (val_ppls[-1] > min(val_ppls) and val_ppls[-2] > min(val_ppls)) or epoch == max_epochs:
            log.info(f'Stopping early after epoch {epoch}.')
            break


if __name__ == '__main__':
    cli()
