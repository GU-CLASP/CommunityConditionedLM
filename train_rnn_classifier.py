import click
import data
from classifier_model import SequenceClassifier
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
from torch.nn.modules.loss import _WeightedLoss

class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


def train(model, batches, vocab_size, criterion, optimizer, log, fields):
    model.train()
    batches.init_epoch()
    batch_loss, batch_entropy = [], []
    for batch_no, batch in enumerate(batches):
        optimizer.zero_grad()
        batch_size_ = len(batch)
        y = batch.community 
        x, x_lens = batch.text
        # print(y.tolist())
        y_hat = model(x, x_lens)
        loss = criterion(y_hat, y).mean()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        batch_entropy += (-y_hat * y_hat.log()).sum(dim=-1).tolist()
        if batch_no % 1 == 0 and batch_no > 0:
            log.info(f"{batch_no:5d}/{len(batches):5d} batches | avg. batch loss {sum(batch_loss)/len(batch_loss):5.4f} | batch entropy {sum(batch_entropy)/len(batch_entropy):5.4f}")
            batch_loss = []
            batch_entropy = []
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
            x, x_lens = batch.text
            y_hat = model(x, x_lens)
            loss = criterion(y_hat, y)
            eval_losses += loss.tolist()
            num_correct += (torch.max(y_hat, dim=-1)[1] == y).sum().item()
    return eval_losses, num_correct 


@click.command()
@click.argument('model_dir', type=click.Path(exists=True))
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--resume-training/--no-resume-training', default=False)
@click.option('--vocab-size', default=40000)
@click.option('--lower-case/--no-lower-case', default=False)
@click.option('--embedding-size', default=128)
@click.option('--hidden-size', default=32)
# @click.option('--num-layers', default=2)
@click.option('--rnn-type', type=click.Choice(['LSTM', 'Unitary']))
@click.option('--dropout', default=0.1)
@click.option('--batch-size', default=16)
@click.option('--max-seq-len', default=64)
@click.option('--lr', default=0.002)
@click.option('--max-epochs', type=int, default=None)
@click.option('--file-limit', type=int, default=None,
        help="Number of examples per file (community).")
@click.option('--gpu-id', type=int, default=None,
        help="ID of the GPU, if traning with CUDA")
def cli(model_dir, data_dir, resume_training,
        vocab_size, lower_case, embedding_size, hidden_size, dropout, rnn_type,
        batch_size, max_seq_len, lr, max_epochs, file_limit, gpu_id):

    model_dir = Path(model_dir)
    model_name = f"lstm_classifier"

    save_dir = model_dir/model_name
    util.mkdir(save_dir)
    log = util.create_logger('train', save_dir/'training.log', True)
    log.info(f"Model will be saved to {save_dir}.")

    log.info(f"Loading data from {data_dir}.")
    fields = data.load_fields(model_dir, data_dir, vocab_size, 
            lower_case=lower_case, use_eosbos=False)
    fields['text'].include_lengths = True
    lower_case = all(w.lower() == w for w in fields['text'].vocab.itos) # really don't want to fuck this up again...
    log.info(f"Using lower-case vocab: {lower_case!s}.")

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
    model = SequenceClassifier(vocab_size, comm_vocab_size, embedding_size, hidden_size, dropout, 
            seq_encoder=rnn_type, agg_seq='max_pool')

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
        train=True)
    val_iterator = tt.data.BucketIterator(
        dev_data,
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        train=False)

    criterion = nn.CrossEntropyLoss(reduction='none')
    eval_criterion = nn.CrossEntropyLoss(reduction='none')
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
        model = train(model, train_iterator, vocab_size, criterion, optimizer, log, fields)
        log.debug(f'Starting epoch {epoch} validation.')
        val_loss, val_correct = evaluate(model, val_iterator, vocab_size, eval_criterion)
        val_loss = sum(val_loss) / len(val_loss)
        val_acc = val_correct / len(dev_data)
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

