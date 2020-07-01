import os
from collections import OrderedDict, defaultdict, Counter
import torchtext as tt
import torch
import csv

def iter_months(year_range=(2017,2018)):
    n_years = year_range[1] - year_range[0]
    n_months = n_years * 12
    for year in range(*year_range):
        for month in range(1,13): # 12 = 13-1 months in a year!
            yield year, month

def iter_comments(data_dir, sub, file_limit):
    for y,m in iter_months():
        with open(data_dir/f"{sub}-{y}-{m:02d}.tokenized.csv", 'r') as f:
            reader = csv.DictReader(f, delimiter='\t', fieldnames=['id','tokenized'])
            items = 0
            for row in reader:
                text = row['tokenized'].lower().strip().split()
                if not text:
                    continue
                yield text
                items += 1
                if items >= file_limit:
                    break

def get_subs(chosen_subs_file, include_excluded=False):
    if not os.path.exists(chosen_subs_file):
        return []
    with open(chosen_subs_file, 'r') as f:
        subs = f.read().strip().split('\n')
    if not include_excluded:
        subs = [sub for sub in subs if not sub.startswith('#')]
    else:
        subs = [sub.lstrip('#') for sub in subs]
    return subs


def create_fields():
    SUB = tt.data.Field(sequential=False, pad_token=None)
    TEXT = tt.data.Field(eos_token='<eos>', tokenize=None)
    return  OrderedDict([('sub', SUB), ('text', TEXT)])

def gen_examples(data_dir, subs, fields, max_seq_len, file_limit):
    fields = fields.items()
    for sub in subs:
        for i, tokens in enumerate(iter_comments(data_dir, sub, file_limit)):
            yield tt.data.Example.fromlist([sub, tokens[:max_seq_len]], fields)

def build_fields(fields, data, subs, vocab_size):
    fields['sub'].vocab = tt.vocab.Vocab(Counter(subs), specials=['<unk>'])
    fields['text'].build_vocab(data, max_size=vocab_size, specials=['<eos>', '<pad>'])
    return fields

def save_fields(fields):
    for field in fields:
        torch.save(fields[field], f"model/{field}.field")

def load_fields():
    fields = ((field, torch.load(f"model/{field}.field")) for field in ('sub', 'text'))
    return OrderedDict(fields)

def load_data_and_fields(data_dir, subs, max_seq_len, vocab_size, rebuild_vocab, file_limit):
    if not rebuild_vocab and os.path.exists('text.field'):
        fields = load_fields()
        dataset = tt.data.Dataset(list(gen_examples(data_dir, subs, fields, max_seq_len, file_limit)), fields=fields)
    else:
        fields = create_fields()
        dataset = tt.data.Dataset(list(gen_examples(data_dir, subs, fields, max_seq_len, file_limit)), fields=fields)
        fields = build_fields(fields, dataset, subs, vocab_size)
        save_fields(fields)
    return dataset, fields

