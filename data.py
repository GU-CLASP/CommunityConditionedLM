import os
from collections import OrderedDict, defaultdict, Counter
import torchtext as tt
import torch
import csv
import util

def get_community_label(file_path):
    base=os.path.basename(file_path)
    return os.path.splitext(base)[0]

def create_fields():
    SUB = tt.data.Field(sequential=False, pad_token=None)
    TEXT = tt.data.Field(eos_token='<eos>', tokenize=None)
    return  OrderedDict([('community', SUB), ('text', TEXT)])

def gen_examples(data_dir, fields, max_seq_len, file_limit):
    fields = fields.items()
    for community, line in util.iter_data(data_dir, file_limit):
        tokens = line.split(' ')[:max_seq_len]
        yield tt.data.Example.fromlist([community, tokens], fields)

def build_fields(fields, data, vocab_size):
    token_counts = Counter()
    for example in data:
        token_counts.update(set(example.text)) # avoid over-counting spammed tokens
    fields['community'].vocab = tt.vocab.Vocab(token_counts, specials=['unk'])
    fields['text'].build_vocab(data, max_size=vocab_size, specials=['<eos>', '<pad>'])
    return fields

def save_fields(fields):
    for field in fields:
        torch.save(fields[field], f"model/{field}.field")

def load_fields():
    fields = ((field, torch.load(f"model/{field}.field")) for field in ('community', 'text'))
    return OrderedDict(fields)

def load_data_and_fields(data_dir, max_seq_len, vocab_size, rebuild_vocab, file_limit):
    if not rebuild_vocab and os.path.exists('model/text.field'):
        print("Using existing fields.")
        fields = load_fields()
        dataset = tt.data.Dataset(list(gen_examples(data_dir, fields, max_seq_len, file_limit)), fields=fields)
    else:
        print("Creating new fields.")
        fields = create_fields()
        dataset = tt.data.Dataset(list(gen_examples(data_dir, fields, max_seq_len, file_limit)), fields=fields)
        fields = build_fields(fields, dataset, vocab_size)
        save_fields(fields)
    return dataset, fields

