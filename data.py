import os
from collections import OrderedDict, defaultdict, Counter
import torchtext as tt
import torch
import csv
import util

def gen_examples(data_dir, fields, split, max_seq_len, file_limit, lower_case):
    fields = fields.items()
    for community, text, i in util.iter_data(data_dir, split, file_limit):
        if lower_case:
            text = text.lower()
        tokens = text.split(' ')[:max_seq_len]
        yield tt.data.Example.fromlist([community, tokens, i], fields)

def build_fields(data_dir, vocab_size):

    fields = OrderedDict([
        ('community', tt.data.Field(sequential=False, pad_token=None, unk_token=None)),
        ('text', tt.data.Field(eos_token='<eos>', init_token='<bos>', tokenize=None)),
        ('example_id', tt.data.Field(sequential=False, use_vocab=False))
    ])

    data = tt.data.Dataset(list(gen_examples(data_dir, fields, 'train', None, None)),
            fields=fields)
    fields['community'].build_vocab(data)
    token_counts = Counter()
    for i, example in enumerate(data):
        token_counts.update(set(example.text)) # avoid over-counting spammed tokens
    fields['text'].vocab = tt.vocab.Vocab(token_counts, max_size=vocab_size,
            specials=['<eos>', '<unk>', '<bos>', '<pad>'])
    return fields

def load_fields(field_dir, data_dir=None, vocab_size):
    text_field_file = os.path.join(field_dir, 'text.field')
    comm_field_file = os.path.join(field_dir, 'community.field')
    print(text_field_file)
    if os.path.exists(text_field_file):
        print("Using existing fields.")
        fields = OrderedDict([
            ('community', torch.load(comm_field_file)),
            ('text', torch.load(text_field_file)),
            ('example_id', tt.data.Field(sequential=False, use_vocab=False))
        ])
    elif (data_dir is not None):
        print("Building new fields.")
        fields = build_fields(data_dir, vocab_size)
        torch.save(fields['text'], text_field_file)
        torch.save(fields['community'], comm_field_file)
    else:
        print(f"{text_field_file} does not exist. Must supply data_dir to build vocab.")
        fields = None
    return fields


def load_data(data_dir, fields, split, max_seq_len, file_limit, lower_case=True):
    dataset = tt.data.Dataset(list(gen_examples(data_dir, fields, split, max_seq_len, file_limit, lower_case)), fields=fields)
    return dataset

