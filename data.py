import os
from collections import OrderedDict, defaultdict, Counter
import torchtext as tt
import torch
import csv
import util

def gen_examples(data_dir, fields, split, max_seq_len, file_limit):
    fields = fields.items()
    for community, line in util.iter_data(data_dir, split, file_limit):
        tokens = line.split(' ')[:max_seq_len]
        yield tt.data.Example.fromlist([community, tokens], fields)

def build_fields(data_dir, vocab_size):

    COMMUNITY = tt.data.Field(sequential=False, pad_token=None, unk_token=None)
    TEXT = tt.data.Field(eos_token='<eos>', init_token='<bos>', tokenize=None)
    fields = OrderedDict([('community', COMMUNITY), ('text', TEXT)])

    data = tt.data.Dataset(list(gen_examples(data_dir, fields, 'train', None, None)), fields=fields)
    fields['community'].build_vocab(data)

    token_counts = Counter()
    for i, example in enumerate(data):
        token_counts.update(set(example.text)) # avoid over-counting spammed tokens
    fields['text'].vocab = tt.vocab.Vocab(token_counts, max_size=vocab_size,
            specials=['<eos>', '<unk>', '<bos>', '<pad>'])

    return fields

def load_fields(field_dir, data_dir=None, vocab_size=None):
    text_field_file = os.path.join(field_dir, 'text.field')
    comm_field_file = os.path.join(field_dir, 'community.field')
    print(text_field_file)
    if os.path.exists(text_field_file):
        print("Using existing fields.")
        fields = OrderedDict([
            ('community', torch.load(comm_field_file)),
            ('text', torch.load(text_field_file))
        ])
    elif (data_dir is not None) and (vocab_size is not None):
        print("Building new fields.")
        fields = build_fields(data_dir, vocab_size)
        torch.save(fields['text'], text_field_file)
        torch.save(fields['community'], comm_field_file)
    else:
        print(f"{text_field_file} does not exist. Must supply data_dir and vocab_size to build.")
        fields = None
    return fields


def load_data(data_dir, fields, split, max_seq_len, file_limit):
    dataset = tt.data.Dataset(list(gen_examples(data_dir, fields, split, max_seq_len, file_limit)), fields=fields)
    return dataset

