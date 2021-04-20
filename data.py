import os
from collections import OrderedDict, defaultdict, Counter
import torchtext as tt
import torch
import csv
import util

def gen_examples(data_dir, fields, split, max_seq_len, file_limit):
    fields = fields.items()
    for community, text, i in util.iter_data(data_dir, split, file_limit):
        tokens = text.lower().split(' ')[:max_seq_len]
        yield tt.data.Example.fromlist([community, tokens, i], fields)

def build_fields(data_dir, min_freq, file_limit):

    fields = OrderedDict([
        ('community', tt.data.Field(sequential=False, pad_token=None, unk_token=None)),
        ('text', tt.data.Field(eos_token='<eos>', init_token='<bos>', tokenize=None)),
        ('example_id', tt.data.Field(sequential=False, use_vocab=False))
    ])

    data = tt.data.Dataset(list(gen_examples(data_dir, fields, 'train', None, file_limit)),
            fields=fields)
    fields['community'].build_vocab(data)
    comms = fields['community'].vocab.itos
    token_counts_comm = {comm: Counter() for comm in comms}
    token_counts_all = Counter()
    for i, example in enumerate(data):
        token_counts_comm[example.community].update(example.text)
    for comm in comms:
        min_count = min_freq * sum(token_counts_comm[comm].values())
        for w,c in token_counts_comm[comm].items():
            if c >= min_count:
                token_counts_all[w] += c
    fields['text'].vocab = tt.vocab.Vocab(token_counts_all,
            specials=['<eos>', '<unk>', '<bos>', '<pad>'])

    return fields

def load_fields(field_dir, data_dir=None, min_freq=0, file_limit=None):
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
        fields = build_fields(data_dir, min_freq, file_limit)
        torch.save(fields['text'], text_field_file)
        torch.save(fields['community'], comm_field_file)
    else:
        print(f"{text_field_file} does not exist. Must supply data_dir to build vocab.")
        fields = None
    return fields


def load_data(data_dir, fields, split, max_seq_len, file_limit):
    dataset = tt.data.Dataset(list(gen_examples(data_dir, fields, split, max_seq_len, file_limit)), fields=fields)
    return dataset

