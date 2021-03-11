import os
from collections import OrderedDict, defaultdict, Counter
import torchtext as tt
import torch
import csv
import util

def create_fields():
    COMMUNITY = tt.data.Field(sequential=False, pad_token=None, unk_token=None)
    TEXT = tt.data.Field(eos_token='<eos>', init_token='<bos>', tokenize=None)
    return  OrderedDict([('community', COMMUNITY), ('text', TEXT)])

def gen_examples(data_dir, fields, max_seq_len, file_limit):
    fields = fields.items()
    for community, line in util.iter_data(data_dir, file_limit):
        tokens = line.split(' ')[:max_seq_len]
        yield tt.data.Example.fromlist([community, tokens], fields)

def build_fields(fields, data, vocab_size):

    fields['community'].build_vocab(data)

    token_counts = Counter()
    for i, example in enumerate(data):
        token_counts.update(set(example.text)) # avoid over-counting spammed tokens
    fields['text'].vocab = tt.vocab.Vocab(token_counts, max_size=vocab_size, 
            specials=['<eos>', '<unk>', '<bos>', '<pad>'])

    return fields

def load_data_and_fields(data_dir, field_dir, max_seq_len, file_limit, vocab_size=None, rebuild_vocab=False):
    text_field_file = os.path.join(field_dir, 'text.field')
    comm_field_file = os.path.join(field_dir, 'community.field')
    if not rebuild_vocab and os.path.exists(text_field_file):
        print("Using existing fields.")
        fields = OrderedDict([
            ('community', torch.load(comm_field_file)),
            ('text', torch.load(text_field_file))
        ])
        dataset = tt.data.Dataset(list(gen_examples(data_dir, fields, max_seq_len, file_limit)), fields=fields)
    else:
        print("Creating new fields.")
        fields = create_fields()
        dataset = tt.data.Dataset(list(gen_examples(data_dir, fields, max_seq_len, file_limit)), fields=fields)
        fields = build_fields(fields, dataset, vocab_size)
        torch.save(fields['text'], text_field_file)
        torch.save(fields['community'], comm_field_file)
    return dataset, fields

