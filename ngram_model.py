import os
from torchtext.data.utils import ngrams_iterator
from collections import Counter, defaultdict

def bigrams_iter(tokens):
    return zip(*[tokens[i:] for i in range(2)])
