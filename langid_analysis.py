import pandas as pd
from pathlib import Path
import data
import numpy as np
import json
import os
from collections import Counter

mode_dir = Path('model/reddit')
fields = data.load_fields(model_family_dir)
comms = fields['community'].vocab.itos

langs = {}
for comm in comms:
    with open(f'data/langid/{comm}.test.txt') as f:
        langs[comm] = Counter([eval(line)[0] for line in f.readlines()])

english_ratio = pd.Series({c: langs[c]['en'] / sum(langs[c].values()) for c in comms})

print(f"ratio English/non-English messages: {english_ratio.mean():0.2f}")
print(f"number of majority English communities: {sum(english_ratio > 0.5)}")
