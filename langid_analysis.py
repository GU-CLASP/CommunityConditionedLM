import pandas as pd
from pathlib import Path
import data
import numpy as np
import json
import os
from collections import Counter

model_family_dir = Path('model/reddit')
fields = data.load_fields(model_family_dir)
comms = fields['community'].vocab.itos
comms_ = []

langs = {}
for comm in comms:
    try:
        with open(f'data/langid/{comm}.test.txt') as f:
            langs[comm] = Counter([eval(line)[0] for line in f.readlines()])
            if sum(langs[comm].values()) == 0:
                comms_.append(comm)
    except:
        comms_.append(comm)

comms = [c for c in comms if not c in comms_]

english_ratio = pd.Series({c: langs[c]['en'] / sum(langs[c].values()) for c in comms})

english_ratio.mean()

