import pandas as pd
from pathlib import Path
import data
import numpy as np
import json

def exp_normalize(x, axis):
    """ https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/ """
    b = x.max(axis=axis)
    y = np.exp(x - b)
    return y / y.sum(axis)

results_agg_ppl = []
results_comm_ppl = []

model_family_dir = Path('model/reddit')
fields = data.load_fields(model_family_dir)
comms = fields['community'].vocab.itos

for architecture in ('lstm', 'transformer'):

    uncond_model_name = f'{architecture}-3'
    uncond_model_dir = model_family_dir/uncond_model_name

    # load the unconditioned model NLLs
    uncond = pd.read_csv(uncond_model_dir/'nll.csv')\
            .rename(columns={'nll': 'uncond_entr'})\
            .drop('length', 1)

    # repalce community indexes with names
    uncond['community'] = uncond['community'].apply(lambda x:comms[x])

    uncond_ppl = np.exp(uncond.uncond_entr.mean())
    uncond_ppl_by_comm = uncond.groupby('community').uncond_entr.mean().apply(np.exp)
    uncond_ppl_by_comm.name = uncond_model_name

    results_comm_ppl.append(uncond_ppl_by_comm)
    results_agg_ppl.append({
            'model': uncond_model_name,
            'test epoch': int(open(uncond_model_dir/'saved-epoch.txt').read()),
            'Ppl': uncond_ppl,
            'IG': None
        })

    for lc in range(4):

        model_name = f'{architecture}-3-{lc}'
        model_dir = model_family_dir/model_name
        print(model_name)

        # load conditioned NLLs (including off-community conditioning)
        df = pd.read_csv(model_dir/'nll.csv')
        # not sure why, but one row had missing values in lstm-3-0. possibly csv i/o problem?
        # (example_id and length, unfortunately, so hard to replace...)
        # drop it, but ensure it really is a limited problem

        n = len(df)
        df.dropna(inplace=True)
        print(f"Dropped {n - len(df)} examples")
        df.example_id = df.example_id.astype(int)
        df.length = df.length.astype(int)
        # repalce community indexes with names
        df['community'] = df['community'].apply(lambda x:comms[x])
        keys = ['community', 'example_id']
        df = pd.merge(df, uncond, left_on=keys, right_on=keys)

        # locate PPL for the "true" condition (the community the message was from)
        df['true_cond_entr'] = df.apply(lambda x: x[x['community']], axis=1)
        ppl = np.exp(df.true_cond_entr.mean())

        ppl_by_comm = df.groupby('community').true_cond_entr.mean().apply(np.exp)
        ppl_by_comm.name = model_name
        results_comm_ppl.append(ppl_by_comm)

        results_agg_ppl.append({
                'model': model_name,
                'test epoch': int(open(model_dir/'saved-epoch.txt').read()),
                'Ppl': ppl,
                'IG': uncond_ppl / ppl
            })

pd.concat(results_comm_ppl, axis=1).to_csv(model_family_dir/'ppl_by_comm.csv')
pd.DataFrame(results_agg_ppl).to_csv(model_family_dir/'ppl_aggregate.csv', index=False)


P_cm = exp_normalize(df[comms].to_numpy().T, axis=0).T

df[comms] = P_cm

df.groupby('community')[comms].sum(axis=0)



