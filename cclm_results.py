import pandas as pd
from pathlib import Path
import data
import numpy as np
import json
import os

model_family_dir = Path('model/reddit')
fields = data.load_fields(model_family_dir)
comms = fields['community'].vocab.itos

def exp_normalize(x, axis):
    """ https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/ """
    b = x.max(axis=axis)
    y = np.exp(x - np.expand_dims(b, axis=axis))
    return y / np.expand_dims(y.sum(axis), axis=axis)

def lmcc_probs(nll_df): 
    nll = nll_df[comms].values # -log(P(m|c)) // shape (m,c)
    P_mc = -nll # -log(P(m|c)) // shape (m,c)
    P_cm = exp_normalize(P_mc, axis=1) # P(c|m) // shape (m,c) 
    lmcc_df = pd.DataFrame(P_cm, columns=comms, index=nll_df.index)
    lmcc_df = pd.merge(nll_df[['community', 'example_id']], lmcc_df, left_index=True, right_index=True)
    return lmcc_df

def lmcc_confusion(lmcc_df):
    total_probs = lmcc_df.groupby('community')[comms].sum()
    return total_probs / total_probs.sum(axis=1)

if __name__ == '__main__':

    results_agg_ppl = []
    results_comm_ppl = []

    uncond_models = ['unigram', 'lstm-3', 'transformer-3']
    cond_models = [
            ['unigram-cond'],
            [f'lstm-3-{i}' for i in range(4)], 
            [f'transformer-3-{i}' for i in range(4)]
    ]

    for uncond_model_name, cond_models in zip(uncond_models, cond_models): 

        print(uncond_model_name)

        uncond_model_dir = model_family_dir/uncond_model_name

        # load the unconditioned model NLLs
        uncond = pd.read_csv(uncond_model_dir/'nll.csv')\
                .rename(columns={'nll': 'uncond_entr'})

        if uncond_model_name == 'unigram':
            test_epoch = None
        else:
            test_epoch = int(open(uncond_model_dir/'saved-epoch.txt').read())

        uncond['entr_per_word'] = uncond.uncond_entr / uncond.length
        uncond_ppl = np.exp(uncond.entr_per_word.mean())
        uncond_ppl_by_comm = uncond.groupby('community').entr_per_word.mean().apply(np.exp)
        uncond_ppl_by_comm.name = uncond_model_name


        results_comm_ppl.append(uncond_ppl_by_comm)
        results_agg_ppl.append({
                'model': uncond_model_name,
                'test epoch': test_epoch,
                'Ppl': uncond_ppl,
                'IG': None
            })

        for model_name in cond_models:

            model_dir = model_family_dir/model_name
            print(model_name)

            # load conditioned NLLs (including off-community conditioning)
            df = pd.read_csv(model_dir/'nll.csv')

            if uncond_model_name == 'unigram':
                test_epoch = None
            else:
                test_epoch = int(open(model_dir/'saved-epoch.txt').read())

            keys = ['community', 'example_id']
            df = pd.merge(df, uncond.drop('length', axis=1), 
                    left_on=keys, right_on=keys, validate='one_to_one')


            # if not os.path.exists(model_dir/'confusion.csv'):
            # compute LMCC probabilities 
            lmcc = pd.DataFrame(lmcc_probs(df))
                    # index=pd.Index(comms, name='community'), columns=comms)
            lmcc.to_csv(model_dir/'lmcc_probs.csv', index=False)
            lmcc_confusion(lmcc).to_csv(model_dir/'confusion.csv')

            # locate PPL for the "true" condition (the community the message was from)
            df['true_cond_entr_per_word'] = df.apply(lambda x: x[x['community']], axis=1) / df.length
            ppl = np.exp(df.true_cond_entr_per_word.mean())
            ppl_by_comm = df.groupby('community').true_cond_entr_per_word.mean().apply(np.exp)
            ppl_by_comm.name = model_name
            results_comm_ppl.append(ppl_by_comm)


            results_agg_ppl.append({
                    'model': model_name,
                    'test epoch': test_epoch,
                    'Ppl': ppl,
                    'IG': uncond_ppl / ppl
                })

    pd.concat(results_comm_ppl, axis=1).to_csv(model_family_dir/'ppl_by_comm.csv')
    pd.DataFrame(results_agg_ppl).to_csv(model_family_dir/'ppl_aggregate.csv', index=False)

