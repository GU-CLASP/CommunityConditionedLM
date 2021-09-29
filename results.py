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

def compute_confusion(nll_df): # TODO per_word switch
    nll_df = nll_df.sort_values(by=['community', 'example_id'])
    nll = nll_df[comms].values * np.expand_dims(nll_df.length.values, axis=1)# -log(P(m|c)) // shape (m,c)
    P_mc = -nll # -log(P(m|c)) // shape (m,c)
    P_cm = exp_normalize(P_mc, axis=1) # P(c|m) // shape (m,c) 
    P_cm_split = np.array(np.split(P_cm,510,axis=0)) # P(c_i|m_j from c_j) // shape (c_j, m_j, c_i)
    P_cm_split_sum = P_cm_split.sum(axis=1) # sum_mj[P(c_i | m_j from c_j)] // shape (c_j, c_i)
    confusion = P_cm_split_sum / P_cm_split_sum.sum(axis=1) # mean_mj[P(c_i | c_j)] // shape (c_j, c_i)
    return confusion

if __name__ == '__main__':

    results_agg_ppl = []
    results_comm_ppl = []

    uncond_models = ['lstm-3', 'transformer-3', 'unigram' ]
    cond_models = [
            [f'lstm-3-{i}' for i in range(1)], 
            [f'transformer-3-{i}' for i in range(1)],
            ['unigram-cond']
    ]

    for uncond_model_name, cond_models in zip(uncond_models, cond_models): 

        print(uncond_model_name)

        uncond_model_dir = model_family_dir/uncond_model_name

        # load the unconditioned model NLLs
        uncond = pd.read_csv(uncond_model_dir/'nll.csv')\
                .rename(columns={'nll': 'uncond_entr'})\
                .drop('length', 1)

        # repalce community indexes with names
        if not uncond_model_name == 'unigram':
            uncond['community'] = uncond['community'].apply(lambda x:comms[x])
            test_epoch = int(open(uncond_model_dir/'saved-epoch.txt').read())
        else:
            test_epoch = None 

        uncond_ppl = np.exp(uncond.uncond_entr.mean())
        uncond_ppl_by_comm = uncond.groupby('community').uncond_entr.mean().apply(np.exp)
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

            # repalce community indexes with names
            if not uncond_model_name == 'unigram':
                df['community'] = df['community'].apply(lambda x:comms[x])
                test_epoch = int(open(model_dir/'saved-epoch.txt').read())
            else:
                test_epoch = None 

            keys = ['community', 'example_id']
            df = pd.merge(df, uncond, left_on=keys, right_on=keys)

            if not os.path.exists(model_dir/'confusion.csv'):
                # compute the community confusion matrix 
                confusion = pd.DataFrame(compute_confusion(df),
                        index=pd.Index(comms, name='community'), columns=comms)
                confusion.to_csv(model_dir/'confusion.csv')

            # locate PPL for the "true" condition (the community the message was from)
            df['true_cond_entr'] = df.apply(lambda x: x[x['community']], axis=1)
            ppl = np.exp(df.true_cond_entr.mean())

            ppl_by_comm = df.groupby('community').true_cond_entr.mean().apply(np.exp)
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

