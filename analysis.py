import pandas as pd
import os
import numpy as np
import torch
from scipy.stats import pearsonr
from scipy.special import entr
from itertools import combinations
from sklearn.decomposition import PCA

model_dir = 'model/reddit2015'
floats_dir = 'paper/floats'

def add_columns(df, new, suffix=''):
    new = new.rename(lambda x: str(x) + suffix, axis=1)
    index_name = df.index.name
    for col in df.columns: # make the operation idempotent
        if col in new.columns:
            df = df.drop(col, axis=1)
    df = pd.merge(df, new, how='outer', left_index=True, right_index=True, 
            validate='one_to_one', sort=True)
    df.index.name = index_name
    return df

def entropy(x):
    return entr(x).sum()

def ppl(x):
    return np.exp(entropy(x))

def cos_sim(v1, v2):
    return (v1 * v2).sum(axis=0) / (np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0))


##### Create community dataframe (with communities as rows)

comms = torch.load(open(os.path.join(model_dir, 'community.field'), 'rb')).vocab.itos[1:]

df_c = pd.DataFrame([], index=pd.Index(comms, name='community'))
df_cc = pd.DataFrame([],index=pd.MultiIndex.from_tuples(combinations(comms, 2), names=['community1', 'community2']))

# Manually assign communities to different types/subjects
# NOTE this shouldn't really be a partition. Some communities clearly belong to multiple types, but we need a more sophisticated viz for that.
comm_cats = {
    'videogames': ['Warframe', 'eu4', 'GlobalOffensive', 'MaddenUltimateTeam', 'heroesofthestorm', 'EDH', 'KerbalSpaceProgram'],
    'female-focused': ['xxfitness', 'femalefashionadvice', 'TwoXChromosomes', 'AskWomen', 'breakingmom', 'BabyBumps'],
    'sports': ['MMA', 'reddevils', 'CFB', 'MLS'],
    'general-interest': ['Advice', 'relationships', 'LifeProTips', 'explainlikeimfive', 'todayilearned'],
    'technology': ['pcmasterrace', 'techsupport', 'jailbreak', 'oculus']
    # Exclude categories with <3 comms
    # 'gamergate': ['Kappa', 'KotakuInAction'],
    # 'meme': ['justneckbeardthings', 'cringe'],
    # 'photos': ['photography', 'EarthPorn'],
    # 'support': ['stopdrinking', 'exjw'],
    # fitness: ['xxfitness', 'bodybuilding']
}
comm_cats = {c:t for t in comm_cats for c in comm_cats[t]} # assumes one type/community
comm_cats = pd.Series([comm_cats.get(c, 'other') for c in comms], index=comms)
df_c['category'] = comm_cats

# Take note if both communites are the same, different, or if one or both are 'other'
def pair_cats(pair):
    cat1, cat2 = map(dict(comm_cats).get, pair)
    if cat1 == 'other' or cat2 == 'other':
        return 'other'
    elif cat1 == cat2:
        return cat1
    else:
        return 'different'

df_cc['category'] =  pd.Series(map(pair_cats, df_cc.index), index=df_cc.index)


##### Create model dataframe (indexed by model architecture)

cond_lstms = [f'lstm-3-{lc}' for lc in range(4)]
cond_transformers = [f'transformer-3-{lc}' for lc in range(4)]
cond_models = cond_lstms + cond_transformers
models = cond_models + ['lstm-3', 'transformer-3']

df_m = pd.DataFrame([], index=pd.Index(models, name='model'))
df_mm = pd.DataFrame([],index=pd.MultiIndex.from_tuples(combinations(cond_models, 2), names=['model1', 'model2']))

# Best model validation epoch (used for testing)
best_epoch = pd.Series([
    int(open(f'{model_dir}/{model}/saved-epoch.txt').read())
    for model in models], index=models)

df_m['best_epoch'] = best_epoch


##### Language model perplexity (CCLM & baseline)

## Model perplexity on test examples
lm_ppl = pd.read_pickle(os.path.join(model_dir, 'test_ppl.pickle'))

## Mean language model perplexity
df_m['lm_ppl'] = lm_ppl[models].mean()

## Mean language model perplexity by community
df_c = add_columns(df_c, lm_ppl.groupby('community').mean(), suffix='_lm_ppl')


#### Community embeddings

comm_w = {model: np.load(os.path.join(model_dir, f'{model}/comm_embed.npy'))[1:] 
        for model in cond_models}

# PCA plots
for model in cond_models:
    pca = PCA(n_components=2)
    embedding_pca = pca.fit_transform(comm_w[model])
    columns = [f'{model}_pca1', f'{model}_pca2']
    pca = pd.DataFrame(embedding_pca, index=comms, columns=columns)
    df_c = add_columns(df_c, pca)

## Pairwise community similarity

def comm_sim(w, comm_pair):
    comm1, comm2 = map(comms.index, comm_pair)
    return cos_sim(w[comm1], w[comm2])

for model in cond_models:
    df_cc[f'{model}_cos_sim'] = [comm_sim(comm_w[model], comm_pair) for comm_pair in df_cc.index]

from comm_author_embed import load_snap_comm_embed
snap_comm_embed = load_snap_comm_embed(comms)

df_cc['snap_cos_sim'] = [comm_sim(snap_comm_embed, comm_pair) for comm_pair in df_cc.index]

## Model embedding community pair similarity correlation with snap
corr = pd.DataFrame([pearsonr(df_cc[f'{model}_cos_sim'], df_cc['snap_cos_sim'])  
    for model in cond_models], 
    index=cond_models, 
    columns=['snap_cos_sim_corr_r', 'snap_cos_sim_corr_p'])

df_m = add_columns(df_m, corr)


#### LMCC 

## Confusion matrix

def model_comm_confusion_matrix(model):
    """ C[i,j] = average_{Posts(cj)}(P(c=ci|m))"""
    P = pd.read_pickle(os.path.join(model_dir, model, 'comm_probs.pickle'))
    C = P.groupby('actual_comm').mean()
    C = C.T # transpose to (pred_comm, actual_comm), as in the paper
    C.index.name = 'pred_comm'
    C = C.sort_index() # sort the rows alphabetically
    C = C[C.index] # sort the columns alphabetically too
    return C

def confusion_plot_format(C):
    """ Format the matrix with comm1, comm2 index for pgfplots matrix plot """
    comms_alpha = {c: i for i,c in enumerate(C.index)}
    C = C.unstack()
    C = C.reset_index()
    C['actual_comm'] = C['actual_comm'].apply(comms_alpha.get)
    C['pred_comm'] = C['pred_comm'].apply(comms_alpha.get)
    C = C.set_index(['actual_comm', 'pred_comm'])
    return C

C = {model: model_comm_confusion_matrix(model) for model in cond_models}
for model in cond_models:
    C[model].name = model

df_confusion = pd.concat([confusion_plot_format(C[model]) 
    for model in cond_models], axis=1)
df_confusion.columns = cond_models

## Linguistic indiscernibility 
for model in cond_models:
    df_c[f'{model}_lmcc_ppl'] = C[model].apply(ppl)

## Pearson correlation of community-wise LMCC perplexity between pairs of models
df_mm['lmcc_ppl_corr_r'], df_mm['lmcc_ppl_corr_p'] = zip(*[pearsonr(df_c[f'{m1}_lmcc_ppl'], df_c[f'{m2}_lmcc_ppl']) for m1, m2 in df_mm.index])


##### Compare LMCC and CCLM perplexity by community

corr = pd.DataFrame([pearsonr(df_c[f'{model}_lmcc_ppl'], df_c[f'{model}_lm_ppl']) 
    for model in cond_models],
    index=cond_models,
    columns=['lmcc_cclm_corr_r', 'lmcc_cclm_corr_p'])

df_m = add_columns(df_m, corr)

## Commuinty indiscrenibility correlation between two example models
r, p = pearsonr(df_c['lstm-3-1_lmcc_ppl'], df_c['transformer-3-3_lmcc_ppl'])
print(f"r = {r:0.4f} p = {p:0.6f}")
# r = 0.9935 p = 0.000000 


##### Information gain (AKA mutual information)

## Model entropy on test examples
lm_entropy = lm_ppl[models].apply(np.log)
lm_entropy['community'] = lm_ppl['community']

lstm_baseline_ppl = np.exp(lm_entropy['lstm-3'].mean())
lstm_cond_ppl = lm_entropy[cond_lstms].mean().apply(np.exp) 
lstm_info_gain = lstm_cond_ppl.apply(lambda x: lstm_baseline_ppl / x)
transformer_baseline_ppl = np.exp(lm_entropy['transformer-3'].mean())
transformer_cond_ppl = lm_entropy[cond_transformers].mean().apply(np.exp) 
transformer_info_gain = transformer_cond_ppl.apply(lambda x: transformer_baseline_ppl / x)
info_gain = pd.concat([lstm_info_gain, transformer_info_gain], axis=0)

df_m['info_gain'] = info_gain

# Mean information gain by community
lm_entropy_comm = lm_entropy.groupby('community').mean()
lstm_baseline_ppl = np.exp(lm_entropy_comm['lstm-3'])
lstm_cond_ppl = lm_entropy_comm[cond_lstms].apply(np.exp) 
lstm_info_gain = lstm_cond_ppl.apply(lambda x: lstm_baseline_ppl / x)
lm_entropy_comm = lm_entropy.groupby('community').mean()
transformer_baseline_ppl = np.exp(lm_entropy_comm['transformer-3'])
transformer_cond_ppl = lm_entropy_comm[cond_transformers].apply(np.exp) 
transformer_info_gain = transformer_cond_ppl.apply(lambda x: transformer_baseline_ppl / x)
info_gain_comm = pd.concat([lstm_info_gain, transformer_info_gain], axis=1)
df_c = add_columns(df_c, info_gain_comm, suffix='_info_gain')

##### Write tabels and CSVs for plots

df_m[['best_epoch', 'lm_ppl', 'info_gain']].to_latex(os.path.join(floats_dir, 'model_results.tex'),
        formatters={
            'lm_ppl': "{:0.2f}".format,
            'info_gain': "{:0.4f}".format
            },
        na_rep='-'
        )

df_c[
        ['lstm-3_lm_ppl'] + [f'lstm-3-{i}_info_gain' for i in range(4)] +
        ['transformer-3_lm_ppl'] + [f'transformer-3-{i}_info_gain' for i in range(4)] 
    ].to_latex(os.path.join(floats_dir, 'community_results.tex'),
        float_format="{:0.2f}".format)

# model_args = list(map(lambda x: x.split('-'), df_m.index))
# model_arch = map(lambda x: {'lstm': 'LSTM', 'transformer': 'Transformer'}[x[0]], model_args)
# model_lc = [args[-1] if len(args) == 3 else None for args in model_args]
# model_args = pd.MultiIndex.from_tuples(zip(model_arch, model_lc), names=['model', 'lc'])
# df_m.index = model_args

# df_m.to_csv(os.path.join(floats_dir, 'model.csv'), sep='\t', na_rep='nan')
# df_mm.to_csv(os.path.join(floats_dir, 'model_model.csv'), sep='\t', na_rep='nan')
# df_c.to_csv(os.path.join(floats_dir, 'comm.csv'), sep='\t', na_rep='nan')
# df_cc.to_csv(os.path.join(floats_dir, 'comm_comm.csv'), sep='\t', na_rep='nan')
# df_confusion.to_csv(os.path.join(floats_dir, 'confusion.csv'), sep='\t', na_rep='nan')

##### Looking for examples 

# lmcc = pd.read_pickle(os.path.join(model_dir, 'lstm-3-1', 'comm_probs.pickle'))
# lmcc['pred'] =  lmcc[comms].idxmax(axis=1)
# lmcc['prob'] =  lmcc[comms].max(axis=1)
# lmcc['comment'] = cclm_ppl['comment']
# lmcc = lmcc.drop(comms, axis=1)
