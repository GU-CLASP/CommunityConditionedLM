import pandas as pd
import os
import numpy as np
import torch
from scipy.stats import pearsonr


model_dir = 'model/reddit2015'
floats_dir = 'paper/floats'

lcs = [None, 0, 1, 2, 3]
architectures = ['LSTM', 'Transformer']
lcs_str = [str(i) for i in lcs]

def model_params_to_name(encoder_arch, lc):
    if encoder_arch == "Transformer":
        prefix = 'transformer'
    elif encoder_arch == "LSTM":
        prefix = 'lstm'
    if lc is None:
        return f"{prefix}-3"
    else:
        return f"{prefix}-3-{lc}"

def model_params_from_name(model_name):
    if model_name.startswith('transformer'):
        encoder_arch = 'Transformer' 
    elif model_name.startswith('lstm'):
        encoder_arch = 'LSTM'
    layers_str = model_name[len(encoder_arch):]
    if layers_str == '-3':
        lc = 'None'
    else:
        lc = layers_str[-1]
    return encoder_arch, lc 

models = [model_params_to_name(arch, lc) for arch in architectures for lc in lcs] 
conditioned_models = [model_params_to_name(arch, lc) for arch in architectures for lc in lcs if lc is not None]

comms = torch.load(open(os.path.join(model_dir, 'community.field'), 'rb')).vocab.itos[1:]
comms = pd.Series(comms, name='community')


cclm_comm_embed = {model: np.load(os.path.join(model_dir, f'{model}/comm_embed.npy'))[1:]
        for model in conditioned_models}

##### BEST EPOCH

def get_best_epoch(model_name):
    return int(open(f'{model_dir}/{model_name}/saved-epoch.txt').read())

best_epoch = pd.DataFrame([{str(lc): get_best_epoch(model_params_to_name(arch, lc))
    for lc in lcs} for arch in architectures], index=architectures)
best_epoch.to_latex(os.path.join(floats_dir, 'best_epoch.tex'))


##### Perplexity by model

ppl = pd.read_pickle(os.path.join(model_dir, 'test_ppl.pickle'))

ppl_mean = ppl[models].mean()
ppl_mean.index = ppl_mean.index.map(model_params_from_name)

ppl_mean_comm = ppl.groupby('community').mean().sort_values('lstm-3')
ppl_mean_comm.columns = ppl_mean_comm.columns.map(model_params_from_name)
ppl_mean_comm.loc['Mean'] = ppl_mean_comm.mean()
rows = ['Mean'] + [c for c in ppl_mean_comm.index if not c == 'Mean']
ppl_mean_comm = ppl_mean_comm.reindex(rows)
ppl_mean_comm.to_latex(os.path.join(floats_dir, 'model_ppls.tex'), 
        float_format="%.2f", multicolumn_format='c')


##### Community embedding PCA

from sklearn.decomposition import PCA

# Manually assign communities to different types/subjects
# NOTE this shouldn't really be a partition. Some communities clearly belong to multiple types,
# but we need a more sophisticated viz for that.
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

def pca(w):
    pca = PCA(n_components=2)
    return pca.fit_transform(w)

def write_pca_table(model_name):
    comm_embed = cclm_comm_embed[model_name]
    comm = pd.DataFrame(pca(comm_embed), index=comms, columns=['pca1', 'pca2'])
    comm['category'] = comm_cats
    comm = comm.reset_index()[['pca1', 'pca2', 'community', 'category']]
    comm.to_csv(os.path.join(floats_dir, f'{model_name}_pca.csv'), sep='\t', 
            float_format="% 5.4f", index=False)
    return comm

for model in conditioned_models:
    write_pca_table(model)


#### Community inference confusion matrix

from comm_author_embed import load_snap_comm_embed

comms_alpha = sorted(list(comms))

def model_comm_confusion_matrix(model_name):
    """ C[i,j] = average_{Posts(cj)}(P(c=ci|m))"""
    P = pd.read_pickle(os.path.join(os.path.join(model_dir, model_name), 'comm_probs.pickle'))
    C = P.groupby('actual_comm').mean()
    C = C.T # transpose to (prob assigned, actual comm), as in the paper
    C = C.sort_index() # sort the rows alphabetically
    C = C[C.index] # sort the columns alphabetically too
    C = C.unstack().reset_index()
    C = C.rename(columns={'level_1': 'confered_to', 0: 'avg_prob'})
    C['actual_comm'] = C['actual_comm'].apply(comms_alpha.index)
    C['confered_to'] = C['confered_to'].apply(comms_alpha.index)
    return C

for model_name in conditioned_models:
    C = model_comm_confusion_matrix(model_name)
    C.to_csv(os.path.join(floats_dir, f'{model_name}_comm_infer_confusion.csv'), sep='\t', float_format="% 5.4f", index=False)

#### Pairwise community similarity scatter

from itertools import combinations

def cos_sim(v1, v2):
    return (v1 * v2).sum(axis=0) / (np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0))

def pairwise_sims(embedding, pairs):
    return pd.Series([cos_sim(*map(lambda x: embedding[list(comms).index(x)], pair)) for pair in pairs], index=pairs)
    
df = pd.DataFrame([], index=pairs)
for model in conditioned_models:
    df[model] = pairwise_sims(cclm_comm_embed[model], pairs)

snap_comm_embed = load_snap_comm_embed(comms)

df['snap'] = pairwise_sims(snap_comm_embed, pairs)

def pair_cats(pair):
    cat1, cat2 = map(dict(comm_cats).get, pair)
    if cat1 == 'other' or cat2 == 'other':
        return 'other'
    elif cat1 == cat2:
        return cat1
    else:
        return 'different'

df['meta'] = pd.Series(map(pair_cats, pairs), index=pairs)


pairs_str = pd.Series(df.index).apply(lambda x: f'{x[0]}/{x[1]}')
pairs_str.index=pairs
df['pair'] = pairs_str

df.to_csv(os.path.join(floats_dir, f'comm_sim.csv'), sep='\t', 
            float_format="% 5.4f", index=False)

df_r = pd.DataFrame([pearsonr(df[model], df['snap']) 
    for model in conditioned_models], index=conditioned_models,
    columns = ['r', 'p'])

def pivot_model_table(df):
    df = df.reset_index()
    df['arch'] = df['index'].apply(lambda x: x.split('-')[0])
    df['arch'] = df['arch'].apply({'lstm': 'LSTM', 'transformer': 'Transformer'}.get)
    df['c'] = df['index'].apply(lambda x: x.split('-')[-1])
    df = df.drop('index', axis=1)
    return df.pivot(index='arch', columns='c')

pivot_model_table(df_r)['r'].to_latex(os.path.join(floats_dir, 'comm_sim.tex')
        float_format="%.2f", multicolumn_format='c')
