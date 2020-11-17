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

cclm_ppl = pd.read_pickle(os.path.join(model_dir, 'test_ppl.pickle'))

##### BEST EPOCH

def get_best_epoch(model_name):
    return int(open(f'{model_dir}/{model_name}/saved-epoch.txt').read())

best_epoch = pd.DataFrame([{str(lc): get_best_epoch(model_params_to_name(arch, lc))
    for lc in lcs} for arch in architectures], index=architectures)
best_epoch.to_latex(os.path.join(floats_dir, 'best_epoch.tex'))


##### Perplexity by model


ppl_mean = cclm_ppl[models].mean()
ppl_mean.index = ppl_mean.index.map(model_params_from_name)

ppl_mean_comm = cclm_ppl.groupby('community').mean().sort_values('lstm-3')
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

#### LMCC confusion matrix

from comm_author_embed import load_snap_comm_embed
from itertools import combinations

comms_alpha = sorted(list(comms))

def model_comm_confusion_matrix(model_name):
    """ C[i,j] = average_{Posts(cj)}(P(c=ci|m))"""
    P = pd.read_pickle(os.path.join(model_dir, model_name), 'comm_probs.pickle')
    C = P.groupby('actual_comm').mean()
    C = C.T # transpose to (prob assigned, actual comm), as in the paper
    C = C.sort_index() # sort the rows alphabetically
    C = C[C.index] # sort the columns alphabetically too
    return C

C = {model_name: model_comm_confusion_matrix(model_name) for model_name in conditioned_models}

for model_name in conditioned_models:
    Cm = C[model_name]
    Cm = Cm.unstack().reset_index()
    Cm = Cm.rename(columns={'level_1': 'confered_to', 0: 'avg_prob'})
    Cm['actual_comm'] = Cm['actual_comm'].apply(comms_alpha.index)
    Cm['confered_to'] = Cm['confered_to'].apply(comms_alpha.index)
    Cm.to_csv(os.path.join(floats_dir, f'{model_name}_comm_infer_confusion.csv'), sep='\t', float_format="% 5.4f", index=False)

I = [] # Linguistic indiscernibility
for model_name in conditioned_models:
    Im = C[model_name].apply(ppl)
    Im.name = model_name
    I.append(Im)
I = pd.concat(I, axis=1)

for m1, m2 in combinations(conditioned_models, 2):
    r, p = pearsonr(I[m1], I[m2])
    print(f"{m1:<15} {m2:<15} {r:0.3f} {p:0.5f}")

# lstm-3-0        lstm-3-1        0.997 0.00000
# lstm-3-0        lstm-3-2        0.995 0.00000
# lstm-3-0        lstm-3-3        0.990 0.00000
# lstm-3-0        transformer-3-0 0.988 0.00000
# lstm-3-0        transformer-3-1 0.995 0.00000
# lstm-3-0        transformer-3-2 0.994 0.00000
# lstm-3-0        transformer-3-3 0.994 0.00000
# lstm-3-1        lstm-3-2        0.998 0.00000
# lstm-3-1        lstm-3-3        0.994 0.00000
# lstm-3-1        transformer-3-0 0.988 0.00000
# lstm-3-1        transformer-3-1 0.991 0.00000
# lstm-3-1        transformer-3-2 0.992 0.00000
# lstm-3-1        transformer-3-3 0.994 0.00000
# lstm-3-2        lstm-3-3        0.995 0.00000
# lstm-3-2        transformer-3-0 0.987 0.00000
# lstm-3-2        transformer-3-1 0.990 0.00000
# lstm-3-2        transformer-3-2 0.992 0.00000
# lstm-3-2        transformer-3-3 0.995 0.00000
# lstm-3-3        transformer-3-0 0.990 0.00000
# lstm-3-3        transformer-3-1 0.988 0.00000
# lstm-3-3        transformer-3-2 0.984 0.00000
# lstm-3-3        transformer-3-3 0.997 0.00000
# transformer-3-0 transformer-3-1 0.990 0.00000
# transformer-3-0 transformer-3-2 0.985 0.00000
# transformer-3-0 transformer-3-3 0.989 0.00000
# transformer-3-1 transformer-3-2 0.989 0.00000
# transformer-3-1 transformer-3-3 0.991 0.00000
# transformer-3-2 transformer-3-3 0.990 0.00000

#### Pairwise community similarity scatter

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
    df = df.pivot(index='arch', columns='c')
    df.index.name = None
    return df

pivot_model_table(df_r)['r'].to_latex(os.path.join(floats_dir, 'comm_sim.tex'),
        float_format="%.2f", multicolumn_format='c')


##### Compare LMCC and CCLM perplexity by community

from scipy.special import entr

def entropy(x):
    return entr(x).sum()

def ppl(x):
    return np.exp(entropy(x))

lmcc_ppl_mean = I
cclm_ppl_mean = cclm_ppl.groupby('community').mean()

df = pd.merge(cclm_ppl_mean, lmcc_ppl_mean, right_index=True, left_index=True, suffixes=['_cclm', '_lmcc'])
df.to_csv(os.path.join(floats_dir, f'cclm_lmcc_ppl.csv'), sep='\t', 
            float_format="% 5.4f", index=True)

cclm_lmcc_ppl_r = {}
for model in conditioned_models:
    r, p = pearsonr(cclm_ppl_mean[model], lmcc_ppl_mean[model])
    print(f"{model:<15} r = {r:0.2f} p = {p:0.4f}")
    cclm_lmcc_ppl_r[model] = ({'r': r,'p': p})

cclm_lmcc_ppl_r = pd.DataFrame(cclm_lmcc_ppl_r)
cclm_lmcc_ppl_r = cclm_lmcc_ppl_r.T
cclm_lmcc_ppl_r.index = pd.MultiIndex.from_tuples(pd.Series(cclm_lmcc_ppl_r.index).apply(model_params_from_name))
cclm_lmcc_ppl_r.index = cclm_lmcc_ppl_r.index.set_names('c', level=1)

cclm_lmcc_ppl_r.to_latex(os.path.join(floats_dir, 'cclm_lmcc_ppl.tex'), 
        formatters=["{:0.2f}".format, "{:0.4f}".format])

# lstm-3-0        r = 0.21 p = 0.1546
# lstm-3-1        r = 0.16 p = 0.2991
# lstm-3-2        r = 0.15 p = 0.3245
# lstm-3-3        r = 0.09 p = 0.5430
# transformer-3-0 r = 0.14 p = 0.3434
# transformer-3-1 r = 0.20 p = 0.1808
# transformer-3-2 r = 0.22 p = 0.1411
# transformer-3-3 r = 0.17 p = 0.2555

r, p = pearsonr(lmcc_mean_entr['lstm-3-1'], lmcc_mean_entr['transformer-3-3'])
print(f"r = {r:0.4f} p = {p:0.6f}")
# r = 0.9935 p = 0.000000 

##### Looking for examples 

lmcc = pd.read_pickle(os.path.join(model_dir, 'lstm-3-1', 'comm_probs.pickle'))
lmcc['pred'] =  lmcc[comms].idxmax(axis=1)
lmcc['prob'] =  lmcc[comms].max(axis=1)
lmcc['comment'] = cclm_ppl['comment']
lmcc = lmcc.drop(comms, axis=1)


