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
    df = df.pivot(index='arch', columns='c')
    df.index.name = None
    return df

pivot_model_table(df_r)['r'].to_latex(os.path.join(floats_dir, 'comm_sim.tex'),
        float_format="%.2f", multicolumn_format='c')


##### Compare LMCC and CCLM perplexity by community

from scipy.special import entr

def ppl(x):
    return np.exp(entr(x).sum())


# This is what we were looking at in the notebook.
# It gives the "perplexity" of the confusion matrix row.
# The problem is, since this is an average over the (comment-level) CC distributions,
# it is not itself a probability distribution.
lmcc_confusion_ppl = []
for model in conditioned_models:
    # Why is there a single comm_probs.pickle file? Isn't it model-dependent? What is in this file?
    #  > There is one for each model in the model directory.
    #  > This file has a row for each comment with the columns "actual_comm" and the remaining columns
    #  > are a probability distribution over communities (the normalisation of P(c_i | m), as below).
    
    # We should: 
    # 1. Compute P(c_i | m)
    #   Given: P(m | c_i); ie. a probability assigned assigned to m, when setting the CCLM to c_i
    #   (this is just the exponential of the negative loss.)
    #   The actual community of m is given by the "actual_comm" column in the 'dataframe'.
    # Then we can to compute P(c_i | m) by normalising the list
    #  of P(m | c_i) (varying i but keeping j fixed) so that ∑_i P(c_i|m) = 1
    # 2. We have too many numbers, so we take the accuracy of the LMCC prediction per *actual* community
    #    P(c_i | c_j)
    #      = P(c_i|m) for a random message m in community c_j.
    #      = ∑_m P(c_i|m) P(m|c_j)
    #                   but here we take every message to be equally "probable" in c_j, so P(m|c_j)=1/|c_j|
    #                   if we take |c_j| to be the number of messages in community c_j 
    #      = ∑_m P(c_i|m) / |c_j|
    #    Is this a probability distribution (when varying c_i?):
    #    ∑_i P(c_i | c_j)
    #                                                                by def.
    #      = ∑_i (∑_m P(c_i|m) / |c_j|)
    #                                                                by linearity of sum.
    #      = ∑_m ∑_i P(c_i|m) / |c_j|
    #                                                                total probability
    #      = ∑_m 1 / |c_j|
    #                                                                size of the domain of m
    #      = |c_j| / |c_j|
    #      = 1
    # 
    # > OK, I see that you're right. The mean of discrete probability distributions 
    # > is itself a probability distribution. I'm not sure it's right to call this
    # > distribution accuracy, however, since I would a prediction (via argmax, for
    # > example) to be involved.
    # > 
    # > It's possible I've convinced myself there's a problem where there isn't one,
    # > but I see two problems with using the entropy of the average distribution:
    # >
    # > 1. This metric can be the same for communities for which the classifier has 
    # > different accuracy.
    # >
    # >   comm_prob =
    # >   m | p(c1 | m) | p(c2 | m ) | actual_comm 
    # >   ----------------------------------------
    # >   0 |       0.7 |       0.3  | c1
    # >   1 |       0.8 |       0.2  | c1
    # >   2 |       0.7 |       0.3  | c2
    # >   3 |       0.8 |       0.2  | c2
    # >
    # >   comm_prob.groupby('actual_comm').mean() =
    # >   actual_comm | p(c1 | m) | p(c2 | m ) 
    # >   ----------------------------------------
    # >            c1 |      0.75 |       0.25  
    # >            c2 |      0.75 |       0.25
    # >
    # >   comm_prob.groupby('actual_comm').mean().apply(ppl) =
    # >   actual_comm | ppl 
    # >   ------------------
    # >            c1 | 1.754    
    # >            c2 | 1.754   
    # >
    # > Even though the classifier is more accurace for c1 than c2, the perplexity is
    # > the same for both.
    # >
    # > 2. Why should we care about the perplexity of the mean distriubtion rather
    # > than the mean of of the perplextity? When we consider the perplexity of the
    # > CCLM, it is the later (via, cross entropy loss over the vocabulary).

    model_ppl = pd.read_pickle(os.path.join(os.path.join(model_dir, model), 'comm_probs.pickle')).groupby('actual_comm').mean().apply(ppl, axis=1)[comms]
    model_ppl.name = model
    lmcc_confusion_ppl.append(model_ppl)

lmcc_confusion_ppl = pd.concat(lmcc_confusion_ppl, axis=1)
lmcc_confusion_ppl.index.name = 'community'

# I think this is what we should be looking at for lmcc_ppl instead (and what is defined in the paper, if I understand correctly).
# It computes the CC perplexity for each comment and averages it over the (actual) community.
lmcc_ppl = []
for model in conditioned_models:
    model_ppl = pd.read_pickle(os.path.join(os.path.join(model_dir, model), 'comm_probs.pickle'))[comms].apply(ppl, axis=1)
    model_ppl.name = model
    lmcc_ppl.append(model_ppl) 
lmcc_ppl= pd.concat(lmcc_ppl, axis=1)
lmcc_ppl['community'] = comms 

lmcc_ppl_mean = lmcc_ppl.groupby('community').mean()
cclm_ppl_mean = cclm_ppl.groupby('community').mean()

df = pd.merge(lmcc_ppl_mean, cclm_ppl_mean, right_index=True, left_index=True, suffixes=['_lmcc', '_cclm'])
df.to_csv(os.path.join(floats_dir, f'cclm_lmcc_ppl.csv'), sep='\t', 
            float_format="% 5.4f", index=True)

for model in conditioned_models:
    r, p = pearsonr(lmcc_confusion_ppl[model],cclm_ppl_mean[model])
    print(f"{model:<15} r = {r:0.2f} p = {p:0.4f}")
# lstm-3-0        r = -0.04 p = 0.8032
# lstm-3-1        r = -0.02 p = 0.8900
# lstm-3-2        r = -0.04 p = 0.8073
# lstm-3-3        r = -0.03 p = 0.8276
# transformer-3-0 r = -0.04 p = 0.7870
# transformer-3-1 r = -0.05 p = 0.7267
# transformer-3-2 r = -0.03 p = 0.8392
# transformer-3-3 r = -0.04 p = 0.7830

for model in conditioned_models:
    r, p = pearsonr(lmcc_ppl_mean[model],cclm_ppl_mean[model])
    print(f"{model:<15} r = {r:0.2f} p = {p:0.4f}")
# lstm-3-0        r = 0.35 p = 0.0185
# lstm-3-1        r = 0.30 p = 0.0444
# lstm-3-2        r = 0.29 p = 0.0504
# lstm-3-3        r = 0.25 p = 0.0878
# transformer-3-0 r = 0.28 p = 0.0591
# transformer-3-1 r = 0.32 p = 0.0298
# transformer-3-2 r = 0.28 p = 0.0620
# transformer-3-3 r = 0.30 p = 0.0456

r, p = pearsonr(lmcc_ppl_mean['lstm-3-1'], lmcc_ppl_mean['transformer-3-3'])
print(f"r = {r:0.4f} p = {p:0.6f}")
# r = 0.9922 p = 0.000000

