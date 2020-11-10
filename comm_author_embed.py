"""
Creates a community embedding for the set of subreddits from the author/community cooccurance matrix
as in community2vec: https://www.aclweb.org/anthology/W17-2904/
"""

import data
import csv
from collections import Counter
from scipy.sparse import coo_matrix, save_npz, load_npz
from sklearn.decomposition import TruncatedSVD, PCA
import numpy as np
import glob
import os

count_files = glob.glob('data/reddit2015_author_comm_counts/2015_author_sub_counts-*.csv')
counts_matrix_file = 'model/reddit2015/comm_author_counts_full.npz'
authors_file = 'model/reddit2015/authors_full.txt'
comms_file = 'model/reddit2015/comms_full.txt'
comms_sorted_file =  'model/reddit2015/comms_full_sorted.txt'

def iter_counts():
    for file in count_files:
        print(f"reading {file}")
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for item in reader:
                yield item['author'], item['subreddit'], int(item['comment_count'])

def save_list(l, file):
    with open(file, 'w') as f:
        f.write('\n'.join(l))

def load_list(file):
    with open(file, 'r') as f:
        return f.read().split('\n') 

def ppmi(counts, cds=False, neg=1, smooth=0):
    """ Based on (copied from) histwords/representations/ppmigen.py """

    # compute marginal probs
    smooth = counts.sum() * smooth
    row_probs = counts.sum(1) + smooth
    col_probs = counts.sum(0) + smooth
    if cds:
        col_probs = np.power(col_probs, 0.75)
    row_probs = row_probs / row_probs.sum()
    col_probs = col_probs / col_probs.sum()

    # build PPMI matrix
    prob_norm = counts.sum() + (counts.shape[0] * counts.shape[1]) * smooth
    row_d = counts.row
    col_d = counts.col
    data_d = counts.data
    neg = np.log(neg)
    for i in range(len(counts.data)):
        if data_d[i] == 0.0:
            continue
        joint_prob = (data_d[i] + smooth) / prob_norm
        denom = row_probs[row_d[i], 0] * col_probs[0, col_d[i]]
        if denom == 0.0:
            raise ValueError("Zero denominator.")
        data_d[i] = np.log(joint_prob / denom)
        data_d[i] = max(data_d[i] - neg, 0)
        # if normalize:
            # data_d[i] /= -1*np.log(joint_prob)
    return coo_matrix((data_d, (row_d, col_d)), shape=counts.shape).tocsr()

def unit_scale_rows(m):
    denom = m.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1 # don't scale zero rows
    return m / denom

def create_comm_author_counts():
    author_stoi, authors = {}, []
    comm_stoi, comms = {}, []

    data, row, col = [], [], []
    for author, comm, count in iter_counts():
        if author == '[deleted]':
            continue
        if author in author_stoi:
            author_idx = author_stoi[author]
        else:
            author_idx = len(authors)
            authors.append(author)
            author_stoi[author] = author_idx
        if comm in comm_stoi:
            comm_idx = comm_stoi[comm]
        else:
            comm_idx = len(comms)
            comms.append(comm)
            comm_stoi[comm] = comm_idx
        data.append(count)
        row.append(comm_idx)
        col.append(author_idx)

    save_list(authors, authors_file)
    save_list(comms, comms_file)

    author_comm_counts = coo_matrix((data, (row, col)))
    author_comm_counts = author_comm_counts.tocsr()
    save_npz(counts_matrix_file, author_comm_counts)

    return authors, comms, author_comm_counts

def pca(w, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(w)

def load_comm_embed(embedding_path, comms=None):
    comms_sorted = load_list(comms_sorted_file)
    if not comms:
        comms = comms_sorted
    comm_idxs = [comms_sorted.index(comm) for comm in comms]
    embed = np.load(embedding_path)
    return embed[comm_idxs]

if __name__ == '__main__':

    if os.path.exists(counts_matrix_file):
        print("Loading comm-author counts matrix.")
        authors = load_list(authors_file)
        comms = load_list(comms_file)
        author_comm_counts = load_npz(counts_matrix_file)
    else:
        print("Creating comm-author counts matrix.")
        authors, comms, author_comm_counts = create_comm_author_counts()

    print("Computing the community co-occurance matrix.")
    # Communities co-occur for each of the users they share (with at least 10 comments in the community)
    author_comm_indicator = (author_comm_counts >= 10).astype(int)
    comm_coocur = author_comm_indicator.dot(author_comm_indicator.T)

    # Sort the coocurrance matrix & list of community names by community size
    comm_size = np.asarray(author_comm_indicator.sum(axis=1)).squeeze() # number of >10 comment authors
    comm_size_rank = (-comm_size).argsort(axis=0) # largest first
    comm_size_dict = dict(zip(comms, comm_size))
    comm_coocur_sorted = comm_coocur[comm_size_rank]
    comms_sorted = sorted(comms, key=comm_size_dict.get, reverse=True)

    comm_embed_explicit = ppmi(comm_coocur_sorted[200:2200].T.astype(float).tocoo()).toarray()
    comm_embed_explicit = unit_scale_rows(comm_embed_explicit)

    comm_embed_pca = ppmi(comm_coocur_sorted[:5000].T.toarray())
    comm_embed_pca = pca(comm_embed_pca, 100)
    comm_embed_pca = unit_scale_rows(comm_embed_pca)


    save_list(comms_sorted, comms_sorted_file)
    np.save('model/reddit2015/comm_embed_explicit.npy', comm_embed_explicit)
    np.save('model/reddit2015/comm_embed_pca.npy', comm_embed_pca)


