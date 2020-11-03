"""
Creates a community embedding for the set of subreddits from the author/community cooccurance matrix
"""

import data
import csv
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np
import glob

count_files = glob.glob('data/reddit2015_author_comm_counts/2015_author_sub_counts-*.csv')

def iter_counts():
    for file in count_files:
        print(f"reading {file}")
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for item in reader:
                yield item['author'], item['subreddit'], int(item['comment_count'])

author_stoi, author_itos = {}, []
comm_stoi, comm_itos = {}, []

print("Compiling counts.")
data, row, col = [], [], []
for author, comm, count in iter_counts():
    if author in author_stoi:
        author_idx = author_stoi[author]
    else:
        author_itos.append(author)
        author_idx = len(author_itos)
        author_stoi[author] = author_idx
    if comm in comm_stoi:
        comm_idx = comm_stoi[comm]
    else:
        comm_itos.append(comm)
        comm_idx = len(comm_itos)
        comm_stoi[comm] = comm_idx
    data.append(count)
    row.append(comm_idx)
    col.append(author_idx)

def save_list(l, file):
    with open(file, 'w') as f:
        f.write('\n'.join(l))

save_list(author_itos, 'model/reddit2015/authors_full.txt')
save_list(comm_itos, 'model/reddit2015/comms_full.txt')


print("Creating counts matrix.")
author_comm_mat = coo_matrix((data, (row, col)))
author_comm_mat = author_comm_mat.tocsr()
np.save('model/reddit2015/comm_author_counts_full', author_comm_mat)

print("Computing SVD.")
svd = TruncatedSVD(n_components=16, n_iter=20) #, algorithm='arpack')
comm_author_embed = svd.fit_transform(author_comm_mat)
np.save('model/reddit2015/comm_author_embed_full_svd16dim', comm_author_embed)

