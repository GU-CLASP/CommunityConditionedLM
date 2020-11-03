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
                yield item['author'], item['subreddit'], item['comment_count']

data, row, col = [], [], []
for author, comm, count in iter_counts():
    data.append(count)
    row.append(comm)
    col.append(author)

print("Creating counts matrix.")
author_comm_mat = coo_matrix((data, (row, col))).tocsr()
np.save('model/reddit2015/comm_author_counts_full', author_comm_mat)

print("Computing SVD.")
svd = TruncatedSVD(n_components=16, n_iter=20) #, algorithm='arpack')
comm_author_embed = svd.fit_transform(author_comm_mat)
np.save('model/reddit2015/comm_author_embed_full_svd16dim', comm_author_embed)

