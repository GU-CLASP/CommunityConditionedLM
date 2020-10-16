"""
Creates a community embedding for the set of subreddits from the author/community cooccurance matrix
"""

import csv
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

dataset, fields = data.load_data_and_fields('data/reddit2015', 'model/reddit2015', 64, 5)
comms = fields['community'].vocab.itos
comms = comms[1:]

def iter_comments(comm):
    for month in range(1,13):
        filename = f'data/subreddit_comments/{comm}/2015-{month:02d}.csv'
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for comment in reader:
                yield comment

authors = set()
for c_idx,c in enumerate(comms):
    print(f'{c_idx+1}/{len(comms)}',end='\r')
    for comment in iter_comments(c):
        authors.add(comment['author'])
authors = list(authors)
author_dict = dict(zip(authors, range(len(authors))))

author_counts = Counter()
for c_idx, c in enumerate(comms):
    print(f'{c_idx+1}/{len(comms)}',end='\r')
    for comment in iter_comments(c):
        author_idx = author_dict[comment['author']]
        author_counts[(c_idx, author_idx)] += 1

data, row, col = [], [], []
for (comm, author), count in author_counts.items():
    data.append(count)
    row.append(comm)
    col.append(author)

author_comm_mat = coo_matrix((data, (row, col))).tocsr()
svd = TruncatedSVD(n_components=16)
comm_author_embed = svd.fit_transform(author_comm_mat)
np.save('model/reddit2015/comm_author_embed_svd16dim', comm_author_embed)

