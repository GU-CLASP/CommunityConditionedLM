import click
import util
from snapy import MinHash, LSH
import json
from multiprocessing import Pool


def find_dupes_in_file(filename):
    n_gram = 9
    content = [(i,line.rstrip('\n')) for i,line in enumerate(open(filename).readlines())]
    content = [(i,c) for i,c in content if len(c.split()) > 10]
    ids, content = zip(*content)
    minhash = MinHash(content, n_gram=n_gram, n_gram_type = 'char', permutations=30, hash_bits=64, seed=3)
    content = dict(zip(ids,content))
    lsh = LSH(minhash, labels=ids)
    lsh.edge_list(min_jaccard=0.5, jaccard_weighted=True)

    dupes = {}
    removed = []
    for i in ids:
        if i in removed:
            continue
        sim = lsh.query(i, min_jaccard=0.5)
        if sim:
            dupes[i] = sim
        for j in sim:
            lsh.remove(j)
            removed.append(j)

    return removed

def iter_nondupes(filename, dupe_ids=[]):
    with open(filename) as f:
        for i, line in enumerate(f.readlines()):
            if i in dupe_ids:
                print(line)
                continue
            yield (i, line.rstrip('\n'))

@click.command()
@click.option('--n-processes', type=int, default=1,
        help="Number of processes for multiprocessing actions.")
def cli(n_processes):
    subs =  util.get_subs()
    args = [(f'../SSSN/data/tokenized/subreddit_comments/2015/{sub}.txt',) for sub in subs]
    with Pool(processes=n_processes) as p:
        dupes = p.starmap(find_dupes_in_file, args)
    with open('data/dupes.json', 'w') as f:
        json.dump(dict(zip(subs, dupes)), f)

if __name__ == '__main__':
    cli()
