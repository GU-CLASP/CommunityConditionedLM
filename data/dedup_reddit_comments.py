from pathlib import Path
import csv

def iter_comments(corpus_dir):
    for comment_file in corpus_dir.glob('*.csv'):
        print(comment_file)
        with open(comment_file, 'r') as f:
            reader = csv.DictReader(f)
            for comment in reader:
                yield comment['id'], comment['body']


def find_dupe_comments(comments):
    min_length = 50
    seen_hashes = set()
    dupe_ids = []
    for comment_id, body in comments: 
        comment_len = len(body) 
        if comment_len < min_length:
            continue
        # only consider last 50 chars (catches most bot-filled forms)
        comment_hash = hash(body[-min_length:])
        if comment_hash in seen_hashes:
            dupe_ids.append(comment_id)
        else:
            seen_hashes.add(comment_hash) 
    return dupe_ids

corpus_dir = Path('reddit_sample')
comment_iter = iter_comments(corpus_dir)
dupe_ids = find_dupe_comments(comment_iter)
with open(corpus_dir/'dupes.txt', 'w') as f:
    f.write('\n'.join(dupe_ids))

