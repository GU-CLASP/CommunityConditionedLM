import os
import logging
import click
from pathlib import Path
import json

def get_subs(chosen_subs_file='data/chosen_subs.txt', include_excluded=False):
    if not os.path.exists(chosen_subs_file):
        return []
    with open(chosen_subs_file, 'r') as f:
        subs = f.read().strip().split('\n')
    if not include_excluded:
        subs = [sub for sub in subs if not sub.startswith('#')]
    else:
        subs = [sub.lstrip('#') for sub in subs]
    return subs

def get_data_filename(data_dir, comm):
    return Path(data_dir)/f'{comm}.txt' 

def create_logger(name, filename, debug):
    logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(name)-12s [%(levelname)-7s] %(message)s',
            datefmt='%m-%d %H:%M', filename=filename, filemode='a')
    console_log_level = logging.DEBUG if debug else logging.INFO
    console = logging.StreamHandler()
    console.setLevel(console_log_level)
    console.setFormatter(logging.Formatter('[%(levelname)-8s] %(message)s'))
    logger = logging.getLogger(name)
    logger.addHandler(console)
    return logger

def iter_data(data_dir, file_limit=None):
    if not os.path.exists('data/dupes.json'):
        raise ValueError("data/dupes.json not found. Run `util find-dupes` first")
    dupes = json.load(open('data/dupes.json', 'r'))
    communities = get_subs()
    for community in communities:
        filename = get_data_filename(data_dir, community)
        for i, line in enumerate(iter_file(filename, dupes=dupes[community])):
            yield community, line
            if file_limit and i >= file_limit:
                break

def iter_file(filename, dupes=None):
    for i,line in enumerate(open(filename).readlines()):
        if dupes and i in dupes:
            continue
        yield line.rstrip('\n')

def find_dupes_in_file(filename, min_tokens):
    seen_hashes = set()
    dupe_ids = []
    for i, s in enumerate(iter_file(filename)):
        if len(s.split()) < min_tokens: # don't consider duplicate
            continue
        s_hash = hash(s)
        if s_hash in seen_hashes:
            dupe_ids.append(i)
        seen_hashes.add(s_hash)
    return dupe_ids

@click.group()
def cli():
    pass

@cli.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--min-tokens', type=int, default=10,
        help="Min length in tokens for a sentence to be considered duplicate.")
@click.option('--n-processes', type=int, default=1,
        help="Number of processes for multiprocessing actions.")
def find_dupes(data_dir, min_tokens, n_processes):
    from multiprocessing import Pool
    communities = get_subs()
    data_files = [get_data_filename(data_dir, c) for c in communities]
    args = [(filename, min_tokens) for filename in data_files]
    with Pool(processes=n_processes) as p:
        dupes = p.starmap(find_dupes_in_file, args)
    dupes = dict(zip(communities, dupes))
    with open('data/dupes.json', 'w') as f:
        json.dump(dupes, f)

if __name__ == '__main__':
    cli()
