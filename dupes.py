import util 
import click
import os
import json
from multiprocessing import Pool

def get_subs(chosen_subs_file, include_excluded=False):
    if not os.path.exists(chosen_subs_file):
        return []
    with open(chosen_subs_file, 'r') as f:
        subs = f.read().strip().split('\n')
    if not include_excluded:
        subs = [sub for sub in subs if not sub.startswith('#')]
    else:
        subs = [sub.lstrip('#') for sub in subs]
    return subs

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

def copy_without_dupes(in_file, out_file, dupes):
    print(in_file)
    with open(out_file, 'w') as f:
        for i, line in enumerate(iter_file(in_file, dupes=dupes)):
            f.write(line + '\n') 

@click.group()
def cli():
    pass

@cli.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('dupes_file', type=click.Path(exists=True))
@click.option('--min-tokens', type=int, default=10,
        help="Min length in tokens for a sentence to be considered duplicate.")
@click.option('--n-processes', type=int, default=1,
        help="Number of processes for multiprocessing actions.")
def find_dupes(data_dir, dupes_file, min_tokens, n_processes):
    communities = get_subs()
    data_files = [get_data_filename(data_dir, c) for c in communities]
    args = [(filename, min_tokens) for filename in data_files]
    with Pool(processes=n_processes) as p:
        dupes = p.starmap(find_dupes_in_file, args)
    dupes = dict(zip(communities, dupes))
    with open(dupes_file, 'w') as f:
        json.dump(dupes, f)

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
@click.argument('dupes_file', type=click.Path(exists=True))
@click.option('--n-processes', type=int, default=1,
        help="Number of processes for multiprocessing actions.")
def remove_dupes(input_dir, output_dir, dupes_file, n_processes):
    dupes = json.load(open(dupes_file, 'r'))
    subs = get_subs('data/chosen_subs.txt') 
    args = [(
        util.get_data_filename(input_dir, sub),
        util.get_data_filename(output_dir, sub),
        dupes[sub]
    ) for sub in subs]
    with Pool(processes=n_processes) as p:
        dupes = p.starmap(copy_without_dupes, args)

if __name__ == '__main__':
    cli()
