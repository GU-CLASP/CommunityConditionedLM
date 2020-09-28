import os

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

