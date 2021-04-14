import os
import logging
import click
from pathlib import Path
import json
import re

def get_community_label(file_path):
    base = os.path.basename(file_path)
    return base.split('.')[0]

def get_communities(data_dir):
    comms = [get_community_label(f) for f in  os.listdir(data_dir) if f.endswith('.train.txt')]
    comms.sort()
    return comms

def data_filename(data_dir, split, comm):
    return Path(data_dir)/f'{comm}.{split}.txt'

def mkdir(path):
    if os.path.exists(path):
        return False
    else:
        os.mkdir(path)
        return True

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

def iter_data(data_dir, split, file_limit=None):
    communities = get_communities(data_dir)
    for community in communities:
        filename = data_filename(data_dir, split, community)
        print(filename)
        for i, line in enumerate(iter_file(filename)):
            yield community, line
            if file_limit and i >= file_limit:
                break

def iter_file(filename):
    for i,line in enumerate(open(filename).readlines()):
        yield line.rstrip('\n')


def read_logged_val_ppls(model_dir):
    log_file = os.path.join(model_dir, 'training.log') 
    log_regex = re.compile("Epoch\s+(\d+)\s+\|\s+val loss\s+(\d+\.\d+)\s+\|\s+ppl\s+(\d+\.\d+)")
    log_contents = open(log_file).read()
    ppls = {}
    for epoch, loss, ppl in re.findall(log_regex, log_contents):
        ppls[int(epoch)] = float(ppl)
    ppls = [ppls[i+1] for i in range(len(ppls))]
    return ppls


