import os
import logging
import click
from pathlib import Path
import json

def get_community_label(file_path):
    base = os.path.basename(file_path)
    return os.path.splitext(base)[0]

def get_communities(data_dir):
    return [get_community_label(f) for f in  os.listdir(data_dir)]

def data_filename(data_dir, comm):
    return Path(data_dir)/f'{comm}.txt'

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

def iter_data(data_dir, file_limit=None):
    communities = get_communities(data_dir)
    for community in communities:
        filename = data_filename(data_dir, community)
        for i, line in enumerate(iter_file(filename)):
            yield community, line
            if file_limit and i >= file_limit:
                break

def iter_file(filename):
    for i,line in enumerate(open(filename).readlines()):
        yield line.rstrip('\n')
