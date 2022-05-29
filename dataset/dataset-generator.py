import os
import gzip
import random
from glob import glob
from tqdm import tqdm
from Bio import SeqIO
import numpy as np
from transforms.kmer import KmerVec

SAMPLES_PER_FILE = 1000
DATA_PATH = '/Volumes/2BIG/RP/dataset/'
SUBSAMPLED_PATH = DATA_PATH + f'subsampled-{SAMPLES_PER_FILE}/'
BASES = ['A', 'C', 'G', 'T']


def compute_dataset(transform):

    for label in ['no', 'cd']:
        for file_path in tqdm(glob(f"{SUBSAMPLED_PATH}{label}/" + '*.txt'), f"Processing {transform.name}, class \'{label}\'"):
            # read the sequences of a sample
            with open(file_path, 'r') as f:
                seqs = [line.rstrip() for line in f]
            embeddings = list(map(transform.encode, seqs))
            dirname = 'k-mer' if 'mer' in transform.name else 'neuroseed'
            filename = '.'.join(os.path.basename(file_path).split('.')[:-1])
            # save_path = f"{DATA_PATH}{dirname}/{transform.name}/{label}/{filename}.npz"
            # np.savez_compressed(save_path, *embeddings)
            save_path = f"{DATA_PATH}{dirname}/{SAMPLES_PER_FILE}/{transform.name}-uncompressed/{label}/{filename}.npz"
            np.savez(save_path, *embeddings)


if __name__ == '__main__':
    for k in range(3, 6, 1):
        transform = KmerVec(k=k)
        compute_dataset(transform)

    # transform = KmerVec(k=3)
    # compute_dataset(transform)



