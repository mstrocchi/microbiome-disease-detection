import os
import gzip
import random
from glob import glob
from tqdm import tqdm
from Bio import SeqIO
import numpy as np
from transforms.kmer import KmerVec


DATA_PATH = '/Volumes/2BIG/RP/dataset/'
SUBSAMPLED_PATH = DATA_PATH + 'subsampled-10000/'
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
            save_path = f"{DATA_PATH}{dirname}/3-mer-uncompressed/{label}/{filename}.npz"
            np.savez(save_path, *embeddings)


# TODO be sure that all the embedding methods use the same set of sequences
if __name__ == '__main__':
    # for k in range(3, 6, 1):
    #     transform = KmerVec(k=k)
    #     compute_dataset(transform)

    transform = KmerVec(k=3)
    compute_dataset(transform)



