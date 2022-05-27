import os
import gzip
import random
from glob import glob
from tqdm import tqdm
from Bio import SeqIO
import numpy as np

DATA_PATH = '/Volumes/2BIG/RP/dataset/'
PROCESSED_PATH = DATA_PATH + 'processed/'
SAVE_PATH = DATA_PATH + 'subsampled-10000/'

BASES = ['A', 'C', 'G', 'T']
SEQUENCES_PER_SAMPLE = 10000


if __name__ == '__main__':

    for label in ['no', 'cd']:
        for file_path in tqdm(glob(f"{PROCESSED_PATH}{label}/" + '*.gz'), f"Processing class \'{label}\'"):

            # read the sequences of a sample
            with gzip.open(file_path, "rt") as file:
                seqs = [str(record.seq) for record in SeqIO.parse(file, "fastq")]

            # filter strings that have letters other than the four bases
            seqs = [string for string in seqs if set(string).issubset(set(BASES))]

            # Shuffle and select only a portion of the sequences in the sample
            random.shuffle(seqs)
            seqs = seqs[:SEQUENCES_PER_SAMPLE]

            filename = '.'.join(os.path.basename(file_path).split('.')[:-2])
            save_path = f"{SAVE_PATH}{label}/{filename}.txt"
            with open(save_path, "w") as f:
                f.write('\n'.join(seqs))
                f.close()


