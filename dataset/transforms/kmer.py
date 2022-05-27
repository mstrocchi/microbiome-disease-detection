import itertools
import numpy as np
from time import time


bases = ['A', 'C', 'G', 'T']


class KmerVec():

    def __init__(self, k=3):
        self.name = f"{k}-mer"
        self.k = k

    def encode(self, sequence: str) -> np.ndarray:
        kmer_counts = {''.join(p): 0 for p in itertools.product(bases, repeat=self.k)}

        # Calculate how many kmers of length k there are
        num_kmers = len(sequence) - self.k + 1

        # Loop over the kmer start positions
        for i in range(num_kmers):
            # Slice the string to get the kmer
            kmer = sequence[i:i + self.k]
            kmer_counts[kmer] += 1

        vector = np.array(list(kmer_counts.values()))
        return vector / np.linalg.norm(vector)


# Example of usage
if __name__ == "__main__":

    embedding_model = KmerVec(k=6)
    seq = 'TACGGAGGATCCGAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGTGGACTGGTAAGTCAGTTGTGAAAGTTTGCGGCTCAACCGTAAAATTGCAGTTGATACTGTCAGTCTTGAGTACAGTAGAGGTGGGCGGAATTCGTGGTGTAGCGGTGAAATGCTTAGATATC'
    start = time()
    embedding = None
    for _ in range(10000):
        embedding = embedding_model.encode(seq)
    print(len(embedding))
    print(f"{time() -start}s")