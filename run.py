import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from time import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, Dataset, DataLoader

from models.models import SetTransformer
from util.loss_functions import AverageMeter


SAMPLE_SIZE = 100
LABEL = {
    'no': 1,
    'cd': 0
}


class CrohnDiseaseDataset(Dataset):
    def __init__(self, dataset_path, transform, target_transform):
        self.samples = glob(dataset_path + '**/*.npz', recursive=True)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, sample_size=SAMPLE_SIZE):
        # FIXME slow operation, loading from compressed file
        sequences = list(np.load(self.samples[idx]).values())
        np.random.shuffle(sequences)
        # FIXME implement smarter sampling techniques (original files are capped to 10k sequences)
        sequences = sequences[:sample_size]
        target = LABEL[self.samples[idx].split('/')[-2]]
        sequences = self.transform(sequences) if self.transform else sequences
        target = self.target_transform(target) if self.target_transform else target
        return sequences, target


def transform(sequences):
    return torch.from_numpy(np.array(sequences)).float()


def target_transform(label):
    return torch.Tensor([[label]]).float()


def train(model, loader, optimizer, loss, device=None, epoch=None):
    avg_loss = AverageMeter()
    model.train()

    for sequences, labels in tqdm(loader, f"Epoch {epoch}, training"):

        if device:
            # move samples to right device
            sequences, labels = sequences.to(device), labels.to(device)

        # forward propagation
        optimizer.zero_grad()
        output = model(sequences)

        # loss and backpropagation
        loss_train = loss(output, labels)
        loss_train.backward()
        optimizer.step()

        # keep track of average loss
        avg_loss.update(loss_train.data.item(), sequences.shape[0])

    return avg_loss.avg


def test(model, loader, loss, device=None, epoch=None):
    avg_loss = AverageMeter()
    model.eval()

    for sequences, labels in tqdm(loader, f"Epoch {epoch}, testing"):

        if device:
            # move samples to right device
            sequences, labels = sequences.to(device), labels.to(device)

        # forward propagation and loss computation
        output = model(sequences)
        loss_val = loss(output, labels).data.item()
        avg_loss.update(loss_val, sequences.shape[0])

    return avg_loss.avg


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2021)
    if device == 'cuda':
        torch.cuda.manual_seed(2021)
    device = torch.device(device)

    dataset_path = '/Volumes/2BIG/RP/dataset/k-mer/3-mer-uncompressed/'
    dataset = CrohnDiseaseDataset(dataset_path, transform, target_transform)
    dataset_classes = ['train', 'val', 'test']

    # FIXME random splitting might not be optimal considered
    #  low amount of samples and class imbalances
    datasets = dict(zip(dataset_classes, random_split(dataset, [700, 150, 211])))
    # TODO The 'vanilla' loader uses a batch sampler by default, this might be optimal for out setup
    loaders = {d_class: DataLoader(datasets[d_class], batch_size=8, shuffle=True) for d_class in datasets}

    model = SetTransformer(dim_input=64, dim_output=1, num_outputs=1)
    if device:
        model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss = nn.BCEWithLogitsLoss()

    # training
    for epoch in range(0, 2):
        t = time()
        loss_train = train(model, loaders['train'], optimizer, loss, device, epoch)
        loss_val = test(model, loaders['val'], loss, device, epoch)

        # print progress
        if True:  # epoch % 5 == 0
            print('Epoch: {:02d}'.format(epoch),
                  'loss_train: {:.6f}'.format(loss_train),
                  'loss_val: {:.6f}'.format(loss_val),
                  'time: {:.4f}s'.format(time() - t))

    # testing
    for dset in loaders.keys():
        avg_loss = test(model, loaders[dset], loss)
        print('Final results {}: loss = {:.6f}'.format(dset, avg_loss))
