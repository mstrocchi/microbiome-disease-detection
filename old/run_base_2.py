import torch
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset

from models.modules import ISAB, PMA, SAB
from torch.utils.tensorboard import SummaryWriter

# model hyperparams
VEC_LEN = 128
NUM_OUTPUTS = 1  # number of seeds
DIM_OUTPUT = 2  # out features
NUM_INDS = 32
DIM_HIDDEN = 128
NUM_HEADS = 4


class CrohnDiseaseDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset = pickle.load(open(dataset_path, "rb"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# Actual SetTransformer implemented for point cloud classification
class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input=VEC_LEN,
        num_outputs=NUM_OUTPUTS,
        dim_output=DIM_OUTPUT,
        num_inds=NUM_INDS,
        dim_hidden=DIM_HIDDEN,
        num_heads=NUM_HEADS,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze()


# Taken from the python notebook in the repo
class SmallSetTransformer(nn.Module):
    def __init__(self,):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=VEC_LEN, dim_out=DIM_HIDDEN, num_heads=NUM_HEADS),
            SAB(dim_in=DIM_HIDDEN, dim_out=DIM_HIDDEN, num_heads=NUM_HEADS),
        )
        self.dec = nn.Sequential(
            PMA(dim=DIM_HIDDEN, num_heads=NUM_HEADS, num_seeds=NUM_OUTPUTS),
            nn.Linear(in_features=DIM_HIDDEN, out_features=DIM_OUTPUT),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)


# TODO perform test with variable number of sequences per set
def gen_data(amount_of_sets=100, set_size=5, seq_len=VEC_LEN):
    # length = np.random.randint(1, max_length + 1) # variable set size
    x = np.random.randint(1, 100, (amount_of_sets, set_size, seq_len))
    y = list(map(lambda e: np.max(e.flatten()), x))
    # FIXME ma il secondo expand dim è necessario?
    y = np.expand_dims(np.expand_dims(y, axis=1), axis=1)
    return x, y


def load_dataset(dataset):
    x, y = zip(*[(x, y) for x, y in dataset])
    x = np.array(list(map(lambda e: e.detach().numpy(), x)))
    y = np.asarray(list(y))
    return x, np.expand_dims(np.expand_dims(y, axis=1), axis=1)


def train_synthetic(model, epochs=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    losses = []
    y_hat = None
    for _ in tqdm(range(epochs)):
        x, y = gen_data()
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return y_hat, losses


def train_actual(dataset, epochs=500):
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = nn.MSELoss()
    # losses = []
    # y_hat = None
    # for _ in tqdm(range(epochs)):
    #     # TODO here the data should be loaded in batches I guess
    #     x, y = load_dataset(dataset)
    #     x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    #     y_hat = model(x)
    #     loss = criterion(y_hat, y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     losses.append(loss.item())
    # return losses, y_hat

    log_dir = "../result/log1.txt"
    writer = SummaryWriter(log_dir)

    model = SetTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(1e-3))
    criterion = nn.CrossEntropyLoss()
    model = nn.DataParallel(model)
    # model = model.cuda()

    train_samples = 700
    train_set, test_set = torch.utils.data.random_split(dataset, [train_samples, len(dataset) - train_samples])

    for epoch in tqdm(range(epochs)):
        model.train()
        losses, total, correct = [], 0, 0
        x, y = load_dataset(train_set)
        # this inner loop might be made with batches in mind

            # imgs = imgs.cuda()
            # lbls = lbls.cuda()

        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

        preds = model(x)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total = y.shape[0]
        correct = (preds.argmax(dim=1) == y).sum().item()

        avg_loss, avg_acc = np.mean(losses), correct / total
        writer.add_scalar("train_loss", avg_loss)
        writer.add_scalar("train_acc", avg_acc)
        print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")

        if epoch % 10 == 0:
            model.eval()
            losses, total, correct = [], 0, 0
            for x, y in load_dataset(test_set):
                x = torch.Tensor(x).cuda()
                y = torch.Tensor(y).long().cuda()
                preds = model(x)
                loss = criterion(preds, y)

                losses.append(loss.item())
                total += y.shape[0]
                correct += (preds.argmax(dim=1) == y).sum().item()
            avg_loss, avg_acc = np.mean(losses), correct / total
            writer.add_scalar("test_loss", avg_loss)
            writer.add_scalar("test_acc", avg_acc)
            print(f"Epoch {epoch}: test loss {avg_loss:.3f} test acc {avg_acc:.3f}")


if __name__ == '__main__':
    dataset = CrohnDiseaseDataset('../resources/classification-sample.pkl')
    model = SetTransformer()
    train_actual(dataset=dataset, epochs=50)
