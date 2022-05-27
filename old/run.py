import torch
import numpy as np
import pickle
import torch.nn as nn
from torch.utils.data import Dataset

from models.modules import ISAB, PMA, SAB


class CrohnDiseaseDataset2(Dataset):
    def __init__(self, dataset_path):
        self.dataset = pickle.load(open(dataset_path, "rb"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class SmallSetTransformer(nn.Module):
    def __init__(self,):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=1, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)

class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input=128,
        num_outputs=1,
        dim_output=2,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
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
        a = self.enc(X)
        a = self.dec(a)
        a = a.squeeze()
        return a


if __name__ == '__main__':

    # dataset_path = './classification-sample.pkl'
    # dataset = CrohnDiseaseDataset2(dataset_path)
    # for sample, label in dataset:
    #     print(f"{sample.shape}: {label}")

    model = SetTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(1e-3))
    criterion = nn.CrossEntropyLoss()
    model = nn.DataParallel(model)
    # model = model.cuda()

    dataset_path = '../resources/classification-sample.pkl'
    dataset = CrohnDiseaseDataset2(dataset_path)
    train_samples = 700
    train_set, val_set = torch.utils.data.random_split(dataset, [train_samples, len(dataset) - train_samples])

    for epoch in range(2000):
        model.train()
        losses, total, correct = [], 0, 0
        for imgs, lbls in train_set:
            # imgs = imgs.cuda()
            # lbls = lbls.cuda()
            preds = model(imgs)
            loss = criterion(preds, lbls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            total += lbls.shape[0]
            correct += (preds.argmax(dim=1) == lbls).sum().item()

        avg_loss, avg_acc = np.mean(losses), correct / total
        writer.add_scalar("train_loss", avg_loss)
        writer.add_scalar("train_acc", avg_acc)
        print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")

        if epoch % 10 == 0:
            model.eval()
            losses, total, correct = [], 0, 0
            for imgs, lbls in test_set:
                imgs = torch.Tensor(imgs).cuda()
                lbls = torch.Tensor(lbls).long().cuda()
                preds = model(imgs)
                loss = criterion(preds, lbls)

                losses.append(loss.item())
                total += lbls.shape[0]
                correct += (preds.argmax(dim=1) == lbls).sum().item()
            avg_loss, avg_acc = np.mean(losses), correct / total
            writer.add_scalar("test_loss", avg_loss)
            writer.add_scalar("test_acc", avg_acc)
            print(f"Epoch {epoch}: test loss {avg_loss:.3f} test acc {avg_acc:.3f}")