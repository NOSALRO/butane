import torch
import butane
import matplotlib.pyplot as plt


if __name__ == "__main__":

    a = torch.arange(1, 300).unsqueeze(0).unsqueeze(-1)
    b = torch.arange(1, 300).unsqueeze(0).unsqueeze(-1)
    zeros = torch.zeros((1, 10, 2))
    ab = torch.cat([a, b], -1)
    ab = torch.cat([zeros + torch.randn(1, 10, 2) * 0.01, ab, zeros], 1)
    ds = butane.data.TrajectoryDataset(ab, 10, shift=15)
    # ds = butane.data.ops.trim(ds, 1.)
    dl = torch.utils.data.DataLoader(ds, batch_size=1)
    for i in dl:
        plt.figure()
        d, t  = i['data'].squeeze(0), i['targets'].squeeze(0)
        plt.scatter(d[:, 0], d[:, 1])
        plt.scatter(t[:, 0], t[:, 1])
        plt.show()
