import torch
import butane
import matplotlib.pyplot as plt


if __name__ == "__main__":

    x = torch.load("data/cifar10/cifar10_data.pt")
    y = torch.load("data/cifar10/cifar10_targets.pt")

    ds = butane.data.PairDataset(
        data=x,
        targets=y,
        data_pair=x,
        targets_pair=y,
    )

    for i in range(10):
        fig, ax = plt.subplots(1, 2)
        b = ds[i]
        ax[0].imshow(b['data'].moveaxis(0,-1).to(torch.uint8))
        ax[1].imshow(b['targets'].moveaxis(0,-1).to(torch.uint8))
        plt.show()
