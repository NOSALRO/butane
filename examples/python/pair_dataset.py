import torch
import butane
import matplotlib.pyplot as plt


if __name__ == "__main__":

    x = torch.load("data/cifar10/cifar10_data.pt")
    y = torch.load("data/cifar10/cifar10_targets.pt")

    ds = butane.data.PairDataset(
        data=x[:3],
        targets=y[:3],
        data_pair=x[:4],
        targets_pair=y[:4],
        deterministic=False,
    )

    dl = torch.utils.data.DataLoader(ds, batch_size=64)
    it = butane.data.utils.InfiniteIterator(dl)
    for i in it:
        pass

    # b = ds[torch.arange(10)]
    # for i in range(10):
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(b['data'][i].moveaxis(0,-1).to(torch.uint8))
        # ax[1].imshow(b['targets'][i].moveaxis(0,-1).to(torch.uint8))
        # plt.show()
