import time
import torch
import butane
import matplotlib.pyplot as plt


if __name__ == "__main__":

    x = torch.load('data/cifar10/cifar10_data.pt')[0]
    x = x.to(torch.uint8)

    # x = butane.utils.center_mask(x, 8)
    x = butane.utils.checkerboard_mask(x, 4)
    # x = butane.utils.random_mask(x, 4)

    # x = butane.data.jigsaw(x, 16)[1]

    fig, ax = plt.subplots()
    ax.imshow(x.moveaxis(0, -1))
    plt.show()

