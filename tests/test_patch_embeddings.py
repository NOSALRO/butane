import torch
import butane

import matplotlib.pyplot as plt

if __name__ == "__main__":

    data = torch.load('data/mnist/mnist_train_data.pt')[0:3]
    patch_size = 8
    pe = butane.nn.PatchEmbeddings([1, 28, 28], patch_size**2, patch_size)

    data=pe(data).cpu().detach()
    print(data)

    print(data.size())
    fig, ax = plt.subplots(1, data.shape[1])
    ax = ax.flatten()
    for i in range(data.size(1)):
        ax[i].imshow(data[0][i].reshape(patch_size, patch_size))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()

    # plt.imshow(data.moveaxis(0,-1))
    # plt.show()
