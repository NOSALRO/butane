import torch
import butane
import matplotlib.pyplot as plt

import numpy as np

if __name__ == "__main__":

    rdata = torch.empty((int(1e+5), 2), device='cuda').uniform_(-1, 1)
    kmeans = butane.clustering.MiniBatchKMeans(n_centroids=200, init='kmeans++', max_iters=500, tol=1e-4, random_state=3)
    kmeans.fit(rdata)
    fig, ax = plt.subplots()
    centroids = kmeans.centroids.cpu().numpy()
    rdata = rdata.cpu().numpy()
    ax.scatter(centroids[:, 0], centroids[:, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.show()