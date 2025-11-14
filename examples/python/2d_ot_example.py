import torch
import numpy as np
import butane
import matplotlib.pyplot as plt


def sample_moons(n):
    x0, y0 = generate_moons(n, noise=0.2)
    return x0 * 3 - 1, y0

if __name__ == "__main__":

    x0 = butane.data.toy.make_eight_normal(n_samples=1_000, scale=5, var=0.05)
    x1, cond = butane.data.toy.make_moons(n_samples=1_000)
    x1 = 3*x1 - 1

    ot = butane.math.OTPlanner(
        cost_func='l2_squared',
        condition_cost_func='cosine'
    )

    # x0, x1 = ot.exact_ot(x0.to('cuda'), x1.to('cuda'), unoptimal=False)
    # x0, x1 = x0.cpu(), x1.cpu()

    # fig, ax = plt.subplots(figsize=(25, 15))
    # plt.ion()
    # ax.scatter(x0[:, 0], x0[:, 1], s=10, edgecolors='blue', facecolor='none', zorder=100)
    # ax.scatter(x1[:, 0], x1[:, 1], s=10, edgecolors='red', facecolor='none', zorder=100)
    # for i in range(x0.size(0)):
    #     ax.scatter(x0[i, 0], x0[i, 1], s=10, c='blue')
    #     ax.scatter(x1[i, 0], x1[i, 1], s=10, c='red')
    #     ax.plot(
    #         [x0[i, 0], x1[i,0]],
    #         [x0[i, 1], x1[i,1]],
    #         c='green',
    #         alpha=0.3,
    #     )
    #     plt.pause(0.5)
    # plt.show()


    x0, x1, _, cond = ot.c2_ot(
        x1=x0,
        x2=x1,
        c1=cond,
        c2=cond,
        r=0.2,
        w=1e+08,
        max_iters=5,
        tol=1e-08,
        unoptimal=False,
    )

    x1_0 = x1[cond == 0]
    x1_1 = x1[cond == 1]

    fig, ax = plt.subplots(figsize=(25, 15))
    plt.ion()
    ax.scatter(x0[:, 0], x0[:, 1], s=10, edgecolors='blue', facecolor='none', zorder=100)
    ax.scatter(x1_0[:, 0], x1_0[:, 1], s=10, edgecolors='red', facecolor='none', zorder=100)
    ax.scatter(x1_1[:, 0], x1_1[:, 1], s=10, edgecolors='green', facecolor='none', zorder=100)
    for i in range(x0.size(0)):
        ax.scatter(x0[i, 0], x0[i, 1], s=10, c='blue')
        ax.scatter(x1[i, 0], x1[i, 1], s=10, c='red' if cond[i] == 0 else 'green')
        ax.plot(
            [x0[i, 0], x1[i,0]],
            [x0[i, 1], x1[i,1]],
            c='green',
            alpha=0.3,
        )
        plt.pause(0.5)
