import argparse
import torch
import butane
import numpy as np
import matplotlib.pyplot as plt


def _spiral():
    data = butane.data.toy.make_spiral(10_000)
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=3)
    plt.show()

def _8gaussian():
    data = butane.data.toy.make_eight_normal(10_000, scale=5, var=0.02)
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=3)
    plt.show()

def _moons():
    data, labels = butane.data.toy.make_moons(10_000)
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=labels, s=3)
    plt.show()

def _spiral_3d():
    data = butane.data.toy.make_spiral(10_000, max_theta=10*np.pi, is_3d=True)
    fig, ax = plt.subplots()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=3)
    plt.show()

def _8gaussian_3d():
    data = butane.data.toy.make_eight_normal(10_000, n_dims=3, scale=2, var=0.003)
    fig, ax = plt.subplots()
    ax = fig.add_subplot(projection='3d')
    s = butane.data.toy.make_spiral(10_000, max_theta=10*np.pi, is_3d=True)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=3)
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=3)
    plt.show()

def _disjoint_circle():
    sampler = butane.data.toy.sampler(butane.data.toy.make_disjoint_circle)
    data, c = sampler(10000)
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=3)
    ax.scatter(c[:, 0], c[:, 1], s=3)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    if args.name == "spiral":
        _spiral()
    if args.name == "8gaussians":
        _8gaussian()
    if args.name == "moons":
        _moons()
    if args.name == "spiral_3d":
        _spiral_3d()
    if args.name == "8gaussians_3d":
        _8gaussian_3d()
    if args.name == "disjoint_circle":
        _disjoint_circle()
