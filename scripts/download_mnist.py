import os
import csv
import kagglehub
import torch
import numpy as np
import pathlib

print("Downloading MNIST...")
path = kagglehub.dataset_download("oddrationale/mnist-in-csv")


pathlib.Path("data/mnist").mkdir(parents=True, exist_ok=True)
os.rename(f"{os.environ['HOME']}/.cache/kagglehub/datasets/oddrationale/mnist-in-csv/versions/2", "data/mnist/")

images, labels = [], []
with open('./data/mnist/mnist_train.csv', 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    next(csvreader)
    for data in csvreader:
        label = int(data[0])
        pixels = data[1:]
        pixels = np.array(pixels, dtype='int64')
        pixels = pixels.reshape((1, 1, 28, 28))
        images.append(pixels)
        labels.append(label)
images = torch.from_numpy(np.vstack(images)).float()
labels = torch.tensor(labels)

torch.save(images, "data/mnist/mnist_train_data.pt")
torch.save(labels, "data/mnist/mnist_train_targets.pt")


images, labels = [], []
with open('./data/mnist/mnist_test.csv', 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    next(csvreader)
    for data in csvreader:
        label = int(data[0])
        pixels = data[1:]
        pixels = np.array(pixels, dtype='int64')
        pixels = pixels.reshape((1, 1, 28, 28))
        images.append(pixels)
        labels.append(label)
images = torch.from_numpy(np.vstack(images)).float()
labels = torch.tensor(labels)

torch.save(images, "data/mnist/mnist_test_data.pt")
torch.save(labels, "data/mnist/mnist_test_targets.pt")
