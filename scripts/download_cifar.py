import os
import urllib.request
import gzip
import numpy as np
import torch
import tarfile
import pathlib

def _unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

print("Downloading CIFAR10...")
pathlib.Path("data/cifar10").mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "data/cifar10/cifar-10-python.tar.gz")

tar = tarfile.open("data/cifar10/cifar-10-python.tar.gz", "r:gz")
tar.extractall("data/cifar10")
tar.close()

data, labels = [], []
for i in range(1,6):
    cifar10 = _unpickle(f"data/cifar10/cifar-10-batches-py/data_batch_{i}")
    labels.append(torch.tensor(cifar10['labels']))
    data.append(torch.from_numpy(cifar10['data'].reshape(-1, 3, 32, 32)))
data = torch.vstack(data)
labels = torch.cat(labels, dim=-1)
torch.save(data.float(), "data/cifar10/cifar10_data.pt")
torch.save(labels, "data/cifar10/cifar10_targets.pt")
np.savetxt("data/cifar10/cifar10_label_names.txt", np.array(_unpickle(f"data/cifar10/cifar-10-batches-py/batches.meta")['label_names']), fmt='%s')

cifar10 = _unpickle(f"data/cifar10/cifar-10-batches-py/test_batch")
labels = torch.tensor(cifar10['labels'])
data = torch.from_numpy(cifar10['data'].reshape(-1, 3, 32, 32))
torch.save(data.float(), "data/cifar10/cifar10_test_data.pt")
torch.save(labels, "data/cifar10/cifar10_test_targets.pt")


os.remove("data/cifar10/cifar-10-python.tar.gz")

print("Downloading CIFAR100...")
pathlib.Path("data/cifar100").mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", "data/cifar100/cifar-100-python.tar.gz")

tar = tarfile.open("data/cifar100/cifar-100-python.tar.gz", "r:gz")
tar.extractall("data/cifar100")
tar.close()

data, labels = [], []
cifar100 = _unpickle(f"data/cifar100/cifar-100-python/train")
torch.save(torch.from_numpy(cifar100['data'].reshape(-1, 3, 32, 32)), "data/cifar100/cifar100_train_data.pt")
torch.save(cifar100['fine_labels'], "data/cifar100/cifar100_train_targets.pt")
np.savetxt("data/cifar100/cifar100_label_names.txt", np.array(_unpickle(f"data/cifar100/cifar-100-python/meta")['fine_label_names']), fmt='%s')

cifar100_test = _unpickle(f"data/cifar100/cifar-100-python/test")
torch.save(torch.from_numpy(cifar100_test['data'].reshape(-1, 3, 32, 32)), "data/cifar100/cifar100_test_data.pt")
torch.save(cifar100_test['fine_labels'], "data/cifar100/cifar100_test_targets.pt")

os.remove("data/cifar100/cifar-100-python.tar.gz")
