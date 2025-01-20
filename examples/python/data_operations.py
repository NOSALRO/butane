import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import butane

if __name__ == "__main__":

    ds = butane.data.Dataset(torch.rand(100, 10, 20))

    print(f"Original \nmean: {ds[:]['data'].mean()} \nstd: {ds[:]['data'].std()}")
    print("-------------------------------------------")

    standard_scaler = butane.data.StandardScaler()
    standard_scaler.fit(ds)
    standard_scaled_data = standard_scaler(ds)

    print(f"Standardized \nmean: {standard_scaled_data.mean()} \nstd: {standard_scaled_data.std()}")
    print("-------------------------------------------")

    min_max_scaler = butane.data.MinMaxScaler()
    min_max_scaler.fit(ds)
    min_max_data = min_max_scaler(ds)

    print(f"Min-Max \nmean: {min_max_data.mean()} \nstd: {min_max_data.std()} \nmin: {min_max_data.min()} \nmax: {min_max_data.max()}")
    print("-------------------------------------------")

    min_max_scaler = butane.data.MinMaxScaler()
    transforms = butane.data.Transforms(
        lambda x : x.transpose(-1, -2),
        lambda x : x.flatten(0),
        min_max_scaler,
    )

    print(f"Before transofrms \nshape: {ds[:]['data'].shape} \nmin: {ds[:]['data'].min()} \nmax: {ds[:]['data'].max()}")

    ds.set_transforms(transforms)
    min_max_scaler.fit(ds)

    print(f"After transofrms \nshape: {ds[:]['data'].shape} \nmin: {ds[:]['data'].min()} \nmax: {ds[:]['data'].max()}")
    print("-------------------------------------------")

    inverse_transforms = butane.data.Transforms(
        min_max_scaler.reverse,
        lambda x : x.reshape(2, -1),
        lambda x : x.transpose(-1, -2),
    )

    print(f"Inverse transofrms \nshape: {inverse_transforms(ds[:]['data']).shape}\
        \nmin: {inverse_transforms(ds[:]['data']).min()}\
        \nmax: {inverse_transforms(ds[:]['data']).max()}")
    print("-------------------------------------------")
