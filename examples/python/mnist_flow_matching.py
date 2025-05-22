from functools import partial
import torch
import numpy as np
import butane
import matplotlib.pyplot as plt
from extra.adapted_unet_v2 import *

from torchcfm.models.unet import UNetModel

if __name__ == "__main__":

    dev = torch.device("cuda")

    ds = butane.data.Dataset(
        torch.jit.load("data/mnist_data.pt").state_dict()["0"],
        torch.jit.load("data/mnist_targets.pt").state_dict()["0"],
    )

    ds.to(dev)
    test_ds = ds.split(0.9)
    butane.data.ops.drop_to_max_size(ds, 8000)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    class_conditioned = False

    model = butane.nn.UNet2d(
        [1, 28, 28],
        channels=[32, 64, 128],
        self_condition=not class_conditioned,
        attention=True,
        use_film=False,
        n_classes=10 if (class_conditioned) else None,
    ).to(dev)

    fm = butane.nn.ConditionalFlowMatching(0.01).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4)
    epochs = 10
    for epoch in range(epochs):
        sum_loss = 0.0
        n_batches = len(dl)
        for batch in dl:
            optimizer.zero_grad()

            x1 = batch["data"]
            label = batch["targets"]
            x0 = torch.randn_like(x1)
            t = fm.sample_timesteps(x1.size(0))
            x_t, ut = fm.forward(x0, x1, t)

            vt = model(x_t.to(dev), t, label if class_conditioned else x1)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

        avg_loss = sum_loss / n_batches
        print(f"Epoch {epoch} -> Loss: {avg_loss}")

    cond = test_ds[:100]["data"]
    cond_label = test_ds[:100]["targets"]
    sampled = (
        fm.flow(
            model,
            x0=torch.randn(cond.size(0), 1, 28, 28),
            n_timesteps=100,
            condition=cond_label if class_conditioned else cond,
        )
        .squeeze()
        .cpu()
    )
    cond = cond.squeeze().cpu()
    for i in range(sampled.size(0)):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(sampled[i])
        ax[1].imshow(cond[i])
        plt.show()
