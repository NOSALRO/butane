import torch
import numpy as np
import matplotlib.pyplot as plt

import butane

@torch.no_grad()
def eval_model(model, diffusion, fpath=None):
    x_T = torch.randn((10, 1, 28, 28))
    generations = diffusion.sample(
        x_T=x_T,
        model=model,
    )
    generations = generations.moveaxis(1, -1).cpu()
    for i in range(generations.size(0)):
        fig, ax = plt.subplots()
        ax.imshow(generations[i])
        if fpath is None:
            plt.show()
        else:
            plt.savefig(f"{fpath}/img_{i}.png")
            plt.close()


if __name__ == "__main__":
    dev = torch.device("cuda")
    ds = butane.data.Dataset(torch.load('data/mnist/mnist_train_data.pt'))
    ds.data = (ds.data * 2)  - 1.
    ds.to(dev)
    butane.data.ops.drop_to_max_size(ds, 5000)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

    model = butane.nn.UNet2d(
        [1, 28, 28],
        channels=[32, 64, 128],
        self_condition=False,
        attention=True,
        use_film=False,
        attention_resolution=[3]
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-04)
    diffusion = butane.nn.Diffusion(
        num_timesteps=1000,
        scheduler='cosine',
        variance_type='fixed_large',
        scale_timesteps=True,
    ).to(dev)
    ema = butane.nn.EMA(model=model, decay=0.9999)

    epochs = 2000
    logger = butane.logger.ModelLogger(".tmp/mnist_ddpm", overwrite=True)
    for epoch in range(epochs):
        sum_loss = 0
        sum_grad_norm = 0
        for batch in dl:
            optimizer.zero_grad()
            x_0 = batch['data']
            t = diffusion.sample_timesteps(x_0.size(0))
            x_t, eps = diffusion(x_0, t)
            out = model(x_t, diffusion.scale_timesteps(t))
            loss = torch.mean((out - eps) ** 2)
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    sum_grad_norm += (p.grad ** 2).sum().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            ema.update()
            sum_loss += loss.item()
        if ((epoch + 1) % 50) == 0:
            logger.checkpoint(epoch + 1, model=model, ema=ema, optimizer=optimizer)
            ema.apply()
            model.eval()
            eval_model(model, diffusion, logger.output_path)
            ema.undo()
            model.train()
        print(f"Epochs {epoch + 1} -> Loss: {sum_loss/len(dl)} Grad Norm: {sum_grad_norm / len(dl)}")

    ema.apply()
    model.eval()
