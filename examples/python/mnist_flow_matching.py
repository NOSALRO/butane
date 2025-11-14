import argparse
from functools import partial
import torch
import numpy as np
import butane
import matplotlib.pyplot as plt

@torch.no_grad()
def eval_model(model, flow_matching, fpath=None):
    x0 = flow_matching.source_distribution().sample((10,))
    test_cond = (torch.jit.load("data/mnist_data.pt").state_dict()["0"] * 2) - 1
    generations = flow_matching.flow(
        model=model,
        x0=x0,
        n_timesteps=100,
        # condition=torch.randn_like(x0),
        condition=test_cond[5000: 5010],
        multiple_gen_per_condition=False,
        keep_record=True,
    )
    generations = generations[0, -1].moveaxis(1, -1).cpu()
    for i in range(generations.size(0)):
        fig, ax = plt.subplots()
        ax.imshow(generations[i])
        if fpath is None:
            plt.show()
        else:
            plt.savefig(f"{fpath}/img_{i}.png")
            plt.close()


if __name__ == "__main__":

    parrser = argparse.ArgumentParser()
    parsers.add_argument("--eval", action='store_true', default=False, dtype=bool)
    parsers.add_argument("--fpath", dtype=str)
    args = parser.parse_args()

    dev = torch.device("cuda")

    ds = butane.data.Dataset(
        (torch.jit.load("data/mnist_data.pt").state_dict()["0"] * 2) - 1,
        torch.jit.load("data/mnist_targets.pt").state_dict()["0"],
        on_demand_device_load=True,
        device='cpu'
    )

    ds.to(dev)
    test_ds = ds.split(0.9)
    butane.data.ops.drop_to_max_size(ds, 40_000)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, pin_memory=False)

    class_conditioned = False

    model = butane.nn.UNet2d(
        [1, 28, 28],
        channels=[32, 64, 128],
        self_condition=True,
        attention_resolution=[3],
        attention=True,
        use_film=True,
        n_classes=None,
        use_scale_shift_norm=True,
        scale_residual=False,
        n_residual_blocks=4,
    ).to(dev)

    fm = butane.nn.ConditionalFlowMatching(0.02).to(dev)
    fm.set_source_distribution(torch.distributions.Independent(torch.distributions.Normal(torch.zeros(1, 28, 28), torch.ones(1, 28, 28)), 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ema = butane.nn.EMA(model=model, decay=0.9999)
    logger = butane.logger.ModelLogger(".tmp/mnist_fm_tb", overwrite=True)

    epochs = 10000
    for epoch in range(epochs):
        sum_loss = 0.0
        sum_grad_norm = 0
        for batch in dl:
            optimizer.zero_grad()
            x1 = batch["data"]
            label = batch["targets"]
            x0 = fm.source_distribution().sample((x1.size(0),)).to(x1.device)
            t = fm.sample_timesteps(x1.size(0))
            x_t, u_t = fm(x0, x1, t)

            v_t = model(x_t.to(dev), t, x1)
            loss = torch.mean((v_t - u_t) ** 2)
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    sum_grad_norm += (p.grad ** 2).sum().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            ema.update()
            sum_loss += loss.item()

        logger.add_stats(loss=sum_loss/len(dl))
        print(f"Epochs {epoch + 1} -> Loss: {sum_loss/len(dl)} Grad Norm: {sum_grad_norm / len(dl)}")
        if ((epoch + 1) % 10) == 0:
            logger.checkpoint(epoch + 1, model=model, optimizer=optimizer, ema=ema)
            ema.apply()
            model.eval()
            eval_model(model, fm, logger.output_path)
            ema.undo()
            model.train()
            logger.checkpoint(epoch+1, model=model, optimizer=optimizer, ema=ema)

    model.eval()
    eval_model(model, flow_matching=fm)
