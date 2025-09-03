from functools import partial
import torch
import numpy as np
import butane
import matplotlib.pyplot as plt

@torch.no_grad()
def eval_model(model, flow_matching, fpath=None):
    x0 = flow_matching.source_distribution().sample((10,))
    generations = flow_matching.flow(
        model=model,
        x0=x0,
        n_timesteps=100,
        condition=torch.randn_like(x0),
        multiple_gen_per_condition=False,
        keep_record=True,
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

    ds = butane.data.Dataset(
        (torch.jit.load("data/mnist_data.pt").state_dict()["0"] * 2) - 1,
        torch.jit.load("data/mnist_targets.pt").state_dict()["0"],
    )

    ds.to(dev)
    test_ds = ds.split(0.9)
    butane.data.ops.drop_to_max_size(ds, 5000)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    class_conditioned = False

    model = butane.nn.UNet2d(
        [1, 28, 28],
        channels=[32, 64, 128],
        self_condition=True,
        attention=True,
        use_film=False,
        n_classes=None,
    ).to(dev)

    fm = butane.nn.ConditionalFlowMatching(0.02).to(dev)
    fm.set_source_distribution(torch.distributions.Independent(torch.distributions.Normal(torch.zeros(1, 28, 28), torch.ones(1, 28, 28)), 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ema = butane.nn.EMA(model=model, decay=0.9999)
    logger = butane.logger.ModelLogger(".tmp/mnist_fm", overwrite=True)

    epochs = 1000
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

            v_t = model(x_t.to(dev), t)
            loss = torch.mean((v_t - u_t) ** 2)
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    sum_grad_norm += (p.grad ** 2).sum().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            ema.update()
            sum_loss += loss.item()

        if ((epoch + 1) % 50) == 0:
            logger.checkpoint(epoch+1, model=model, optimizer=optimizer, ema=ema)
            ema.apply()
            model.eval()
            eval_model(model, fm, logger.output_path)
            ema.undo()
            model.train()
        print(f"Epochs {epoch + 1} -> Loss: {sum_loss/len(dl)} Grad Norm: {sum_grad_norm / len(dl)}")
    # pretrained_dict = torch.load(".tmp/mnist_fm/checkpoint_1000/model.pt")
    # model_dict = model.state_dict()
    # filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(filtered_dict)
    # model.load_state_dict(model_dict)
    # eval_model(model, flow_matching=fm)
