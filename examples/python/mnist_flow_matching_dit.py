import argparse
from functools import partial
import torch
import numpy as np
import butane
import matplotlib.pyplot as plt

@torch.no_grad()
def eval_model(model, flow_matching, fpath=None):
    x0 = flow_matching.source_distribution().sample((2, 10,))
    test_cond = (torch.load('data/mnist/mnist_train_data.pt') * 2) - 1
    generations, v = flow_matching.log_likelihood(
        model=model,
        x0=x0,
        n_timesteps=100,
        # condition=torch.randn_like(x0),
        condition=test_cond[5000: 5010],
        multiple_gen_per_condition=True,
        method='euler',
        edm_time_grid=True,
        # keep_record=False,
        # edm_time_grid=True,
        # return_model_outputs=True,
    )
    print(generations.size(), v.size())
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument("--fpath", type=str)
    args = parser.parse_args()

    dev = torch.device("xpu")

    ds = butane.data.Dataset(
        (torch.load('data/mnist/mnist_train_data.pt') * 2) - 1,
        torch.load("data/mnist/mnist_train_targets.pt"),
        on_demand_device_load=True,
        device='cpu'
    )

    ds.to(dev)
    test_ds = ds.split(0.9)
    butane.data.ops.drop_to_max_size(ds, 4_00)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, pin_memory=False)

    class_conditioned = False

    model = butane.nn.DiT2d(
        [1, 28, 28],
        depth=2,
        hidden_dims=384,
        patch_size=4,
        output_channels=None,
        time_embedding_size=64,
        embedding_size=None,
        embedder=None,
        learnable_embeddings=False,
        learnable_input_embeddings=False,
        learnable_condition_embeddings=False,
        adaLN_zero=True,
        cross_attention_condition=False,
        additive_condition=False,
        in_context_condition=False,
        ctx_dims=None,
        ctx_patch_size=None,
        n_classes=None,
    ).to(dev)

    fm = butane.nn.ConditionalFlowMatching(0.02).to(dev)
    fm.set_source_distribution(torch.distributions.Independent(torch.distributions.Normal(torch.zeros(1, 28, 28), torch.ones(1, 28, 28)), 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ema = butane.nn.EMA(model=model, decay=0.9999)
    logger = butane.logger.ModelLogger(".tmp/mnist_fm_dit", overwrite=True)

    epochs = 1
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

        logger.add_stats(loss=sum_loss/len(dl))
        print(f"Epochs {epoch + 1} -> Loss: {sum_loss/len(dl)} Grad Norm: {sum_grad_norm / len(dl)}")
        if ((epoch + 1) % 10) == 0:
            logger.checkpoint(epoch + 1, model=model, optimizer=optimizer, ema=ema)
            ema.enable()
            model.eval()
            eval_model(model, fm, logger.output_path)
            ema.disable()
            model.train()
            logger.checkpoint(epoch+1, model=model, optimizer=optimizer, ema=ema)

    model.eval()
    eval_model(model, flow_matching=fm)
