import torch
import numpy as np
import butane
import matplotlib.pyplot as plt

from torchcfm.models.unet import UNetModel


class TimeDependentModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        c_enc = butane.nn.Conv2dBlock(
            input_dims = [1, 28, 28],
            channels = [32, 32],
            activation_function = [torch.nn.GELU()],
            conv_stride = [1, 1],
            conv_bias = [True, True],
            pool_kernels = [0, 0],
            normalization = [True, False]
        )

        mlp_enc = butane.nn.MLPBlock(
            input_dims = c_enc.output_size.prod().item(),
            output_dims=2,
            hidden_dims=[64, 64],
            activation_function=[torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Tanh()],
            output_activation=False)

        mlp_dec = butane.nn.MLPBlock(
            input_dims = 2,
            output_dims=c_enc.output_size.prod().item(),
            hidden_dims=[64, 64],
            activation_function=[torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU()],
            output_activation=False)

        c_dec = butane.nn.ConvTranspose2dBlock(
            input_dims = [c_enc.output_size[0].item(), c_enc.output_size[1].item(), c_enc.output_size[2].item()],
            channels = [32, 1],
            activation_function = [torch.nn.GELU(), torch.nn.Sigmoid()],
            conv_stride = [1, 1],
            conv_bias = [True, True],
            output_activation = False,
            normalization = [False, False])

        mlp = torch.nn.Sequential(c_enc, torch.nn.Flatten(1), mlp_enc, mlp_dec, butane.nn.Unflatten(1, c_enc.output_size), c_dec)

        self.time_proj = torch.nn.Sequential(
            butane.nn.SinusoidalEmbeddings(2),
            butane.nn.MLPBlock(
                input_dims = 2,
                output_dims = 2,
                hidden_dims=[256, 256],
                activation_function=[torch.nn.GELU()],
                output_activation=False),
        )
        self.mlp = butane.nn.MLPBlock(
            input_dims = 4,
            output_dims=2,
            hidden_dims=[256, 256],
            activation_function=[torch.nn.GELU()],
            output_activation=False)

    def forward(self, x, t, cond=None):
        # return self.mlp(x + self.time_proj(t))
        return self.mlp(torch.hstack([x, self.time_proj(t)]))

if __name__ == "__main__":
    dev = torch.device('cuda')
    ds = butane.data.Dataset(torch.jit.load("data/mnist_data.pt").state_dict()['0'], torch.jit.load("data/mnist_targets.pt").state_dict()['0'])
    ds.to(dev)
    butane.data.ops.drop_to_max_size(ds, 8000)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    # mlp = butane.nn.MLPBlock(
    #     input_dims = 784,
    #     output_dims=784,
    #     hidden_dims=[1024, 512],
    #     activation_function=[torch.nn.GELU(), torch.nn.GELU(), torch.nn.Tanh()],
    #     output_activation=False)
    # model = butane.nn.wrappers.TimeDependent(mlp).to(dev)
    model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(dev)
    fm = butane.nn.ConditionalFlowMatching(0.01).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4)
    epochs=10
    for epoch in range(epochs):
        sum_loss = 0.
        n_batches = len(dl)
        for batch in dl:
            optimizer.zero_grad()

            x1 = batch['data']
            x0 = torch.randn_like(x1)
            t = fm.sample_timesteps(x1.size(0))
            x_t, ut = fm.forward(x0, x1, t)

            vt = model(x_t.to(dev), t)
            # loss = torch.mean((vt - ut) ** 2)
            loss = torch.nn.functional.mse_loss(vt, ut)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

        avg_loss = sum_loss / n_batches
        print(f"Epoch {epoch} -> Loss: {avg_loss}")

    sampled = fm.sample(model, [100, 1, 28, 28], torch.randn, 100)
    sampled = sampled.reshape(-1, 28, 28).cpu()
    for i in sampled:
        plt.imshow(i)
        plt.show()
