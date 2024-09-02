import torch
import numpy as np
from butane.nn import Conv2dBlock, ConvTranspose2dBlock, MLPBlock, ProbabilisticMLPBlock, Quantizer, MLVQVAE
from butane.nn.utils import ModelTrainer

class MNIST(torch.utils.data.Dataset):
     def __init__(self):
         self.data = torch.jit.load("data/mnist_data.pt").state_dict()['0']

     def __len__(self):
         return len(self.data)

     def __getitem__(self,index):
         return self.data[index]

if __name__ == "__main__":
    dev = torch.device('cuda')
    ds = MNIST()
    ds.data = ds.data[:3000].to(dev)
    dl = torch.utils.data.DataLoader(ds, batch_size=64)

    c_enc = Conv2dBlock(
        input_dims = [1, 28, 28],
        channels = [32, 32],
        activation_function = [torch.nn.GELU(), torch.nn.GELU()],
        conv_stride = [[1], [1]],
        conv_bias = [True, True],
        pool_kernels = [[0], [0]],
        normalization = [True, True]
    )

    mlp_enc = MLPBlock(
        input_dims = c_enc.output_size.prod().item(),
        output_dims=100,
        hidden_dims=[64, 64],
        activation_function=[torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Tanh()],
        output_activation=False)

    mlp_dec = MLPBlock(
        input_dims = 100,
        output_dims=c_enc.output_size.prod().item(),
        hidden_dims=[64, 64],
        activation_function=[torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU()],
        output_activation=False)

    c_dec = ConvTranspose2dBlock(
        input_dims = [32, 28, 28],
        channels = [32, 1],
        activation_function = [torch.nn.GELU(), torch.nn.Sigmoid()],
        conv_stride = [[1], [1]],
        conv_bias = [True, True],
        pool_kernels = [[0], [0]],
        output_activation = False,
        normalization = [False, False])

    quantizer = Quantizer (100, 10, dev)
    quantizer.set_beta(1.25)

    encoder = torch.nn.Sequential(c_enc, torch.nn.Flatten(1), mlp_enc)
    decoder = torch.nn.Sequential(mlp_dec, torch.nn.Unflatten(1, (32, 24, 24)), c_dec)
    model = MLVQVAE(encoder, decoder, quantizer).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-03)
    trainer = ModelTrainer(model, dl, optimizer)
    trainer(100, model.loss_fn)
